import transformers
from DataCleaning import parse_genres, clean_text, collate_fn
from BookDataset import BookDataset
from GenreEmbeddings import compute_all_embeddings
from GenreModel import DistilBertForMultiLabelClassification
from ImageCreation import average_line_graph, plot_wordcloud_for_genre, create_heatmap, create_classification_image, \
    create_genre_pie_chart, run_bertopic_on_summaries
from Prediction import get_predictions
transformers.logging.set_verbosity_error()
import torch
from torch.utils.data import DataLoader
from torch import nn
from transformers import DistilBertTokenizer
from torch.optim import AdamW
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from sklearn.metrics import classification_report

#ensure that GPU is being used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#loading dataset
column_names = ["summary", "genres"]
try:
    df = pd.read_csv("cleaned_genre_info", names=column_names, quoting=1, on_bad_lines='skip', engine="python", delimiter='\t')
except FileNotFoundError:
    print("Dataset not found.")
    exit()

#parse genres and clean text
df['genres'] = df['genres'].apply(parse_genres)

#normalize genre capitalization
df['genres'] = df['genres'].apply(lambda genres: [genre.capitalize() for genre in genres if isinstance(genre, str)])
df['summary'] = df['summary'].astype(str).apply(clean_text)

# Remove rows where genres is NaN or contains 'N'
df = df.dropna(subset=['genres'])
df['genres'] = df['genres'].apply(lambda g: [genre for genre in g if str(genre).lower() != 'nan'] if isinstance(g, list) else g)

#create a pie chart with count of summaries in each genre type
create_genre_pie_chart(df)

#label genres
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(df['genres'])
print(f"Genres: {mlb.classes_}")

#seperate dataset into training and testing sets
genre_model_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
training_summaries, testing_summaries, training_genre_labels, testing_genre_labels = train_test_split(df['summary'].tolist(), labels, test_size=0.2, random_state=50)


#load into BookDataset
training_model_data = BookDataset(training_summaries, training_genre_labels, genre_model_tokenizer)
testing_model_data = BookDataset(testing_summaries, testing_genre_labels, genre_model_tokenizer)

#load data into DataLoader
training_dataloader = DataLoader(training_model_data, batch_size=4, shuffle=True, collate_fn=collate_fn)
testing_dataloader = DataLoader(testing_model_data, batch_size=4, shuffle=False, collate_fn=collate_fn)

#create instance of model class, feeding in multi-hot vectors that represent the unique genres
my_genre_model = DistilBertForMultiLabelClassification(num_labels=len(mlb.classes_))

my_genre_model.to(device)

#AdamW optimizer
optimize = AdamW(my_genre_model.parameters(), lr=2e-5)

#loss function for multi-label classification
genre_model_criteria = nn.BCEWithLogitsLoss()

#each epoch is 1 full pass through the dataset for training
epoch_amount = 3

#set initial value for best_value_loss, keeps track of best performing model (whichever has the lowest validation loss)
best_value_loss = float('inf')

#get current script directory and save best model to this directory location
this_script_dir = os.path.dirname(os.path.abspath(__file__))
genre_model_path = os.path.join(this_script_dir, "distilbert_genre_model.pt")
tokenizer_path = os.path.join(this_script_dir, "distilbert_tokenizer")

# list for creating loss line graph with all epoch info
loss_line_graph_info = []

#list for average validation loss per epoch
validation_losses = []
validation_losses_best_epoch = []

# training model
for epoch in range(epoch_amount):
    print(f"\nEpoch: {epoch + 1}/{epoch_amount}")
    my_genre_model.train()
    total_loss = 0
    ##loss_line_graph_info.clear()

    #for each batch number, batch is a dictionary containing inputs and labels
    for batch_number, batch in enumerate(training_dataloader):
        #clear previous gradients
        optimize.zero_grad()

        #moves batched inputs and labels into the device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        #runs a forward pass through current batch and outputs raw logits for predicted scores
        outputs = my_genre_model(input_ids=input_ids, attention_mask=attention_mask)

        #computes loss between the predictions and true labels
        genre_loss = genre_model_criteria(outputs, labels)

        #backward pass for backpropagation
        genre_loss.backward()

        #updates model weights using optimizer
        optimize.step()

        #loss value added for later reporting
        total_loss += genre_loss.item()
        loss_line_graph_info.append(genre_loss.item())

        #prints an update to the console every 100 batches with loss amount of monitoring progress during training
        if batch_number % 100 == 0:
            print(f"Training batch # {batch_number} with loss: {genre_loss.item():.4f}")


    #after training loop, calculate average training loss per epoch and store for loss analysis
    average_training_loss = total_loss / len(training_dataloader)
    print(f"\nAverage training loss: {average_training_loss:.4f}")
    validation_losses.append(average_training_loss)

    #sets model to evaluation mode for validation testing, and initialize variables for validation testing
    my_genre_model.eval()
    value_loss = 0
    all_genre_predictions = []
    all_true_predictions = []

    #prevents calculating gradients during evaluation and stops accidental updating of model weights
    with torch.no_grad():
        #takes batches from your testing set in dataloader
        for my_batch in testing_dataloader:
            #move batch data to device (aka GPU)
            input_ids = my_batch['input_ids'].to(device)
            attention_mask = my_batch['attention_mask'].to(device)
            labels = my_batch['labels'].to(device)

            #forward pass through batch
            outputs = my_genre_model(input_ids=input_ids, attention_mask=attention_mask)

            #calculate loss per batch and add to running loss total
            genre_loss = genre_model_criteria(outputs, labels)
            value_loss += genre_loss.item()

            #convert previous logits to prediction percentages, applies as positive a genre if over 50% certain
            genre_predictions = torch.sigmoid(outputs).cpu().numpy() > 0.5

            #add predictions to all_predictions list and true labels to all_true_predictions list
            all_genre_predictions.extend(genre_predictions)
            all_true_predictions.extend(labels.cpu().numpy())

    #calculates average validation loss per epoch
    average_value_loss = value_loss / len(testing_dataloader)
    print(f"\nValidation loss: {average_value_loss:.4f}")

    #checks if current epoch of model is the most accurate on unseen data when compared to previous epochs
    if average_value_loss < best_value_loss:
        #if this newest epoch is the best, save the new best model and tokenizer data
        best_value_loss = average_value_loss
        torch.save({
            'genre_model_state_dict': my_genre_model.state_dict(),
            'genre_classes': mlb.classes_
        }, genre_model_path)
        genre_model_tokenizer.save_pretrained(tokenizer_path)

        #indicates to console that new best epoch was found
        print(f"\nBest model was saved at epoch # {epoch + 1} with loss {best_value_loss:.4f}")

        #save loss/validation info for new best epoch
        best_loss_epoch = loss_line_graph_info.copy()
        validation_losses_best_epoch = validation_losses.copy()
        print("Best_loss_epoch and validation_losses updated.")

#indicates to console that model training is completed and visualizations for the application are being created from the model info
print("\nTraining is completed!")
print("Please wait as application visualizations are created...")

#get predictions on the testing data
true_labels, predicted_labels = get_predictions(my_genre_model, testing_dataloader, device)

#classification report to verify model within console
#print("\nClassification Report Information: ")
#print(classification_report(all_true_predictions, all_genre_predictions, target_names=mlb.classes_))

#create classification report image
create_classification_image(all_true_predictions, all_genre_predictions, mlb)

#compute embeddings for the current version of df
print(f"Computing embeddings for {len(predicted_labels)} genres...")
embeddings = compute_all_embeddings(df)

#check alignment of embeddings
if embeddings.shape[0] != len(df):
    raise ValueError(f"Mismatch: embeddings ({embeddings.shape[0]}) != df rows ({len(df)})")

#run bertopic on embeddings
print(f"Running BERTopic on embeddings...")
df, topic_model = run_bertopic_on_summaries(df, embedding_model=embeddings)

#create wordclouds for all unique genres
print(f"\nCreating wordclouds for {len(df)} genres...")
unique_genres = df['genres'].explode().dropna().unique()
for genre in unique_genres:
    plot_wordcloud_for_genre(topic_model, df, genre)

#line graph of training and testing loss info
print(f"\nCreating line graph of training and validation information...")
average_line_graph(best_loss_epoch, validation_losses_best_epoch)

#heatmap of data
print(f"\nCreating a heatmap of training and validation information...")
create_heatmap(all_genre_predictions, all_true_predictions, mlb)

print(f"\nTraining and image creation are both complete! Please see 'GenreGUI.py' to run application. Thank you!")