import transformers
from DataCleaning import parse_genres, clean_text, collate_fn
from GenreDataset import GenreDataset
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
from sklearn.metrics import classification_report
import pandas as pd
import os

#ensure that GPU is being used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loading and preprocessing dataset
column_names = ["summary", "genres"]
try:
    df = pd.read_csv("cleaned_genre_info", names=column_names, quoting=1, on_bad_lines='skip', engine="python", delimiter='\t')
except FileNotFoundError:
    print("Dataset not found.")
    exit()

# Parse genres and clean text
df['genres'] = df['genres'].apply(parse_genres)

# normalize genre capitalization
df['genres'] = df['genres'].apply(
    lambda genres: [genre.capitalize() for genre in genres if isinstance(genre, str)]
)

df['summary'] = df['summary'].astype(str).apply(clean_text)

# Remove rows where genres is NaN or contains 'N'
df = df.dropna(subset=['genres'])
df['genres'] = df['genres'].apply(lambda g: [genre for genre in g if str(genre).lower() != 'nan'] if isinstance(g, list) else g)

#create a pie chart with count of summaries in each genre type
create_genre_pie_chart(df)

mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(df['genres'])
print(f"Genres: {mlb.classes_}")

genre_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
training_texts, testing_texts, training_labels, testing_labels = train_test_split(df['summary'].tolist(), labels, test_size=0.2, random_state=50)


training_data = GenreDataset(training_texts, training_labels, genre_tokenizer)
testing_data = GenreDataset(testing_texts, testing_labels, genre_tokenizer)

training_dataloader = DataLoader(training_data, batch_size=4, shuffle=True, collate_fn=collate_fn)
testing_dataloader = DataLoader(testing_data, batch_size=4, shuffle=False, collate_fn=collate_fn)

my_genre_model = DistilBertForMultiLabelClassification(num_labels=len(mlb.classes_))
my_genre_model.to(device)

optimize = AdamW(my_genre_model.parameters(), lr=2e-5)
criteria = nn.BCEWithLogitsLoss()

epoch_amount = 3
best_value_loss = float('inf')
script_dir = os.path.dirname(os.path.abspath(__file__))
model_save_path = os.path.join(script_dir, "distilbert_genre_model.pt")
tokenizer_save_path = os.path.join(script_dir, "distilbert_tokenizer")

# list for creating loss line graph with all epoch info
loss_line_graph_info = []

#list for average validation loss per epoch
validation_losses = []

# training model
for epoch in range(epoch_amount):
    print(f"\nEpoch: {epoch + 1}/{epoch_amount}")
    my_genre_model.train()
    total_loss = 0
    ##loss_line_graph_info.clear()

    for batch_idx, batch in enumerate(training_dataloader):
        optimize.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = my_genre_model(input_ids=input_ids, attention_mask=attention_mask)
        genre_loss = criteria(outputs, labels)  # calculate the loss

        genre_loss.backward()
        optimize.step()

        total_loss += genre_loss.item()

        loss_line_graph_info.append(genre_loss.item())

        if batch_idx % 100 == 0:
            print(f"Training batch {batch_idx} with loss: {genre_loss.item():.4f}")


    average_training_loss = total_loss / len(training_dataloader)
    print(f"Average training loss: {average_training_loss:.4f}")

    my_genre_model.eval()
    value_loss = 0
    all_preds = []
    all_true = []

    with torch.no_grad():
        for my_batch in testing_dataloader:
            input_ids = my_batch['input_ids'].to(device)
            attention_mask = my_batch['attention_mask'].to(device)
            labels = my_batch['labels'].to(device)

            outputs = my_genre_model(input_ids=input_ids, attention_mask=attention_mask)
            genre_loss = criteria(outputs, labels)
            value_loss += genre_loss.item()

            preds = torch.sigmoid(outputs).cpu().numpy() > 0.5
            all_preds.extend(preds)
            all_true.extend(labels.cpu().numpy())

    average_value_loss = value_loss / len(testing_dataloader)
    print(f"Validation loss: {average_value_loss:.4f}")

    if average_value_loss < best_value_loss:
        best_value_loss = average_value_loss
        torch.save({
            'genre_model_state_dict': my_genre_model.state_dict(),
            'genre_classes': mlb.classes_
        }, model_save_path)
        genre_tokenizer.save_pretrained(tokenizer_save_path)
        print(f"Best model was found and saved at epoch {epoch + 1} with value loss {best_value_loss:.4f}")

        # save loss/validation info for best epoch
        best_loss_epoch = loss_line_graph_info.copy()
        validation_losses.append(average_value_loss)
        print("Best_loss_epoch and validation_losses updated.")

print("\nTraining is completed!")

# VISUAL - classification report to verify model without GUI
print("\nClassification Report Information: ")
print(classification_report(all_true, all_preds, target_names=mlb.classes_))

create_classification_image(all_true, all_preds, mlb)

# Get predictions on the testing data
true_labels, predicted_labels = get_predictions(my_genre_model, testing_dataloader, device)

# Recompute embeddings for the current version of df
print(f"Number of summaries for BERTopic: {len(df)}")
embeddings = compute_all_embeddings(df)

# Check alignment
if embeddings.shape[0] != len(df):
    raise ValueError(f"Mismatch: embeddings ({embeddings.shape[0]}) != df rows ({len(df)})")

# Run BERTopic
df, topic_model = run_bertopic_on_summaries(df, embedding_model=embeddings)



#wordcloud
unique_genres = df['genres'].explode().dropna().unique()
for genre in unique_genres:
    plot_wordcloud_for_genre(topic_model, df, genre)

#line graph
average_line_graph(best_loss_epoch, validation_losses)

#heatmap
create_heatmap(all_preds, all_true, mlb)