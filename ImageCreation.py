import numpy as np
from GenreEmbeddings import compute_all_embeddings
from sklearn.metrics import confusion_matrix
from bertopic import BERTopic
import seaborn as sns
from wordcloud import WordCloud
import random
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import pandas as pd
import os

#create a folder for graph images to be located
GRAPHS_LOCAL = "graph_plot_images"
os.makedirs(GRAPHS_LOCAL, exist_ok=True)

#create a folder for wordcloud images to be located
WORDCLOUD_IMAGE_LOCAL = "wordcloud_images"
os.makedirs(WORDCLOUD_IMAGE_LOCAL, exist_ok=True)


#created loss line graph
def average_line_graph(best_loss_epoch, validation_losses):
    #verify loss is available
    if best_loss_epoch:
        # Compute average loss over every 100 batches
        averaged_batch_losses = []
        for i in range(0, len(best_loss_epoch), 100):
            chunk = best_loss_epoch[i:i + 100]
            average_loss = sum(chunk) / len(chunk)
            averaged_batch_losses.append(average_loss)

        #generate x-axis values
        batch_for_x = list(range(1, len(averaged_batch_losses) + 1))

        #generate validation loss on same graph for each corresponding position in the 100-batch groups
        epoch_for_x = [(i + 1) * (len(batch_for_x) / len(validation_losses)) for i in range(len(validation_losses))]

        #plot graph
        plt.figure(figsize=(12, 6))
        plt.plot(batch_for_x, averaged_batch_losses, marker='o', linestyle='-', label='Avg Training Loss (per 100 batches)')
        plt.plot(epoch_for_x, validation_losses, marker='s', linestyle='--', color='orange', label='Average Validation Loss (per epoch)')

        #format graph
        plt.xlabel('Batch Group / Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save the graph
        graph_path = os.path.join(GRAPHS_LOCAL, "point_loss_chart.png")
        plt.savefig(graph_path)
        print(f"\n{os.path.basename(graph_path)} saved!")
    else:
        #skips plotting if there is no info
        print("No loss info to plot.")




#assign each summary topic labels
def run_bertopic_on_summaries(df, embedding_model=None):
    print("Running BERTopic on summaries...")

    #convert summaries to a list
    summary_text = df["summary"].astype(str).tolist()

    #use precomputed embeddings if available, compute embeddings if there are none saved
    if embedding_model is None:
        embeddings = compute_all_embeddings(df)
    else:
        embeddings = embedding_model

    #running BERTopic model, returning summary genres and their probabilities
    genre_topic_model = BERTopic(verbose=True)
    topics, probs = genre_topic_model.fit_transform(summary_text, embeddings)

    #add topic to DataFrame
    df["topic"] = topics

    #identifies amount of unique topics
    print(f"Identified {len(set(topics))} topics.")

    #return results
    return df, genre_topic_model


# Define genre to color palette mapping
GENRE_COLORS = {
    'sci_fi': ['#00FFE0','#006D8F','#9400D3','#101820','#7FFFD4'],
    'romance': ['#FFB6C1','#FF1493','#DB7093','#FFF0F5','#C71585'],
    'fantasy': ['#7B68EE','#6A5ACD','#228B22','#D2691E','#FFD700'],
    'horror': ['#8B0000','#FF4500','#4B0082','#2F4F4F','#6B0000'],
    'adventure': ['#FFA500','#228B22','#1E90FF','#8B4513','#FFD700'],
    'historical_fiction': ['#8B4513','#A0522D','#C0C0C0','#708090','#F5DEB3'],
    'mystery': ['#2F4F4F','#800000','#000000','#4682B4','#A9A9A9'],
    'non_fiction': ['#4682B4','#2E8B57','#D2B48C','#708090','#FFFFFF'],
    'children___ya': ['#FF69B4','#9370DB','#40E0D0','#FFD700','#00BFFF']
}


#generate a random color for wordcloud
def random_color_func(wordcloud_colors):
    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        return random.choice(wordcloud_colors)
    return color_func

def plot_wordcloud_for_genre(topic_model, df, genre):
    # Filter summaries that include this genre
    genre_df = df[df['genres'].apply(lambda x: genre in x)]

    if genre_df.empty:
        print(f"No data for genre: {genre}")
        return

    #get topic frequencies
    topic_counts = genre_df['topic'].value_counts()

    #aggregate keywords and their weights weighted by topic frequency
    keyword_weights = {}

    for topic_id, count in topic_counts.items():
        if topic_id == -1:
            continue
        keywords = topic_model.get_topic(topic_id)
        for word, weight in keywords:
            keyword_weights[word] = keyword_weights.get(word, 0) + weight * count

    if not keyword_weights:
        print(f"No keywords found for genre: {genre}")
        return

    # Boost the genre name in ALL CAPS (with spaces, no underscores)
    genre_display = genre.strip().upper()
    max_weight = max(keyword_weights.values())
    keyword_weights[genre_display] = max_weight * 2

    #color palette if available or default grayscale
    colors = GENRE_COLORS.get(genre.lower(), ['#333333'])

    #create wordcloud
    wc = WordCloud(
        width=900,
        height=500,
        background_color='white',
        collocations=False
    )
    wc.generate_from_frequencies(keyword_weights)

    #color with custom random color function
    wc_recolored = wc.recolor(color_func=random_color_func(colors))

    #plot
    plt.figure(figsize=(8, 6))
    plt.imshow(wc_recolored, interpolation='bilinear')
    plt.axis('off')

    #eliminate whitespace
    safe_genre_name = "".join(c if c.isalnum() else "_" for c in genre)
    plt.savefig(
        os.path.join(WORDCLOUD_IMAGE_LOCAL, f"wordcloud_{safe_genre_name}.png"),
        bbox_inches='tight',
        pad_inches=0
    )

    print(f"Saved word cloud for: {genre_display} as wordcloud_{safe_genre_name}.png")


#heatmap
def create_heatmap(all_predictions, all_true_info, mlb):
    #convert multilabel to single-label arrays
    y_prediction_info = np.argmax(all_predictions, axis=1)
    y_true_info = np.argmax(all_true_info, axis=1)

    #create confusion matrix
    cm = confusion_matrix(y_true_info, y_prediction_info)

    #plot
    plt.figure(figsize=(8, 7))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=mlb.classes_, yticklabels=mlb.classes_)
    plt.xlabel('Predicted Genre')
    plt.ylabel('Verified Genre')
    plt.title('Confusion Matrix Heatmap')
    plt.tight_layout()

    #save heatmap
    plt.savefig(os.path.join(GRAPHS_LOCAL, f"confusion_matrix_heatmap.png"), bbox_inches='tight', pad_inches=0)

    print("\nconfusion_matrix_heatmap.png saved!")

#create image with classification info
def create_classification_image(all_true_genres, all_predictions, mlb, max_width=500):

    genre_model_report = classification_report(all_true_genres, all_predictions, target_names=mlb.classes_, output_dict=True)
    df_report = pd.DataFrame(genre_model_report).transpose()
    df_report = df_report[['precision', 'recall', 'f1-score', 'support']]
    df_report = df_report.round(2)

    #dynamic height
    row_height = 0.3
    fig_width = 5
    fig_height = len(df_report) * row_height
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')

    table = ax.table(
        cellText=df_report.values,
        colLabels=[col.upper() for col in df_report.columns],
        rowLabels=[str(label).upper() for label in df_report.index],
        cellLoc='center',
        loc='center'
    )
    #make column headers bold
    for key, cell in table.get_celld().items():
        row, col = key
        if row == 0 or col == -1:
            cell.set_text_props(weight='bold')

    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(0.8, 0.8)
    plt.tight_layout(pad=0.2)

    # save dimensions for UI
    graph_path = os.path.join(GRAPHS_LOCAL, "classification_report.png")
    plt.savefig(graph_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\n{os.path.basename(graph_path)} saved!")



def create_genre_pie_chart(df, min_genre_percent=5):
    #explode and count genres
    all_genres = df['genres'].dropna().explode()
    genre_counts = Counter(all_genres)
    total = sum(genre_counts.values())

    #group genres with fewer examples into 'other' on visualization
    primary_genres = {}
    other_genres = 0
    for my_genre, count in genre_counts.items():
        genre_percent = (count / total) * 100
        if genre_percent >= min_genre_percent:
            primary_genres[my_genre] = count
        else:
            other_genres += count

    if other_genres > 0:
        primary_genres['Other - Genres <5%'] = other_genres

    #prepare data for visualization
    labels = list(primary_genres.keys())
    sizes = list(primary_genres.values())

    #plot
    plt.figure(figsize=(12, 8))
    wedges, texts, autotexts = plt.pie(
        sizes,
        startangle=140,
        autopct='%1.1f%%',
        wedgeprops={'edgecolor': 'white'}
    )

    #add legend with labels
    plt.legend(wedges, labels, title="Genres", loc="center left", bbox_to_anchor=(1.05, 0.5))

    plt.title('Genre Distribution', fontsize=16)
    plt.axis('equal')
    plt.tight_layout()
    graph_path = os.path.join(GRAPHS_LOCAL, "genre_pie_chart.png")
    plt.savefig(graph_path, bbox_inches='tight')

    #print to console
    print(f"\n{os.path.basename(graph_path)} saved!")
