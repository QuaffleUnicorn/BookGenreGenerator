from transformers import DistilBertTokenizerFast, DistilBertModel
import torch
import numpy as np
import os

#general DistilBERT Hugging Face documentation used for development: https://huggingface.co/transformers/v2.9.1/model_doc/distilbert.html

#initialize the genre model and tokenizer globally
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def get_embeddings_batch(summary_texts, batch_size=16):
    #initialize list
    these_embeddings = []

    #loop through all batches and breaks the summaries into chunks
    for i in range(0, len(summary_texts), batch_size):
        genre_batch = summary_texts[i:i + batch_size]

        #tokenize batch per hugging face tokenizer
        genre_inputs = tokenizer(genre_batch, return_tensors="pt", padding=True, truncation=True, max_length=512)

        #moves input to GPU
        genre_inputs = {k: v.to(device) for k, v in genre_inputs.items()}

        #run with no gradient tracking
        with torch.no_grad():
            genre_output = model(**genre_inputs)

            #extract embeddings
            cls_embeddings = genre_output.last_hidden_state[:, 0, :]

            #store embeddings to CPU
            these_embeddings.append(cls_embeddings.cpu().numpy())

    return np.vstack(these_embeddings)


def compute_all_embeddings(df, cache_path="embeddings.npy"):
    #checks for previously cached embeddings
    if os.path.exists(cache_path):
        print("Loading previously cached embeddings...")
        return np.load(cache_path)

    #if no embeddings are found, generate embeddings
    print("Generating embeddings for clustering...")
    summaries = df["summary"].astype(str).tolist()
    genre_model_embeddings = get_embeddings_batch(summaries, batch_size=16)
    np.save(cache_path, genre_model_embeddings)
    return genre_model_embeddings