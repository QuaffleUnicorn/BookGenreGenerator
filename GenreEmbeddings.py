from transformers import DistilBertTokenizerFast, DistilBertModel
import torch
import numpy as np
import os

# Initialize model and tokenizer globally
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def get_embeddings_batch(texts, batch_size=16):
    """
    Compute [CLS] embeddings for a list of texts using batching.

    Args:
        texts (List[str]): List of summaries/texts.
        batch_size (int): Number of texts to embed per batch.

    Returns:
        np.ndarray: 2D array of shape (num_texts, hidden_dim)
    """
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            all_embeddings.append(cls_embeddings.cpu().numpy())
    return np.vstack(all_embeddings)


def compute_all_embeddings(df, cache_path="embeddings.npy"):
    if os.path.exists(cache_path):
        print("Loading cached embeddings...")
        return np.load(cache_path)

    print("Generating embeddings for clustering...")
    summaries = df["summary"].astype(str).tolist()
    embeddings = get_embeddings_batch(summaries, batch_size=16)
    np.save(cache_path, embeddings)
    return embeddings