import transformers
transformers.logging.set_verbosity_error()
import os
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel



class DistilBertForMultiLabelClassification(nn.Module):
    def __init__(self, num_labels):
        super(DistilBertForMultiLabelClassification, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = nn.Linear(self.distilbert.config.hidden_size, 256)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(256, num_labels)

    def forward(self, input_ids, attention_mask):
        batch_size, num_chunks, max_len = input_ids.shape

        input_ids = input_ids.view(batch_size * num_chunks, max_len)
        attention_mask = attention_mask.view(batch_size * num_chunks, max_len)

        output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output.last_hidden_state
        pooled_output = hidden_state[:, 0]

        x = self.pre_classifier(pooled_output)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        logits = self.classifier(x)

        logits = logits.view(batch_size, num_chunks, -1)

        aggregation_logits = torch.mean(logits, dim=1)

        return aggregation_logits

def load_model_and_tokenizer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "distilbert_genre_model.pt")
    tokenizer_path = os.path.join(script_dir, "distilbert_tokenizer")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file NOT found at: {model_path}")

    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer folder NOT found at: {tokenizer_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    genre_classes = checkpoint.get('genre_classes')
    if genre_classes is None:
        genre_classes = checkpoint.get('classes')
    if genre_classes is None:
        raise KeyError("No genre_classes or classes key found in the checkpoint.")

    if hasattr(genre_classes, 'tolist'):
        genre_classes = genre_classes.tolist()

    genre_classes = [gc.decode() if isinstance(gc, bytes) else gc for gc in genre_classes]
    num_labels = len(genre_classes)

    model = DistilBertForMultiLabelClassification(num_labels)
    state_dict = checkpoint.get('genre_model_state_dict') or checkpoint.get('model_state_dict')
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)

    return model, tokenizer, genre_classes, device