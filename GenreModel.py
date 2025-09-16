import transformers
transformers.logging.set_verbosity_error()
import os
import sys
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

    #defines as forward pass
    def forward(self, input_ids, attention_mask):
        #unpacks the input tensor shape
        batch_size, num_chunks, max_len = input_ids.shape

        #reshape tensors to merge batch and chunk dimensions for independent processing
        input_ids = input_ids.view(batch_size * num_chunks, max_len)
        attention_mask = attention_mask.view(batch_size * num_chunks, max_len)

        #passes chunk tokens into DistilBERT model
        output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)

        #gets embedding info for each token in the input sequence
        hidden_state = output.last_hidden_state
        pooled_output = hidden_state[:, 0]

        #feedforward layer
        x = self.pre_classifier(pooled_output)

        #helps with learning complex labelling relationships
        x = nn.ReLU()(x)

        #regularization
        x = self.dropout(x)

        #generate logits
        logits = self.classifier(x)

        #reshapes logits in order to restore chunk dimension
        logits = logits.view(batch_size, num_chunks, -1)

        #aggregates chunk logits by averages across chunks of each sample
        aggregation_logits = torch.mean(logits, dim=1)

        #returns final logits
        return aggregation_logits

def load_model_and_tokenizer():
    #ensures model is run of GPU when possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #get directories needed
    if getattr(sys, 'frozen', False):
        #running in a bundle
        this_script_directory = sys._MEIPASS
    else:
        #running in Python
        this_script_directory = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(this_script_directory, "distilbert_genre_model.pt")
    tokenizer_path = os.path.join(this_script_directory, "distilbert_tokenizer")

    #if model or tokenizer is not found
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file NOT found at: {model_path}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer folder NOT found at: {tokenizer_path}")

    #save PyTorch checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    #check for genres
    genre_classes = checkpoint.get('genre_classes')
    if genre_classes is None:
        genre_classes = checkpoint.get('classes')
    if genre_classes is None:
        raise KeyError("No genre_classes or classes key found in the checkpoint.")

    #check if genre_classes has a 'tolist' method - if it does, call it and replace genre_classes with the result
    if hasattr(genre_classes, 'tolist'):
        genre_classes = genre_classes.tolist()

    #convert bytes to strings if needed
    genre_classes = [gc.decode() if isinstance(gc, bytes) else gc for gc in genre_classes]

    #count amount of genres that exist
    genre_labels = len(genre_classes)

    #load new instance of model
    genre_model = DistilBertForMultiLabelClassification(genre_labels)

    #extract previously saved model weights from checkpoint
    state_dict = checkpoint.get('genre_model_state_dict') or checkpoint.get('model_state_dict')

    #load model weights into new model
    genre_model.load_state_dict(state_dict)

    #run model on GPU
    genre_model.to(device)

    #set model to evaluation mode
    genre_model.eval()

    #loads pretrained tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)

    #return the model, tokenizer, list of genres, and the device that computations were done on
    return genre_model, tokenizer, genre_classes, device