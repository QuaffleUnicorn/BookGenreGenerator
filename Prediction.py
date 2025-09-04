import transformers
transformers.logging.set_verbosity_error()
import torch
import numpy as np

# Function to get predictions from a dataloader // set threshold to .55 from .5 for testing purposes
def get_predictions(model, dataloader, device, threshold=0.55):
    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).cpu().numpy()
            true = labels.cpu().numpy()

            all_preds.append(preds)
            all_true.append(true)

    all_preds = np.vstack(all_preds)
    all_true = np.vstack(all_true)
    return all_true, all_preds