import transformers
transformers.logging.set_verbosity_error()
import torch
import numpy as np

#function gets predictions from a dataloader
def get_predictions(model, dataloader, device, threshold=0.50):

    #set model to evaluation mode and initialize 2 lists
    model.eval()
    all_predictions = []
    all_true_info = []

    with torch.no_grad():
        #loop through each batch from the dataloader and moves info to GPU
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            #pass through model to get raw logits
            summary_output = model(input_ids=input_ids, attention_mask=attention_mask)

            #get each genre's probability for the summary
            this_probability = torch.sigmoid(summary_output)

            #applies threshold of >50% as true
            model_predictions = (this_probability > threshold).cpu().numpy()

            #convert truth labels to NumPy
            true = labels.cpu().numpy()

            #adds batch predictions/labels to lists
            all_predictions.append(model_predictions)
            all_true_info.append(true)

    #stacks results into NumPy array
    all_predictions = np.vstack(all_predictions)
    all_true_info = np.vstack(all_true_info)

    #returns actual and predicted labels
    return all_true_info, all_predictions