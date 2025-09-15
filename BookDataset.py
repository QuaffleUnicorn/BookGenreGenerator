import transformers
transformers.logging.set_verbosity_error()
import torch
from torch.utils.data import Dataset

#created class to work with DataLoader
class BookDataset(Dataset):
    def __init__(self, summaries, labels, tokenizer, max_length=512):
        self.summaries = summaries
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    #returns dataset length
    def __len__(self):
        return len(self.summaries)

    #returns the tokenized and chunked summary and label
    def __getitem__(self, idx):
        #summary gets tokenized
        summary = str(self.summaries[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            summary,
            add_special_tokens=True,
            return_tensors='pt',
            truncation=False, #manual chunking
            padding=False
        )

        #obtain token IDs and attention mask
        tensor_ids = encoding['input_ids'][0]
        attention_mask = encoding['attention_mask'][0]

        #loop through tokenized input by chunk size of max_length
        chunk_input_id = []
        chunk_attention_mask = []

        for i in range(0, len(tensor_ids), self.max_length):
            chunk_id = tensor_ids[i:i+self.max_length]
            chunk_mask = attention_mask[i:i+self.max_length]

            #pad chunks that are less than max_length
            padding = self.max_length - len(chunk_id)
            if padding > 0:
                chunk_id = torch.cat([chunk_id, torch.zeros(padding, dtype=torch.long)])
                chunk_mask = torch.cat([chunk_mask, torch.zeros(padding, dtype=torch.long)])

            #collect chunks in a way that is compatible for batching
            chunk_input_id.append(chunk_id.unsqueeze(0))
            chunk_attention_mask.append(chunk_mask.unsqueeze(0))

        #combine final tensors
        input_ids_tensor = torch.cat(chunk_input_id, dim=0)
        attention_mask_tensor = torch.cat(chunk_attention_mask, dim=0)

        return {
            'input_ids': input_ids_tensor,
            'attention_mask': attention_mask_tensor,
            'labels': torch.tensor(label, dtype=torch.float)
        }