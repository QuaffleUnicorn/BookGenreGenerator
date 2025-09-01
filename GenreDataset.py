import transformers
transformers.logging.set_verbosity_error()
import torch
from torch.utils.data import Dataset

class GenreDataset(Dataset):
    def __init__(self, summaries, labels, tokenizer, max_length=512):
        self.summaries = summaries
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.summaries)

    def __getitem__(self, idx):
        summary = str(self.summaries[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            summary,
            add_special_tokens=True,
            return_tensors='pt',
            truncation=False,
            padding=False
        )

        tensor_ids = encoding['input_ids'][0]
        attention_mask = encoding['attention_mask'][0]

        chunk_input_id = []
        chunk_attention_mask = []

        for i in range(0, len(tensor_ids), self.max_length):
            chunk_id = tensor_ids[i:i+self.max_length]
            chunk_mask = attention_mask[i:i+self.max_length]

            padding = self.max_length - len(chunk_id)
            if padding > 0:
                chunk_id = torch.cat([chunk_id, torch.zeros(padding, dtype=torch.long)])
                chunk_mask = torch.cat([chunk_mask, torch.zeros(padding, dtype=torch.long)])

            chunk_input_id.append(chunk_id.unsqueeze(0))
            chunk_attention_mask.append(chunk_mask.unsqueeze(0))

        input_ids_tensor = torch.cat(chunk_input_id, dim=0)
        attention_mask_tensor = torch.cat(chunk_attention_mask, dim=0)

        return {
            'input_ids': input_ids_tensor,
            'attention_mask': attention_mask_tensor,
            'labels': torch.tensor(label, dtype=torch.float)
        }