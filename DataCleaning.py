import transformers
transformers.logging.set_verbosity_error()
import torch
import ast
from nltk.corpus import stopwords

def clean_genre(genre):
    return genre.strip().strip('\\').strip('\'"')

def parse_genres(genres_as_string):
    if not genres_as_string or str(genres_as_string).strip() == "":
        return []
    try:
        genres_list = ast.literal_eval(genres_as_string)
        if isinstance(genres_list, str):
            genres_list = [genres_list]
    except Exception:
        genres_list = [g.strip() for g in str(genres_as_string).split(',')]
    return [clean_genre(g) for g in genres_list if g and g.strip()]


stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    words = text.lower().split()
    filtered = [word for word in words if word not in stop_words]
    return ' '.join(filtered)


# collate function
def collate_fn(my_batch):
    my_batch_ids = [item['input_ids'] for item in my_batch]
    my_batch_mask = [item['attention_mask'] for item in my_batch]
    my_batch_labels = [item['labels'] for item in my_batch]

    max_chunks = max(x.shape[0] for x in my_batch_ids)

    padded_id = []
    padded_att_mask = []

    for input_ids, my_mask in zip(my_batch_ids, my_batch_mask):
        pad_chunks = max_chunks - input_ids.shape[0]
        if pad_chunks > 0:
            pad_shape = (pad_chunks, input_ids.shape[1])
            input_ids = torch.cat([input_ids, torch.zeros(pad_shape, dtype=torch.long)], dim=0)
            my_mask = torch.cat([my_mask, torch.zeros(pad_shape, dtype=torch.long)], dim=0)
        padded_id.append(input_ids.unsqueeze(0))
        padded_att_mask.append(my_mask.unsqueeze(0))

    input_ids_batch = torch.cat(padded_id, dim=0)
    attention_mask_batch = torch.cat(padded_att_mask, dim=0)
    labels_batch = torch.stack(my_batch_labels)

    return {
        'input_ids': input_ids_batch,
        'attention_mask': attention_mask_batch,
        'labels': labels_batch
    }