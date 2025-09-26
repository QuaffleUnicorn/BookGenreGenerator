import transformers
transformers.logging.set_verbosity_error()
import torch
import ast
from nltk.corpus import stopwords

def clean_slash_from_genre(genre):
    return genre.strip().strip('\\').strip('\'"')

def parse_genres(genre_string):
    #if string is empty, return an empty list
    if not genre_string or str(genre_string).strip() == "":
        return []
    #tries to evaluate string into a python object for formatting
    try:
        genres_list = ast.literal_eval(genre_string)
        if isinstance(genres_list, str):
            genres_list = [genres_list]

    #if it fails, split by commas
    except Exception:
        genres_list = [g.strip() for g in str(genre_string).split(',')]

    #return cleaned genres
    return [clean_slash_from_genre(g) for g in genres_list if g and g.strip()]

#stop words to remove from summaries
stop_words = set(stopwords.words('english'))

def clean_text(text):
    #if text isn't a string, return an empty string
    if not isinstance(text, str):
        return ""

    #convert text to all lowercase
    words = text.lower().split()

    #remove any stop words
    filtered = [word for word in words if word not in stop_words]

    #recombine and return words
    return ' '.join(filtered)


# collate function
def collate_fn(my_batch):
    #3 keys from item in my_batch
    my_batch_ids = [item['input_ids'] for item in my_batch]
    my_batch_mask = [item['attention_mask'] for item in my_batch]
    my_batch_labels = [item['labels'] for item in my_batch]

    #find the maximum number of chunks
    max_chunks = max(x.shape[0] for x in my_batch_ids)

    #lists for padded input_ids and attention_masks
    padded_id = []
    padded_att_mask = []

    #
    for input_ids, my_mask in zip(my_batch_ids, my_batch_mask):
        #find how many chunks are missing from the max
        pad_chunks = max_chunks - input_ids.shape[0]

        #pad with zeros
        if pad_chunks > 0:
            pad_shape = (pad_chunks, input_ids.shape[1])
            input_ids = torch.cat([input_ids, torch.zeros(pad_shape, dtype=torch.long)], dim=0)
            my_mask = torch.cat([my_mask, torch.zeros(pad_shape, dtype=torch.long)], dim=0)

        #add batch dimension
        padded_id.append(input_ids.unsqueeze(0))
        padded_att_mask.append(my_mask.unsqueeze(0))

    #combine into batch tensors
    input_ids_batch = torch.cat(padded_id, dim=0)
    attention_mask_batch = torch.cat(padded_att_mask, dim=0)
    labels_batch = torch.stack(my_batch_labels)

    #return a single dictionary with the padded batch that can be run through the model
    return {
        'input_ids': input_ids_batch,
        'attention_mask': attention_mask_batch,
        'labels': labels_batch
    }