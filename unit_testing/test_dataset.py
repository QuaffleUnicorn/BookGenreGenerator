import torch
from BookDataset import BookDataset
from transformers import DistilBertTokenizer

def test_genre_dataset_length():
    #verifies that items are not being dropped, miscounted, or ignored
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    texts = ["Book summary one", "Book summary two"]
    labels = [[1, 0, 0], [0, 1, 0]]
    dataset = BookDataset(texts, labels, tokenizer)
    assert len(dataset) == 2

def test_genre_dataset_output_keys():
    #confirms the data being fed into the tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    dataset_test = BookDataset(["Book Summary"], [[1, 0, 1]], tokenizer)
    test_item = dataset_test[0]
    assert 'input_ids' in test_item
    assert 'attention_mask' in test_item
    assert 'labels' in test_item
    assert isinstance(test_item['labels'], torch.Tensor)