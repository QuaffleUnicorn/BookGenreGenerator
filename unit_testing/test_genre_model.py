import torch
from GenreModel import DistilBertForMultiLabelClassification

def test_model_output_shape():
    #creates instance of model with 5 possible genres
    model_fake = DistilBertForMultiLabelClassification(num_labels=5)

    #batch of tokenized fake input that is a batch of 3 with 15 labels
    input_fake_ids = torch.randint(0, 1000, (3, 15))
    attention_mask = torch.ones_like(input_fake_ids)

    #run model from provided 3 batches
    output_from_test_model = model_fake(input_ids=input_fake_ids, attention_mask=attention_mask)

    #verify the model shape matches inputs provided above
    assert output_from_test_model.shape == (3, 5)