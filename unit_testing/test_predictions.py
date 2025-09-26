import torch
from unittest.mock import MagicMock

from Prediction import get_predictions

def test_get_predictions():
    #creates fake model
    mock_genre_model = MagicMock()
    mock_genre_model.eval = MagicMock()
    mock_genre_model.to = MagicMock()
    mock_genre_model.__call__ = lambda input_ids, attention_mask: torch.tensor([[0.8, 0.1], [0.2, 0.9]])

    #creates fake DataLoader
    fake_dataloader = [
        {
            'input_ids': torch.tensor([[1, 2], [3, 4]]),
            'attention_mask': torch.ones(2, 2),
            'labels': torch.tensor([[1, 0], [0, 1]])
        }
    ]

    #confirms that you get predictions for the amount of samples you provided
    #confirms that function does not crash or skip items
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    true_labels, predicted_labels = get_predictions(mock_genre_model, fake_dataloader, device)
    assert len(true_labels) == 2
    assert len(predicted_labels) == 2
