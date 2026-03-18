import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.preprocessing import (
    basic_text_cleaning,
    text_to_sequence,
)

class AGNewsDataset(Dataset):
    def __init__(self, 
                 texts: pd.Series, 
                 labels: pd.Series, 
                 vocab: dict[str, int], 
                 max_len: int
                 ) -> None:
        self.texts = texts # all the news texts
        self.labels = labels # corresponding labels
        self.vocab = vocab # index dictionary
        self.max_len = max_len # fixed length for sequences

    def __len__(self) -> int:
        """
        Returns the number of examples in the dataset.
        """
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns one example.
        """
        tokens = basic_text_cleaning(self.texts.iloc[idx])
        seq = text_to_sequence(tokens, self.vocab)

        seq = seq[:self.max_len]
        padding = [0] * (self.max_len - len(seq))
        seq = seq + padding

        return torch.tensor(seq, dtype=torch.long), torch.tensor(self.labels.iloc[idx] - 1, dtype=torch.long)

def make_loaders(X_train, y_train, X_dev, y_dev, X_test, y_test, vocab, max_len: int, batch_size: int = 64):
    train_ds = AGNewsDataset(X_train, y_train, vocab, max_len)
    dev_ds   = AGNewsDataset(X_dev,   y_dev,   vocab, max_len)
    test_ds  = AGNewsDataset(X_test,  y_test,  vocab, max_len)
    
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(dev_ds,   batch_size=batch_size, shuffle=False),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False),
        test_ds
    )