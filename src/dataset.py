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

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = self.labels[idx]

        # tokenizer does several things at once:
        # 1. splits into tokens
        # 2. converts tokens to integers (IDs)
        # 3. pads/truncates to max_len
        # 4. creates attention_mask
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),    # token IDs
            "attention_mask": encoding["attention_mask"].squeeze(),  # 1 if real token, 0 if padded
            "label": torch.tensor(label, dtype=torch.long)
        }

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