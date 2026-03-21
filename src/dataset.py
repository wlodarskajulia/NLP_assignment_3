import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class AGNewsDataset(Dataset):
    def __init__(
        self,
        texts: pd.Series,
        labels: list,
        tokenizer,          # HuggingFace PreTrainedTokenizer
        max_len: int,
    ) -> None:
        self.texts = texts.reset_index(drop=True)
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
 
    def __len__(self) -> int:
        return len(self.texts)
 
    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = self.labels[idx]
 
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
 
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),       # (max_len,)
            "attention_mask": encoding["attention_mask"].squeeze(0),  # (max_len,)
            "label":          torch.tensor(label, dtype=torch.long),
        }
    
def make_loaders(X_train, y_train, X_dev, y_dev, X_test, y_test, tokenizer, max_len: int, batch_size: int = 64):
    train_ds = AGNewsDataset(X_train, y_train, tokenizer, max_len)
    dev_ds   = AGNewsDataset(X_dev,   y_dev,   tokenizer, max_len)
    test_ds  = AGNewsDataset(X_test,  y_test,  tokenizer, max_len)
    
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(dev_ds,   batch_size=batch_size, shuffle=False),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False),
        test_ds
    )