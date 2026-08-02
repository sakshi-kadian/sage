import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import PreTrainedTokenizer

class HHRLHFDataset(Dataset):
    """
    Dataset class for HH-RLHF.
    Returns tokenized chosen and rejected responses.
    """
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        self.df = pd.read_parquet(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Tokenize chosen response
        chosen_enc = self.tokenizer(
            row['chosen'],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Tokenize rejected response
        rejected_enc = self.tokenizer(
            row['rejected'],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'chosen_input_ids': chosen_enc['input_ids'].squeeze(0),
            'chosen_attention_mask': chosen_enc['attention_mask'].squeeze(0),
            'rejected_input_ids': rejected_enc['input_ids'].squeeze(0),
            'rejected_attention_mask': rejected_enc['attention_mask'].squeeze(0)
        }

def get_dataloader(data_path: str, tokenizer, batch_size: int = 8, shuffle: bool = True, num_workers: int = 4):
    dataset = HHRLHFDataset(data_path, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
