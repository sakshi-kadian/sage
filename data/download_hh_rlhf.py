import os
from datasets import load_dataset
import pandas as pd

def download_and_save():
    print("Downloading Anthropic HH-RLHF dataset...")
    # Load dataset
    dataset = load_dataset("Anthropic/hh-rlhf")
    
    # Create data directory if it doesn't exist
    os.makedirs("data/raw", exist_ok=True)
    
    # Save train and test splits to parquet for fast loading
    print("Saving to data/raw/...")
    dataset['train'].to_parquet("data/raw/train.parquet")
    dataset['test'].to_parquet("data/raw/test.parquet")
    
    print("Download complete. Files saved to data/raw/")

if __name__ == "__main__":
    download_and_save()
