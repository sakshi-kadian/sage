import pandas as pd
import numpy as np

def create_stratified_sample(input_path: str, output_path: str, n_samples: int = 10000, random_state: int = 42):
    """
    Creates a stratified sample of the HH-RLHF dataset to ensure 
    a balanced distribution of topics and lengths.
    """
    print(f"Loading raw data from {input_path}...")
    df = pd.read_parquet(input_path)
    
    # Basic stratification heuristic: segment by length of the chosen response
    df['length_bucket'] = pd.qcut(df['chosen'].str.len(), q=4, labels=['short', 'medium', 'long', 'very_long'])
    
    print(f"Sampling {n_samples} examples stratified by response length...")
    sampled_df = df.groupby('length_bucket', group_keys=False).apply(
        lambda x: x.sample(n=int(n_samples/4), random_state=random_state)
    ).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    
    # Drop the temporary bucket column
    sampled_df = sampled_df.drop(columns=['length_bucket'])
    
    print(f"Saving stratified sample to {output_path}...")
    sampled_df.to_parquet(output_path)
    print("Done!")

if __name__ == "__main__":
    import os
    os.makedirs("data/processed", exist_ok=True)
    create_stratified_sample("data/raw/train.parquet", "data/processed/train_10k.parquet", n_samples=10000)
    create_stratified_sample("data/raw/test.parquet", "data/processed/test_1k.parquet", n_samples=1000)
