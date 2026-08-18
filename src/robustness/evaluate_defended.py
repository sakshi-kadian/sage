import os
import sys
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.reward_model import DeBERTaRewardModel
from data.dataloader import get_dataloader

def main():
    print("--- Starting Defended Model Evaluation ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading architecture and weights...")
    model_name = "microsoft/deberta-v3-large"
    model = DeBERTaRewardModel(model_name)
    
    checkpoint_path = "models/ablation/reward_model_adv_25pct.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print("Weights loaded successfully!")

    print("Loading test dataset...")
    test_path = "data/processed/test_1k.parquet"
    eval_batch_size = 4
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    test_dataloader = get_dataloader(
        data_path=test_path,
        tokenizer=tokenizer,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0
    )

    results = []
    print("Running inference on test set (1,000 examples)...")
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            chosen_ids = batch['chosen_input_ids'].to(device)
            chosen_mask = batch['chosen_attention_mask'].to(device)
            rejected_ids = batch['rejected_input_ids'].to(device)
            rejected_mask = batch['rejected_attention_mask'].to(device)
            
            with torch.cuda.amp.autocast(enabled=True):
                chosen_rewards = model(input_ids=chosen_ids, attention_mask=chosen_mask)
                rejected_rewards = model(input_ids=rejected_ids, attention_mask=rejected_mask)
            
            batch_size = chosen_ids.size(0)
            
            for i in range(batch_size):
                results.append({
                    'chosen_score': chosen_rewards[i].item(),
                    'rejected_score': rejected_rewards[i].item(),
                    'is_correct': chosen_rewards[i].item() > rejected_rewards[i].item()
                })

    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/defended_test_results.csv", index=False)
    print(f"Saved detailed results to results/defended_test_results.csv")
    print(f"Overall Accuracy: {df['is_correct'].mean() * 100:.2f}%")

if __name__ == "__main__":
    main()
