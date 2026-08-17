import os
import sys
import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.reward_model import DeBERTaRewardModel
from src.adversarial.mlm_attack import MLMAttack
from src.adversarial.filters import StealthFilter

def evaluate_clean_accuracy(model, tokenizer, test_pairs, device='cpu', batch_size=8):
    """Evaluate pairwise accuracy on clean data."""
    model.eval()
    correct = 0
    total = len(test_pairs)
    
    with torch.no_grad():
        for i in range(0, total, batch_size):
            batch = test_pairs[i:i+batch_size]
            
            chosen_texts = [p['chosen'] for p in batch]
            rejected_texts = [p['rejected'] for p in batch]
            
            inputs_c = tokenizer(chosen_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            inputs_r = tokenizer(rejected_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            
            inputs_c.pop("token_type_ids", None)
            inputs_r.pop("token_type_ids", None)
            
            rewards_c = model(**inputs_c).squeeze(-1)
            rewards_r = model(**inputs_r).squeeze(-1)
            
            correct += (rewards_c > rewards_r).sum().item()
            
    return correct / total if total > 0 else 0.0

def evaluate_robustness(model, tokenizer, texts, device='cpu'):
    """Evaluate how much the reward drops under the MLM stealth attack."""
    mlm_attack = MLMAttack(model, tokenizer, device=device)
    sf = StealthFilter(device=device)
    
    total_drop = 0
    valid_attacks = 0
    
    for text in tqdm(texts, desc="Running MLM Attack"):
        res = mlm_attack.attack(text, true_label=1, budget=2)
        filter_res = sf.check(text, res['adversarial_text'])
        
        # Only count attacks that pass the stealth filter
        if filter_res['passes']:
            drop = res['original_reward'] - res['adversarial_reward']
            total_drop += drop
            valid_attacks += 1
            
    avg_drop = total_drop / valid_attacks if valid_attacks > 0 else 0
    return avg_drop

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Checkpoints to evaluate
    checkpoints = {
        "0% (Baseline)": "checkpoints/baseline_epoch_3.pt",
        "10% Adv": "models/ablation/reward_model_adv_10pct.pt",
        "25% Adv": "models/ablation/reward_model_adv_25pct.pt",
        "50% Adv": "models/ablation/reward_model_adv_50pct.pt"
    }
    
    # Load test data
    data_path = "data/test_data.json"
    with open(data_path, 'r') as f:
        all_data = json.load(f)
        
    clean_test_pairs = all_data[:100]  # Use 100 pairs for fast clean accuracy
    attack_texts = [p['chosen'] for p in all_data[100:120]]  # Use 20 texts for robust eval
    
    # Load tokenizer and base model
    model_name = "microsoft/deberta-v3-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = DeBERTaRewardModel(model_name)
    
    results = []
    
    for ratio_name, ckpt_path in checkpoints.items():
        print(f"\n{'='*50}\nEvaluating: {ratio_name}\n{'='*50}")
        if not os.path.exists(ckpt_path):
            print(f"File not found: {ckpt_path}. Skipping.")
            continue
            
        # Load weights (cast to float32 before sending to GPU like we did in training!)
        state_dict = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        model.float()
        model.to(device)
        
        # 1. Clean Accuracy
        print("Evaluating Clean Accuracy...")
        clean_acc = evaluate_clean_accuracy(model, tokenizer, clean_test_pairs, device)
        print(f"Clean Accuracy: {clean_acc:.4f}")
        
        # 2. Adversarial Robustness (Average Reward Drop)
        print("Evaluating Adversarial Robustness...")
        avg_drop = evaluate_robustness(model, tokenizer, attack_texts, device)
        print(f"Average Reward Drop (Lower is Better): {avg_drop:.4f}")
        
        results.append({
            "Ratio": ratio_name,
            "Clean Accuracy": clean_acc,
            "Reward Drop": avg_drop
        })
        
    # Save results
    df = pd.DataFrame(results)
    out_path = "results/ablation_tradeoff.csv"
    df.to_csv(out_path, index=False)
    print(f"\nTradeoff results saved to {out_path}")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
