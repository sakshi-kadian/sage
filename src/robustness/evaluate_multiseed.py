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
from data.dataloader import get_dataloader

def evaluate_ood_accuracy(model, tokenizer, device='cpu'):
    """Evaluate pairwise accuracy on OOD data (1k test set) using fast dataloader."""
    test_path = "data/processed/test_1k.parquet"
    eval_batch_size = 4
    
    test_dataloader = get_dataloader(
        data_path=test_path,
        tokenizer=tokenizer,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0
    )

    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating OOD Acc", leave=False):
            chosen_ids = batch['chosen_input_ids'].to(device)
            chosen_mask = batch['chosen_attention_mask'].to(device)
            rejected_ids = batch['rejected_input_ids'].to(device)
            rejected_mask = batch['rejected_attention_mask'].to(device)
            
            with torch.cuda.amp.autocast(enabled=True):
                chosen_rewards = model(input_ids=chosen_ids, attention_mask=chosen_mask)
                rejected_rewards = model(input_ids=rejected_ids, attention_mask=rejected_mask)
            
            correct += (chosen_rewards > rejected_rewards).sum().item()
            total += chosen_ids.size(0)
            
    return correct / total if total > 0 else 0.0

def evaluate_robustness(model, tokenizer, texts, device='cpu'):
    """Evaluate how much the reward drops under the MLM stealth attack."""
    mlm_attack = MLMAttack(model, tokenizer, device=device)
    sf = StealthFilter(device=device)
    
    total_drop = 0
    valid_attacks = 0
    
    model.eval()
    for text in tqdm(texts, desc="Running MLM Attack", leave=False):
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
        "Baseline (Seed 1)": "models/multi_seed/baseline_seed_1.pt",
        "Baseline (Seed 2)": "models/multi_seed/baseline_seed_2.pt",
        "Baseline (Seed 3)": "models/multi_seed/baseline_seed_3.pt",
        "Defended 25% (Seed 1)": "models/multi_seed/defended_25pct_seed_1.pt",
        "Defended 25% (Seed 2)": "models/multi_seed/defended_25pct_seed_2.pt",
        "Defended 25% (Seed 3)": "models/multi_seed/defended_25pct_seed_3.pt"
    }
    
    # Load 20 texts for attack evaluation
    data_path = "data/test_data.json"
    with open(data_path, 'r') as f:
        all_data = json.load(f)
    attack_texts = [p['chosen'] for p in all_data[100:120]]
    
    # Load tokenizer and base model
    model_name = "microsoft/deberta-v3-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = DeBERTaRewardModel(model_name)
    
    results = []
    
    for name, ckpt_path in checkpoints.items():
        print(f"\n{'='*50}\nEvaluating: {name}\n{'='*50}")
        if not os.path.exists(ckpt_path):
            print(f"File not found: {ckpt_path}. Skipping.")
            continue
            
        state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(state_dict, strict=False)
        model.float()
        model.to(device)
        
        print("Evaluating Clean OOD Accuracy (1000 pairs)...")
        ood_acc = evaluate_ood_accuracy(model, tokenizer, device)
        print(f"Clean OOD Accuracy: {ood_acc:.4f}")
        
        print("Evaluating Adversarial Robustness (MLM Attack)...")
        avg_drop = evaluate_robustness(model, tokenizer, attack_texts, device)
        print(f"Average Reward Drop: {avg_drop:.4f}")
        
        results.append({
            "Model": name,
            "Clean OOD Accuracy": ood_acc,
            "Reward Drop": avg_drop
        })
        
    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    out_path = "results/multi_seed_results.csv"
    df.to_csv(out_path, index=False)
    
    # Save as JSON too for easy loading
    json_path = "results/multi_seed_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nResults saved to {out_path} and {json_path}")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
