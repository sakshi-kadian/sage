import os
import sys
import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.reward_model import DeBERTaRewardModel
from src.adversarial.model_wrapper import RewardModelWrapper
from src.adversarial.blackbox_attacks import TextFoolerRewardAttack
from src.adversarial.mechanistic_attack import MechanisticAttack
from src.adversarial.mlm_attack import MLMAttack
from src.adversarial.filters import StealthFilter

def load_test_data(data_path: str, num_samples: int = 50):
    """Load a subset of test data for evaluation."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    # Use only 'chosen' texts for attack evaluation (we try to minimize their reward)
    texts = [item['chosen'] for item in data[:num_samples]]
    return texts

def evaluate_attacks(model, tokenizer, texts, device='cpu'):
    """Run all three attacks on the given texts and filter the results."""
    
    print("\nInitializing Attackers and Filters...")
    model_wrapper = RewardModelWrapper(model, tokenizer, device=device)
    
    blackbox = TextFoolerRewardAttack(model_wrapper, max_substitutions=3)
    mechanistic = MechanisticAttack(model, tokenizer, model_wrapper, device=device)
    mlm_attack = MLMAttack(model, tokenizer, device=device)
    
    sf = StealthFilter(device=device)
    
    results = []
    
    print(f"\nEvaluating {len(texts)} samples...")
    for i, text in enumerate(tqdm(texts)):
        # 1. Black-Box Attack
        bb_res = blackbox.attack(text)
        bb_filter = sf.check(text, bb_res['adversarial_text'])
        
        results.append({
            'sample_idx': i,
            'attack_type': 'Black-Box (WordNet)',
            'original_reward': bb_res['original_reward'],
            'adversarial_reward': bb_res['adversarial_reward'],
            'reward_drop': bb_res['original_reward'] - bb_res['adversarial_reward'],
            'passes_filter': bb_filter['passes'],
            'similarity': bb_filter['similarity']
        })
        
        # 2. Mechanistic Attack
        mech_res = mechanistic.attack(text)
        mech_filter = sf.check(text, mech_res['adversarial_text'])
        
        results.append({
            'sample_idx': i,
            'attack_type': 'Mechanistic (WordNet)',
            'original_reward': mech_res['original_reward'],
            'adversarial_reward': mech_res['adversarial_reward'],
            'reward_drop': mech_res['original_reward'] - mech_res['adversarial_reward'],
            'passes_filter': mech_filter['passes'],
            'similarity': mech_filter['similarity']
        })
        
        # 3. BERT MLM Attack
        mlm_res = mlm_attack.attack(text, true_label=1, budget=3)
        mlm_filter = sf.check(text, mlm_res['adversarial_text'])
        
        results.append({
            'sample_idx': i,
            'attack_type': 'BERT MLM',
            'original_reward': mlm_res['original_reward'],
            'adversarial_reward': mlm_res['adversarial_reward'],
            'reward_drop': mlm_res['original_reward'] - mlm_res['adversarial_reward'],
            'passes_filter': mlm_filter['passes'],
            'similarity': mlm_filter['similarity']
        })

    return pd.DataFrame(results)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load Model
    print("Loading reward model...")
    model_name = "microsoft/deberta-v3-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = DeBERTaRewardModel(model_name)
    
    model_path = os.path.join("models", "deberta_reward_model.pt")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded trained weights.")
    else:
        print("WARNING: No trained weights found, using untrained model.")
        
    model.to(device)
    model.eval()

    # Load Data
    data_path = os.path.join("data", "test_data.json")
    if not os.path.exists(data_path):
        print(f"Error: Could not find {data_path}. Run Day 1 data pipeline first.")
        return
        
    texts = load_test_data(data_path, num_samples=50) # Run on 50 for local evaluation
    
    # Run Evaluation
    df = evaluate_attacks(model, tokenizer, texts, device=device)
    
    # Save Results
    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "attack_eval_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved evaluation results to {out_path}")
    
    # Print quick summary
    print("\n--- Summary Statistics ---")
    summary = df.groupby('attack_type').agg(
        avg_drop=('reward_drop', 'mean'),
        filter_pass_rate=('passes_filter', 'mean')
    )
    print(summary)

if __name__ == "__main__":
    main()
