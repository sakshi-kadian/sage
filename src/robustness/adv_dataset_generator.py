"""
Adversarial Dataset Generator
==============================
Combines verified stealthy adversarial examples with clean training pairs
to build the defense dataset for adversarial training.

The resulting dataset contains two types of pairs:
  - Clean pairs: (chosen, rejected) from the original HH-RLHF dataset.
  - Adversarial pairs: (adversarial_chosen, rejected) where the chosen
    response has been perturbed by the BERT MLM attack but still passed
    the SBERT + GPT-2 stealth filter. The model must correctly prefer the
    unmodified rejected text over this stealthy but perturbed response.

This teaches the reward model to be robust against stealthy adversarial
perturbations without sacrificing accuracy on clean data.
"""

import os
import json
import csv
import random
from typing import Optional


def load_clean_pairs(data_path: str) -> list[dict]:
    """Load the clean chosen/rejected pairs from the test dataset."""
    with open(data_path, 'r') as f:
        return json.load(f)


def load_adversarial_results(
    csv_path: str,
    attack_type: Optional[str] = None,
    min_reward_drop: float = 0.001
) -> list[dict]:
    """
    Load adversarial examples that passed the stealth filter.

    Args:
        csv_path: Path to the attack_eval_results.csv file.
        attack_type: If provided, only include rows from this attack type.
                     If None, include all attack types that passed the filter.
        min_reward_drop: Only include adversarial examples that actually
                         dropped the reward by at least this amount. This
                         filters out substitutions that were benign (no effect).

    Returns:
        List of dicts with keys: sample_idx, attack_type, original_reward,
        adversarial_reward, reward_drop, similarity.
    """
    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['passes_filter'] != 'True':
                continue
            if attack_type and row['attack_type'] != attack_type:
                continue
            if float(row['reward_drop']) < min_reward_drop:
                continue
            results.append({
                'sample_idx': int(row['sample_idx']),
                'attack_type': row['attack_type'],
                'original_reward': float(row['original_reward']),
                'adversarial_reward': float(row['adversarial_reward']),
                'reward_drop': float(row['reward_drop']),
                'similarity': float(row['similarity'])
            })
    return results


def generate_adversarial_dataset(
    clean_data_path: str,
    eval_csv_path: str,
    output_path: str,
    adv_ratio: float = 0.25,
    attack_type: str = 'BERT MLM',
    seed: int = 42
) -> dict:
    """
    Generate the combined defense training dataset.

    For each adversarial example that passed the stealth filter, we create
    a new training pair where the model must prefer the ORIGINAL clean text
    over the stealthy adversarial variant. This is the core of the defense.

    Args:
        clean_data_path: Path to the test_data.json file.
        eval_csv_path: Path to the attack_eval_results.csv file.
        output_path: Where to save the combined adv_training_pairs.json.
        adv_ratio: Fraction of training pairs that should be adversarial.
                   0.25 means 25% adversarial, 75% clean.
        attack_type: Which attack type to use for adversarial pairs.
        seed: Random seed for reproducibility.

    Returns:
        Dict with statistics about the generated dataset.
    """
    random.seed(seed)

    # Load data
    clean_pairs = load_clean_pairs(clean_data_path)
    adv_results = load_adversarial_results(eval_csv_path, attack_type=attack_type)

    print(f"Loaded {len(clean_pairs)} clean pairs.")
    print(f"Found {len(adv_results)} valid stealthy adversarial examples "
          f"(attack: {attack_type}, passed filter + reward drop > 0.001).")

    if not adv_results:
        raise ValueError(
            "No valid adversarial examples found. "
            "Check that evaluate_attacks.py has been run and the CSV exists."
        )

    import sys
    import torch
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from transformers import AutoTokenizer
    from src.reward_model import DeBERTaRewardModel
    from src.adversarial.mlm_attack import MLMAttack
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Loading model to regenerate adversarial texts...")
    model_name = "microsoft/deberta-v3-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = DeBERTaRewardModel(model_name)
    model_path = os.path.join("models", "deberta_reward_model.pt")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    mlm_attack = MLMAttack(model, tokenizer, device=device)
    
    adv_pairs = []
    print(f"Regenerating {len(adv_results)} adversarial texts...")
    for result in adv_results:
        idx = result['sample_idx']
        if idx >= len(clean_pairs):
            continue
        original_text = clean_pairs[idx]['chosen']
        
        # Regenerate the adversarial text that was evaluated in evaluate_attacks.py
        # We only do this for the valid ones that passed the filter.
        print(f"  Regenerating sample {idx}...")
        mlm_res = mlm_attack.attack(original_text, true_label=1, budget=3)
        adversarial_text = mlm_res['adversarial_text']
        
        adv_pairs.append({
            'chosen': original_text,
            'rejected': adversarial_text,  # The model should reject the stealthy attack
            'sample_idx': idx,
            'reward_drop': result['reward_drop'],
            'similarity': result['similarity'],
            'is_adversarial': True
        })

    # Combine: sample clean pairs to fill the non-adversarial portion
    n_adv = len(adv_pairs)
    n_clean_needed = int(n_adv / adv_ratio) - n_adv
    n_clean_needed = min(n_clean_needed, len(clean_pairs))

    sampled_clean = random.sample(clean_pairs, n_clean_needed)
    clean_training_pairs = [
        {
            'chosen': pair['chosen'],
            'rejected': pair['rejected'],
            'is_adversarial': False
        }
        for pair in sampled_clean
    ]

    combined = clean_training_pairs + adv_pairs
    random.shuffle(combined)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(combined, f, indent=2)

    stats = {
        'total_pairs': len(combined),
        'clean_pairs': len(clean_training_pairs),
        'adversarial_pairs': n_adv,
        'actual_adv_ratio': round(n_adv / len(combined), 3),
        'avg_reward_drop': round(
            sum(r['reward_drop'] for r in adv_results) / len(adv_results), 4
        ),
        'avg_similarity': round(
            sum(r['similarity'] for r in adv_results) / len(adv_results), 4
        ),
        'output_path': output_path
    }

    print("\n--- Dataset Generation Complete ---")
    print(f"Total pairs:        {stats['total_pairs']}")
    print(f"Clean pairs:        {stats['clean_pairs']}")
    print(f"Adversarial pairs:  {stats['adversarial_pairs']}")
    print(f"Actual adv ratio:   {stats['actual_adv_ratio']:.1%}")
    print(f"Avg reward drop:    {stats['avg_reward_drop']:.4f}")
    print(f"Avg SBERT similarity: {stats['avg_similarity']:.4f}")
    print(f"Saved to:           {output_path}")

    return stats


if __name__ == "__main__":
    stats = generate_adversarial_dataset(
        clean_data_path=os.path.join("data", "test_data.json"),
        eval_csv_path=os.path.join("results", "attack_eval_results.csv"),
        output_path=os.path.join("data", "adv_training_pairs.json"),
        adv_ratio=0.25,
        attack_type="BERT MLM"
    )
