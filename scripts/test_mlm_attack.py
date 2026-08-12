import sys
import os
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.adversarial.mlm_attack import MLMAttack
from src.adversarial.filters import StealthFilter
from src.reward_model import DeBERTaRewardModel

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. Load Reward Model
    print("Loading reward model...")
    model_name = "microsoft/deberta-v3-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = DeBERTaRewardModel(model_name)
    
    # Load state dict if available, otherwise just use untrained for pilot
    model_path = os.path.join("models", "deberta_reward_model.pt")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded trained weights.")
    else:
        print("WARNING: No trained weights found, using untrained model for pilot.")
        
    model.to(device)
    model.eval()

    # 2. Initialize Attacker and Filter
    print("\nInitializing BERT MLM Attacker...")
    attacker = MLMAttack(model, tokenizer, device=device)
    
    print("\nInitializing Stealth Filter...")
    sf = StealthFilter(device=device)

    # 3. Run Pilot Test
    test_sentences = [
        "I strongly recommend that you carefully evaluate the financial risks before investing.",
        "The capital of France is Paris, and it is known for the Eiffel Tower.",
        "I am not entirely sure about this, but I think it might possibly be correct."
    ]

    print(f"\n{'='*80}")
    print("Running MLM Attack + Stealth Filter...")
    print(f"{'='*80}\n")
    
    passed_filter = 0
    total_drop = 0.0

    for i, text in enumerate(test_sentences):
        print(f"--- Example {i+1} ---")
        print(f"[Original]: {text}")
        
        # Attack!
        result = attacker.attack(text, true_label=1, budget=3)
        
        orig_reward = result['original_reward']
        adv_reward = result['adversarial_reward']
        adv_text = result['adversarial_text']
        subs = result['substitutions']
        
        print(f"Orig Reward: {orig_reward:7.4f}")
        print(f"[Adversarial]: {adv_text}")
        print(f"Adv Reward:  {adv_reward:7.4f}  (Drop: {orig_reward - adv_reward:.4f}, Subs: {subs})")
        
        # Filter!
        filter_res = sf.check(text, adv_text)
        sim = filter_res['similarity']
        
        if filter_res['passes']:
            print(f"Filter: PASS (Similarity: {sim:.3f})")
            passed_filter += 1
        else:
            print(f"Filter: FAIL (Similarity: {sim:.3f} - {filter_res['fail_reason']})")
            
        total_drop += (orig_reward - adv_reward)
        print()

    print(f"{'='*80}")
    print(f"Pilot Summary:")
    print(f"Average Reward Drop: {total_drop / len(test_sentences):.4f}")
    print(f"Stealth Pass Rate:   {passed_filter} / {len(test_sentences)} ({(passed_filter/len(test_sentences))*100:.1f}%)")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
