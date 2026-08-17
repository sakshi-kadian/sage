import os
import json
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.reward_model import DeBERTaRewardModel

def evaluate_model_on_dataset(model, tokenizer, data, device):
    """Evaluates the model's accuracy on the given preference dataset."""
    model.eval()
    correct = 0
    
    with torch.no_grad():
        for item in tqdm(data, desc="Evaluating"):
            prompt = item['prompt']
            chosen = item['chosen']
            rejected = item['rejected']
            
            # Format inputs
            chosen_text = f"Context: {prompt}\n\nResponse: {chosen}"
            rejected_text = f"Context: {prompt}\n\nResponse: {rejected}"
            
            chosen_inputs = tokenizer(
                chosen_text, return_tensors="pt", truncation=True, max_length=512, padding="max_length"
            ).to(device)
            chosen_inputs.pop("token_type_ids", None)
            
            rejected_inputs = tokenizer(
                rejected_text, return_tensors="pt", truncation=True, max_length=512, padding="max_length"
            ).to(device)
            rejected_inputs.pop("token_type_ids", None)
            
            # Get rewards
            reward_chosen = model(**chosen_inputs)
            reward_rejected = model(**rejected_inputs)
            
            if reward_chosen.item() > reward_rejected.item():
                correct += 1
                
    accuracy = correct / len(data)
    return accuracy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Paths
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, "data", "shp_test_subset.json")
    baseline_path = os.path.join(base_dir, "checkpoints", "baseline_epoch_3.pt")
    defended_path = os.path.join(base_dir, "models", "ablation", "reward_model_adv_25pct.pt")
    
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Please run download_shp.py first.")
        return
        
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print(f"Loaded {len(data)} OOD examples from Stanford SHP dataset.")
    
    # Load tokenizer
    model_name = "microsoft/deberta-v3-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    results = {}
    
    # 1. Evaluate Baseline Model (0%)
    print("\n--- Evaluating Baseline Model (0% Adv) ---")
    if not os.path.exists(baseline_path):
        print(f"Could not find baseline checkpoint at {baseline_path}")
    else:
        baseline_model = DeBERTaRewardModel(model_name).to(device)
        baseline_model.load_state_dict(torch.load(baseline_path, map_location=device, weights_only=True))
        
        baseline_acc = evaluate_model_on_dataset(baseline_model, tokenizer, data, device)
        print(f"Baseline OOD Accuracy: {baseline_acc:.4f}")
        results["Baseline (0%)"] = baseline_acc
        
        del baseline_model
        torch.cuda.empty_cache()
        
    # 2. Evaluate Defended Model (25%)
    print("\n--- Evaluating 25% Defended Model ---")
    if not os.path.exists(defended_path):
        print(f"Could not find defended checkpoint at {defended_path}")
    else:
        defended_model = DeBERTaRewardModel(model_name).to(device)
        defended_model.load_state_dict(torch.load(defended_path, map_location=device, weights_only=True))
        
        defended_acc = evaluate_model_on_dataset(defended_model, tokenizer, data, device)
        print(f"25% Defended OOD Accuracy: {defended_acc:.4f}")
        results["25% Defended"] = defended_acc
        
        del defended_model
        torch.cuda.empty_cache()
        
    # Save results
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "ood_results.json")
    
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nOOD Evaluation Complete. Results saved to {results_path}")

if __name__ == "__main__":
    main()
