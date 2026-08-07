import os
import torch
import yaml
import json
import pandas as pd
from tqdm import tqdm
from src.reward_model import DeBERTaRewardModel
from src.reward_model import bradley_terry_loss
from src.utils import AverageMeter, calculate_accuracy, set_seed
from data.dataloader import get_dataloader

def main():
    print("--- Starting Baseline Evaluation ---")
    
    # 1. Setup
    with open("configs/kaggle_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    set_seed(config['training']['seed'])
    # Automatically use GPU if available, else fallback to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Load Model Architecture & Weights
    print("Loading architecture and weights...")
    model = DeBERTaRewardModel(
        model_name_or_path="microsoft/deberta-v3-large"
    )
    
    checkpoint_path = "checkpoints/baseline_epoch_3.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Please ensure it is downloaded!")
        
    # Load the 434M parameter state dict
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # Put model in inference mode
    print("Weights loaded successfully!")

    # 3. Load Test Data
    print("Loading test dataset...")
    # Using local relative path for the test set
    test_path = "data/processed/test_1k.parquet"
    if not os.path.exists(test_path):
         raise FileNotFoundError(f"Test data not found at {test_path}")

    # Evaluate batch size can be a bit larger since we don't store gradients
    eval_batch_size = 4
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name_or_path'])
    
    test_dataloader = get_dataloader(
        data_path=test_path,
        tokenizer=tokenizer,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0 # Safest for evaluation on Windows
    )

    # 4. Evaluation Loop
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    results = []

    print("Running inference on test set (1,000 examples)...")
    with torch.no_grad(): # Extremely important: turns off gradient tracking to save memory
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            chosen_ids = batch['chosen_input_ids'].to(device)
            chosen_mask = batch['chosen_attention_mask'].to(device)
            rejected_ids = batch['rejected_input_ids'].to(device)
            rejected_mask = batch['rejected_attention_mask'].to(device)
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=True):
                chosen_rewards = model(input_ids=chosen_ids, attention_mask=chosen_mask)
                rejected_rewards = model(input_ids=rejected_ids, attention_mask=rejected_mask)
                
                loss = bradley_terry_loss(chosen_rewards, rejected_rewards)
            
            # Metrics
            batch_size = chosen_ids.size(0)
            loss_meter.update(loss.item(), batch_size)
            acc = calculate_accuracy(chosen_rewards, rejected_rewards)
            acc_meter.update(acc, batch_size)
            
            # Store individual instance results so we can plot them later
            for i in range(batch_size):
                results.append({
                    'chosen_score': chosen_rewards[i].item(),
                    'rejected_score': rejected_rewards[i].item(),
                    'is_correct': chosen_rewards[i].item() > rejected_rewards[i].item()
                })

    print(f"\nFinal Results:")
    print(f"Test Loss: {loss_meter.avg:.4f}")
    print(f"Test Accuracy: {acc_meter.avg * 100:.2f}%")

    # 5. Save Results to Disk
    os.makedirs("results", exist_ok=True)
    
    # Save raw CSV for plotting
    df = pd.DataFrame(results)
    df.to_csv("results/baseline_test_results.csv", index=False)
    
    # Save top-level metrics
    metrics = {
        "test_loss": loss_meter.avg,
        "test_accuracy": acc_meter.avg
    }
    with open("results/baseline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"\nSaved detailed results to results/baseline_test_results.csv")

if __name__ == "__main__":
    main()
