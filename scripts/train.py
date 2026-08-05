import os
import sys
import yaml
import torch
from transformers import AutoTokenizer

# Add src to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.reward_model import DeBERTaRewardModel
from src.trainer import RewardTrainer
from src.utils import set_seed
from data.dataloader import get_dataloader

def main():
    print("--- Starting Full Baseline Training ---")
    
    # 1. Load Config
    try:
        with open("configs/kaggle_config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("[ERROR] configs/kaggle_config.yaml not found!")
        return
        
    set_seed(config['training']['seed'])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model_name = config['model']['name_or_path']
    print(f"Loading {model_name} tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 2. Data
    print("Loading 10K stratified data...")
    train_loader = get_dataloader(
        config['data']['train_path'], 
        tokenizer, 
        batch_size=config['training']['batch_size'], 
        shuffle=True
    )
    
    # 3. Model
    print("Initializing architecture...")
    model = DeBERTaRewardModel(model_name)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=float(config['training']['learning_rate']),
        weight_decay=config['training']['weight_decay']
    )
    
    # 4. Trainer
    trainer = RewardTrainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_loader,
        device=device,
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        fp16=config['training']['fp16']
    )
    
    # 5. Train Loop
    epochs = config['training']['num_train_epochs']
    print(f"\nStarting training for {epochs} epochs on full dataset...")
    os.makedirs("checkpoints", exist_ok=True)
    
    for epoch in range(1, epochs + 1):
        loss, acc = trainer.train_epoch(epoch)
        print(f"Epoch {epoch} | Loss: {loss:.4f} | Acc: {acc:.4f}")
        
        # Save checkpoint
        torch.save(model.state_dict(), f"checkpoints/baseline_epoch_{epoch}.pt")
        print(f"Saved checkpoint: checkpoints/baseline_epoch_{epoch}.pt")
        
    print("\n[SUCCESS] Training complete! Models saved to checkpoints/")

if __name__ == "__main__":
    main()
