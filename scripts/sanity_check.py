import os
import sys
import torch
from transformers import AutoTokenizer

# Add src to python path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.reward_model import DeBERTaRewardModel
from src.trainer import RewardTrainer
from src.utils import set_seed
from data.dataloader import get_dataloader

def run_sanity_check():
    """
    Trains the model on a single batch for 10 epochs. 
    If the codebase is completely bug-free, the loss should drop to near 0 and accuracy should hit 1.0.
    """
    print("--- Starting Single Batch Sanity Check ---")
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load Dummy Data (Using lightweight small model for fast local CPU testing)
    model_name = "microsoft/deberta-v3-small"
    print(f"Loading {model_name} tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create dataloader (assumes Day 1 data exists)
    try:
        dataloader = get_dataloader("data/processed/train_10k.parquet", tokenizer, batch_size=2, shuffle=False)
    except FileNotFoundError:
        print("[ERROR] Dataset not found! Please ensure you have run the Day 1 data scripts to generate data/processed/train_10k.parquet")
        return

    # Extract just ONE batch
    print("Extracting a single batch of chosen/rejected pairs...")
    single_batch = next(iter(dataloader))
    
    # Create a dummy dataloader that just yields this single batch repeatedly
    class SingleBatchLoader:
        def __iter__(self):
            yield single_batch
        def __len__(self):
            return 1
            
    dummy_loader = SingleBatchLoader()
    
    # 2. Init Model and Optimizer
    print(f"Initializing {model_name} architecture...")
    model = DeBERTaRewardModel(model_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    
    # 3. Setup Trainer
    trainer = RewardTrainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=dummy_loader,
        device=device,
        gradient_accumulation_steps=1,
        fp16=False # Keep it simple for the sanity check
    )
    
    # 4. Overfit
    print("\nAttempting to overfit on the single batch for up to 50 epochs...")
    print("Goal: Loss should decrease towards 0.0, and Accuracy should reach 1.0\n")
    
    for epoch in range(1, 51):
        loss, acc = trainer.train_epoch(epoch)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch} | Loss: {loss:.4f} | Acc: {acc:.4f}")
        
        if acc == 1.0 and loss < 0.1:
            print(f"\n[SUCCESS] SANITY CHECK PASSED at Epoch {epoch}! The model successfully memorized the batch.")
            print("Your architecture, loss function, and training loop are completely bug-free!")
            return
            
    print("\n[FAILED] SANITY CHECK FAILED! The model failed to converge. There is a bug in the gradients or loss function.")

if __name__ == "__main__":
    run_sanity_check()
