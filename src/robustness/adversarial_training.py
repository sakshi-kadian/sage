"""
Adversarial Training - Defense Ablation Study
===============================================
Fine-tunes the DeBERTa reward model on a mix of clean and adversarial pairs
to make it robust against stealthy BERT MLM attacks.

Ablation Study Design:
  Three separate models are trained with different adversarial mixing ratios
  to find the optimal trade-off between clean accuracy and adversarial robustness:
    - 10% adversarial ratio  : Baseline defense (minimal fine-tuning)
    - 25% adversarial ratio  : Balanced defense (our primary hypothesis)
    - 50% adversarial ratio  : Aggressive defense (maximum robustness)

Each model is saved as a separate checkpoint so Figure 3 can plot the
clean accuracy vs. adversarial robustness trade-off curve.

Designed to run on Kaggle GPU (T4/P100) for Day 15 compute.
"""

import os
import sys
import json
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.reward_model import DeBERTaRewardModel, bradley_terry_loss
from src.utils import AverageMeter, calculate_accuracy


# Config

MODEL_NAME = "microsoft/deberta-v3-large"
BASE_CHECKPOINT = "models/deberta_reward_model.pt"     # Day 3 trained weights
ADV_DATA_PATH = "data/adv_training_pairs.json"         # Day 13 generated dataset
OUTPUT_DIR = "models/ablation"                          # Saved checkpoints
RESULTS_DIR = "results"

# Training hyperparameters - kept identical to the original training (Day 3)
# to isolate the effect of the adversarial data, not the hyperparameters.
LEARNING_RATE = 1e-5
EPOCHS = 2                  # Fine-tuning - fewer epochs than original training
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 16  # Effective batch size = 16
MAX_LENGTH = 256
SEED = 42

# The three mixing ratios to test in the ablation study
ADV_RATIOS = [0.10, 0.25, 0.50]


# Dataset

class PairwiseDataset(Dataset):
    """
    Pairwise dataset for reward model training.
    Each item returns tokenized chosen and rejected responses.
    """
    def __init__(self, pairs: list[dict], tokenizer, max_length: int = 256):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        chosen = self.tokenizer(
            item['chosen'],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        rejected = self.tokenizer(
            item['rejected'],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'chosen_input_ids': chosen['input_ids'].squeeze(0),
            'chosen_attention_mask': chosen['attention_mask'].squeeze(0),
            'rejected_input_ids': rejected['input_ids'].squeeze(0),
            'rejected_attention_mask': rejected['attention_mask'].squeeze(0),
            'is_adversarial': int(item.get('is_adversarial', False))
        }


def build_dataset_for_ratio(
    all_pairs: list[dict],
    adv_ratio: float,
    seed: int = SEED
) -> list[dict]:
    """
    Sub-sample the full dataset to produce a dataset with exactly `adv_ratio`
    fraction of adversarial examples.

    Args:
        all_pairs: The full 108-pair dataset from adv_training_pairs.json.
        adv_ratio: Target fraction of adversarial pairs (e.g., 0.10, 0.25, 0.50).
        seed: Random seed for reproducibility.

    Returns:
        A sub-sampled list of pairs with the target adversarial ratio.
    """
    random.seed(seed)

    clean = [p for p in all_pairs if not p.get('is_adversarial', False)]
    adv = [p for p in all_pairs if p.get('is_adversarial', False)]

    n_clean = len(clean)
    n_adv = len(adv)
    
    # We want: final_adv / (final_adv + final_clean) = adv_ratio
    # Case 1: We have plenty of clean data, so we limit clean data based on n_adv.
    target_clean = int(n_adv / adv_ratio) - n_adv
    
    if target_clean <= n_clean:
        final_adv = n_adv
        final_clean = target_clean
    else:
        # Case 2: We don't have enough clean data. Limit ADV data instead.
        final_adv = int((adv_ratio * n_clean) / (1.0 - adv_ratio))
        final_clean = n_clean

    sampled_clean = random.sample(clean, final_clean)
    sampled_adv = random.sample(adv, final_adv)
    
    combined = sampled_clean + sampled_adv
    random.shuffle(combined)

    actual_ratio = final_adv / len(combined)
    print(f"  Built dataset for ratio={adv_ratio:.0%}: "
          f"{len(combined)} pairs "
          f"({final_adv} adv + {final_clean} clean), "
          f"actual ratio = {actual_ratio:.1%}")
    return combined


# Training Loop

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
    epoch_idx: int,
    gradient_accumulation: int = GRADIENT_ACCUMULATION
) -> tuple[float, float]:
    """Run one full training epoch. Returns (avg_loss, avg_accuracy)."""
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    pbar = tqdm(dataloader, desc=f"  Epoch {epoch_idx}")
    optimizer.zero_grad()

    for step, batch in enumerate(pbar):
        chosen_ids = batch['chosen_input_ids'].to(device)
        chosen_mask = batch['chosen_attention_mask'].to(device)
        rejected_ids = batch['rejected_input_ids'].to(device)
        rejected_mask = batch['rejected_attention_mask'].to(device)

        chosen_rewards = model(input_ids=chosen_ids, attention_mask=chosen_mask)
        rejected_rewards = model(input_ids=rejected_ids, attention_mask=rejected_mask)

        loss = bradley_terry_loss(chosen_rewards, rejected_rewards)
        loss = loss / gradient_accumulation
        loss.backward()

        batch_size = chosen_ids.size(0)
        loss_meter.update(loss.item() * gradient_accumulation, batch_size)
        acc = calculate_accuracy(chosen_rewards, rejected_rewards)
        acc_meter.update(acc, batch_size)

        if (step + 1) % gradient_accumulation == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        pbar.set_postfix({
            "loss": f"{loss_meter.avg:.4f}",
            "acc":  f"{acc_meter.avg:.4f}"
        })

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str
) -> tuple[float, float]:
    """Evaluate the model on a dataloader. Returns (avg_loss, avg_accuracy)."""
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for batch in tqdm(dataloader, desc="  Evaluating"):
        chosen_ids = batch['chosen_input_ids'].to(device)
        chosen_mask = batch['chosen_attention_mask'].to(device)
        rejected_ids = batch['rejected_input_ids'].to(device)
        rejected_mask = batch['rejected_attention_mask'].to(device)

        chosen_rewards = model(input_ids=chosen_ids, attention_mask=chosen_mask)
        rejected_rewards = model(input_ids=rejected_ids, attention_mask=rejected_mask)

        loss = bradley_terry_loss(chosen_rewards, rejected_rewards)
        batch_size = chosen_ids.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc = calculate_accuracy(chosen_rewards, rejected_rewards)
        acc_meter.update(acc, batch_size)

    return loss_meter.avg, acc_meter.avg


# Main Ablation Runner

def run_ablation_study(
    adv_ratios: list[float] = ADV_RATIOS,
    epochs: int = EPOCHS,
    device: str = None
) -> list[dict]:
    """
    Run the full ablation study by training a separate model for each
    adversarial mixing ratio.

    Returns a list of result dicts (one per ratio) with training metrics.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load tokenizer once
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load the full adversarial dataset once
    print(f"Loading adversarial dataset: {ADV_DATA_PATH}")
    with open(ADV_DATA_PATH, 'r') as f:
        all_pairs = json.load(f)
    print(f"  Loaded {len(all_pairs)} total pairs.")

    all_results = []

    for ratio in adv_ratios:
        ratio_label = f"{int(ratio * 100)}pct"
        print(f"\n{'='*60}")
        print(f"ABLATION: adv_ratio = {ratio:.0%} ({ratio_label})")
        print(f"{'='*60}")

        # Build dataset for this ratio
        pairs = build_dataset_for_ratio(all_pairs, adv_ratio=ratio)
        dataset = PairwiseDataset(pairs, tokenizer, max_length=MAX_LENGTH)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        # Load a fresh copy of the base model for each ablation
        print(f"\n  Loading base checkpoint: {BASE_CHECKPOINT}")
        model = DeBERTaRewardModel(MODEL_NAME)
        if os.path.exists(BASE_CHECKPOINT):
            model.load_state_dict(torch.load(BASE_CHECKPOINT, map_location=device))
            print("  Loaded trained weights.")
        else:
            print("  WARNING: No trained weights found, using untrained model.")
        model.to(device)

        # Optimizer and scheduler - same as original training
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        total_steps = (len(dataloader) // GRADIENT_ACCUMULATION) * epochs
        warmup_steps = int(0.06 * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Training loop
        ratio_metrics = []
        for epoch in range(1, epochs + 1):
            print(f"\n  --- Epoch {epoch}/{epochs} ---")
            train_loss, train_acc = train_one_epoch(
                model, dataloader, optimizer, scheduler, device, epoch
            )
            # For evaluation, re-use the training data (no separate val set here;
            # the final evaluation on clean test data is done in Day 16's notebook)
            print(f"  Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
            ratio_metrics.append({
                'epoch': epoch,
                'train_loss': round(train_loss, 4),
                'train_acc': round(train_acc, 4)
            })

        # Save checkpoint for this ratio
        ckpt_path = os.path.join(OUTPUT_DIR, f"reward_model_adv_{ratio_label}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"\n  Saved checkpoint: {ckpt_path}")

        all_results.append({
            'adv_ratio': ratio,
            'ratio_label': ratio_label,
            'n_pairs': len(pairs),
            'epochs': epochs,
            'metrics_per_epoch': ratio_metrics,
            'final_train_acc': ratio_metrics[-1]['train_acc'],
            'checkpoint': ckpt_path
        })

    # Save summary results JSON for Day 16 notebook to pick up
    results_path = os.path.join(RESULTS_DIR, 'ablation_training_summary.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nAll ablation runs complete. Summary saved to: {results_path}")

    return all_results


# Entry Point

if __name__ == "__main__":
    results = run_ablation_study()

    print("\n\n--- FINAL SUMMARY ---")
    for r in results:
        print(f"  {r['adv_ratio']:.0%} ratio | "
              f"final train acc = {r['final_train_acc']:.4f} | "
              f"checkpoint = {r['checkpoint']}")
