import torch
import torch.nn as nn
from tqdm import tqdm
from .utils import AverageMeter, calculate_accuracy
from .reward_model import bradley_terry_loss

class RewardTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dataloader,
        val_dataloader=None,
        device: str = "cuda",
        gradient_accumulation_steps: int = 1,
        fp16: bool = False
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.scaler = torch.cuda.amp.GradScaler(enabled=fp16)
        self.fp16 = fp16

    def train_epoch(self, epoch_idx: int):
        self.model.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch_idx}")
        self.optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            # Move to device
            chosen_ids = batch['chosen_input_ids'].to(self.device)
            chosen_mask = batch['chosen_attention_mask'].to(self.device)
            rejected_ids = batch['rejected_input_ids'].to(self.device)
            rejected_mask = batch['rejected_attention_mask'].to(self.device)

            with torch.cuda.amp.autocast(enabled=self.fp16):
                # Forward pass for chosen and rejected
                chosen_rewards = self.model(input_ids=chosen_ids, attention_mask=chosen_mask)
                rejected_rewards = self.model(input_ids=rejected_ids, attention_mask=rejected_mask)
                
                # Calculate loss
                loss = bradley_terry_loss(chosen_rewards, rejected_rewards)
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            self.scaler.scale(loss).backward()

            # Track metrics
            batch_size = chosen_ids.size(0)
            loss_meter.update(loss.item() * self.gradient_accumulation_steps, batch_size)
            acc = calculate_accuracy(chosen_rewards, rejected_rewards)
            acc_meter.update(acc, batch_size)

            if (step + 1) % self.gradient_accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}", "acc": f"{acc_meter.avg:.4f}"})
        
        return loss_meter.avg, acc_meter.avg
