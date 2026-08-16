import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class DeBERTaRewardModel(nn.Module):
    """
    Reward Model using DeBERTa-v3-large as the backbone.
    Outputs a single scalar reward for a given input sequence.
    """
    def __init__(self, model_name_or_path: str = "microsoft/deberta-v3-large"):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        # Load the base model without any classification head
        self.backbone = AutoModel.from_pretrained(model_name_or_path, config=self.config)
        
        # Scalar reward head
        self.reward_head = nn.Sequential(
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(self.config.hidden_size, 1)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Returns the scalar reward for each item in the batch.
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False
        )
        # Use the [CLS] token equivalent (first token) for DeBERTa
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Get scalar reward (cast to match the head's dtype to prevent Kaggle float16 mismatches)
        pooled_output = pooled_output.to(self.reward_head[1].weight.dtype)
        reward = self.reward_head(pooled_output)
        return reward.squeeze(-1)

def bradley_terry_loss(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> torch.Tensor:
    """
    Bradley-Terry Pairwise Ranking Loss.
    
    L = -log(sigmoid(r_chosen - r_rejected))
    
    The model is penalized if the rejected response gets a higher reward than the chosen one.
    """
    # Calculate the difference in rewards
    reward_diff = chosen_rewards - rejected_rewards
    
    # Compute the log-sigmoid of the difference
    loss = -nn.functional.logsigmoid(reward_diff).mean()
    return loss
