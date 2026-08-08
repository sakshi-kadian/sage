"""
Attention Rollout for DeBERTa-v3-large Reward Model
=====================================================
Implements the residual-aware multi-layer attention rollout algorithm
from Abnar & Zuidema (2020): "Quantifying Attention Flow in Transformers".

Key Insight:
    Raw attention weights from a single layer only tell half the story.
    Because of residual connections, information also flows through a
    "skip path" that bypasses attention entirely. Rollout accounts for
    this by adding the identity matrix (the residual) to each layer's
    attention matrix before recursively multiplying through all layers.

Algorithm:
    For layer l, attention matrix A_l (averaged over heads):
        A_hat_l = 0.5 * A_l + 0.5 * I   (mix attention + residual)
        A_hat_l = normalize rows to sum to 1
    
    Rollout (recursive product from layer 1 to L):
        R_1 = A_hat_1
        R_l = A_hat_l @ R_{l-1}
    
    Final R[0, :] gives the influence of every token on the [CLS] token,
    which is what drives the scalar reward prediction.
"""

import torch
import numpy as np
from transformers import AutoTokenizer
from src.reward_model import DeBERTaRewardModel


def extract_attention_weights(
    model: DeBERTaRewardModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor
) -> list[torch.Tensor]:
    """
    Run a forward pass and extract raw attention weights from all 24 layers.

    Args:
        model: The loaded DeBERTaRewardModel.
        input_ids: Tokenized input [1, seq_len].
        attention_mask: Attention mask [1, seq_len].

    Returns:
        List of 24 attention matrices, each of shape [num_heads, seq_len, seq_len].
    """
    model.eval()
    with torch.no_grad():
        outputs = model.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,    # This is the key flag
            output_hidden_states=False
        )
    # outputs.attentions is a tuple of 24 tensors
    # Each tensor: [batch=1, num_heads, seq_len, seq_len]
    attention_weights = [attn[0] for attn in outputs.attentions]
    return attention_weights


def compute_rollout(
    attention_weights: list[torch.Tensor],
    head_fusion: str = "mean"
) -> np.ndarray:
    """
    Compute Attention Rollout across all 24 layers.

    Args:
        attention_weights: List of per-layer attention tensors
                           [num_heads, seq_len, seq_len].
        head_fusion: How to combine multiple attention heads.
                     Options: "mean", "max", "min".

    Returns:
        rollout_matrix: numpy array of shape [seq_len, seq_len].
                        rollout_matrix[0, :] gives the influence of each
                        token on the final [CLS] token representation.
    """
    num_layers = len(attention_weights)
    seq_len = attention_weights[0].shape[-1]

    # Start rollout as identity (before any layers, nothing has happened)
    rollout = torch.eye(seq_len)

    for layer_idx in range(num_layers):
        attn = attention_weights[layer_idx]  # [num_heads, seq_len, seq_len]

        # Fuse attention heads
        if head_fusion == "mean":
            attn_fused = attn.mean(dim=0)   # [seq_len, seq_len]
        elif head_fusion == "max":
            attn_fused = attn.max(dim=0).values
        elif head_fusion == "min":
            attn_fused = attn.min(dim=0).values
        else:
            raise ValueError(f"Unknown head_fusion: {head_fusion}")

        # Add residual connection: mix 50/50 with identity
        identity = torch.eye(seq_len)
        attn_with_residual = 0.5 * attn_fused + 0.5 * identity

        # Row-normalize so rows sum to 1 (keeps it a valid probability dist)
        row_sums = attn_with_residual.sum(dim=-1, keepdim=True)
        attn_normalized = attn_with_residual / (row_sums + 1e-9)

        # Recursive product: chain the current layer with all previous layers
        rollout = attn_normalized @ rollout

    return rollout.numpy()


def run_attention_rollout(
    model: DeBERTaRewardModel,
    tokenizer,
    text: str,
    device: str = "cpu",
    head_fusion: str = "mean"
) -> dict:
    """
    Full pipeline: tokenize text → extract attentions → compute rollout.

    Args:
        model: Loaded DeBERTaRewardModel.
        tokenizer: Corresponding tokenizer.
        text: The raw text string to analyse.
        device: "cpu" or "cuda".
        head_fusion: Head aggregation method.

    Returns:
        Dictionary with:
            - "tokens": list of decoded token strings
            - "cls_influence": 1D array of per-token influence on [CLS]
            - "rollout_matrix": full 2D rollout matrix
            - "reward": scalar reward score for this text
    """
    # Tokenize
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,   # Use shorter length for interpretability speed
        padding=False
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # Get reward score
    with torch.no_grad():
        reward = model(input_ids=input_ids, attention_mask=attention_mask)
    reward_score = reward.item()

    # Extract attention weights from all 24 layers
    attn_weights = extract_attention_weights(model, input_ids, attention_mask)

    # Compute rollout
    rollout_matrix = compute_rollout(attn_weights, head_fusion=head_fusion)

    # CLS influence: row 0 of rollout shows what [CLS] attends to overall
    cls_influence = rollout_matrix[0]

    # Decode tokens for labelling
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    return {
        "tokens": tokens,
        "cls_influence": cls_influence,
        "rollout_matrix": rollout_matrix,
        "reward": reward_score,
        "seq_len": len(tokens)
    }
