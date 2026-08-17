"""
Visualizer for Attention Rollout Results
=========================================
Generates publication-quality heatmap figures from attention rollout data.

Produces:
    - Token influence bar chart (which words drive the reward score most)
    - Full 2D attention rollout heatmap (token-to-token influence matrix)
    - Side-by-side comparison of chosen vs. rejected response influence
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch


# Publication-quality style settings
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlepad": 14,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})


def plot_token_influence(
    tokens: list[str],
    cls_influence: np.ndarray,
    reward: float,
    title: str = "Token Influence on Reward Score",
    save_path: str = None,
    top_k: int = 20
) -> plt.Figure:
    """
    Bar chart showing per-token influence on the [CLS] reward token.
    Only shows top_k most influential tokens for readability.

    Args:
        tokens: List of token strings.
        cls_influence: 1D array of influence scores (same length as tokens).
        reward: The scalar reward score for this text.
        title: Plot title string.
        save_path: If provided, saves the figure to this path.
        top_k: Number of top tokens to display.

    Returns:
        matplotlib Figure object.
    """
    # Normalise influence scores to [0, 1] range
    influence = cls_influence.copy()
    influence = (influence - influence.min()) / (influence.max() - influence.min() + 1e-9)

    # Skip [CLS], [SEP], and padding tokens for cleaner visuals
    skip_tokens = {"[CLS]", "[SEP]", "<s>", "</s>", "[PAD]", "▁"}
    valid_indices = [
        i for i, t in enumerate(tokens)
        if t not in skip_tokens and not t.startswith("[PAD")
    ]
    filtered_tokens = [tokens[i] for i in valid_indices]
    filtered_influence = influence[valid_indices]

    # Sort by influence and take top_k
    sorted_idx = np.argsort(filtered_influence)[::-1][:top_k]
    top_tokens = [filtered_tokens[i] for i in sorted_idx]
    top_scores = filtered_influence[sorted_idx]

    # Build colour map: high influence = Darker Purple, low = light tint
    cmap = mcolors.LinearSegmentedColormap.from_list("DarkPurple", ["#F5EEF8", "#7D3C98"])
    colours = [cmap(s * 0.85 + 0.15) for s in top_scores]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(top_tokens)), top_scores, color=colours, edgecolor="white", linewidth=0.5)

    ax.set_yticks(range(len(top_tokens)))
    ax.set_yticklabels(top_tokens, fontsize=9)
    ax.invert_yaxis()  # Most influential at the top
    ax.set_xlabel("Normalised Influence on [CLS] Token")
    ax.set_title(f"{title}\nReward Score: {reward:.4f}", pad=12)
    ax.set_xlim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate bars with their scores
    for bar, score in zip(bars, top_scores):
        ax.text(
            score + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}", va="center", fontsize=8, color="#333333"
        )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_rollout_heatmap(
    tokens: list[str],
    rollout_matrix: np.ndarray,
    title: str = "Attention Rollout Heatmap",
    save_path: str = None,
    max_tokens: int = 30
) -> plt.Figure:
    """
    Full 2D heatmap of the rollout matrix.
    Each cell (i, j) shows how much token j influenced token i after rollout.

    Args:
        tokens: List of token strings.
        rollout_matrix: 2D numpy array [seq_len, seq_len].
        title: Plot title.
        save_path: Optional save path.
        max_tokens: Truncate to this many tokens for legibility.

    Returns:
        matplotlib Figure object.
    """
    n = min(len(tokens), max_tokens)
    matrix = rollout_matrix[:n, :n]
    labels = tokens[:n]

    fig, ax = plt.subplots(figsize=(12, 10))
    # Create custom Dark Purple colormap for the heatmap (avoids pure white)
    cmap = mcolors.LinearSegmentedColormap.from_list("DarkPurple", ["#F5EEF8", "#7D3C98"])
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0)

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Source Token (attending from)")
    ax.set_ylabel("Target Token (attended to)")
    ax.set_title(title, pad=12)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Rollout Influence")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_chosen_vs_rejected(
    chosen_result: dict,
    rejected_result: dict,
    save_path: str = None,
    top_k: int = 15
) -> plt.Figure:
    """
    Side-by-side comparison of token influence for a chosen vs rejected pair.
    This is the key figure for the paper — it shows WHY the model prefers one response.

    Args:
        chosen_result: Output dict from run_attention_rollout() for chosen text.
        rejected_result: Output dict from run_attention_rollout() for rejected text.
        save_path: Optional path to save the figure.
        top_k: Number of top tokens to show per side.

    Returns:
        matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"Chosen (reward={chosen_result['reward']:.3f}) vs "
        f"Rejected (reward={rejected_result['reward']:.3f})",
        fontsize=13, fontweight="bold"
    )

    for ax, result, label, colour_map in zip(
        axes,
        [chosen_result, rejected_result],
        ["Chosen Response", "Rejected Response"],
        [
            mcolors.LinearSegmentedColormap.from_list("DarkPurple", ["#F5EEF8", "#7D3C98"]),
            mcolors.LinearSegmentedColormap.from_list("DarkCoral", ["#FDEDEC", "#CB4335"])
        ]
    ):
        tokens = result["tokens"]
        influence = result["cls_influence"].copy()
        influence = (influence - influence.min()) / (influence.max() - influence.min() + 1e-9)

        skip = {"[CLS]", "[SEP]", "<s>", "</s>", "[PAD]", "▁"}
        valid = [(t, s) for t, s in zip(tokens, influence) if t not in skip]
        if len(valid) == 0:
            continue
        valid_tokens, valid_scores = zip(*valid)

        sorted_idx = np.argsort(valid_scores)[::-1][:top_k]
        top_tokens = [valid_tokens[i] for i in sorted_idx]
        top_scores = np.array([valid_scores[i] for i in sorted_idx])
        colours = [colour_map(s * 0.85 + 0.15) for s in top_scores]

        ax.barh(range(len(top_tokens)), top_scores, color=colours, edgecolor="white")
        ax.set_yticks(range(len(top_tokens)))
        ax.set_yticklabels(top_tokens, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Normalised Influence")
        ax.set_title(label, fontsize=11)
        ax.set_xlim(0, 1.1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig
