import torch
import sys
import os

# Ensure we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoTokenizer
from src.reward_model import DeBERTaRewardModel
from src.adversarial.model_wrapper import RewardModelWrapper
from src.adversarial.blackbox_attacks import TextFoolerRewardAttack

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Loading reward model to {DEVICE}...")

# Load Model
model = DeBERTaRewardModel('microsoft/deberta-v3-large')
state_dict = torch.load('checkpoints/baseline_epoch_3.pt', map_location=DEVICE, weights_only=False)
model.load_state_dict(state_dict)
model.to(DEVICE)

tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-large')

# Wrap model and initialize attacker
wrapper = RewardModelWrapper(model, tokenizer, device=DEVICE)
attacker = TextFoolerRewardAttack(wrapper, max_substitutions=3)

# Test sentences covering different domains and reward profiles
test_sentences = [
    "I deeply appreciate your thoughtful and detailed explanation of this complex topic.",
    "The capital of France is Paris, and it is known for the Eiffel Tower.",
    "I am not entirely sure about this, but I think it might possibly be correct.",
    "I strongly recommend that you carefully evaluate the financial risks before investing.",
    "This product is absolutely exceptional and I strongly recommend it to everyone.",
]

for text in test_sentences:
    print(f"\n{'='*80}")
    print(f"[Original Text]: {text}")
    print("Running Black-Box Attack...")
    result = attacker.attack(text)
    print(f"Original Reward:    {result['original_reward']:.4f}")
    print(f"Adversarial Text:   {result['adversarial_text']}")
    print(f"Adversarial Reward: {result['adversarial_reward']:.4f}")
    print(f"Total Reward Drop:  {result['reward_drop']:.4f}")
    print(f"Words Substituted:  {result['substitutions']}")
print(f"\n{'='*80}")
print("All tests complete!")
