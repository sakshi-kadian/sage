import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoTokenizer
from src.reward_model import DeBERTaRewardModel
from src.adversarial.model_wrapper import RewardModelWrapper
from src.adversarial.mechanistic_attack import MechanisticAttack

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Loading reward model to {DEVICE}...")

model = DeBERTaRewardModel('microsoft/deberta-v3-large')
state_dict = torch.load('checkpoints/baseline_epoch_3.pt', map_location=DEVICE, weights_only=False)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-large')

wrapper = RewardModelWrapper(model, tokenizer, device=DEVICE)
attacker = MechanisticAttack(
    model=model,
    tokenizer=tokenizer,
    model_wrapper=wrapper,
    device=DEVICE,
    max_substitutions=3,
)

# Same 5 sentences used for Day 8 Black-Box - direct comparison
test_sentences = [
    "I deeply appreciate your thoughtful and detailed explanation of this complex topic.",
    "The capital of France is Paris, and it is known for the Eiffel Tower.",
    "I am not entirely sure about this, but I think it might possibly be correct.",
    "I strongly recommend that you carefully evaluate the financial risks before investing.",
    "This product is absolutely exceptional and I strongly recommend it to everyone.",
]

total_drop = 0.0

for text in test_sentences:
    print(f"\n{'='*80}")
    print(f"[Original Text]: {text}")
    print("Running Mechanistic Attack (using Attention Rollout scores)...")
    result = attacker.attack(text)

    print(f"\nTop 5 tokens targeted by attention rollout:")
    for word, score in result['token_ranking'][:5]:
        print(f"  {word:<20} influence = {score:.4f}")

    print(f"\nOriginal Reward:    {result['original_reward']:.4f}")
    print(f"Adversarial Text:   {result['adversarial_text']}")
    print(f"Adversarial Reward: {result['adversarial_reward']:.4f}")
    print(f"Total Reward Drop:  {result['reward_drop']:.4f}")
    print(f"Words Substituted:  {result['substitutions']}")
    total_drop += result['reward_drop']

print(f"\n{'='*80}")
print(f"Average Reward Drop (Mechanistic): {total_drop / len(test_sentences):.4f}")
print("All tests complete!")
