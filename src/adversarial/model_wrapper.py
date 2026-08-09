import torch

class RewardModelWrapper:
    """
    Wraps the DeBERTa reward model to provide a clean API for adversarial attacks.
    Takes plain text strings and returns continuous scalar reward scores.
    """
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        
    def get_reward(self, text: str) -> float:
        """Returns the scalar reward for a single text string."""
        enc = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=128, padding=False)
        with torch.no_grad():
            score = self.model(enc['input_ids'].to(self.device), enc['attention_mask'].to(self.device))
        return score.item()

    def get_rewards_batch(self, texts: list[str], batch_size: int = 16) -> list[float]:
        """Scores a batch of texts efficiently."""
        rewards = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            enc = self.tokenizer(batch_texts, return_tensors='pt', truncation=True, max_length=128, padding=True)
            with torch.no_grad():
                scores = self.model(enc['input_ids'].to(self.device), enc['attention_mask'].to(self.device))
            # Flatten to 1D array to safely handle any batch size (including 1)
            rewards.extend(scores.view(-1).tolist())
        return rewards
