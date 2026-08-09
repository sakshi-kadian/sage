import re
import nltk
from nltk.corpus import wordnet
from src.adversarial.model_wrapper import RewardModelWrapper

class TextFoolerRewardAttack:
    """
    A Black-Box greedy word substitution attack inspired by TextFooler.
    Adapted specifically for continuous scalar Reward Models.
    
    Algorithm:
    1. Rank words by importance (by deleting them one-by-one and checking reward drop).
    2. Find synonyms for top important words using WordNet.
    3. Greedily substitute words with synonyms that maximally drop the reward score.
    """
    def __init__(self, model_wrapper: RewardModelWrapper, max_substitutions: int = 5):
        self.model = model_wrapper
        self.max_substitutions = max_substitutions
        
        # Ensure NLTK resources are available
        try:
            wordnet.synsets('hello')
        except LookupError:
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            
    def get_synonyms(self, word: str) -> list[str]:
        """Fetch synonyms using NLTK WordNet."""
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                # Filter out multi-word synonyms and the original word itself
                synonym = lemma.name().replace('_', ' ').lower()
                if synonym != word.lower() and ' ' not in synonym:
                    synonyms.add(synonym)
        return list(synonyms)
        
    def get_word_importance(self, words: list[str], original_reward: float) -> list[tuple[str, int, float]]:
        """Rank words by how much removing them drops the reward."""
        if not words:
            return []
            
        masked_texts = []
        for i in range(len(words)):
            # Remove the i-th word to see its impact
            masked_words = words[:i] + words[i+1:]
            masked_texts.append(" ".join(masked_words))
            
        # Batch score all masked texts
        masked_rewards = self.model.get_rewards_batch(masked_texts)
        
        # Importance = Original Reward - Masked Reward 
        # (Higher importance means the word was boosting the score)
        importance_scores = []
        for i in range(len(words)):
            importance = original_reward - masked_rewards[i]
            importance_scores.append((words[i], i, importance))
            
        # Sort by importance descending
        importance_scores.sort(key=lambda x: x[2], reverse=True)
        return importance_scores

    def attack(self, text: str, original_reward: float = None) -> dict:
        """
        Executes the greedy synonym substitution attack to minimize the reward.
        Returns a dictionary with attack stats and the final adversarial text.
        """
        if original_reward is None:
            original_reward = self.model.get_reward(text)
            
        # Simple tokenization by words
        words = re.findall(r'\b\w+\b', text)
        
        # 1. Rank words by importance
        importance_ranking = self.get_word_importance(words, original_reward)
        
        current_text = text
        current_reward = original_reward
        substitutions_made = 0
        
        # 2. Iterate through most important words
        for word, idx, importance in importance_ranking:
            if substitutions_made >= self.max_substitutions:
                break
                
            # Skip very short words (stop words like 'a', 'is', 'the')
            if len(word) <= 3:
                continue
                
            synonyms = self.get_synonyms(word)
            if not synonyms:
                continue
                
            # 3. Generate candidate texts with synonyms
            candidate_texts = []
            for syn in synonyms:
                candidate_words = words.copy()
                candidate_words[idx] = syn
                candidate_texts.append(" ".join(candidate_words))
                
            candidate_rewards = self.model.get_rewards_batch(candidate_texts)
            
            # 4. Find the synonym that drops the reward the most
            min_reward = min(candidate_rewards)
            best_syn_idx = candidate_rewards.index(min_reward)
            
            # If the best synonym successfully drops the reward lower than current
            if min_reward < current_reward:
                current_reward = min_reward
                words[idx] = synonyms[best_syn_idx]  # Update word list for future iterations
                current_text = candidate_texts[best_syn_idx]
                substitutions_made += 1
                
        return {
            'original_text': text,
            'adversarial_text': current_text,
            'original_reward': original_reward,
            'adversarial_reward': current_reward,
            'reward_drop': original_reward - current_reward,
            'substitutions': substitutions_made
        }
