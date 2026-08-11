"""
Mechanistically-Guided White-Box Adversarial Attack
====================================================
Uses Attention Rollout influence scores (computed in Day 7) to directly
target the highest-influence tokens in a response.

Key difference from TextFooler (Black-Box):
  - TextFooler: Ranks words by running N deletion passes (blind guessing).
  - This attack: Uses the model's internal attention flow to KNOW which
    tokens matter most before making a single substitution attempt.

This is the ICLR Flex: interpretability directly powering adversarial attacks.
"""

import re
import nltk
from nltk.corpus import wordnet

from src.adversarial.model_wrapper import RewardModelWrapper
from src.interpretability.attention_rollout import run_attention_rollout


class MechanisticAttack:
    """
    White-Box adversarial attack guided by Attention Rollout scores.

    Algorithm:
    1. Run Attention Rollout on the input text to get per-token CLS influence.
    2. Rank tokens by their influence score (descending) — no guessing needed.
    3. For each high-influence token, fetch WordNet synonyms.
    4. Greedily substitute with the synonym that causes the largest reward drop.

    Requires access to model internals (attention weights) — hence White-Box.
    """

    def __init__(
        self,
        model,
        tokenizer,
        model_wrapper: RewardModelWrapper,
        device: str = 'cpu',
        max_substitutions: int = 5,
        head_fusion: str = 'mean',
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.wrapper = model_wrapper
        self.device = device
        self.max_substitutions = max_substitutions
        self.head_fusion = head_fusion

        # Ensure NLTK resources are available
        try:
            wordnet.synsets('hello')
        except LookupError:
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)

    def get_synonyms(self, word: str) -> list[str]:
        """Fetch synonyms using NLTK WordNet, filtering multi-word entries."""
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ').lower()
                if synonym != word.lower() and ' ' not in synonym:
                    synonyms.add(synonym)
        return list(synonyms)

    def attack(self, text: str) -> dict:
        """
        Execute the mechanistic attack.

        Returns a dict with:
          - original_text, adversarial_text
          - original_reward, adversarial_reward, reward_drop
          - substitutions: number of words swapped
          - token_ranking: the attention rollout influence scores used
        """
        # Step 1: Run Attention Rollout to get token influence scores
        rollout_result = run_attention_rollout(
            model=self.model,
            tokenizer=self.tokenizer,
            text=text,
            device=self.device,
            head_fusion=self.head_fusion,
        )

        original_reward = rollout_result['reward']
        subtokens = rollout_result['tokens']          # e.g. ['▁I', '▁deeply', ...]
        influence = rollout_result['cls_influence']   # shape: (n_tokens,)

        # Step 2: Map subtokens to whole words, aggregate influence
        # DeBERTa uses SentencePiece; leading ▁ means start of a new word.
        word_influence = {}   # word_str -> max influence across its subtokens
        word_positions = {}   # word_str -> list of subtoken indices

        current_word = ""
        current_indices = []
        for i, tok in enumerate(subtokens):
            # Skip special tokens
            if tok in ('[CLS]', '[SEP]', '<s>', '</s>', '[PAD]'):
                continue
            clean = tok.lstrip('▁Ġ')   # strip SentencePiece / GPT-2 prefix
            if tok.startswith('▁') or tok.startswith('Ġ') or not current_word:
                if current_word:
                    key = current_word.lower().strip('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
                    if key:
                        word_influence[key] = max(word_influence.get(key, 0),
                                                  max(influence[j] for j in current_indices))
                        word_positions.setdefault(key, current_indices[:])
                current_word = clean
                current_indices = [i]
            else:
                current_word += clean
                current_indices.append(i)

        # flush last word
        if current_word:
            key = current_word.lower().strip('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
            if key:
                word_influence[key] = max(word_influence.get(key, 0),
                                          max(influence[j] for j in current_indices))
                word_positions.setdefault(key, current_indices[:])

        # Step 3: Sort words by influence (highest first)
        ranked_words = sorted(word_influence.items(), key=lambda x: x[1], reverse=True)

        # Step 4: Greedy synonym substitution
        # Work on a plain word-list for easy in-place substitution
        words = re.findall(r'\b\w+\b', text)
        current_text = text
        current_reward = original_reward
        substitutions_made = 0

        token_ranking = [(w, float(s)) for w, s in ranked_words]

        for word, score in ranked_words:
            if substitutions_made >= self.max_substitutions:
                break
            if len(word) <= 3:
                continue

            synonyms = self.get_synonyms(word)
            if not synonyms:
                continue

            # Build candidate texts by swapping this word wherever it appears
            candidate_texts = []
            for syn in synonyms:
                # Case-insensitive replacement of the first occurrence
                candidate = re.sub(
                    rf'\b{re.escape(word)}\b',
                    syn,
                    current_text,
                    count=1,
                    flags=re.IGNORECASE
                )
                candidate_texts.append(candidate)

            if not candidate_texts:
                continue

            candidate_rewards = self.wrapper.get_rewards_batch(candidate_texts)
            min_reward = min(candidate_rewards)
            best_idx = candidate_rewards.index(min_reward)

            if min_reward < current_reward:
                current_reward = min_reward
                current_text = candidate_texts[best_idx]
                substitutions_made += 1

        return {
            'original_text': text,
            'adversarial_text': current_text,
            'original_reward': original_reward,
            'adversarial_reward': current_reward,
            'reward_drop': original_reward - current_reward,
            'substitutions': substitutions_made,
            'token_ranking': token_ranking,   # useful for the notebook visualization
        }
