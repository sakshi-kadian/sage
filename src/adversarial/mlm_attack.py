"""
BERT Masked Language Model (MLM) Adversarial Attack
===================================================

This module upgrades the naive WordNet substitution with contextual
substitution using BERT's fill-mask capabilities.

By masking a target word and having BERT predict the replacement,
the generated adversarial examples remain semantically coherent
and fluent in context, vastly increasing the stealth of the attack.
"""

import re
import torch
from transformers import pipeline

from src.adversarial.mechanistic_attack import run_attention_rollout

class MLMAttack:
    """
    Context-aware White-Box adversarial attack using BERT Masked Language Modeling.

    Parameters
    ----------
    model       : Target reward model to attack (e.g. DeBERTa)
    tokenizer   : Tokenizer for the reward model
    mlm_model   : HuggingFace model name for the fill-mask pipeline (default: bert-base-uncased)
    device      : 'cpu' or 'cuda'
    """

    def __init__(self, model, tokenizer, mlm_model='bert-base-uncased', device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        print(f"  Loading MLM ({mlm_model})...")
        # Use pipeline for easy fill-mask API
        dev_id = 0 if device == 'cuda' else -1
        self.mlm = pipeline('fill-mask', model=mlm_model, device=dev_id, top_k=10)
        self.mask_token = self.mlm.tokenizer.mask_token
        print("  MLM Attacker ready.")

    def attack(self, text: str, true_label: int = 1, budget: int = 3) -> dict:
        """
        Attack the input text using Attention Rollout + BERT MLM.

        Parameters
        ----------
        text       : Original input text
        true_label : 1 if text is 'chosen', 0 if 'rejected' (determines attack direction)
        budget     : Max number of words to substitute

        Returns
        -------
        dict with keys:
          - original_text
          - adversarial_text
          - original_reward
          - adversarial_reward
          - substitutions: number of words swapped
          - token_ranking: the attention rollout influence scores used
        """
        # Step 1: Run Attention Rollout to get token influence scores
        rollout_result = run_attention_rollout(
            model=self.model,
            tokenizer=self.tokenizer,
            text=text,
            device=self.device
        )
        
        orig_reward = rollout_result['reward']
        subtokens = rollout_result['tokens']
        influence = rollout_result['cls_influence']

        # Step 2: Map subtokens to whole words, aggregate influence
        word_influence = {}
        word_positions = {}
        
        # DeBERTa uses SentencePiece; leading ▁ means start of a new word.
        # Mask special tokens [CLS], [SEP]
        for i, sub in enumerate(subtokens):
            if sub in ['[CLS]', '[SEP]']:
                continue
                
            clean_sub = sub.replace('▁', '')
            if not clean_sub.strip():
                continue
                
            if sub.startswith('▁'):
                # New word
                key = clean_sub
                word_positions[key] = [i]
                word_influence[key] = influence[i]
            else:
                # Continuation
                if word_positions:
                    # Find last key
                    last_key = list(word_positions.keys())[-1]
                    new_key = last_key + clean_sub
                    
                    # Update dicts
                    current_indices = word_positions.pop(last_key)
                    current_indices.append(i)
                    
                    word_influence.pop(last_key)
                    
                    word_influence[new_key] = max(word_influence.get(new_key, 0), 
                                                  max(influence[j] for j in current_indices))
                    word_positions.setdefault(new_key, current_indices[:])

        # Step 3: Sort words by influence (highest first)
        # Filter out short words and punctuation
        ranked_words = []
        for w, inf in word_influence.items():
            clean_w = re.sub(r'[^a-zA-Z0-9]', '', w)
            if len(clean_w) > 2:
                ranked_words.append((clean_w, inf))
                
        ranked_words = sorted(ranked_words, key=lambda x: x[1], reverse=True)

        # Step 4: Greedy context-aware substitution
        current_text = text
        subs_made = 0
        best_reward = orig_reward
        
        # Determine goal: minimize reward for chosen (1), maximize for rejected (0)
        minimize = (true_label == 1)

        for target_word, _ in ranked_words:
            if subs_made >= budget:
                break
                
            # Create masked text
            # Ensure we only replace whole words
            pattern = r'\b' + re.escape(target_word) + r'\b'
            if not re.search(pattern, current_text):
                continue
                
            masked_text = re.sub(pattern, self.mask_token, current_text, count=1)
            
            # Ask BERT for suggestions
            try:
                predictions = self.mlm(masked_text)
            except Exception as e:
                # Fallback if something goes wrong (e.g. sequence too long)
                continue
                
            best_local_text = current_text
            best_local_reward = best_reward
            replaced = False

            # Test BERT suggestions against the reward model
            for pred in predictions:
                candidate_word = pred['token_str'].strip()
                
                # Filter bad suggestions (same word, subwords, non-alphabetic)
                if not candidate_word.isalpha():
                    continue
                if candidate_word.lower() == target_word.lower():
                    continue
                    
                candidate_text = masked_text.replace(self.mask_token, candidate_word)
                
                # Score candidate text
                inputs = self.tokenizer(candidate_text, return_tensors='pt', truncation=True, max_length=512).to(self.device)
                with torch.no_grad():
                    cand_reward = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask']).squeeze().item()
                
                # Check if it improves the attack
                if (minimize and cand_reward < best_local_reward) or \
                   (not minimize and cand_reward > best_local_reward):
                    best_local_reward = cand_reward
                    best_local_text = candidate_text
                    replaced = True
            
            if replaced:
                current_text = best_local_text
                best_reward = best_local_reward
                subs_made += 1

        return {
            'original_text': text,
            'adversarial_text': current_text,
            'original_reward': orig_reward,
            'adversarial_reward': best_reward,
            'substitutions': subs_made,
            'token_ranking': ranked_words[:10]
        }
