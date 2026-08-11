"""
Stealth Filters for Adversarial Examples
=========================================
Ensures generated adversarial texts are both:
  1. Semantically equivalent  (SBERT cosine similarity >= threshold)
  2. Fluent English            (GPT-2 perplexity <= threshold)

A substitution that fails either filter is discarded as "detectable"
and the original text is returned instead.

Usage:
    from src.adversarial.filters import StealthFilter
    sf = StealthFilter()
    result = sf.check(original_text, adversarial_text)
    if result['passes']:
        # attack is stealthy — use it
"""

import math
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


class StealthFilter:
    """
    Dual stealth filter combining semantic and fluency checks.

    Parameters
    ----------
    sbert_model   : HuggingFace model name for SentenceTransformer
    gpt2_model    : HuggingFace model name for GPT-2 perplexity scorer
    sbert_threshold : Minimum cosine similarity to accept (default 0.85)
    ppl_threshold   : Maximum GPT-2 perplexity to accept (default 200.0)
    device          : 'cpu' or 'cuda'
    """

    def __init__(
        self,
        sbert_model: str = 'all-MiniLM-L6-v2',
        gpt2_model: str = 'gpt2',
        sbert_threshold: float = 0.85,
        ppl_threshold: float = 200.0,
        device: str = 'cpu',
    ):
        self.sbert_threshold = sbert_threshold
        self.ppl_threshold = ppl_threshold
        self.device = device

        print(f"  Loading SBERT ({sbert_model})...")
        self.sbert = SentenceTransformer(sbert_model, device=device)

        print(f"  Loading GPT-2 ({gpt2_model})...")
        self.gpt2_tokenizer = GPT2TokenizerFast.from_pretrained(gpt2_model)
        self.gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model).to(device)
        self.gpt2_model.eval()
        print("  Stealth filter ready.")

    # Semantic Similarity

    def semantic_similarity(self, text_a: str, text_b: str) -> float:
        """Returns cosine similarity between two sentences using SBERT."""
        embs = self.sbert.encode([text_a, text_b], convert_to_tensor=True)
        score = util.cos_sim(embs[0], embs[1]).item()
        return score

    # GPT-2 Perplexity

    def perplexity(self, text: str) -> float:
        """
        Computes GPT-2 perplexity of a text string.
        Lower perplexity = more fluent / natural English.
        """
        enc = self.gpt2_tokenizer(text, return_tensors='pt')
        input_ids = enc['input_ids'].to(self.device)

        if input_ids.shape[1] == 0:
            return float('inf')

        with torch.no_grad():
            loss = self.gpt2_model(input_ids, labels=input_ids).loss

        return math.exp(loss.item())

    # Combined Check

    def check(self, original: str, adversarial: str) -> dict:
        """
        Run both filters and return a result dict.

        Returns
        -------
        dict with keys:
          passes         : bool — True if both filters passed
          similarity     : float — SBERT cosine similarity
          adv_perplexity : float — GPT-2 perplexity of adversarial text
          orig_perplexity: float — GPT-2 perplexity of original text
          fail_reason    : str or None — which filter failed (if any)
        """
        similarity = self.semantic_similarity(original, adversarial)
        orig_ppl = self.perplexity(original)
        adv_ppl = self.perplexity(adversarial)

        passes = True
        fail_reason = None

        if similarity < self.sbert_threshold:
            passes = False
            fail_reason = (
                f"SBERT similarity {similarity:.3f} < threshold {self.sbert_threshold}"
            )
        elif adv_ppl > self.ppl_threshold:
            passes = False
            fail_reason = (
                f"GPT-2 perplexity {adv_ppl:.1f} > threshold {self.ppl_threshold}"
            )

        return {
            'passes': passes,
            'similarity': similarity,
            'orig_perplexity': orig_ppl,
            'adv_perplexity': adv_ppl,
            'fail_reason': fail_reason,
        }

    def filter_batch(self, pairs: list[tuple[str, str]]) -> list[dict]:
        """
        Run the filter over a list of (original, adversarial) pairs.
        Returns a list of result dicts (one per pair).
        """
        results = []
        for original, adversarial in pairs:
            result = self.check(original, adversarial)
            result['original'] = original
            result['adversarial'] = adversarial
            results.append(result)
        return results
