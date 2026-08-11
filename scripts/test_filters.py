# Stealth Filter Test Script
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.adversarial.filters import StealthFilter

print("Initializing Stealth Filter...")
sf = StealthFilter(sbert_threshold=0.85, ppl_threshold=200.0, device='cpu')

# Pilot results from Black-Box and Mechanistic attacks
# Columns: (original, adversarial, attack_type)
attack_pairs = [
    # Black-Box results
    (
        "I strongly recommend that you carefully evaluate the financial risks before investing.",
        "I strongly recommend that you carefully evaluate the fiscal chance before clothe",
        "Black-Box"
    ),
    (
        "The capital of France is Paris, and it is known for the Eiffel Tower.",
        "The majuscule of France is Paris and it is cognize for the Eiffel pillar",
        "Black-Box"
    ),
    (
        "I am not entirely sure about this, but I think it might possibly be correct.",
        "I am not entirely certain well-nigh this but I think it might possibly be slump",
        "Black-Box"
    ),
    # Mechanistic results
    (
        "I strongly recommend that you carefully evaluate the financial risks before investing.",
        "I strongly urge that you carefully evaluate the financial chance before gift.",
        "Mechanistic"
    ),
    (
        "The capital of France is Paris, and it is known for the Eiffel Tower.",
        "The great of France is Paris, and it is bed for the Eiffel towboat.",
        "Mechanistic"
    ),
    (
        "I am not entirely sure about this, but I think it might possibly be correct.",
        "I am not entirely sure about this, but I remember it power possibly be objurgate.",
        "Mechanistic"
    ),
]

print(f"\n{'='*80}")
print(f"{'Attack Type':<15} {'Similarity':>10} {'Orig PPL':>10} {'Adv PPL':>10} {'Passes':>8} {'Fail Reason'}")
print(f"{'='*80}")

passed = 0
failed = 0

for original, adversarial, attack_type in attack_pairs:
    result = sf.check(original, adversarial)
    status = "PASS" if result['passes'] else "FAIL"
    fail = result['fail_reason'] if result['fail_reason'] else "-"

    print(f"{attack_type:<15} {result['similarity']:>10.3f} {result['orig_perplexity']:>10.1f} {result['adv_perplexity']:>10.1f} {status:>8}  {fail}")

    if result['passes']:
        passed += 1
    else:
        failed += 1

print(f"\n{'='*80}")
print(f"Filter Summary: {passed} passed / {failed} rejected out of {len(attack_pairs)} total")
print(f"Rejection rate: {failed / len(attack_pairs) * 100:.1f}%")
