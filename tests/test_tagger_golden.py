"""Golden test: the hand-reassembled logits must match SigLIP's own combined forward,
so the scores callers depend on (confidence, threshold/margin selection) don't drift.

  .venv/bin/python tests/test_tagger_golden.py     # or pytest, if installed
"""
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.settings import SIGLIP_ID  # noqa: E402
from app.stages import tagger as tagger_mod  # noqa: E402
from app.stages.tagger import ZeroShotTagger  # noqa: E402

TOL = 1e-4
IMAGES = ["examples/images/apple.jpg", "examples/images/photo.jpg"]
# A few representative attribute label sets, rendered exactly as the pipeline would.
LABEL_SETS = [
    ["a photo of a red shoe", "a photo of a blue shoe", "a photo of a black shoe"],
    ["a photo of a shoe made of leather", "a photo of a shoe made of suede", "a photo of a shoe made of canvas"],
    ["a photo of shoes", "a photo of bags", "a photo of tops", "a photo of pants"],
]

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _reference_softmax(tagger, image, labels):
    """The original combined path: one model() forward, softmax over logits_per_image."""
    inputs = tagger.processor(text=labels, images=image, padding="max_length", return_tensors="pt").to(tagger.device)
    with torch.no_grad():
        return tagger.model(**inputs).logits_per_image[0].softmax(dim=-1)


def _candidate_softmax(tagger, img_emb, labels):
    """Refactored core: cached text embeddings + reconstructed logits, one template
    per label so it must equal the combined forward."""
    txt_emb = tagger._encode_texts(labels)
    return tagger._logits(img_emb, txt_emb).softmax(dim=-1)


def _load_image(rel):
    return Image.open(os.path.join(_ROOT, rel)).convert("RGB")


def test_scores_match_combined_forward():
    tagger = ZeroShotTagger(SIGLIP_ID)
    worst = 0.0
    for rel in IMAGES:
        image = _load_image(rel)
        img_emb = tagger.encode_image(image)
        for labels in LABEL_SETS:
            ref = _reference_softmax(tagger, image, labels)
            cand = _candidate_softmax(tagger, img_emb, labels)
            diff = float((ref - cand).abs().max())
            worst = max(worst, diff)
            assert diff < TOL, f"{rel} / {labels[0]!r}: max abs diff {diff:.2e} >= {TOL:.0e}"
    print(f"  scores match combined forward — worst diff {worst:.2e} (tol {TOL:.0e})")


def test_text_cache_is_populated_and_reused():
    tagger_mod._TEXT_CACHE.clear()
    tagger = ZeroShotTagger(SIGLIP_ID)
    labels = LABEL_SETS[0]
    tagger._encode_texts(labels)
    assert len(tagger_mod._TEXT_CACHE) == len(labels), "every rendered label should be cached"
    # Second pass must hit the cache: no encode calls, identical keys.
    keys_before = set(tagger_mod._TEXT_CACHE)
    tagger._encode_texts(labels)
    assert set(tagger_mod._TEXT_CACHE) == keys_before, "reuse must not create new keys"
    print(f"  text cache populated and reused — {len(tagger_mod._TEXT_CACHE)} entries")


def test_cache_key_changes_with_label_text():
    """Editing the rendered label (prompt/option/category) must yield a new key —
    this is what makes stale entries impossible."""
    tagger_mod._TEXT_CACHE.clear()
    tagger = ZeroShotTagger(SIGLIP_ID)
    tagger._encode_texts(["a photo of a red shoe"])
    tagger._encode_texts(["a photo of a crimson shoe"])
    assert (SIGLIP_ID, "a photo of a red shoe") in tagger_mod._TEXT_CACHE
    assert (SIGLIP_ID, "a photo of a crimson shoe") in tagger_mod._TEXT_CACHE
    print("  distinct label text -> distinct cache keys (no staleness)")


def main():
    tests = [
        test_scores_match_combined_forward,
        test_text_cache_is_populated_and_reused,
        test_cache_key_changes_with_label_text,
    ]
    for t in tests:
        print(f"{t.__name__}:")
        t()
    print(f"\nAll {len(tests)} golden tests passed.")


if __name__ == "__main__":
    main()
