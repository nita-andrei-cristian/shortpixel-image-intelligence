"""Offline accuracy harness — same scoring as the client's "Run test set".

Loads each NN-name.json sidecar, runs the pipeline, and scores attributes by set
equality (every expected value present, no extras; one point each).

  .venv/bin/python scripts/eval.py [TESTSET_DIR]   # default: ~/dev/images/testset
"""
import glob
import json
import os
import sys
import time

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.classes.pipeline import ProductIntelligencePipeline  # noqa: E402
from app.classes.taxonomy import Taxonomy  # noqa: E402
from app.settings import HINT_BOOST, SIGLIP_ID  # noqa: E402

DEFAULT_DIR = "/home/nita/dev/images/testset"
USE_TITLE_HINTS = False  # True = feed title words into `known` (upper-bound probe)


def cmp_list(exp, got):
    """match / partial / miss — mirrors the client's cmpList exactly."""
    exp, got = exp or [], got or []
    g = set(got)
    hit = sum(1 for x in exp if x in g)
    if hit == len(exp) and len(got) == len(exp):
        return "match"
    return "partial" if hit > 0 else "miss"


def cmp_best(exp, accepts, got):
    """Best status over the primary expected list and any accepted alternatives."""
    best = "miss"
    for candidate in [exp, *(accepts or [])]:
        st = cmp_list(candidate, got)
        if st == "match":
            return "match"
        if st == "partial":
            best = "partial"
    return best


def _title_hints(title: str, taxonomy: dict, category_id: str) -> dict:
    import re

    words = set(re.findall(r"[a-z]+", title.lower()))
    cat = next(c for c in taxonomy["categories"] if c["id"] == category_id)
    hints = {}
    for attr in cat["attributes"]:
        for opt in attr.get("options", []):
            if opt in words:  # whole-word match only
                hints[attr["key"]] = opt
                break
    return hints


def main(testset_dir):
    pipe = ProductIntelligencePipeline()
    cases = sorted(glob.glob(os.path.join(testset_dir, "[0-9]*.json")))
    # Multi-category sets keep one shared taxonomy.json instead of one per sidecar.
    shared_path = os.path.join(testset_dir, "taxonomy.json")
    shared_taxonomy = json.load(open(shared_path)) if os.path.exists(shared_path) else None

    total_pts = total_tot = 0
    rows = []
    for path in cases:
        case = json.load(open(path))
        image = Image.open(os.path.join(testset_dir, case["image"])).convert("RGB")
        meta = case.get("meta", {})
        taxonomy = Taxonomy(case.get("taxonomy") or shared_taxonomy)
        known = _title_hints(meta.get("title", ""), case.get("taxonomy") or shared_taxonomy,
                             case["expected"]["category"]) if USE_TITLE_HINTS else {}

        t0 = time.perf_counter()
        resp = pipe.analyze(image, taxonomy, meta, known, tagging=True)
        ms = (time.perf_counter() - t0) * 1000

        exp = case["expected"]
        accept = exp.get("accept", {})
        pts = tot = 0
        attr_bits = []
        for k, ev in exp["attributes"].items():
            tot += 1
            got = (resp["attributes"].get(k) or {}).get("value", [])
            st = cmp_best(ev, accept.get(k), got) if k in resp["attributes"] else "na"
            if st == "match":
                pts += 1
            else:
                attr_bits.append(f"{k}:{','.join(ev)}→{','.join(got) or '∅'}")
        cat_ok = resp["category"] == exp["category"]
        total_pts += pts
        total_tot += tot
        rows.append((case["image"], case.get("difficulty", "?"), "✓" if cat_ok else "✗",
                     f"{pts}/{tot}", round(ms), "  ".join(attr_bits)))

    w = max(len(r[0]) for r in rows)
    print(f"{'image':<{w}}  diff    cat  score   ms     misses")
    for img, diff, cat, score, ms, misses in rows:
        print(f"{img:<{w}}  {diff:<6}  {cat}    {score:<6}  {ms:<5}  {misses}")
    print(f"\nTOTAL: {total_pts}/{total_tot} attribute points  ({100*total_pts/total_tot:.0f}%)   "
          f"model={SIGLIP_ID}  hint_boost={HINT_BOOST}  known_from_title={USE_TITLE_HINTS}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR)
