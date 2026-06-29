"""Build a multi-category test set from Unsplash photos.

For each item in the manifest below it searches Unsplash, downloads the top hit, and
writes a shared taxonomy.json plus per-image NN-slug.jpg + sidecar. Labels come from the
search query (so verify them — the alt-text is saved in each sidecar's note).

  UNSPLASH_ACCESS_KEY=xxxx .venv/bin/python scripts/build_testset.py [OUT_DIR]
  .venv/bin/python scripts/build_testset.py --dry-run        # taxonomy + manifest only
  .venv/bin/python scripts/build_testset.py --placeholders   # local swatches, no key

Resumable (skips existing .jpg), so re-run in batches — the free Unsplash tier caps at
~50 requests/hour. OUT_DIR defaults to ~/dev/images/testset-v2.
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

OUT_DEFAULT = Path.home() / "dev/images/testset-v2"
API = "https://api.unsplash.com/search/photos"

# Shared option pools.
COLORS = ["black", "white", "gray", "silver", "blue", "red", "green", "pink", "gold", "beige"]
EC_COLORS = ["black", "white", "silver", "space gray", "blue", "green", "gold", "red", "purple"]

# taxonomy_v3: visual attributes the tagger picks, plus non-visual specs (no options)
# that only ever come from known/meta — never guessed from the photo.
def _opt(key, options, type_="multi_option", custom=False):
    return {"key": key, "type": type_, "options": options, "allow_custom_values": custom}

def _spec(key, type_="number"):
    return {"key": key, "type": type_, "allow_custom_values": True}

TAXONOMY = {
    "taxonomy_id": "taxonomy_v3",
    "name": "Multi-category product taxonomy",
    "categories": [
        {"id": "shoes", "label": "Shoes", "attributes": [
            _opt("color", COLORS + ["brown", "burgundy", "orange", "multicolor"]),
            _opt("material", ["leather", "suede", "canvas", "mesh", "knit", "rubber", "synthetic"], custom=True),
            _opt("gender", ["men", "women", "unisex", "kids"], "single_option"),
            _opt("style", ["casual", "formal", "sport", "streetwear", "skate", "outdoor"], custom=True)]},
        {"id": "bags", "label": "Bag", "attributes": [
            _opt("color", COLORS + ["brown", "burgundy", "orange", "multicolor"]),
            _opt("material", ["leather", "suede", "canvas", "nylon", "synthetic"], custom=True),
            _opt("style", ["casual", "formal", "luxury", "streetwear", "tote", "crossbody"], custom=True)]},
        {"id": "tops", "label": "Top", "attributes": [
            _opt("color", COLORS + ["brown", "burgundy", "orange", "multicolor"]),
            _opt("material", ["cotton", "knit", "linen", "polyester", "synthetic"], custom=True),
            _opt("gender", ["men", "women", "unisex", "kids"], "single_option"),
            _opt("style", ["casual", "formal", "sport", "streetwear"], custom=True),
            _opt("pattern", ["solid", "graphic", "striped", "logo"])]},
        {"id": "pants", "label": "Pants", "attributes": [
            _opt("color", COLORS + ["khaki", "brown", "multicolor"]),
            _opt("material", ["denim", "cotton", "satin", "leather", "synthetic"], custom=True),
            _opt("gender", ["men", "women", "unisex", "kids"], "single_option"),
            _opt("style", ["casual", "formal", "sport", "streetwear"], custom=True)]},
        {"id": "phones", "label": "Smartphone", "attributes": [
            _opt("color", EC_COLORS),
            _opt("body_material", ["glass", "aluminum", "plastic", "titanium"], custom=True),
            _spec("storage_gb"), _spec("brand", "text")]},
        {"id": "laptops", "label": "Laptop", "attributes": [
            _opt("color", EC_COLORS),
            _opt("form_factor", ["clamshell", "2-in-1", "ultrabook"], "single_option"),
            _opt("body_material", ["aluminum", "plastic", "magnesium"], custom=True),
            _spec("ram_gb"), _spec("storage_gb"), _spec("brand", "text")]},
        {"id": "headphones", "label": "Headphones", "attributes": [
            _opt("color", EC_COLORS),
            _opt("type", ["over-ear", "on-ear", "in-ear", "earbuds"], "single_option"),
            _opt("connectivity", ["wireless", "wired"], "single_option"),
            _spec("brand", "text")]},
        {"id": "lamps", "label": "Lamp", "attributes": [
            _opt("color", COLORS + ["brass", "copper"]),
            _opt("material", ["metal", "wood", "glass", "ceramic", "plastic"], custom=True),
            _opt("style", ["modern", "vintage", "industrial", "minimalist"], custom=True)]},
        {"id": "lipsticks", "label": "Lipstick", "attributes": [
            _opt("color", ["red", "pink", "nude", "coral", "burgundy", "brown", "purple", "orange"]),
            _opt("finish", ["matte", "glossy", "satin"], "single_option")]},
    ],
}


def _items():
    """~10 items per category. Expected only covers attributes the query reliably
    implies — verify the rest against the saved photo."""
    items = []

    def add(cat, slug, query, expected, difficulty="medium", meta=None):
        items.append({"category": cat, "slug": slug, "query": query,
                      "difficulty": difficulty, "expected": expected, "meta": meta or {}})

    # Fashion. Queries stay short — over-specific ones return no Unsplash results.
    for c in ["red", "black", "white", "blue", "green", "pink", "gray", "beige", "brown", "multicolor"]:
        q = "colorful sneakers" if c == "multicolor" else f"{c} sneakers"
        add("shoes", f"{c}-sneaker", q,
            {"color": ["multicolor" if c == "multicolor" else c], "style": ["sport"]},
            "easy" if c in ("red", "black", "white") else "medium")
    for c, m in [("black", "leather"), ("brown", "leather"), ("tan", "suede"), ("red", "leather"),
                 ("white", "canvas"), ("green", "nylon"), ("blue", "canvas"), ("pink", "leather"),
                 ("gray", "synthetic"), ("beige", "canvas")]:
        add("bags", f"{c}-{m}-bag", f"{c} {m} handbag product photo",
            {"color": [c], "material": [m]}, "medium")
    for c, p in [("white", "solid"), ("black", "graphic"), ("blue", "striped"), ("red", "solid"),
                 ("gray", "graphic"), ("green", "solid"), ("pink", "striped"), ("beige", "solid"),
                 ("white", "logo"), ("black", "striped")]:
        add("tops", f"{c}-{p}-tshirt", f"{c} {p} t-shirt on white background",
            {"color": [c], "pattern": [p]}, "medium")
    for c, m in [("blue", "denim"), ("black", "denim"), ("beige", "cotton"), ("white", "cotton"),
                 ("gray", "cotton"), ("black", "leather"), ("khaki", "cotton"), ("pink", "satin"),
                 ("green", "cotton"), ("brown", "cotton")]:
        add("pants", f"{c}-{m}-pants", f"{c} {m} pants product photo",
            {"color": [c], "material": [m]}, "medium")

    # Electronics. Specs (storage/ram/brand) ride in meta, not expected.
    for c in EC_COLORS + ["beige"]:
        add("phones", f"{c}-smartphone", f"{c} smartphone",
            {"color": [c.replace(' ', '-') if c == 'space gray' else c]}, "medium",
            meta={"title": f"{c} smartphone 128GB", "brand": "Generic"})
    for c in EC_COLORS + ["beige"]:
        add("laptops", f"{c}-laptop", f"{c} laptop",
            {"color": [c.replace(' ', '-') if c == 'space gray' else c], "form_factor": ["clamshell"]},
            "medium", meta={"title": f"{c} ultrabook laptop 16GB RAM 512GB"})
    for c, t in [("black", "over-ear"), ("white", "over-ear"), ("silver", "on-ear"), ("blue", "earbuds"),
                 ("red", "in-ear"), ("green", "earbuds"), ("gold", "over-ear"), ("gray", "on-ear"),
                 ("white", "earbuds"), ("black", "in-ear")]:
        add("headphones", f"{c}-{t}-headphones", f"{c} {t} headphones product photo",
            {"color": [c], "type": [t]}, "hard")

    # Home + beauty.
    for c, m in [("black", "metal"), ("white", "ceramic"), ("gold", "metal"), ("wood", "wood"),
                 ("gray", "ceramic"), ("blue", "glass"), ("green", "glass"), ("brass", "metal"),
                 ("white", "plastic"), ("black", "wood")]:
        col = "brown" if c == "wood" else c
        add("lamps", f"{col}-{m}-lamp", f"{col} {m} table lamp product photo",
            {"color": [col], "material": [m]}, "medium")
    for c, f in [("red", "matte"), ("pink", "glossy"), ("nude", "satin"), ("coral", "matte"),
                 ("burgundy", "matte"), ("brown", "satin"), ("purple", "glossy"), ("red", "glossy"),
                 ("pink", "matte"), ("orange", "matte")]:
        add("lipsticks", f"{c}-{f}-lipstick", f"{c} {f} lipstick product photo",
            {"color": [c], "finish": [f]}, "medium")
    return items


# Rough name->RGB for placeholder swatches.
_RGB = {"black": (30, 30, 30), "white": (240, 240, 240), "gray": (128, 128, 128),
        "silver": (192, 192, 195), "space gray": (90, 92, 96), "blue": (40, 90, 200),
        "red": (200, 40, 40), "green": (40, 160, 70), "pink": (235, 150, 175),
        "gold": (210, 175, 80), "beige": (225, 210, 180), "brown": (120, 80, 50),
        "burgundy": (120, 30, 50), "orange": (230, 130, 40), "khaki": (190, 175, 120),
        "purple": (130, 70, 170), "nude": (220, 185, 160), "coral": (240, 120, 90),
        "brass": (180, 150, 90), "tan": (200, 170, 120), "multicolor": (120, 120, 120)}


def _placeholder(item):
    """Solid-color swatch so the set is runnable without fetching. Only 'color' is
    meaningful until real photos replace it."""
    from PIL import Image, ImageDraw

    color = (item["expected"].get("color") or ["gray"])[0]
    img = Image.new("RGB", (600, 600), _RGB.get(color, (128, 128, 128)))
    d = ImageDraw.Draw(img)
    fg = (20, 20, 20) if sum(_RGB.get(color, (128,) * 3)) > 380 else (245, 245, 245)
    d.text((20, 20), f"PLACEHOLDER\n{item['category']}\n{item['slug']}", fill=fg)
    return img


def _search(query, key):
    url = f"{API}?{urllib.parse.urlencode({'query': query, 'per_page': 1, 'orientation': 'squarish'})}"
    req = urllib.request.Request(url, headers={"Authorization": f"Client-ID {key}",
                                               "Accept-Version": "v1"})
    data = json.loads(urllib.request.urlopen(req, timeout=20).read())
    results = data.get("results", [])
    if not results:
        return None
    r = results[0]
    return {"download": r["urls"]["regular"], "alt": r.get("alt_description") or "",
            "credit": (r.get("user") or {}).get("name", ""), "register": r["links"]["download_location"]}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry = "--dry-run" in sys.argv
    out = Path(args[0]) if args else OUT_DEFAULT
    out.mkdir(parents=True, exist_ok=True)

    # Number items so filenames sort by category then variant.
    items = _items()
    for i, it in enumerate(items, 1):
        it["image"] = f"{i:02d}-{it['category']}-{it['slug']}.jpg"

    (out / "taxonomy.json").write_text(json.dumps(TAXONOMY, indent=2))
    (out / "manifest.json").write_text(json.dumps(items, indent=2))
    print(f"wrote {out/'taxonomy.json'} ({len(TAXONOMY['categories'])} categories)")
    print(f"wrote {out/'manifest.json'} ({len(items)} items)")
    if dry:
        print("--dry-run: skipping fetch")
        return

    if "--placeholders" in sys.argv:
        for it in items:
            jpg = out / it["image"]
            if jpg.exists():
                continue
            _placeholder(it).save(jpg, "JPEG", quality=80)
            jpg.with_suffix(".json").write_text(json.dumps({
                "image": it["image"], "difficulty": it["difficulty"],
                "note": "PLACEHOLDER swatch — replace via the Unsplash fetch", "meta": it["meta"],
                "expected": {"category": it["category"], "attributes": it["expected"]},
            }, indent=2))
        print(f"wrote {len(items)} placeholder images + sidecars -> {out}")
        return

    key = os.getenv("UNSPLASH_ACCESS_KEY")
    if not key:
        sys.exit("Set UNSPLASH_ACCESS_KEY to fetch images (https://unsplash.com/developers). "
                 "taxonomy.json + manifest.json were written; re-run with the key to download.")

    # Round-robin across categories so a rate-limited partial run still spans all of them.
    by_cat = {}
    for it in items:
        by_cat.setdefault(it["category"], []).append(it)
    order = []
    while any(by_cat.values()):
        for cat in list(by_cat):
            if by_cat[cat]:
                order.append(by_cat[cat].pop(0))

    ok = skip = fail = 0
    for it in order:
        jpg = out / it["image"]
        sidecar = jpg.with_suffix(".json")
        if jpg.exists():
            skip += 1
            continue
        try:
            hit = _search(it["query"], key)
            if not hit:
                print(f"  no result: {it['query']!r}")
                fail += 1
                continue
            img = urllib.request.urlopen(hit["download"] + "&w=900&q=80", timeout=30).read()
            jpg.write_bytes(img)
            # (Skipping Unsplash's download-registration ping to spend 1 API call per item
            # instead of 2 — doubles throughput on the free tier.)
            sidecar.write_text(json.dumps({
                "image": it["image"], "difficulty": it["difficulty"],
                "note": f"unsplash: {hit['alt']} (photo: {hit['credit']}) — verify labels",
                "meta": it["meta"],
                "expected": {"category": it["category"], "attributes": it["expected"]},
            }, indent=2))
            ok += 1
            print(f"  ✓ {it['image']}  ({hit['credit']})")
            time.sleep(1.0)  # be polite to the API
        except Exception as e:
            print(f"  ✗ {it['image']}: {type(e).__name__}: {str(e)[:80]}")
            fail += 1
    print(f"\ndone: {ok} fetched, {skip} skipped (exist), {fail} failed -> {out}")


if __name__ == "__main__":
    main()
