import math

import cv2
import numpy as np

PALETTE_CLUSTERS = 7
MAX_KMEANS_PIXELS = 20000
HUE_FAMILY_RADIUS = 14
MIN_COLORED_PIXELS = 100
NEUTRAL_COLORED_SHARE = 0.35  # below this the product is treated as white/gray/black


def rgb_to_hex(rgb) -> str:
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def circular_hue_distance(a, b) -> float:
    diff = abs(float(a) - float(b)) % 180
    return min(diff, 180 - diff)


def circular_mean_hue(hues, weights) -> float:
    angles = np.asarray(hues, dtype=np.float32) / 180 * 2 * np.pi
    weights = np.asarray(weights, dtype=np.float32)
    angle = math.atan2(np.sum(np.sin(angles) * weights), np.sum(np.cos(angles) * weights))
    return float((angle % (2 * np.pi)) / (2 * np.pi) * 180)


def circular_median_hue(hues) -> int:
    hues = np.asarray(hues, dtype=np.float32)
    if len(hues) > 5000:
        hues = hues[np.random.default_rng(0).choice(len(hues), size=5000, replace=False)]
    candidates = np.arange(180, dtype=np.float32)
    distances = np.abs(candidates[:, None] - hues[None, :]) % 180
    distances = np.minimum(distances, 180 - distances)
    return int(candidates[np.argmin(np.sum(distances, axis=1))])


def clean_mask(mask) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8) * 255
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask_u8 = cv2.erode(mask_u8, np.ones((3, 3), np.uint8))
    return mask_u8 > 0


def hsv_percentile_color(hsv) -> tuple:
    hue = circular_median_hue(hsv[:, 0])
    sat = np.percentile(hsv[:, 1], 55)
    val = np.percentile(hsv[:, 2], 68)
    rgb = cv2.cvtColor(np.array([[[hue, sat, val]]], dtype=np.uint8), cv2.COLOR_HSV2RGB)[0, 0]
    return tuple(int(x) for x in rgb)


def kmeans_palette(pixels) -> list:
    sample = pixels
    if len(sample) > MAX_KMEANS_PIXELS:
        sample = sample[np.random.default_rng(0).choice(len(sample), size=MAX_KMEANS_PIXELS, replace=False)]
    clusters = min(PALETTE_CLUSTERS, len(sample))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.2)
    _, labels, centers = cv2.kmeans(np.float32(sample), clusters, None, criteria, 5, cv2.KMEANS_PP_CENTERS)

    labels = labels.reshape(-1)
    centers = np.clip(centers, 0, 255).astype(np.uint8)
    centers_hsv = cv2.cvtColor(centers.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)

    palette = []
    for i, center in enumerate(centers):
        share = (labels == i).sum() / len(labels)
        hue, sat, val = [int(x) for x in centers_hsv[i]]
        palette.append({
            "rgb": tuple(int(x) for x in center),
            "hex": rgb_to_hex(center),
            "share": round(float(share), 4),
            "hsv": [hue, sat, val],
            "colored": bool(sat > 25 and 35 < val < 252),
        })
    return sorted(palette, key=lambda item: item["share"], reverse=True)


def build_hue_families(palette) -> list:
    families = []
    for item in (p for p in palette if p["colored"]):
        hue = item["hsv"][0]
        target = next((f for f in families if circular_hue_distance(hue, f["center_hue"]) <= HUE_FAMILY_RADIUS), None)
        if target is None:
            target = {"clusters": [], "center_hue": hue}
            families.append(target)
        target["clusters"].append(item)
        target["share"] = round(sum(c["share"] for c in target["clusters"]), 4)
        target["center_hue"] = circular_mean_hue(
            [c["hsv"][0] for c in target["clusters"]], [c["share"] for c in target["clusters"]],
        )
    return sorted(families, key=lambda item: item["share"], reverse=True)


def confidence_from_families(families) -> dict:
    if not families:
        return {"score": 0.0, "level": "low"}

    top = families[0]["share"]
    second = families[1]["share"] if len(families) > 1 else 0.0
    gap = top - second
    dominant = max(c["share"] for c in families[0]["clusters"])

    score = (
        min(top / 0.55, 1.0) * 0.55
        + min(gap / 0.25, 1.0) * 0.25
        + min(dominant / 0.35, 1.0) * 0.20
    )
    if top < 0.40:
        score *= 0.65
    if gap < 0.10:
        score *= 0.75

    score = round(float(score), 3)
    level = "low" if score < 0.55 else "medium" if score < 0.72 else "high"
    return {"score": score, "level": level}


class ColorExtractor:
    """Dominant product color from a masked image."""

    def extract(self, image, mask) -> dict:
        rgb = np.array(image)
        mask = clean_mask(mask)
        pixels = rgb[mask]
        hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)

        colored = (hsv[:, 1] > 25) & (hsv[:, 2] > 35) & (hsv[:, 2] < 252)

        # Mostly neutral product (white/gray/black): the chromatic logic would latch onto
        # tiny colored accents, so take the dominant cluster over ALL pixels instead.
        if colored.mean() < NEUTRAL_COLORED_SHARE or colored.sum() < MIN_COLORED_PIXELS:
            palette = kmeans_palette(pixels)
            top = palette[0]
            return _result(top["rgb"], {"score": top["share"], "level": _level(top["share"])}, palette[:3])

        pixels, hsv = pixels[colored], hsv[colored]
        palette = kmeans_palette(pixels)
        families = build_hue_families(palette)

        if families:
            family_mask = np.zeros(len(hsv), dtype=bool)
            for cluster in families[0]["clusters"]:
                family_mask |= np.array([
                    circular_hue_distance(h, cluster["hsv"][0]) <= HUE_FAMILY_RADIUS for h in hsv[:, 0]
                ])
            family_hsv = hsv[family_mask] if family_mask.sum() >= MIN_COLORED_PIXELS else hsv
            selected_rgb = hsv_percentile_color(family_hsv)
        else:
            selected_rgb = palette[0]["rgb"]

        return _result(selected_rgb, confidence_from_families(families), palette[:3])


def _level(share: float) -> str:
    return "high" if share >= 0.6 else "medium" if share >= 0.4 else "low"


def _result(rgb, confidence: dict, palette_top3: list) -> dict:
    return {
        "selected_rgb": list(rgb),
        "selected_hex": rgb_to_hex(rgb),
        "confidence": confidence,
        "palette_top3": palette_top3,
    }
