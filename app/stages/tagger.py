import re
from collections import OrderedDict

from app.classes.base import AIModel
from app.settings import HINT_BOOST, TEXT_CACHE_MAX

OPTION_TYPES = ("single_option", "multi_option")

# Cache of normalized text embeddings, keyed on (model_id, rendered label). Labels are
# static per taxonomy, so we encode each once. Keying on the final string means an edited
# prompt/option gets a fresh vector instead of a stale hit. Bounded LRU.
_TEXT_CACHE: "OrderedDict[tuple[str, str], object]" = OrderedDict()


def _cache_get(key):
    vec = _TEXT_CACHE.get(key)
    if vec is not None:
        _TEXT_CACHE.move_to_end(key)
    return vec


def _cache_put(key, vec):
    _TEXT_CACHE[key] = vec
    _TEXT_CACHE.move_to_end(key)
    while len(_TEXT_CACHE) > TEXT_CACHE_MAX:
        _TEXT_CACHE.popitem(last=False)


class ZeroShotTagger(AIModel):
    """SigLIP zero-shot selection of taxonomy options — picks, never generates."""

    def _load(self):
        from transformers import AutoModel, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        return AutoModel.from_pretrained(self.model_id).eval().to(self.device)

    def encode_image(self, image):
        """Encode the image once; the embedding is reused for the category and every
        attribute, so the vision tower runs a single time per request."""
        import torch

        model = self.model  # triggers lazy load (and sets self.processor)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feats = model.get_image_features(**inputs).pooler_output
        return _l2norm(feats)

    def _encode_texts(self, labels: list[str]):
        """Normalized text embeddings, cached per label. Misses go in one batched pass."""
        import torch

        misses = [l for l in labels if _cache_get((self.model_id, l)) is None]
        if misses:
            model = self.model
            inputs = self.processor(text=misses, return_tensors="pt", padding="max_length").to(self.device)
            with torch.no_grad():
                feats = _l2norm(model.get_text_features(**inputs).pooler_output)
            for label, vec in zip(misses, feats):
                _cache_put((self.model_id, label), vec)
        return torch.stack([_cache_get((self.model_id, l)) for l in labels])

    def _logits(self, img_emb, txt_emb):
        """SigLIP logits: img·textᵀ · logit_scale.exp(). The bias is dropped — it's a
        constant that cancels in the softmax. Returns raw logits so a prior can be added
        first. Matches the model's combined forward to ~1e-6 (see the golden test)."""
        import torch

        with torch.no_grad():
            return (img_emb @ txt_emb.t())[0] * self.model.logit_scale.exp()

    def _option_embeddings(self, options: list[str], templates: list[str], category_label: str):
        """One embedding per option, averaged over several phrasings. Smooths out the
        quirks of any single phrasing that let generic labels ("synthetic") win."""
        import torch

        vecs = []
        for opt in options:
            labels = [t.replace("{category}", category_label.lower()).format(option=opt) for t in templates]
            vecs.append(_l2norm(self._encode_texts(labels).mean(dim=0, keepdim=True))[0])
        return torch.stack(vecs)

    def _score(self, img_emb, options, templates, category_label, prior=None) -> dict:
        """{option: probability}; `prior` is an optional additive boost in logit space."""
        txt_emb = self._option_embeddings(options, templates, category_label)
        logits = self._logits(img_emb, txt_emb)
        if prior is not None:
            logits = logits + prior
        probs = logits.softmax(dim=-1)
        return {o: float(probs[i]) for i, o in enumerate(options)}

    def pick_category(self, img_emb, taxonomy, meta: dict) -> str:
        categories = taxonomy.categories
        if len(categories) == 1:
            return categories[0]["id"]
        labels = {c["label"]: c["id"] for c in categories}
        prior = self._title_prior(meta, list(labels))
        scores = self._score(img_emb, list(labels), CATEGORY_TEMPLATES, "", prior)
        best = max(scores, key=scores.get)
        return labels[best]

    def tag(self, img_emb, category: dict, meta: dict) -> dict:
        chosen = {}
        for attr in category["attributes"]:
            if attr["type"] not in OPTION_TYPES:
                continue
            options = attr["options"]
            templates = _TEMPLATES.get(attr["key"], _DEFAULT_TEMPLATES)
            prior = self._title_prior(meta, options)
            scores = self._score(img_emb, options, templates, category["label"], prior)
            chosen[attr["key"]] = _select(scores)
        return chosen

    def _title_prior(self, meta: dict, options: list[str]):
        """Boost any option named in the title by HINT_BOOST logits. Whole-word match so
        'tan' doesn't fire on 'rectangular'. For hard specs, use `known` instead."""
        import torch

        if not HINT_BOOST:
            return None
        words = set(re.findall(r"[a-z]+", (meta.get("title") or "").lower()))
        boosts = [HINT_BOOST if o in words else 0.0 for o in options]
        if not any(boosts):
            return None
        return torch.tensor(boosts, device=self.device)


def _l2norm(feats):
    return feats / feats.norm(p=2, dim=-1, keepdim=True)


# Phrasings averaged per option. {category} and {option} are filled in per call.
_TEMPLATES = {
    "color":    ["a photo of a {option} {category}", "a {option} {category}", "a {option}-colored {category}"],
    "material": ["a photo of a {category} made of {option}", "a {option} {category}",
                 "a close-up of {option} texture", "a {category} in {option}"],
    # one tight phrasing — looser ones drift to trendy labels (formal->luxury)
    "style":    ["a photo of a {option}-style {category}"],
    "gender":   ["a photo of a {category} for {option}", "a photo of a {option}'s {category}"],
    "pattern":  ["a photo of a {category} with a {option} pattern", "a {option} {category}",
                 "a photo of a {category} with {option} print"],
}
_DEFAULT_TEMPLATES = ["a photo of {option} {category}", "a {option} {category}"]
CATEGORY_TEMPLATES = ["a photo of {option}", "a photo of a {option}", "a product photo of {option}"]


def _select(scores: dict) -> dict:
    """Highest-scoring option."""
    top, top_score = max(scores.items(), key=lambda kv: kv[1])
    return {"value": [top], "score": round(float(top_score), 3)}
