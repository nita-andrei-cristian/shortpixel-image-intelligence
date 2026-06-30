import re

from app.classes.base import AIModel
from app.classes.text_embedding_cache import TextEmbeddingCache
from app.settings import HINT_BOOST, TEXT_CACHE_MAX
from app.prompts.templates import CATEGORY_TEMPLATES, DEFAULT_TEMPLATES, TEMPLATES

"""Picks taxonomy options for an image — it chooses from a list, it never writes text.

The trick is "embeddings". The model turns both the image and each candidate label
(e.g. "a photo of a red shoe") into a vector — a list of numbers that captures meaning.
Similar things land close together, so we can score how well the image matches each
label just by comparing vectors. No generation, no LLM: every answer is one of the
options the caller handed us.

What it receives: an image, plus a taxonomy — a list of categories, each with attributes
(color, material, ...) and, for each attribute, the list of allowed options. For every
attribute we embed its options, compare them to the image, and keep the best match.

Comparing gives a raw score per option ("logits") — bigger means a closer match. Softmax
turns those scores into probabilities that add up to 1, and we pick the highest.

The image is embedded once and reused for every comparison (the vision pass is the slow
part). Label embeddings are cached, since the same labels come back on every request.
"""

OPTION_TYPES = ("single_option", "multi_option")

# Shared across the process: labels are static per taxonomy, so we encode each one once.
_TEXT_CACHE = TextEmbeddingCache(TEXT_CACHE_MAX)


class ZeroShotTagger(AIModel):
    """SigLIP zero-shot selection of taxonomy options — picks, never generates."""

    def _load(self):
        from transformers import AutoModel, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        return AutoModel.from_pretrained(self.model_id).eval().to(self.device)

    def encode_image(self, image):
        """Turn the image into one vector. Done once per request and reused for the
        category and every attribute, so the slow vision pass runs a single time."""
        import torch

        model = self.model  # triggers lazy load (and sets self.processor)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feats = _pooled(model.get_image_features(**inputs))
        return _l2norm(feats)

    def _encode_texts(self, labels: list[str]):
        """One vector per label, normalized. Cache hits are free; misses get embedded
        together in a single batched pass, then stored for next time."""
        import torch

        misses = [l for l in labels if _TEXT_CACHE.get((self.model_id, l)) is None]
        if misses:
            model = self.model
            inputs = self.processor(text=misses, return_tensors="pt", padding="max_length").to(self.device)
            with torch.no_grad():
                feats = _l2norm(_pooled(model.get_text_features(**inputs)))
            for label, vec in zip(misses, feats):
                _TEXT_CACHE.put((self.model_id, label), vec)
        return torch.stack([_TEXT_CACHE.get((self.model_id, l)) for l in labels])

    def _logits(self, img_emb, txt_emb):
        """Score the image against each label by comparing vectors (a dot product —
        higher means more alike), scaled the way SigLIP does. The bias is dropped: it's
        a constant that cancels once we softmax. Raw scores come back so a prior can be
        added first. Matches the model's own forward to ~1e-6 (see the golden test)."""
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
        # logits = raw match scores (image vs each option); softmax turns them into
        # probabilities. The prior is added here, before softmax, so it nudges fairly.
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
            templates = TEMPLATES.get(attr["key"], DEFAULT_TEMPLATES)
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


def _pooled(out):
    # transformers 5.x returns an object with .pooler_output; 4.x returns the tensor itself.
    return out.pooler_output if hasattr(out, "pooler_output") else out


def _l2norm(feats):
    return feats / feats.norm(p=2, dim=-1, keepdim=True)


def _select(scores: dict) -> dict:
    """Highest-scoring option."""
    top, top_score = max(scores.items(), key=lambda kv: kv[1])
    return {"value": [top], "score": round(float(top_score), 3)}
