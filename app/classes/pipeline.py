from app.settings import DEVICE, SIGLIP_ID
from app.stages.tagger import OPTION_TYPES, ZeroShotTagger


def _as_list(value) -> list:
    return [str(v) for v in value] if isinstance(value, list) else [str(value)]


def _constrain(values: list, options: list, allow_custom: bool) -> list:
    if allow_custom or not options:
        return values
    return [v for v in values if v in options]


class ProductIntelligencePipeline:
    """Runs the zero-shot tagger and maps the result onto the caller's taxonomy."""

    def __init__(self, device: str = DEVICE):
        self.tagger = ZeroShotTagger(SIGLIP_ID, device)

    def analyze(self, image, taxonomy, meta: dict, known: dict, tagging: bool = True) -> dict:
        meta, known = meta or {}, known or {}
        img_emb = self.tagger.encode_image(image) if tagging else None  # encode once, reuse
        category_id = self.tagger.pick_category(img_emb, taxonomy, meta) if tagging else taxonomy.category_ids()[0]
        category = taxonomy.category(category_id)
        tagged = self.tagger.tag(img_emb, category, meta) if tagging else {}
        return self._assemble(taxonomy, category_id, tagged, known)

    def _assemble(self, taxonomy, category_id: str, tagged: dict, known: dict) -> dict:
        attributes = {}
        for attr in taxonomy.attributes(category_id):
            key = attr["key"]
            if key in known:
                attributes[key] = {"value": _as_list(known[key]), "source": "provided"}
            elif attr["type"] in OPTION_TYPES and key in tagged:
                value = _constrain(tagged[key]["value"], attr.get("options", []), attr.get("allow_custom_values", False))
                if value:
                    attributes[key] = {"value": value, "source": "tagger", "confidence": tagged[key]["score"]}

        tags = [v for a in attributes.values() for v in a["value"]]  # flat list for search
        return {
            "taxonomy_id": taxonomy.id,
            "category": category_id,
            "attributes": attributes,
            "tags": tags,
            "source": "poc-v1",
        }
