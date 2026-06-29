class Taxonomy:
    """Thin wrapper over the caller's taxonomy schema so the pipeline never pokes raw dicts."""

    def __init__(self, schema: dict):
        self.schema = schema
        self.id = schema.get("taxonomy_id", "unknown")
        self.categories = schema["categories"]

    def category_ids(self) -> list[str]:
        return [c["id"] for c in self.categories]

    def category(self, category_id: str) -> dict:
        return next(c for c in self.categories if c["id"] == category_id)

    def attributes(self, category_id: str) -> list[dict]:
        return self.category(category_id)["attributes"]

    def options_for(self, category_id: str, key: str) -> list[str]:
        attr = next(a for a in self.attributes(category_id) if a["key"] == key)
        return attr.get("options", [])
