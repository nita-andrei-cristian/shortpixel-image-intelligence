from pydantic import BaseModel


class Meta(BaseModel):
    title: str | None = None
    description: str | None = None
    sku: str | None = None
    brand: str | None = None


class AnalyzePayload(BaseModel):
    taxonomy: dict
    image_url: str | None = None
    meta: Meta = Meta()
    known: dict = {}
    tagging: bool = True  # run the zero-shot tagger; False -> color + provided only


class AttributeResult(BaseModel):
    value: list[str]
    source: str
    confidence: float | None = None


class AnalyzeResponse(BaseModel):
    taxonomy_id: str
    category: str
    attributes: dict[str, AttributeResult]
    tags: list[str]
    source: str
    processing_ms: int | None = None
