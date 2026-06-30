from typing import Optional

from pydantic import BaseModel


class Meta(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    sku: Optional[str] = None
    brand: Optional[str] = None


class AnalyzePayload(BaseModel):
    taxonomy: dict
    image_url: Optional[str] = None
    meta: Meta = Meta()
    known: dict = {}
    tagging: bool = True  # run the zero-shot tagger; False -> color + provided only


class AttributeResult(BaseModel):
    value: list[str]
    source: str
    confidence: Optional[float] = None


class AnalyzeResponse(BaseModel):
    taxonomy_id: str
    category: str
    attributes: dict[str, AttributeResult]
    tags: list[str]
    source: str
    processing_ms: Optional[int] = None
