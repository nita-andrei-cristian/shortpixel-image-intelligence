# Product Intelligence API — PoC v1

A small HTTP API for e-commerce image intelligence. Give it a product image, optional metadata, and a
taxonomy; it returns one enriched JSON that maps the image onto that taxonomy. Everything works by
**selecting** from the taxonomy's predefined options — no text generation.

## Pipeline

```
image ──►  analysis stage  ──►  attributes selected from the taxonomy's options
           (model backend)                        │
                                                  ▼
                                 Taxonomy mapping ──► enriched JSON
```

The pipeline is backend-agnostic: an analysis stage turns an image into taxonomy selections, and
the rest maps those onto the caller's taxonomy. Today that stage is a SigLIP zero-shot tagger; a
different backend — e.g. a vision-language model — can slot in behind the same interface without
changing the API.

- **`app/stages/`** — analysis backends. Current: `ZeroShotTagger` (SigLIP zero-shot selection over
  the taxonomy options; every attribute, including color, is a named-option pick).
- **`app/classes/`** — building blocks: `AIModel` (base), `Taxonomy`, `ProductIntelligencePipeline`.
- **`legacy/`** — the old BiRefNet + kmeans color stack, kept for reference, not wired in.

Every backend subclasses `AIModel` (lazy weight loading), so adding or swapping a model is one new
subclass — the API contract doesn't change.

## Setup

```bash
scripts/install.sh        # creates .venv and installs requirements
```

## Run

```bash
scripts/run.sh            # starts the API on :8000 (PORT=... to override)
```

First run downloads SigLIP (~370MB, fast on CPU). The tagger is always available; each request
decides whether to run it via the `tagging` field (see below).

## Use

By image URL (JSON body):

```bash
curl -X POST localhost:8000/analyze -H 'content-type: application/json' \
  -d '{"image_url":"https://...","taxonomy":{...}}'
```

Response:

```json
{
  "taxonomy_id": "fashion_v1",
  "category": "shoes",
  "attributes": {
    "color":    {"value": ["white"], "source": "tagger", "confidence": 0.93},
    "material": {"value": ["leather"], "source": "tagger", "confidence": 0.71}
  },
  "tags": ["white", "leather"],
  "source": "poc-v1"
}
```

Each attribute's `source` is `tagger` (SigLIP) or `provided` (sent in `known`). Every value is
guaranteed to be one of the taxonomy's allowed options. `tags` is a flat list of the selected
values (for search) — not generated.

## Request fields

```jsonc
{
  "taxonomy": { ...full schema, required... },   // categories + attributes with their options
  "image_url": "https://...",                    // only for the JSON body form
  "meta":  { "title": "...", "brand": "..." },   // optional context
  "known": { "material": ["leather"] },          // optional, overrides what the model picks
  "tagging": true                                // optional (default true); false -> provided/known only
}
```

## Config (`app/settings.py`)

Runs on CPU. A few constants live in `app/settings.py`:

- `SIGLIP_ID` — which SigLIP checkpoint to load.
- `TEXT_CACHE_MAX` — how many text embeddings to keep cached.
- `HINT_BOOST` — nudge for an option named in the request title (0 disables).

Whether the tagger runs is a **per-request** choice (`tagging` in the payload), not a server setting.
