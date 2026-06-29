import io
import ipaddress
import json
import socket
import time
from urllib.parse import urlparse

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image

from app.classes.pipeline import ProductIntelligencePipeline
from app.classes.taxonomy import Taxonomy
from app.logging_setup import get_logger
from app.schemas import AnalyzePayload, AnalyzeResponse
from app.settings import DEVICE

logger = get_logger()
app = FastAPI(title="Product Intelligence API")
pipeline = ProductIntelligencePipeline()

IMAGE_TIMEOUT = 5                     # seconds
MAX_IMAGE_BYTES = 15 * 1024 * 1024


@app.exception_handler(Exception)
async def on_error(request: Request, exc: Exception):
    # Log the traceback, but never hand internal error text back to the caller.
    logger.exception("500 on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"error": "internal_error"})


def fetch_image_url(url: str) -> bytes:
    # Only public http(s) URLs — keeps the server from being used to reach internal
    # services or the cloud metadata endpoint (SSRF).
    parsed = urlparse(url or "")
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise HTTPException(400, "image_url must be a public http(s) URL")
    try:
        addrs = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror:
        raise HTTPException(400, "image_url host does not resolve")
    if any(_is_private(info[4][0]) for info in addrs):
        raise HTTPException(400, "image_url points to a non-public address")

    resp = requests.get(url, timeout=IMAGE_TIMEOUT, stream=True)
    resp.raise_for_status()
    data = resp.raw.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(400, "image too large")
    return data


def _is_private(addr: str) -> bool:
    ip = ipaddress.ip_address(addr)
    return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: Request):
    t0 = time.perf_counter()
    if "multipart" in request.headers.get("content-type", ""):
        form = await request.form()
        payload = AnalyzePayload(**json.loads(form["payload"]))
        image = Image.open(io.BytesIO(await form["image"].read()))
    else:
        payload = AnalyzePayload(**await request.json())
        image = Image.open(io.BytesIO(fetch_image_url(payload.image_url)))

    image = image.convert("RGB")
    taxonomy = Taxonomy(payload.taxonomy)
    result = pipeline.analyze(image, taxonomy, payload.meta.model_dump(), payload.known, payload.tagging)
    result["processing_ms"] = round((time.perf_counter() - t0) * 1000)
    logger.info("analyze ok: category=%s attrs=%s %dms", result["category"], list(result["attributes"]), result["processing_ms"])
    return result
