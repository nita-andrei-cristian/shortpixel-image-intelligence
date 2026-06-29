import math

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

from app.classes.base import AIModel

MAX_SIDE = 1024
SIZE_MULTIPLE = 32

_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_image(image: Image.Image) -> Image.Image:
    """Convert to RGB and cap the longest side."""
    image = image.convert("RGB")
    w, h = image.size
    if max(w, h) > MAX_SIDE:
        scale = MAX_SIDE / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return image


def pad_to_multiple(image: Image.Image, multiple: int):
    w, h = image.size
    new_w = math.ceil(w / multiple) * multiple
    new_h = math.ceil(h / multiple) * multiple
    left = (new_w - w) // 2
    top = (new_h - h) // 2
    padded = ImageOps.expand(image, border=(left, top, new_w - w - left, new_h - h - top), fill=(0, 0, 0))
    return padded, (left, top, left + w, top + h)


class Segmenter(AIModel):
    """BiRefNet foreground/background mask."""

    def _load(self):
        model = AutoModelForImageSegmentation.from_pretrained(
            self.model_id, trust_remote_code=True, dtype=torch.float32,
        )
        return model.float().eval().to(self.device)

    def mask(self, image: Image.Image) -> np.ndarray:
        padded, (left, top, right, bottom) = pad_to_multiple(image, SIZE_MULTIPLE)
        x = _TRANSFORM(padded).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            pred = self.model(x)[-1].sigmoid()[0, 0].cpu().numpy()
        return (pred > 0.5)[top:bottom, left:right]
