# legacy/ — old CV color pipeline (not wired in)

Kept for reference only. This is the original computer-vision color stack that the API used before
color moved to the SigLIP zero-shot tagger:

- `segmenter.py` — BiRefNet (`ZhengPeng7/BiRefNet_lite`) foreground/background mask.
- `color_extractor.py` — dominant color via kmeans + hue families (with a neutral-product branch).
- `color_namer.py` — snaps an extracted RGB to the nearest named taxonomy option.

It was dropped because it is chromatic-biased: low-saturation products (white/gray/black, pastels)
broke its accuracy. SigLIP picks the named color holistically and handles those cases. These files
are not imported by the app; they need `einops`, `kornia`, `timm`, and `opencv-python` to run.
