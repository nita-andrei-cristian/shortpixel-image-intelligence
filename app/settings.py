# Config constants. Kept out of the environment on purpose — the server's behavior
# lives in the codebase, not in the host's env. Edit here to change it.

DEVICE = "cpu"  # "cuda" on a GPU box

# SigLIP at 384px (not 224) so texture survives downscaling — needed to tell
# leather/suede apart. so400m is stronger but wants a GPU.
SIGLIP_ID = "google/siglip-base-patch16-384"

TEXT_CACHE_MAX = 4096  # bounded LRU of text-embedding vectors (~tens of MB)

# Boost (logit units) for an option named in the request title. Nudges close calls,
# doesn't override clear visual evidence. 0 disables.
HINT_BOOST = 2.0
