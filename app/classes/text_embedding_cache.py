from collections import OrderedDict


class TextEmbeddingCache:
    """Remembers the vector we computed for a piece of text so we don't recompute it.

    Turning a label like "a photo of a red shoe" into an embedding (a vector) means
    running it through the model — the slow part. But taxonomy labels repeat across
    every request, so we keep each result here, keyed on (model_id, exact label text).

    Keying on the *exact* text matters: if a prompt or option is edited, the new text
    is a different key, so you get a fresh vector instead of a stale one.

    It's a small LRU — once it's full, the least-recently-used entry is dropped, so
    memory stays bounded. One instance is shared across the whole process.
    """

    def __init__(self, max_size: int):
        self.max_size = max_size
        self._store: "OrderedDict[tuple[str, str], object]" = OrderedDict()

    def get(self, key):
        """Return the cached vector, or None on a miss. A hit becomes most-recent."""
        vec = self._store.get(key)
        if vec is not None:
            self._store.move_to_end(key)
        return vec

    def put(self, key, vec):
        """Store a vector, evicting the oldest entry if we're over capacity."""
        self._store[key] = vec
        self._store.move_to_end(key)
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)

    def clear(self):
        self._store.clear()

    def __len__(self):
        return len(self._store)

    def __contains__(self, key):
        return key in self._store

    def __iter__(self):
        return iter(self._store)
