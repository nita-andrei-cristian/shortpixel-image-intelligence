from abc import ABC, abstractmethod


class AIModel(ABC):
    """Local AI model: lazy-loads its weights on first use, pinned to a device."""

    def __init__(self, model_id: str, device: str = "cpu"):
        self.model_id = model_id
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = self._load()  # heavy load happens once, on demand
        return self._model

    @abstractmethod
    def _load(self):
        ...
