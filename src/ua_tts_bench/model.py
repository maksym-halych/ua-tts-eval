"""The interface a TTS model implements to be benchmarked."""

from abc import ABC, abstractmethod

import numpy as np


class TtsModel(ABC):
    name: str
    device: str
    params: dict = {}

    @abstractmethod
    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Return (samples, sample_rate), with float samples in [-1.0, 1.0]."""
