from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    bbox: list[float]
    score: float
    label: str
    mask: np.ndarray | None = None


class DetectorBackend(ABC):
    @abstractmethod
    def load_model(self, model_path: str | None = None) -> None:
        ...

    @abstractmethod
    def unload_model(self) -> None:
        ...

    @abstractmethod
    def detect(
        self,
        image: np.ndarray,
        query: str,
        confidence: float | None = None,
        max_detections: int = 10,
    ) -> list[Detection]:
        ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        ...
