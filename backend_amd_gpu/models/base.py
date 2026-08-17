from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class SegmentationBackend(ABC):
    @abstractmethod
    def load_model(self, model_path: str) -> None:
        pass

    @abstractmethod
    def unload_model(self) -> None:
        pass

    @abstractmethod
    def set_image(self, image: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(
        self,
        points: list[list[float]] | None = None,
        labels: list[int] | None = None,
        bboxes: list[float] | None = None,
        mask_input: np.ndarray | None = None,
        multimask_output: bool = True,
    ) -> dict[str, Any]:
        pass

    @abstractmethod
    def reset_image(self) -> None:
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        pass

    @property
    @abstractmethod
    def has_image(self) -> bool:
        pass
