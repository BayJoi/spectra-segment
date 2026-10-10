from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager

import torch

LOGGER = logging.getLogger(__name__)

MAX_THREADS = int(os.environ.get("SPECTRA_MAX_THREADS", "32"))

_cudnn_toggle_lock = threading.Lock()


def resolve_threads(env_var: str) -> int:
    raw = os.environ.get(env_var)
    if raw:
        try:
            return max(1, min(int(raw), MAX_THREADS))
        except ValueError:
            LOGGER.warning("Invalid %s=%r — using default", env_var, raw)
    return max(1, min(os.cpu_count() or 4, MAX_THREADS))


@contextmanager
def cpu_threads(threads: int | None):
    if threads is None:
        yield
        return
    prev = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        yield
    finally:
        torch.set_num_threads(prev)


@contextmanager
def cudnn_disabled():
    backend = getattr(torch.backends, "cudnn", None)
    if backend is None or not torch.cuda.is_available():
        yield
        return
    with _cudnn_toggle_lock:
        prev = backend.enabled
        backend.enabled = False
        try:
            yield
        finally:
            backend.enabled = prev
