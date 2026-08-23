from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager

import torch

LOGGER = logging.getLogger(__name__)

cudnn_toggle_lock = threading.Lock()


def resolve_threads(env_var: str) -> int:
    raw = os.environ.get(env_var)
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            LOGGER.warning("Ignoring invalid %s=%r", env_var, raw)
    return max(1, min(os.cpu_count() or 4, 6))


@contextmanager
def cpu_threads(threads: int | None):
    """Temporarily set torch intra-op thread count, restoring the previous value."""
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
    """Process-wide exclusive cudnn disable, safe to nest across backends."""
    backend = getattr(torch.backends, "cudnn", None)
    if backend is None or not torch.cuda.is_available():
        yield
        return
    with cudnn_toggle_lock:
        prev = backend.enabled
        backend.enabled = False
        try:
            yield
        finally:
            backend.enabled = prev
