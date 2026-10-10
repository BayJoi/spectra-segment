from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager

LOGGER = logging.getLogger(__name__)

_OFFLINE_ENV_VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")

_lock = threading.RLock()


def _apply(offline: bool) -> None:
    for name in _OFFLINE_ENV_VARS:
        if offline:
            os.environ[name] = "1"
        else:
            os.environ.pop(name, None)

    try:
        import huggingface_hub.constants as hub_constants

        hub_constants.HF_HUB_OFFLINE = offline
    except Exception:
        pass

    try:
        import transformers.utils.hub as tf_hub

        # `is_offline_mode()` returns this module global, not the environment,
        # so patching it is the only way to flip transformers at runtime.
        tf_hub._is_offline_mode = offline
    except Exception:
        pass


def is_offline() -> bool:
    try:
        import huggingface_hub.constants as hub_constants

        if hub_constants.HF_HUB_OFFLINE:
            return True
    except Exception:
        pass
    return (
        os.environ.get("HF_HUB_OFFLINE") == "1"
        or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    )


@contextmanager
def offline_guard(offline: bool = True):
    """Force (or allow) network access for HuggingFace-backed loads.

    Cached model loads run under `offline_guard(True)` so a fully downloaded
    model never contacts the Hub again. Downloads run under
    `offline_guard(False)` to opt back into the network for that call only.
    """
    with _lock:
        previous = is_offline()
        _apply(offline)
    try:
        yield
    finally:
        with _lock:
            _apply(previous)
