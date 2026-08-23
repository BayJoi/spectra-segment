from __future__ import annotations

import logging
import urllib.request
import ssl

LOGGER = logging.getLogger(__name__)

_PINNED_HOSTS = [
    "https://huggingface.co",
    "https://github.com",
]


def is_connected(timeout: float = 1.5) -> bool:
    ctx = ssl.create_default_context()
    for host in _PINNED_HOSTS:
        try:
            req = urllib.request.Request(host, method="HEAD")
            urllib.request.urlopen(req, timeout=timeout, context=ctx)
            return True
        except Exception:
            continue
    LOGGER.debug("connectivity check failed — all %d hosts unreachable", len(_PINNED_HOSTS))
    return False
