from __future__ import annotations

import logging
import re

LOGGER = logging.getLogger(__name__)


class StderrInterceptor:
    _MAX_BUF = 4096
    _MIN_PCT_STEP = 5

    def __init__(self, model_name: str = ""):
        self._buf = ""
        self._model = model_name
        self._last_pct = -5
        self._last_desc: str | None = None

    def write(self, data: str) -> None:
        self._buf += data
        self._drain()
        if len(self._buf) > self._MAX_BUF:
            self._buf = ""

    def _drain(self) -> None:
        while True:
            nl = self._buf.find("\n")
            cr = self._buf.find("\r")
            if nl < 0 and cr < 0:
                return
            if nl >= 0 and (cr < 0 or nl < cr):
                idx = nl
            else:
                idx = cr
            chunk = self._buf[:idx]
            self._buf = self._buf[idx + 1:]
            self._handle(chunk)

    def _handle(self, chunk: str) -> None:
        chunk = chunk.strip()
        if not chunk:
            return
        m = re.search(r"(\d+)%", chunk)
        if not m:
            LOGGER.debug("%s: %s", self._model or "HF", chunk)
            return
        pct = int(m.group(1))
        desc_m = re.match(r"^([^:]+?):\s", chunk)
        desc = desc_m.group(1).strip() if desc_m else ""
        if desc and desc != self._last_desc:
            self._last_desc = desc
            self._last_pct = -5
        if pct < 100 and pct < self._last_pct + self._MIN_PCT_STEP:
            return
        self._last_pct = pct
        if desc:
            LOGGER.info("%s: %s %d%% complete", self._model or "HF", desc, pct)
        else:
            LOGGER.info("%s: %d%% complete", self._model or "HF", pct)

    def flush(self) -> None:
        if self._buf.strip():
            self._handle(self._buf)
            self._buf = ""
