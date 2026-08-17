from __future__ import annotations

import hmac
import logging
import re
import secrets
import time
from collections import deque

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

LOGGER = logging.getLogger(__name__)

ALLOWED_HOSTS = {"127.0.0.1", "localhost"}

BACKEND_PORT = 8000

MAX_UPLOAD_BYTES = 50 * 1024 * 1024

RATE_LIMIT_PER_IP = 300
RATE_LIMIT_WINDOW_IP = 60


def _generate_token() -> str:
    return secrets.token_urlsafe(32)


def get_or_create_token() -> str:
    return _generate_token()


class HostValidationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        host_header = request.headers.get("host", "").lower()
        host_name = host_header.rsplit(":", 1)[0]
        if host_name not in ALLOWED_HOSTS:
            LOGGER.debug("Rejected request with invalid Host: %s", host_header)
            return JSONResponse(
                status_code=403,
                content={"detail": "Forbidden: invalid Host header"},
            )
        if ":" in host_header:
            port_str = host_header.rsplit(":", 1)[1]
            if not port_str.isdigit():
                LOGGER.debug("Rejected request with invalid port: %s", host_header)
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Forbidden: invalid port"},
                )
            if int(port_str) != BACKEND_PORT:
                LOGGER.debug("Rejected request with wrong port: %s", port_str)
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Forbidden: invalid port"},
                )
        return await call_next(request)


class TokenAuthMiddleware(BaseHTTPMiddleware):
    _EXEMPT_PATHS = frozenset({"/health", "/api/token", "/api/logs/stream", "/api/models/stream"})
    _EXEMPT_PATTERNS = (re.compile(r"^/api/sessions/[^/]+/release$"),)

    def __init__(self, app, token_getter):
        super().__init__(app)
        self._token_getter = token_getter

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path in self._EXEMPT_PATHS or any(p.match(path) for p in self._EXEMPT_PATTERNS):
            return await call_next(request)

        token = request.headers.get("x-local-token", "")
        current_token = self._token_getter()
        if not token or not hmac.compare_digest(token, current_token):
            return JSONResponse(
                status_code=401,
                content={"detail": "Unauthorized"},
            )
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self._ip_hits: dict[str, deque[float]] = {}
        self._last_cleanup: float = time.time()

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        if now - self._last_cleanup > RATE_LIMIT_WINDOW_IP * 2:
            self._last_cleanup = now
            stale = [ip for ip, dq in self._ip_hits.items()
                     if dq and dq[0] < now - RATE_LIMIT_WINDOW_IP]
            for ip in stale:
                del self._ip_hits[ip]

        if client_ip not in self._ip_hits:
            self._ip_hits[client_ip] = deque()

        hits = self._ip_hits[client_ip]
        cutoff = now - RATE_LIMIT_WINDOW_IP

        while hits and hits[0] < cutoff:
            hits.popleft()

        if len(hits) >= RATE_LIMIT_PER_IP:
            LOGGER.debug("Rate limit exceeded for %s", client_ip)
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests"},
            )

        hits.append(now)
        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    _HEADERS = {
        "x-content-type-options": "nosniff",
        "referrer-policy": "no-referrer",
    }

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        for name, value in self._HEADERS.items():
            response.headers.setdefault(name, value)
        return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                size = int(content_length)
                if size > MAX_UPLOAD_BYTES:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": f"Request too large. Max: {MAX_UPLOAD_BYTES // (1024*1024)}MB"},
                    )
            except ValueError:
                pass
            return await call_next(request)

        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > MAX_UPLOAD_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"detail": f"Request too large. Max: {MAX_UPLOAD_BYTES // (1024*1024)}MB"},
                )
        if body:
            request._body = bytes(body)
        return await call_next(request)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    SKIP_PATHS = {"/api/logs/stream", "/api/models/stream"}

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.SKIP_PATHS:
            return await call_next(request)

        path = request.url.path.replace("\r", "\\r").replace("\n", "\\n")
        start = time.monotonic()
        try:
            response = await call_next(request)
        except Exception:
            LOGGER.exception(
                "REQUEST %s %s -> unhandled error",
                request.method,
                path,
            )
            raise

        duration_ms = (time.monotonic() - start) * 1000
        status = response.status_code
        msg = "REQUEST %s %s -> %d in %.1fms" % (
            request.method,
            path,
            status,
            duration_ms,
        )
        if status >= 500:
            LOGGER.error(msg)
        elif status >= 400:
            LOGGER.warning(msg)
        else:
            LOGGER.info(msg)
        return response
