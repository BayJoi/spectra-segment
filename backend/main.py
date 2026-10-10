from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import time
import posixpath
import re
import sys
import threading
import zipfile
from collections import deque
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

MODEL_WEIGHTS_DIR = Path(__file__).resolve().parent / "model_weights"
_HF_CACHE = MODEL_WEIGHTS_DIR / "hf_cache"
os.environ["HF_HOME"] = str(_HF_CACHE)
os.environ["HF_HUB_CACHE"] = str(_HF_CACHE / "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(_HF_CACHE / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(_HF_CACHE / "hub")
os.environ["HF_MODULES_CACHE"] = str(_HF_CACHE / "modules")
os.environ["HF_XET_CACHE"] = str(_HF_CACHE / "xet")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_DATASETS_CACHE"] = str(MODEL_WEIGHTS_DIR / "hf_datasets")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(MODEL_WEIGHTS_DIR / "sentence_transformers")
os.environ["TORCH_HOME"] = str(MODEL_WEIGHTS_DIR / "torch_cache")
os.environ["TRITON_CACHE_DIR"] = str(MODEL_WEIGHTS_DIR / "triton")
os.environ["NUMBA_CACHE_DIR"] = str(MODEL_WEIGHTS_DIR / "numba")
os.environ["ULTRALYTICS_HOME"] = str(MODEL_WEIGHTS_DIR)
os.environ["YOLO_CONFIG_DIR"] = str(MODEL_WEIGHTS_DIR)
os.environ["MPLCONFIGDIR"] = str(MODEL_WEIGHTS_DIR / "matplotlib")
os.environ["XDG_CACHE_HOME"] = str(MODEL_WEIGHTS_DIR / "cache")
os.environ["XDG_CONFIG_HOME"] = str(MODEL_WEIGHTS_DIR / "config")
os.environ["XDG_DATA_HOME"] = str(MODEL_WEIGHTS_DIR / "data")
os.environ["TMPDIR"] = str(MODEL_WEIGHTS_DIR / "tmp")
os.environ["TEMP"] = str(MODEL_WEIGHTS_DIR / "tmp")
os.environ["TMP"] = str(MODEL_WEIGHTS_DIR / "tmp")
for _dir in (
    _HF_CACHE,
    _HF_CACHE / "hub",
    _HF_CACHE / "modules",
    MODEL_WEIGHTS_DIR / "hf_datasets",
    MODEL_WEIGHTS_DIR / "sentence_transformers",
    MODEL_WEIGHTS_DIR / "torch_cache",
    MODEL_WEIGHTS_DIR / "triton",
    MODEL_WEIGHTS_DIR / "numba",
    MODEL_WEIGHTS_DIR / "matplotlib",
    MODEL_WEIGHTS_DIR / "cache",
    MODEL_WEIGHTS_DIR / "config",
    MODEL_WEIGHTS_DIR / "data",
    MODEL_WEIGHTS_DIR / "tmp",
):
    _dir.mkdir(parents=True, exist_ok=True)
del _dir

import cv2
import numpy as np
import torch as _torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, field_validator

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.models.manager import ModelManager
from backend.utils.compositing import mask_to_png_b64
from backend.utils.device import detect_device
from backend.utils.humantime import format_duration
from backend.utils.security import (
    HostValidationMiddleware,
    MAX_UPLOAD_BYTES,
    RateLimitMiddleware,
    RequestLoggingMiddleware,
    RequestSizeLimitMiddleware,
    SecurityHeadersMiddleware,
    TokenAuthMiddleware,
    get_or_create_token,
)
from PIL import Image, ImageOps

_log_dir = Path(__file__).resolve().parent / "logs"
os.makedirs(_log_dir, exist_ok=True)

from backend.utils import logctx

_file_handler = RotatingFileHandler(
    _log_dir / "backend.log",
    maxBytes=int(os.environ.get("SPECTRA_LOG_MAX_MB", "10")) * 1024 * 1024,
    backupCount=int(os.environ.get("SPECTRA_LOG_BACKUPS", "3")),
    encoding="utf-8",
)
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] rid=%(rid)s sid=%(sid)s %(name)s: %(message)s"
))
logging.root.setLevel(logging.DEBUG)
logging.root.addHandler(_file_handler)

LOGGER = logging.getLogger(__name__)

_log_buffer: deque[dict[str, Any]] = deque(maxlen=300)


class _CtxFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.rid = logctx.rid()
        record.sid = logctx.sid()
        return True


_ctx_filter = _CtxFilter()
_file_handler.addFilter(_ctx_filter)

_log_subscribers: list[asyncio.Queue[dict[str, str] | None]] = []
_sub_lock = threading.Lock()
MAX_SSE_SUBSCRIBERS = 100
MAX_SSE_PER_IP = 5
MAX_SSE_LIFETIME_SECONDS = 1800
_sse_clients: dict[str, int] = {}


_ANSI_ESC_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_ERASE_LINE_RE = re.compile(r"\[K")
_BOX_DRAWING_RE = re.compile(r"\u2501|\u2500|\u2577")

_DOWNLOAD_RES = (
    re.compile(r"^(?P<name>[^:]{1,80}):\s+\S+\s+(?P<pct>\d{1,3})%\s+complete"),
    re.compile(r"^(?P<name>[^:]{1,80}):\s+(?P<pct>\d{1,3})%\s+—"),
    re.compile(r"^(?P<name>[^:]{1,80}):\s+(?P<pct>\d{1,3})%\|"),
)


def _clean_message(msg: str) -> str:
    msg = _ANSI_ESC_RE.sub("", msg)
    msg = _ERASE_LINE_RE.sub("", msg)
    msg = _BOX_DRAWING_RE.sub("-", msg)
    return msg.strip()


class _BufferHandler(logging.Handler):
    _local = threading.local()

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(_BufferHandler._local, "in_emit", False):
            return
        _BufferHandler._local.in_emit = True
        try:
            msg = _clean_message(record.getMessage())
            if not msg:
                return
            entry: dict[str, Any] = {
                "level": record.levelname,
                "message": msg,
                "ts": time.strftime("%H:%M:%S"),
                "rid": getattr(record, "rid", "-"),
                "sid": getattr(record, "sid", "-"),
            }
            for key in ("model", "detector", "phase", "dur_ms"):
                val = getattr(record, key, None)
                if val is not None:
                    entry[key] = val
            dl = getattr(record, "dl", None)
            if isinstance(dl, dict):
                for k, v in dl.items():
                    if k in ("level", "message", "ts"):
                        continue
                    entry[k] = v
                entry["level"] = "DOWNLOAD"
            else:
                for rx in _DOWNLOAD_RES:
                    m = rx.match(msg)
                    if m and int(m.group("pct")) <= 100:
                        entry["level"] = "DOWNLOAD"
                        entry["name"] = m.group("name").strip()
                        entry["pct"] = int(m.group("pct"))
                        break
            _log_buffer.append(entry)
            loop = _get_main_loop()
            if loop is None or loop.is_closed():
                return
            with _sub_lock:
                for q in list(_log_subscribers):
                    try:
                        loop.call_soon_threadsafe(q.put_nowait, entry)
                    except Exception:
                        pass
        except Exception:
            LOGGER.error("_BufferHandler.emit() failed", exc_info=True)
        finally:
            _BufferHandler._local.in_emit = False


_buf_handler = _BufferHandler()
_ui_level_name = os.environ.get("SPECTRA_UI_LOG_LEVEL", "INFO").upper()
_buf_handler.setLevel(getattr(logging, _ui_level_name, logging.INFO))
_buf_handler.addFilter(_ctx_filter)
logging.root.addHandler(_buf_handler)


def _quiet_third_party() -> None:
    levels = {
        "ultralytics": "WARNING",
        "transformers": "WARNING",
        "huggingface_hub": "ERROR",
        "huggingface_hub.utils": "ERROR",
        "urllib3": "WARNING",
        "fsspec": "WARNING",
        "matplotlib": "WARNING",
        "sentence_transformers": "WARNING",
        "PIL": "WARNING",
        "asyncio": "WARNING",
    }
    for name, lvl in levels.items():
        logging.getLogger(name).setLevel(getattr(logging, lvl, logging.WARNING))


_quiet_third_party()

manager: ModelManager | None = None
local_token: str = ""
_main_loop: asyncio.AbstractEventLoop | None = None


def _get_main_loop() -> asyncio.AbstractEventLoop | None:
    return _main_loop

MAX_IMAGE_PIXELS = 4096 * 4096
MAX_UPLOAD_PIXELS = 32_000_000
MAX_IMAGE_DIMENSION = 8192

if Image is not None:
    Image.MAX_IMAGE_PIXELS = MAX_UPLOAD_PIXELS

IMAGE_MAGIC_BYTES = {
    b"\xff\xd8\xff": "jpeg",
    b"\x89PNG": "png",
}

ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
ALLOWED_IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}


def get_manager() -> ModelManager:
    if manager is None:
        raise RuntimeError("ModelManager not initialized")
    return manager


def _sanitize_filename(name: str) -> str:
    name = name.replace("\x00", "")
    name = re.sub(r"[/\\]", "_", name)
    name = re.sub(r"\.\.", "_", name)
    name = re.sub(r"[^\w.\- ]", "_", name)
    return name[:128] or "upload"


def _validate_image_magic(data: bytes) -> bool:
    if len(data) < 12:
        return False
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return True
    for magic in IMAGE_MAGIC_BYTES:
        if data[:len(magic)] == magic:
            return True
    return False


def _decode_image(raw: bytes) -> np.ndarray:
    if Image:
        try:
            with Image.open(io.BytesIO(raw)) as _probe:
                _pw, _ph = _probe.size
            if (
                _pw * _ph > MAX_UPLOAD_PIXELS
                or _pw > MAX_IMAGE_DIMENSION * 4
                or _ph > MAX_IMAGE_DIMENSION * 4
            ):
                LOGGER.warning("Rejected image: header reports %dx%d (over hard ceiling)", _pw, _ph)
                return None
        except Image.DecompressionBombError:
            LOGGER.warning("Rejected image: exceeds hard pixel ceiling")
            return None
        except Exception:
            pass

    def _fit(w: int, h: int) -> tuple[int, int]:
        scale = min(
            1.0,
            (MAX_IMAGE_PIXELS / max(1, w * h)) ** 0.5,
            MAX_IMAGE_DIMENSION / max(1, w, h),
        )
        if scale >= 1.0:
            return w, h
        return max(1, int(w * scale)), max(1, int(h * scale))

    if Image and ImageOps:
        try:
            pil_img = Image.open(io.BytesIO(raw))
            try:
                pil_img = ImageOps.exif_transpose(pil_img)
                w, h = pil_img.size
                nw, nh = _fit(w, h)
                if (nw, nh) != (w, h):
                    LOGGER.info("Resizing upload %dx%d -> %dx%d", w, h, nw, nh)
                    pil_img = pil_img.resize((nw, nh), Image.LANCZOS)
                rgb = np.array(pil_img)
                if rgb.ndim == 2:
                    return cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
                if rgb.shape[2] == 4:
                    return cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            finally:
                pil_img.close()
        except Image.DecompressionBombError:
            LOGGER.warning("Rejected image: exceeds hard pixel ceiling")
            return None
        except Exception:
            pass

    arr = np.frombuffer(raw, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if decoded is not None:
        h, w = decoded.shape[:2]
        nw, nh = _fit(w, h)
        if (nw, nh) != (w, h):
            LOGGER.info("Resizing upload %dx%d -> %dx%d", w, h, nw, nh)
            decoded = cv2.resize(decoded, (nw, nh), interpolation=cv2.INTER_AREA)
    return decoded


@asynccontextmanager
async def lifespan(app: FastAPI):
    global manager, local_token, _main_loop
    _main_loop = asyncio.get_running_loop()

    vram_cap = os.environ.get("SPECTRA_VRAM_HARD_CAP", "0.90")
    if _torch.cuda.is_available() and vram_cap:
        try:
            cap = min(float(vram_cap), 0.95)
            if cap > 0.0:
                _torch.cuda.set_per_process_memory_fraction(cap)
                LOGGER.info(
                    "VRAM: hard cap set to %.0f%% of device memory", cap * 100,
                    extra={"phase": "vram", "reason": "hard-cap"},
                )
        except Exception as e:
            LOGGER.warning("Could not set VRAM hard cap: %s", e)

    if _torch.cuda.is_available() and not _torch.version.hip:
        try:
            if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.8,max_split_size_mb:512,expandable_segments:True"
        except Exception:
            pass
        try:
            _torch.backends.cudnn.benchmark = True
            _torch.backends.cudnn.benchmark_limit = 4
        except Exception:
            pass

    device_info = detect_device()

    if not _torch.cuda.is_available():
        LOGGER.info("Running on CPU")
    else:
        try:
            props = _torch.cuda.get_device_properties(0)
            LOGGER.info(
                "%s · %.0f GB · compute %d.%d",
                _torch.cuda.get_device_name(0),
                props.total_memory / (1024**3),
                props.major,
                props.minor,
            )
        except Exception:
            pass

    manager = ModelManager(device=device_info.torch_device)
    manager._status_callback = _notify_model_status
    local_token = get_or_create_token()

    async def _eviction_loop():
        while True:
            await asyncio.sleep(10)
            try:
                mgr = manager
                if mgr is not None:
                    evicted = await asyncio.to_thread(mgr.evict_idle_sessions)
                    if evicted:
                        LOGGER.debug("Eviction loop: removed %d idle session(s)", evicted)
            except Exception:
                LOGGER.exception("Session eviction error")

    eviction_task = asyncio.create_task(_eviction_loop())

    LOGGER.info("Server ready.")
    sys.stdout.write("backend started\n")
    sys.stdout.flush()
    yield
    eviction_task.cancel()
    try:
        await eviction_task
    except asyncio.CancelledError:
        pass
    if manager:
        manager.destroy_all_sessions()


app = FastAPI(
    title="Spectra Segment",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(TokenAuthMiddleware, token_getter=lambda: local_token)
app.add_middleware(RequestSizeLimitMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(HostValidationMiddleware)

app.add_middleware(SecurityHeadersMiddleware)

app.add_middleware(RequestLoggingMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-Local-Token"],
)

VRAM_OOM_DETAIL = (
    "Not enough GPU memory for the current memory profile. "
    "Switch to a higher profile (or CPU mode) in the launcher "
    "or lower the SAM3 encode dimension."
)

_FLUSH_EVERY_REQUEST = os.environ.get("SPECTRA_FLUSH_EVERY_REQUEST", "") == "1"


async def _maybe_flush_cache() -> None:
    if _FLUSH_EVERY_REQUEST and _torch.cuda.is_available():
        await asyncio.to_thread(_torch.cuda.empty_cache)


@app.exception_handler(_torch.OutOfMemoryError)
async def _handle_vram_oom(request: Request, exc: _torch.OutOfMemoryError):
    free_mb = total_mb = None
    if _torch.cuda.is_available():
        try:
            free, total = _torch.cuda.mem_get_info()
            free_mb = free / (1024 * 1024)
            total_mb = total / (1024 * 1024)
        except Exception:
            pass
    LOGGER.error(
        "VRAM out of memory (cap active)%s: %s",
        (
            f" — free {free_mb:.0f} MB / total {total_mb:.0f} MB"
            if free_mb is not None and total_mb is not None
            else ""
        ),
        exc,
        extra={"phase": "vram", "reason": "oom"},
    )
    return JSONResponse(status_code=507, content={"detail": VRAM_OOM_DETAIL})


class CreateSessionRequest(BaseModel):
    model_name: str = "sam2.1_b.pt"

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        from backend.models.ultralytics_backend import ULTRALYTICS_MODELS
        from backend.models.sam3_backend import SAM3_MODEL
        allowed = set(ULTRALYTICS_MODELS.keys()) | {SAM3_MODEL}
        if v not in allowed:
            raise ValueError(f"Model '{v}' not in allowed list")
        return v


class CreateSessionResponse(BaseModel):
    session_id: str
    model_name: str


class StrokeRequest(BaseModel):
    object_id: int = 0
    points: list[list[float]] | None = None
    labels: list[int] | None = None
    bboxes: list[float] | None = None

    @field_validator("points")
    @classmethod
    def validate_points(cls, v):
        if v is not None:
            if len(v) > 10000:
                raise ValueError("Too many points (max 10000)")
            for p in v:
                if len(p) != 2:
                    raise ValueError("Each point must be [x, y]")
                if not all(-10_000 <= c <= 10_000 for c in p):
                    raise ValueError("Point coordinates are out of range")
        return v

    @field_validator("labels")
    @classmethod
    def validate_labels(cls, v):
        if v is not None:
            if len(v) > 10000:
                raise ValueError("Too many labels (max 10000)")
            if not all(lb in (0, 1) for lb in v):
                raise ValueError("labels must be 0 (negative) or 1 (positive)")
        return v

    @field_validator("bboxes")
    @classmethod
    def validate_bboxes(cls, v):
        if v is not None:
            if len(v) != 4:
                raise ValueError("bboxes must be [x1, y1, x2, y2]")
            if not all(-10_000 <= c <= 10_000 for c in v):
                raise ValueError("bbox coordinates are out of range")
            if v[2] < v[0] or v[3] < v[1]:
                raise ValueError("bbox must be [x1, y1, x2, y2] with x2>=x1 and y2>=y1")
        return v

    @field_validator("object_id")
    @classmethod
    def validate_object_id(cls, v):
        if v < 0 or v > 1_000_000:
            raise ValueError("object_id out of range")
        return v


class PredictResponse(BaseModel):
    masks: list[str]
    scores: list[float]
    object_masks: dict[str, str] = {}
    object_history: dict[str, dict[str, int]] = {}


class ExportZipFileItem(BaseModel):
    path: str
    mask_b64: str

    @field_validator("path")
    @classmethod
    def validate_path(cls, v):
        if not v or len(v) > 500:
            raise ValueError("path must be 1-500 characters")
        return v

    @field_validator("mask_b64")
    @classmethod
    def validate_mask_b64(cls, v):
        if not v:
            raise ValueError("mask_b64 is required")
        return v


class ExportZipRequest(BaseModel):
    root: str = "export"
    files: list[ExportZipFileItem] = []
    include_whole: bool = False
    format: str = "png"
    background_color: list[int] | None = None
    feather_radius: int = 3

    @field_validator("root")
    @classmethod
    def validate_root(cls, v):
        if len(v) > 200:
            raise ValueError("root must be 0-200 characters")
        return v

    @field_validator("files")
    @classmethod
    def validate_files(cls, v):
        if len(v) > 1000:
            raise ValueError("files must contain at most 1000 masks")
        return v

    @field_validator("format")
    @classmethod
    def validate_format(cls, v):
        return _validate_export_format(v)

    @field_validator("feather_radius")
    @classmethod
    def validate_feather(cls, v):
        if not 0 <= v <= 50:
            raise ValueError("Feather radius must be 0-50")
        return v

    @field_validator("background_color")
    @classmethod
    def validate_bg_color(cls, v):
        if v is not None:
            if len(v) != 3 or not all(0 <= c <= 255 for c in v):
                raise ValueError("background_color must be [R, G, B] with values 0-255")
        return v


class ExportImageRequest(BaseModel):
    files: list[ExportZipFileItem] = []
    format: str = "png"
    background_color: list[int] | None = None
    feather_radius: int = 3

    @field_validator("files")
    @classmethod
    def validate_files(cls, v):
        if len(v) > 1000:
            raise ValueError("files must contain at most 1000 masks")
        return v

    @field_validator("format")
    @classmethod
    def validate_format(cls, v):
        return _validate_export_format(v)

    @field_validator("feather_radius")
    @classmethod
    def validate_feather(cls, v):
        if not 0 <= v <= 50:
            raise ValueError("Feather radius must be 0-50")
        return v

    @field_validator("background_color")
    @classmethod
    def validate_bg_color(cls, v):
        if v is not None:
            if len(v) != 3 or not all(0 <= c <= 255 for c in v):
                raise ValueError("background_color must be [R, G, B] with values 0-255")
        return v


class DetectRequest(BaseModel):
    query: str
    confidence: float | None = None
    max_detections: int = 10
    use_yoloe_masks: bool = False

    @field_validator("query")
    @classmethod
    def validate_query(cls, v):
        if not v or len(v) > 500:
            raise ValueError("Query must be 1-500 characters")
        return v.strip()

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v):
        return _validate_confidence(v)

    @field_validator("max_detections")
    @classmethod
    def validate_max_detections(cls, v):
        if not 1 <= v <= 100:
            raise ValueError("max_detections must be 1-100")
        return v


class LoadDetectorRequest(BaseModel):
    detector_name: str


class SegmentBatchRequest(BaseModel):
    bboxes: list[list[float]]

    @field_validator("bboxes")
    @classmethod
    def validate_bboxes(cls, v):
        if not v:
            raise ValueError("At least one bbox required")
        if len(v) > 200:
            raise ValueError("Too many bboxes (max 200)")
        for i, bb in enumerate(v):
            if len(bb) != 4:
                raise ValueError(f"Bbox {i} must have 4 coordinates, got {len(bb)}")
            if any(c < 0 for c in bb):
                raise ValueError(f"Bbox {i} coordinates must be non-negative")
        return v


class Sam3PromptRequest(BaseModel):
    text: str
    confidence: float | None = None

    @field_validator("text")
    @classmethod
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError("Prompt text must not be empty")
        if len(v) > 500:
            raise ValueError("Prompt text must be 1-500 characters")
        return v.strip()

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v):
        return _validate_confidence(v)


class Sam3PromptResponse(BaseModel):
    masks: list[str]
    scores: list[float]
    bboxes: list[list[float]]


class Sam3RemoveInstanceRequest(BaseModel):
    prompt_index: int
    instance_index: int

    @field_validator("prompt_index", "instance_index")
    @classmethod
    def validate_index(cls, v):
        if v < 0:
            raise ValueError("Indexes must be non-negative")
        return v


class Sam3SettingsRequest(BaseModel):
    keep_loaded: bool | None = None
    encode_dim: int | None = None

    @field_validator("encode_dim")
    @classmethod
    def validate_encode_dim(cls, v):
        if v is not None and not 384 <= v <= 1500:
            raise ValueError("encode_dim must be 384-1500")
        return v


@app.get("/health")
async def health():
    return {"status": "ok", "active_sessions": get_manager().get_active_session_count()}


@app.get("/api/logs")
async def get_logs(limit: int = 300):
    limit = max(0, min(int(limit), len(_log_buffer)))
    entries = list(_log_buffer)[-limit:] if limit else []
    return {"entries": entries, "total": len(_log_buffer)}


@app.get("/api/logs/stream")
async def stream_logs(request: Request):
    client_ip = request.client.host if request.client else "unknown"
    if _sse_clients.get(client_ip, 0) >= MAX_SSE_PER_IP:
        raise HTTPException(429, "Too many SSE connections from this client")
    q: asyncio.Queue[dict[str, str] | None] = asyncio.Queue(maxsize=128)
    with _sub_lock:
        if len(_log_subscribers) >= MAX_SSE_SUBSCRIBERS:
            raise HTTPException(503, "Too many SSE subscribers")
        _log_subscribers.append(q)
        _sse_clients[client_ip] = _sse_clients.get(client_ip, 0) + 1
    deadline = time.monotonic() + MAX_SSE_LIFETIME_SECONDS

    async def event_generator():
        try:
            while True:
                if time.monotonic() > deadline:
                    break
                try:
                    entry = await asyncio.wait_for(q.get(), timeout=30)
                    if entry is None:
                        break
                    yield f"data: {json.dumps(entry)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            with _sub_lock:
                if q in _log_subscribers:
                    _log_subscribers.remove(q)
                _sse_clients[client_ip] = max(0, _sse_clients.get(client_ip, 1) - 1)
                if not _sse_clients.get(client_ip):
                    _sse_clients.pop(client_ip, None)

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    origin = request.headers.get("origin")
    if origin in ("http://localhost:3000", "http://127.0.0.1:3000"):
        headers["Access-Control-Allow-Origin"] = origin
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=headers,
    )


@app.get("/api/models/status")
async def model_status():
    statuses = await asyncio.to_thread(_check_model_statuses)
    return {"models": statuses}


_model_status_subscribers: list[asyncio.Queue[str | None]] = []


async def _broadcast_model_status(event: str | None = None) -> None:
    statuses = await asyncio.to_thread(_check_model_statuses)
    payload = json.dumps({"models": statuses, "event": event})
    for q in list(_model_status_subscribers):
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            pass


def _notify_model_status(event: str) -> None:
    try:
        loop = _main_loop
        if loop is None or loop.is_closed():
            return
        asyncio.run_coroutine_threadsafe(_broadcast_model_status(event), loop)
    except Exception:
        LOGGER.debug("Failed to broadcast model status", exc_info=True)


def _is_hf_model_downloaded(name: str) -> bool:
    from backend.models.grounding_detector import HF_MODEL_IDS, hf_weights_downloaded
    hf_id = HF_MODEL_IDS.get(name, name)
    return hf_weights_downloaded(hf_id)


def _check_model_statuses() -> list[dict]:
    from backend.models.ultralytics_backend import ULTRALYTICS_MODELS, MODEL_METADATA
    from backend.models.grounding_detector import GROUNDING_DETECTORS, DETECTOR_METADATA
    from backend.models.yoloe_detector import YOLOE_MODELS, YOLOE_METADATA
    from backend.models.sam3_backend import SAM3_MODEL, SAM3_MODEL_DISPLAY
    model_dir = MODEL_WEIGHTS_DIR
    mgr = get_manager()
    statuses = []
    for name, display in ULTRALYTICS_MODELS.items():
        meta = MODEL_METADATA.get(name, {})
        statuses.append({
            "name": name, "display_name": display, "type": "segment",
            "downloaded": (model_dir / name).exists(),
            "loaded": name == mgr._current_model,
            "tier": meta.get("tier", "medium"), "perf": meta.get("perf", ""),
        })
    statuses.append({
        "name": SAM3_MODEL, "display_name": SAM3_MODEL_DISPLAY, "type": "sam3",
        "downloaded": (model_dir / SAM3_MODEL).exists(),
        "loaded": SAM3_MODEL == mgr._current_model,
        "tier": "medium", "perf": "text-driven semantic segmentation",
    })
    for name, display in GROUNDING_DETECTORS.items():
        meta = DETECTOR_METADATA.get(name, {})
        dl = _is_hf_model_downloaded(name)
        statuses.append({
            "name": name, "display_name": display, "type": "detector",
            "detector_type": "grounding",
            "downloaded": dl,
            "loaded": name == mgr._current_detector,
            "tier": meta.get("tier", "medium"), "perf": meta.get("perf", ""),
        })
    for name, display in YOLOE_MODELS.items():
        meta = YOLOE_METADATA.get(name, {})
        weight_file = f"{name}.pt"
        statuses.append({
            "name": name, "display_name": display, "type": "detector",
            "detector_type": "yoloe",
            "downloaded": (model_dir / weight_file).exists(),
            "loaded": name == mgr._current_detector,
            "tier": meta.get("tier", "medium"), "perf": meta.get("perf", ""),
        })
    return statuses


@app.get("/api/models/stream")
async def stream_model_status(request: Request):
    q: asyncio.Queue[str | None] = asyncio.Queue(maxsize=64)
    with _sub_lock:
        if len(_model_status_subscribers) >= MAX_SSE_SUBSCRIBERS:
            raise HTTPException(503, "Too many SSE subscribers")
        _model_status_subscribers.append(q)

    async def generator():
        last = None
        try:
            current = await asyncio.to_thread(_check_model_statuses)
            last = json.dumps({"models": current}, sort_keys=True)
            yield f"data: {last}\n\n"
            while True:
                try:
                    item = await asyncio.wait_for(q.get(), timeout=3)
                    if item is None:
                        break
                    yield f"data: {item}\n\n"
                    last = item
                except asyncio.TimeoutError:
                    current = await asyncio.to_thread(_check_model_statuses)
                    current_json = json.dumps({"models": current}, sort_keys=True)
                    if current_json != last:
                        yield f"data: {current_json}\n\n"
                        last = current_json
        finally:
            with _sub_lock:
                if q in _model_status_subscribers:
                    _model_status_subscribers.remove(q)

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    origin = request.headers.get("origin")
    if origin in ("http://localhost:3000", "http://127.0.0.1:3000"):
        headers["Access-Control-Allow-Origin"] = origin
    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers=headers,
    )


@app.get("/api/token")
async def get_token():
    return {"token": local_token}

@app.post("/api/sessions", response_model=CreateSessionResponse)
async def create_session(req: CreateSessionRequest):
    try:
        session = await asyncio.to_thread(get_manager().create_session, req.model_name)
    except RuntimeError as e:
        raise HTTPException(429, str(e))
    LOGGER.info(
        "Session %s created with %s", session.session_id, session.model_name,
        extra={"phase": "session", "model": session.model_name,
               "sid": session.session_id},
    )
    return CreateSessionResponse(
        session_id=session.session_id,
        model_name=session.model_name,
    )


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    info = await asyncio.to_thread(get_manager().get_session_info, session_id)
    if info is None:
        raise HTTPException(404, "Session not found")
    return info


@app.delete("/api/sessions/{session_id}")
async def destroy_session(session_id: str):
    await asyncio.to_thread(get_manager().destroy_session, session_id)
    LOGGER.info(
        "Session %s destroyed", session_id,
        extra={"phase": "session", "sid": session_id},
    )
    return {"status": "destroyed"}


@app.post("/api/sessions/{session_id}/release")
async def release_session(session_id: str):
    get_manager().destroy_session(session_id)
    LOGGER.info(
        "Session %s released", session_id,
        extra={"phase": "session", "sid": session_id},
    )
    return {"status": "destroyed"}


@app.post("/api/sessions/{session_id}/image")
async def upload_image(session_id: str, file: UploadFile = File(...)):
    logctx.set_session(session_id)
    session = get_manager().get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    raw = await file.read(MAX_UPLOAD_BYTES + 1)

    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large. Max: {MAX_UPLOAD_BYTES // (1024*1024)}MB")

    filename = (file.filename or "").lower()
    extension = filename.rsplit(".", 1)[-1] if "." in filename else ""
    if extension not in ALLOWED_IMAGE_EXTENSIONS and not (
        file.content_type and file.content_type in ALLOWED_IMAGE_MIME_TYPES
    ):
        raise HTTPException(
            400,
            "Unsupported file format. Supported: .jpg, .jpeg, .png, .webp",
        )

    if not _validate_image_magic(raw):
        LOGGER.warning(
            "Upload rejected: not a valid image (magic bytes check failed)",
            extra={"phase": "upload", "sid": session_id},
        )
        raise HTTPException(400, "File is not a valid image (magic bytes check failed)")

    t_decode = time.time()
    image = _decode_image(raw)
    if image is None:
        LOGGER.warning(
            "Upload rejected: image could not be decoded or exceeds the ceiling",
            extra={"phase": "upload", "sid": session_id},
        )
        raise HTTPException(400, "Failed to decode image")

    image_id = _sanitize_filename(file.filename or "upload")
    LOGGER.info(
        "Upload accepted: %s (%dx%d), decoding took %s",
        extension.upper() or "image", image.shape[1], image.shape[0],
        format_duration(time.time() - t_decode),
        extra={
            "phase": "upload", "sid": session_id,
            "dur_ms": int((time.time() - t_decode) * 1000),
        },
    )
    await asyncio.to_thread(get_manager().load_image, session_id, image, image_id)

    return {
        "image_id": image_id,
        "width": image.shape[1],
        "height": image.shape[0],
    }


def _validate_confidence(value: Any) -> Any:
    if value is not None and not 0.0 <= value <= 1.0:
        raise ValueError("Confidence must be 0.0-1.0")
    return value


def _validate_export_format(value: Any) -> Any:
    if value not in ("png", "jpg", "jpeg"):
        raise ValueError("Format must be 'png' or 'jpg'")
    return value


def _as_tolist(value: Any) -> list:
    return value.tolist() if hasattr(value, "tolist") else []


def _masks_to_png_list(masks: Any) -> list[str]:
    if masks is None:
        return []
    out: list[str] = []
    for m in masks:
        arr = np.asarray(m)
        if arr.size == 0:
            continue
        out.append(mask_to_png_b64(arr))
    return out


def _decode_mask_resized(mask_b64: str, h: int, w: int) -> np.ndarray:
    if h < 1 or w < 1 or h * w > MAX_IMAGE_PIXELS:
        raise HTTPException(400, "Invalid mask dimensions")
    mask = _decode_mask_b64(mask_b64)
    if mask.shape != (h, w):
        mask_img = Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)
        mask = np.array(mask_img) > 127
    return mask


def _masks_to_predict_response(result: dict) -> PredictResponse:
    masks = result.get("all_masks")
    if masks is None:
        masks = result.get("masks", [])
    scores = result.get("all_scores")
    if scores is None:
        scores = result.get("scores", [])
    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    return PredictResponse(
        masks=_masks_to_png_list(masks),
        scores=list(scores),
        object_masks=_masks_to_png_dict(result.get("object_masks")),
        object_history=result.get("object_history") or {},
    )


def _masks_positional_png_list(masks: Any) -> list[str | None]:
    if masks is None:
        return []
    out: list[str | None] = []
    for m in masks:
        arr = np.asarray(m)
        if arr.size == 0:
            out.append(None)
            continue
        out.append(mask_to_png_b64(arr))
    return out


def _masks_to_png_dict(masks: Any) -> dict[str, str]:
    if not masks:
        return {}
    out: dict[str, str] = {}
    for key, m in masks.items():
        arr = np.asarray(m)
        if arr.size == 0:
            continue
        out[str(key)] = mask_to_png_b64(arr)
    return out


@app.post("/api/sessions/{session_id}/predict", response_model=PredictResponse)
async def run_prediction(session_id: str, req: StrokeRequest):
    try:
        result = await asyncio.to_thread(
            get_manager().predict,
            session_id=session_id,
            object_id=req.object_id,
            points=req.points,
            labels=req.labels,
            bboxes=req.bboxes,
        )
    except ValueError as e:
        LOGGER.exception("Prediction failed for session %s", session_id)
        raise HTTPException(400, f"Invalid prediction request: {e}")

    await _maybe_flush_cache()
    return _masks_to_predict_response(result)


@app.post("/api/sessions/{session_id}/segment-batch")
async def segment_batch(session_id: str, req: SegmentBatchRequest):
    try:
        masks = await asyncio.to_thread(
            get_manager().segment_batch,
            session_id=session_id,
            bboxes=req.bboxes,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except _torch.OutOfMemoryError:
        LOGGER.exception("VRAM OOM during batch segmentation for session %s", session_id)
        raise HTTPException(507, VRAM_OOM_DETAIL)
    except Exception:
        LOGGER.exception("Batch segmentation failed for session %s", session_id)
        raise HTTPException(500, "Batch segmentation failed")
    await _maybe_flush_cache()
    return {"masks": _masks_positional_png_list(masks)}


@app.post("/api/sessions/{session_id}/undo", response_model=PredictResponse)
async def undo_stroke(session_id: str, object_id: int = 0):
    result = await asyncio.to_thread(get_manager().undo, session_id, object_id)
    if result is None:
        exists = await asyncio.to_thread(get_manager().session_exists, session_id)
        if not exists:
            raise HTTPException(404, "Session not found")
        raise HTTPException(409, "Nothing to undo for this subject")
    return _masks_to_predict_response(result)


@app.post("/api/sessions/{session_id}/redo", response_model=PredictResponse)
async def redo_stroke(session_id: str, object_id: int = 0):
    result = await asyncio.to_thread(get_manager().redo, session_id, object_id)
    if result is None:
        exists = await asyncio.to_thread(get_manager().session_exists, session_id)
        if not exists:
            raise HTTPException(404, "Session not found")
        raise HTTPException(409, "Nothing to redo for this subject")
    return _masks_to_predict_response(result)


@app.post("/api/sessions/{session_id}/clear-object", response_model=PredictResponse)
async def clear_object(session_id: str, object_id: int = 0):
    result = await asyncio.to_thread(get_manager().clear_object, session_id, object_id)
    if result is None:
        raise HTTPException(404, "Session not found")
    return _masks_to_predict_response(result)


def _masks_to_sam3_response(result: dict) -> Sam3PromptResponse:
    return Sam3PromptResponse(
        masks=_masks_to_png_list(result.get("masks")),
        scores=_as_tolist(result.get("scores")),
        bboxes=_as_tolist(result.get("bboxes")),
    )


@app.post("/api/sessions/{session_id}/sam3-prompt", response_model=Sam3PromptResponse)
async def sam3_prompt(session_id: str, req: Sam3PromptRequest):
    try:
        result = await asyncio.to_thread(
            get_manager().sam3_predict,
            session_id=session_id,
            text=req.text,
            confidence=req.confidence,
        )
    except ValueError as e:
        LOGGER.exception("SAM3 prompt failed for session %s (text=%r)", session_id, req.text)
        raise HTTPException(400, str(e))
    except _torch.OutOfMemoryError:
        LOGGER.exception("VRAM OOM during SAM3 prompt for session %s (text=%r)", session_id, req.text)
        raise HTTPException(507, VRAM_OOM_DETAIL)
    except Exception:
        LOGGER.exception("SAM3 prompt failed for session %s (text=%r)", session_id, req.text)
        raise HTTPException(500, "SAM3 prompt failed")

    await _maybe_flush_cache()
    return _masks_to_sam3_response(result)


@app.post("/api/sessions/{session_id}/sam3-undo", response_model=Sam3PromptResponse)
async def sam3_undo(session_id: str):
    result = await asyncio.to_thread(get_manager().sam3_undo, session_id)
    if result is None:
        raise HTTPException(404, "Nothing to undo")
    return _masks_to_sam3_response(result)


@app.post("/api/sessions/{session_id}/sam3-redo", response_model=Sam3PromptResponse)
async def sam3_redo(session_id: str):
    result = await asyncio.to_thread(get_manager().sam3_redo, session_id)
    if result is None:
        raise HTTPException(404, "Nothing to redo")
    return _masks_to_sam3_response(result)


@app.post("/api/sessions/{session_id}/sam3-remove-instance", response_model=Sam3PromptResponse)
async def sam3_remove_instance(session_id: str, req: Sam3RemoveInstanceRequest):
    try:
        result = await asyncio.to_thread(
            get_manager().sam3_remove_instance,
            session_id=session_id,
            prompt_index=req.prompt_index,
            instance_index=req.instance_index,
        )
    except ValueError as e:
        LOGGER.exception("SAM3 remove instance failed for session %s", session_id)
        raise HTTPException(400, str(e))
    except _torch.OutOfMemoryError:
        LOGGER.exception("VRAM OOM during SAM3 remove instance for session %s", session_id)
        raise HTTPException(507, VRAM_OOM_DETAIL)
    except Exception:
        LOGGER.exception("SAM3 remove instance failed for session %s", session_id)
        raise HTTPException(500, "SAM3 remove instance failed")
    if result is None:
        raise HTTPException(404, "No prompts to remove from")
    return _masks_to_sam3_response(result)


@app.post("/api/settings/sam3")
async def sam3_settings(req: Sam3SettingsRequest):
    try:
        return await asyncio.to_thread(
            get_manager().set_sam3_settings,
            keep_loaded=req.keep_loaded,
            encode_dim=req.encode_dim,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except _torch.OutOfMemoryError:
        LOGGER.exception("VRAM OOM while updating SAM3 settings")
        raise HTTPException(507, VRAM_OOM_DETAIL)
    except Exception:
        LOGGER.exception("Failed to update SAM3 settings")
        raise HTTPException(500, "Failed to update SAM3 settings")


@app.get("/api/detectors")
async def list_detectors():
    return await asyncio.to_thread(get_manager().list_detectors)


@app.post("/api/detectors/load")
async def load_detector(req: LoadDetectorRequest):
    try:
        await asyncio.to_thread(get_manager().load_detector, req.detector_name)
    except ValueError as e:
        raise HTTPException(400, f"Invalid detector name: {e}")
    except _torch.OutOfMemoryError:
        LOGGER.exception("VRAM OOM loading detector %s", req.detector_name)
        raise HTTPException(507, VRAM_OOM_DETAIL)
    except Exception:
        LOGGER.exception("Failed to load detector %s", req.detector_name)
        raise HTTPException(500, "Failed to load detector")
    return {"status": "loaded", "detector": req.detector_name}


@app.post("/api/sessions/{session_id}/detect")
async def run_detection(session_id: str, req: DetectRequest):
    try:
        result = await asyncio.to_thread(
            get_manager().detect,
            session_id=session_id,
            query=req.query,
            confidence=req.confidence,
            max_detections=req.max_detections,
            use_yoloe_masks=req.use_yoloe_masks,
        )
    except ValueError as e:
        LOGGER.exception("Detection validation failed for session %s (query=%r, max_detections=%s)", session_id, req.query, req.max_detections)
        raise HTTPException(400, f"Invalid detection request: {e}")
    except _torch.OutOfMemoryError:
        LOGGER.exception("VRAM OOM during detection for session %s (query=%r)", session_id, req.query)
        raise HTTPException(507, VRAM_OOM_DETAIL)
    except Exception:
        LOGGER.exception("Detection failed for session %s (query=%r)", session_id, req.query)
        raise HTTPException(500, "Detection failed")
    return result


def _safe_arcname(root: str, path: str) -> str:
    root_norm = root.replace("\\", "/").strip("/")
    base = posixpath.basename(root_norm)
    if base in (".", ".."):
        base = ""

    p = posixpath.normpath(path.replace("\\", "/"))
    if p.startswith("/") or p.startswith("..") or "/.." in p or ":" in p:
        raise HTTPException(400, f"Invalid zip path: {path}")
    return posixpath.join(base, p)


def _decode_mask_b64(mask_b64: str) -> np.ndarray:
    if Image is None:
        raise HTTPException(400, "PIL is required for mask export")
    try:
        data = base64.b64decode(mask_b64)
        img = Image.open(io.BytesIO(data))
        if img.width * img.height > MAX_IMAGE_PIXELS:
            raise HTTPException(400, "Mask dimensions too large")
        img = img.convert("L")
    except (ValueError, OSError) as exc:
        raise HTTPException(400, "Invalid mask data") from exc
    except Image.DecompressionBombError as exc:
        raise HTTPException(400, "Mask image too large") from exc
    return np.array(img) > 127


@app.post("/api/sessions/{session_id}/export-zip")
async def export_zip(session_id: str, req: ExportZipRequest):
    from backend.utils.compositing import composite_and_encode

    session = get_manager().get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    if session.image_rgb is None:
        raise HTTPException(400, "No image loaded")
    if not req.files:
        raise HTTPException(400, "No masks to export")

    image_rgb = session.image_rgb
    h, w = image_rgb.shape[:2]
    bg_color = tuple(req.background_color) if req.background_color else None
    bg_img = np.full_like(image_rgb, bg_color) if bg_color is not None else None
    feather = req.feather_radius

    combined = np.zeros((h, w), dtype=bool) if req.include_whole else None
    count = 0
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in req.files:
            mask = _decode_mask_resized(item.mask_b64, h, w)
            if combined is not None:
                combined |= mask
            count += 1
            png_bytes = await asyncio.to_thread(
                composite_and_encode,
                mask=mask,
                image_rgb=image_rgb,
                output_format="png",
                background_color=bg_color,
                feather_radius=feather,
                bg_image=bg_img,
            )
            zf.writestr(_safe_arcname(req.root, item.path), png_bytes)

        if req.include_whole:
            use_jpg = req.format in ("jpg", "jpeg") and bg_color is not None
            whole_bytes = await asyncio.to_thread(
                composite_and_encode,
                mask=combined,
                image_rgb=image_rgb,
                output_format="jpg" if use_jpg else "png",
                background_color=bg_color,
                feather_radius=feather,
                bg_image=bg_img,
            )
            whole_name = f"whole.{'jpg' if use_jpg else 'png'}"
            zf.writestr(_safe_arcname(req.root, whole_name), whole_bytes)

    zip_bytes = zip_buf.getvalue()
    return {
        "data": base64.b64encode(zip_bytes).decode("ascii"),
        "format": "zip",
        "count": count,
    }


@app.post("/api/sessions/{session_id}/export-image")
async def export_image(session_id: str, req: ExportImageRequest):
    from backend.utils.compositing import composite_and_encode

    session = get_manager().get_session(session_id)
    if not session or session.image_rgb is None:
        raise HTTPException(404, "Session not found")
    if not req.files:
        raise HTTPException(400, "No masks to export")

    image_rgb = session.image_rgb
    h, w = image_rgb.shape[:2]
    bg_color = tuple(req.background_color) if req.background_color else None
    bg_img = np.full_like(image_rgb, bg_color) if bg_color is not None else None

    combined = np.zeros((h, w), dtype=bool)
    for item in req.files:
        mask = _decode_mask_resized(item.mask_b64, h, w)
        combined |= mask

    use_jpg = req.format in ("jpg", "jpeg") and bg_color is not None
    out_bytes = await asyncio.to_thread(
        composite_and_encode,
        mask=combined,
        image_rgb=image_rgb,
        output_format="jpg" if use_jpg else "png",
        background_color=bg_color,
        feather_radius=req.feather_radius,
        bg_image=bg_img,
    )
    media_type = "image/jpeg" if use_jpg else "image/png"
    return Response(content=out_bytes, media_type=media_type)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
