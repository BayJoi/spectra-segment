from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable

import cv2
import numpy as np
import torch as _torch

from .base import SegmentationBackend
from .detector_base import DetectorBackend
from .ultralytics_backend import ULTRALYTICS_MODELS, UltralyticsBackend
from .grounding_detector import GROUNDING_DETECTORS, DETECTOR_METADATA, GroundingDetector, HF_MODEL_IDS, hf_weights_downloaded
from .yoloe_detector import YOLOE_MODELS, YOLOE_METADATA, YOLEDetector
from .sam3_backend import SAM3_MODEL, SAM3Backend, DEFAULT_ENCODE_DIM, ENCODE_DIM_MAX
from ..utils.compositing import mask_to_png_b64
from ..utils.device import MODEL_WEIGHTS_DIR, release_gpu_memory
from ..utils.humantime import format_duration

LOGGER = logging.getLogger(__name__)

IDLE_TIMEOUT = 600

SAM3_IDLE_UNLOAD_TIMEOUT = int(os.environ.get("SAM3_IDLE_UNLOAD_TIMEOUT", "300"))

SAM3_ENCODE_DIM_MIN = 384
SAM3_ENCODE_DIM_MAX = ENCODE_DIM_MAX

DETECTOR_IDLE_TIMEOUT = 300

MAX_STROKE_HISTORY = int(os.environ.get("SPECTRA_MAX_HISTORY", "200"))

SAM3_MAX_HISTORY = int(os.environ.get("SAM3_MAX_HISTORY", "100"))

MAX_REPLAY_POINTS = 2048

MAX_OBJECTS_PER_SESSION = int(os.environ.get("SPECTRA_MAX_OBJECTS", "50"))

MAX_SESSIONS = int(os.environ.get("SPECTRA_MAX_SESSIONS", "100"))


@dataclass(slots=True)
class Session:
    session_id: str
    model_name: str
    backend: SegmentationBackend
    image_id: str | None = None
    image_width: int = 0
    image_height: int = 0
    image_rgb: np.ndarray | None = None
    image_bgr: np.ndarray | None = None
    last_active: float = field(default_factory=time.time)
    objects: dict[int, ObjectState] = field(default_factory=dict)
    sam3_prompt_history: list[tuple[SAM3PromptEntry, SAM3Snapshot]] = field(default_factory=list)
    sam3_redo_stack: list[tuple[SAM3PromptEntry, SAM3Snapshot]] = field(default_factory=list)
    sam3_last_masks: np.ndarray | None = None
    sam3_last_scores: np.ndarray | None = None
    sam3_last_bboxes: np.ndarray | None = None
    has_image: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass(slots=True)
class MaskSnapshot:
    low_res: np.ndarray | None = None


@dataclass(slots=True)
class ObjectState:
    object_id: int
    stroke_history: list[StrokeEntry] = field(default_factory=list)
    redo_stack: list[tuple[StrokeEntry, MaskSnapshot | None]] = field(default_factory=list)
    last_mask: np.ndarray | None = None
    last_low_res_mask: np.ndarray | None = None
    mask_snapshots: list[MaskSnapshot] = field(default_factory=list)


@dataclass(slots=True)
class StrokeEntry:
    points: list[list[float]]
    labels: list[int]
    bboxes: list[float] | None = None


@dataclass(slots=True)
class SAM3PromptEntry:
    text: str
    confidence: float | None = None


@dataclass(slots=True)
class SAM3Snapshot:
    masks: np.ndarray | None = None
    scores: np.ndarray | None = None
    bboxes: np.ndarray | None = None
    mask_shape: tuple[int, ...] | None = None


class ModelManager:
    def __init__(self, device: str = "cpu") -> None:
        self._device = device
        self._sessions: dict[str, Session] = {}
        self._loaded_models: dict[str, SegmentationBackend] = {}
        self._current_model: str | None = None
        self._detector: DetectorBackend | None = None
        self._current_detector: str | None = None
        self._last_detector_name: str | None = None
        self._last_seg_use: float = time.time()
        self._last_detector_use: float = time.time()
        self._keep_sam3_loaded: bool = False
        self._no_co_residency: bool = os.environ.get("SPECTRA_NO_CO_RESIDENCY", "") == "1"
        self._sam3_encode_dim: int = DEFAULT_ENCODE_DIM
        self._lock = threading.Lock()
        self._model_load_locks: dict[str, threading.Lock] = {}
        self._status_callback: Callable[[str], None] | None = None

    def _can_unload(self, obj: Any) -> bool:
        if not obj.is_loaded:
            return False
        infer_lock = getattr(obj, "_infer_lock", None)
        if infer_lock is None:
            return True
        try:
            acquired = infer_lock.acquire(blocking=False)
        except Exception:
            return False
        if acquired:
            infer_lock.release()
            return True
        return False

    def _safe_unload(self, obj: Any) -> bool:
        if not self._can_unload(obj):
            LOGGER.debug("Residency: unload skipped (inference in flight)", extra={"phase":"residency"})
            return False
        try:
            obj.unload_model()
            return True
        except Exception:
            LOGGER.exception("Error unloading model for VRAM headroom")
            return False

    def _evict_loaded_models(self, keep_backend: SegmentationBackend | None = None) -> None:
        if not _torch.cuda.is_available() or self._device == "cpu":
            return
        evict: list[Any] = []
        with self._lock:
            if self._detector is not None and self._detector.is_loaded:
                if self._can_unload(self._detector):
                    evict.append(self._detector)
                    self._detector = None
                    self._current_detector = None
                else:
                    LOGGER.debug("Residency: detector left resident (inference in flight)", extra={"phase":"residency","detector":self._current_detector})
            for name in list(self._loaded_models.keys()):
                obj = self._loaded_models[name]
                if obj is keep_backend or not obj.is_loaded:
                    continue
                if not self._can_unload(obj):
                    LOGGER.debug("Residency: %s left resident (inference in flight)", name, extra={"phase":"residency","model":name})
                    continue
                evict.append(obj)
                del self._loaded_models[name]
                if self._current_model == name:
                    self._current_model = None
        for obj in evict:
            self._safe_unload(obj)
        if _torch.cuda.is_available():
            try:
                _torch.cuda.empty_cache()
                _torch.cuda.ipc_collect()
            except Exception:
                pass

    def _ensure_vram_headroom(self, required_mb: int, keep_backend: SegmentationBackend | None = None) -> None:
        if not _torch.cuda.is_available() or self._device == "cpu":
            return
        try:
            _torch.cuda.empty_cache()
            free, total = _torch.cuda.mem_get_info()
        except Exception:
            return
        free_mb = free / (1024 * 1024)
        total_mb = total / (1024 * 1024)
        try:
            fraction = min(float(os.environ.get("SPECTRA_VRAM_FRACTION", "1.0")), 0.95)
        except (ValueError, TypeError):
            fraction = 1.0
        if fraction <= 0.0:
            return
        budget_mb = fraction * total_mb
        projected_mb = (total_mb - free_mb) + required_mb
        if projected_mb <= budget_mb:
            return

        LOGGER.info(
            "VRAM: freeing headroom â€” over budget by %.0f MB "
            "(free %.0f / total %.0f MB, need %.0f MB)",
            max(0.0, projected_mb - budget_mb), free_mb, total_mb, required_mb,
            extra={"phase":"vram","reason":"over-budget"},
        )
        self._evict_loaded_models(keep_backend=keep_backend)

    def _ensure_encode_headroom(self, keep_backend: SegmentationBackend | None = None) -> None:
        self._evict_loaded_models(keep_backend=keep_backend)

    def get_or_load_model(self, model_name: str) -> SegmentationBackend:
        with self._lock:
            if (
                self._current_model == model_name
                and model_name in self._loaded_models
                and self._loaded_models[model_name].is_loaded
            ):
                LOGGER.debug("Model already loaded: %s", model_name)
                self._last_seg_use = time.time()
                return self._loaded_models[model_name]

        if model_name == SAM3_MODEL:
            pass
        elif model_name not in ULTRALYTICS_MODELS:
            raise ValueError(
                f"Unknown segmentation model '{model_name}'. "
                f"Available: {sorted(set(ULTRALYTICS_MODELS) | {SAM3_MODEL})}"
            )

        model_dir = MODEL_WEIGHTS_DIR
        model_file = model_dir / model_name
        if not model_file.exists():
            LOGGER.info("%s not cached â€” downloading", model_name)
            if self._status_callback:
                self._status_callback("downloading")
            from backend_amd_gpu.utils.net_check import is_connected
            if not is_connected():
                raise RuntimeError(
                    f"No internet connection and '{model_name}' not found locally.\n"
                    f"  Connect to the internet to download, or place the file in:\n"
                    f"  {model_dir}"
                )
            from backend_amd_gpu.utils.model_integrity import download_model_file
            download_model_file(model_name, model_file)
            if self._status_callback:
                self._status_callback("downloaded")
        else:
            from backend_amd_gpu.utils.model_integrity import verify_model
            try:
                verified = verify_model(model_file, model_name)
            except Exception as exc:
                LOGGER.warning("Model integrity check failed (%s) â€” continuing: %s", model_name, exc)
            else:
                if not verified:
                    raise RuntimeError(
                        f"Model integrity check failed for '{model_name}'.\n"
                        f"  Local file may be corrupted or tampered with.\n"
                        f"  Delete the file and re-download it:\n"
                        f"  {model_file}"
                    )

        with self._model_load_lock(model_name):
            with self._lock:
                if (
                    self._current_model == model_name
                    and model_name in self._loaded_models
                    and self._loaded_models[model_name].is_loaded
                ):
                    self._last_seg_use = time.time()
                    return self._loaded_models[model_name]

            if self._no_co_residency and self._detector is not None and self._detector.is_loaded:
                LOGGER.info("Residency: unloading detector %s (no co-residency)", self._current_detector, extra={"phase":"residency","detector":self._current_detector,"reason":"no-co-residency"})
                self._safe_unload(self._detector)
                self._detector = None
                self._current_detector = None

            if self._current_model and self._current_model in self._loaded_models:
                LOGGER.info("Residency: unloading previous model %s", self._current_model, extra={"phase":"residency","model":self._current_model,"reason":"model-switch"})
                self._safe_unload(self._loaded_models[self._current_model])
                del self._loaded_models[self._current_model]
                self._current_model = None

            if model_name == SAM3_MODEL:
                backend: SegmentationBackend = SAM3Backend(device=self._device)
                backend._encode_dim = self._sam3_encode_dim
            else:
                backend = UltralyticsBackend(device=self._device)
            if self._status_callback:
                self._status_callback("loading")
            try:
                req_mb = int(model_file.stat().st_size / (1024 * 1024)) * 2 + 512
            except Exception:
                req_mb = 4096
            self._ensure_vram_headroom(req_mb)
            _load_t0 = time.time()
            backend.load_model(model_name)

            with self._lock:
                self._loaded_models[model_name] = backend
                self._current_model = model_name
                self._last_seg_use = time.time()
            if self._status_callback:
                self._status_callback("loaded")
            LOGGER.info("Model %s ready in %s", model_name, format_duration(time.time() - _load_t0), extra={"phase":"residency","model":model_name,"dur_ms":int((time.time()-_load_t0)*1000)})
            return backend

    @contextmanager
    def _model_load_lock(self, model_name: str):
        with self._lock:
            lock = self._model_load_locks.get(model_name)
            if lock is None:
                lock = threading.Lock()
                self._model_load_locks[model_name] = lock
        lock.acquire()
        try:
            yield
        finally:
            lock.release()

    def create_session(self, model_name: str) -> Session:
        self.evict_idle_sessions()
        with self._lock:
            if len(self._sessions) >= MAX_SESSIONS:
                raise RuntimeError(f"Maximum sessions ({MAX_SESSIONS}) reached")
            # Reserve the slot under the same lock so N concurrent creations
            # cannot all pass the MAX_SESSIONS check. The value stays None until
            # the backend exists, and every consumer skips reserved slots.
            session_id = str(uuid.uuid4())
            self._sessions[session_id] = None  # type: ignore[assignment]
        try:
            backend = self.get_or_load_model(model_name)
        except Exception:
            with self._lock:
                self._sessions.pop(session_id, None)
            raise

        session = Session(
            session_id=session_id,
            model_name=model_name,
            backend=backend,
        )
        with self._lock:
            self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Session | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.last_active = time.time()
        return session

    def session_exists(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.get(session_id) is not None

    def destroy_session(self, session_id: str) -> None:
        unload_model: SegmentationBackend | None = None
        unload_model_name: str | None = None
        unload_detector: DetectorBackend | None = None
        unload_detector_name: str | None = None
        with self._lock:
            session = self._sessions.pop(session_id, None)
            if session and not self._sessions:
                unload_model_name = self._current_model
                if unload_model_name:
                    unload_model = self._loaded_models.pop(unload_model_name, None)
                    self._current_model = None
                unload_detector_name = self._current_detector
                unload_detector = self._detector
                self._detector = None
                self._current_detector = None
        if session:
            if session.backend.has_image:
                if self._can_unload(session.backend):
                    session.backend.reset_image()
                else:
                    LOGGER.debug("Session %s backend busy â€” skipping reset_image", session_id)
            session.image_rgb = None
            session.image_bgr = None
            session.objects.clear()
            LOGGER.debug("Destroyed session %s", session_id)
            if unload_model is not None and self._safe_unload(unload_model):
                LOGGER.info("Residency: unloading %s (idle)", unload_model_name, extra={"phase":"residency","model":unload_model_name,"reason":"idle"})
            if unload_detector is not None and self._safe_unload(unload_detector):
                LOGGER.info("Residency: unloading detector %s (idle)", unload_detector_name, extra={"phase":"residency","detector":unload_detector_name,"reason":"idle"})
            if _torch.cuda.is_available():
                try:
                    _torch.cuda.empty_cache()
                except Exception:
                    pass

    def destroy_all_sessions(self) -> None:
        with self._lock:
            sids = list(self._sessions.keys())
        for sid in sids:
            self.destroy_session(sid)

    def load_image(self, session_id: str, image: np.ndarray, image_id: str) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        if session.backend.has_image:
            session.backend.reset_image()

        session.image_id = image_id
        session.image_height, session.image_width = image.shape[:2]
        session.image_bgr = image
        session.image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        session.objects.clear()
        session.sam3_prompt_history.clear()
        session.sam3_redo_stack.clear()
        session.sam3_last_masks = None
        session.sam3_last_scores = None
        session.sam3_last_bboxes = None

        LOGGER.debug("Encoding %dx%d image for session %s ...",
                     image.shape[1], image.shape[0], session_id,
                     extra={"phase": "encode", "sid": session_id})
        with self._lock:
            self._last_seg_use = time.time()
        t0 = time.time()
        try:
            if session.backend.is_loaded:
                self._ensure_encode_headroom(session.backend)
            session.backend.set_image(image)
            elapsed = time.time() - t0
            LOGGER.debug("Encoding complete in %.2fs", elapsed, extra={"phase":"encode","sid":session.session_id,"dur_ms":int(elapsed*1000)})
            session.has_image = True
        except Exception:
            LOGGER.exception(
                "SAM encoding failed for session %s â€” continuing without features", session_id
            )
            session.has_image = False
            if self._can_unload(session.backend):
                session.backend.reset_image()

        session.last_active = time.time()
        LOGGER.info("Encoded %dx%d image in %s", session.image_width, session.image_height, format_duration(time.time() - t0), extra={"phase":"encode","sid":session.session_id,"dur_ms":int((time.time()-t0)*1000)})

    def _ensure_model_loaded(self, session: Session) -> None:
        with self._lock:
            self._last_seg_use = time.time()

        if session.backend.is_loaded:
            if session.backend.has_image or session.image_rgb is None:
                return
            LOGGER.debug("Encoding image for session %s", session.session_id, extra={"phase":"encode","sid":session.session_id})
            self._ensure_encode_headroom(session.backend)
            session.backend.set_image(session.image_bgr)
            return

        if self._status_callback:
            self._status_callback("loading")
        try:
            LOGGER.debug("Reloading %s", session.model_name)
            self._ensure_encode_headroom(None)
            backend = self.get_or_load_model(session.model_name)
            session.backend = backend
            if session.image_rgb is not None:
                backend.set_image(session.image_bgr)
        finally:
            if self._status_callback:
                self._status_callback("loaded")

    def _ensure_detector_loaded(self) -> DetectorBackend | None:
        with self._lock:
            det = self._detector
        if det is not None and det.is_loaded:
            return det
        with self._lock:
            name = self._last_detector_name
        if not name:
            return None
        LOGGER.info("Reloading detector %s", name)
        self.load_detector(name)
        with self._lock:
            return self._detector

    def predict(
        self,
        session_id: str,
        object_id: int,
        points: list[list[float]] | None = None,
        labels: list[int] | None = None,
        bboxes: list[float] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        self._ensure_model_loaded(session)
        if not session.backend.has_image:
            raise ValueError("No image loaded in session")

        if object_id not in session.objects:
            if len(session.objects) >= MAX_OBJECTS_PER_SESSION:
                raise ValueError(
                    f"Maximum objects per session ({MAX_OBJECTS_PER_SESSION}) reached"
                )
            session.objects[object_id] = ObjectState(object_id=object_id)

        obj = session.objects[object_id]

        appended = False
        if points or bboxes:
            with session.lock:
                if points is not None and len(points) > MAX_REPLAY_POINTS:
                    n = len(points)
                    keep = sorted(set(np.round(np.linspace(0, n - 1, MAX_REPLAY_POINTS)).astype(int).tolist()))
                    points = [points[i] for i in keep]
                    if labels is not None and len(labels) == n:
                        labels = [labels[i] for i in keep]
                    LOGGER.debug("Stroke points capped to %d", len(points))
                obj.stroke_history.append(
                    StrokeEntry(
                        points=points or [],
                        labels=labels or [],
                        bboxes=bboxes,
                    )
                )
                obj.redo_stack.clear()
                appended = True

                if len(obj.stroke_history) > MAX_STROKE_HISTORY:
                    excess = len(obj.stroke_history) - MAX_STROKE_HISTORY
                    del obj.stroke_history[:excess]
                    del obj.mask_snapshots[:excess]

        LOGGER.debug("Predicting for subject %d in session %s ...", object_id, session_id, extra={"phase":"decode","sid":session_id,"model":session.model_name})
        t0 = time.time()
        result = self._compute_mask(session, obj)
        elapsed = time.time() - t0
        mask_count = len(result.get("masks", [])) if hasattr(result.get("masks", []), "__len__") else 0
        LOGGER.info("Subject %d mask ready in %s (%d mask(s))", object_id, format_duration(elapsed), mask_count, extra={"phase":"decode","sid":session_id,"dur_ms":int(elapsed*1000),"model":session.model_name})

        if appended:
            with session.lock:
                masks = result["masks"]
                has_mask = masks is not None and len(masks) > 0
                obj.mask_snapshots.append(
                    MaskSnapshot(
                        low_res=result.get("low_res_masks") if has_mask else None,
                    )
                )
                if has_mask:
                    obj.last_mask = masks
                    obj.last_low_res_mask = result.get("low_res_masks")
                else:
                    obj.last_mask = None
                    obj.last_low_res_mask = None

        session.last_active = time.time()
        result["all_masks"], result["all_scores"], result["object_masks"] = self._collect_all_masks(session)
        result["object_history"] = self._object_history(session)
        return result

    def _object_history(self, session: Session) -> dict[str, dict[str, int]]:
        with session.lock:
            return {
                str(oid): {
                    "undo": len(obj.stroke_history),
                    "redo": len(obj.redo_stack),
                    "strokes": len(obj.stroke_history),
                    "has_mask": obj.last_mask is not None and len(obj.last_mask) > 0,
                }
                for oid, obj in session.objects.items()
            }

    def _collect_all_masks(self, session: Session) -> tuple[list[np.ndarray], list[float], dict[str, np.ndarray]]:
        masks: list[np.ndarray] = []
        scores: list[float] = []
        object_masks: dict[str, np.ndarray] = {}
        with session.lock:
            oids = sorted(session.objects.keys())
        for oid in oids:
            with session.lock:
                obj = session.objects.get(oid)
                if obj is None or obj.last_mask is None or len(obj.last_mask) == 0:
                    continue
                m = np.asarray(obj.last_mask)[0]
            masks.append(m)
            scores.append(1.0)
            object_masks[str(oid)] = m
        return masks, scores, object_masks

    def segment_batch(
        self,
        session_id: str,
        bboxes: list[list[float]],
    ) -> list[list[list[bool]]]:
        with self._lock:
            session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        self._ensure_model_loaded(session)
        if not session.backend.has_image:
            raise ValueError("No image loaded in session")
        if isinstance(session.backend, SAM3Backend):
            raise ValueError("Batch segmentation is not supported for SAM3 sessions")

        self._ensure_full_image_encoded(session)
        LOGGER.debug(
            "Batch segment: %d bboxes for session %s", len(bboxes), session_id,
        )
        t0 = time.time()
        masks = session.backend.predict_batch(bboxes)
        elapsed = time.time() - t0
        LOGGER.info("Batch segment complete in %s â€” %d mask(s)", format_duration(elapsed), len(masks), extra={"phase":"segment","sid":session_id,"dur_ms":int(elapsed*1000),"model":session.model_name})
        session.last_active = time.time()
        return masks

    def undo(self, session_id: str, object_id: int) -> dict[str, Any] | None:
        with self._lock:
            session = self._sessions.get(session_id)
        if not session:
            return None

        obj = session.objects.get(object_id)
        if not obj or len(obj.stroke_history) == 0:
            return None

        with session.lock:
            popped = obj.stroke_history.pop()
            if obj.mask_snapshots:
                saved_snapshot = obj.mask_snapshots.pop()
            else:
                saved_snapshot = None
            obj.redo_stack.append((popped, saved_snapshot))
            if len(obj.redo_stack) > MAX_STROKE_HISTORY:
                excess = len(obj.redo_stack) - MAX_STROKE_HISTORY
                del obj.redo_stack[:excess]

        session.last_active = time.time()
        if not session.backend.is_loaded:
            self._ensure_model_loaded(session)
        result = self._replay_history(session, obj, full_replay=True)
        result["all_masks"], result["all_scores"], result["object_masks"] = self._collect_all_masks(session)
        result["object_history"] = self._object_history(session)
        return result

    def redo(self, session_id: str, object_id: int) -> dict[str, Any] | None:
        with self._lock:
            session = self._sessions.get(session_id)
        if not session:
            return None

        obj = session.objects.get(object_id)
        if not obj or len(obj.redo_stack) == 0:
            return None

        with session.lock:
            entry, saved_snapshot = obj.redo_stack.pop()
            obj.stroke_history.append(entry)
            obj.mask_snapshots.append(saved_snapshot if saved_snapshot is not None else MaskSnapshot(low_res=None))

        session.last_active = time.time()
        if not session.backend.is_loaded:
            self._ensure_model_loaded(session)
        result = self._replay_history(session, obj, full_replay=True)
        result["all_masks"], result["all_scores"], result["object_masks"] = self._collect_all_masks(session)
        result["object_history"] = self._object_history(session)
        return result

    def clear_object(self, session_id: str, object_id: int) -> dict[str, Any] | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            with session.lock:
                session.objects.pop(object_id, None)
        LOGGER.debug("Cleared object %d in session %s", object_id, session_id)
        session.last_active = time.time()
        all_masks, all_scores, object_masks = self._collect_all_masks(session)
        return {
            "all_masks": all_masks,
            "all_scores": all_scores,
            "object_masks": object_masks,
            "object_history": self._object_history(session),
        }

    def sam3_predict(
        self,
        session_id: str,
        text: str,
        confidence: float | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        self._ensure_model_loaded(session)
        if not session.backend.has_image:
            raise ValueError("No image loaded in session")
        if not isinstance(session.backend, SAM3Backend):
            raise ValueError(f"Session {session_id} is not a SAM3 session")

        entry = SAM3PromptEntry(text=text, confidence=confidence)
        LOGGER.debug("Running SAM3 prompt for session %s: %r", session_id, text)
        t0 = time.time()
        result = session.backend.predict_text(text, confidence=confidence)
        elapsed = time.time() - t0
        mask_count = len(result.get("masks", [])) if hasattr(result.get("masks", []), "__len__") else 0
        LOGGER.info("SAM3 prompt complete in %s â€” %d mask(s)", format_duration(elapsed), mask_count, extra={"phase":"sam3","sid":session_id,"dur_ms":int(elapsed*1000),"model":SAM3_MODEL})

        with session.lock:
            session.sam3_prompt_history.append((entry, self._sam3_snapshot(result)))
            session.sam3_redo_stack.clear()
            if len(session.sam3_prompt_history) > SAM3_MAX_HISTORY:
                excess = len(session.sam3_prompt_history) - SAM3_MAX_HISTORY
                del session.sam3_prompt_history[:excess]
        session.sam3_last_masks = result["masks"]
        session.sam3_last_scores = result["scores"]
        session.sam3_last_bboxes = result["bboxes"]
        with self._lock:
            self._last_seg_use = time.time()
            session.last_active = time.time()
        return result

    def sam3_undo(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            session = self._sessions.get(session_id)
        if not session:
            return None

        if not session.sam3_prompt_history:
            return None

        with session.lock:
            popped = session.sam3_prompt_history.pop()
            session.sam3_redo_stack.append(popped)
            if len(session.sam3_redo_stack) > SAM3_MAX_HISTORY:
                excess = len(session.sam3_redo_stack) - SAM3_MAX_HISTORY
                del session.sam3_redo_stack[:excess]
        prev = session.sam3_prompt_history[-1] if session.sam3_prompt_history else None
        if prev is not None:
            session.sam3_last_masks = self._unpack_masks(prev[1].masks, prev[1].mask_shape)
            session.sam3_last_scores = prev[1].scores
            session.sam3_last_bboxes = prev[1].bboxes
        else:
            session.sam3_last_masks = None
            session.sam3_last_scores = None
            session.sam3_last_bboxes = None
        return self._sam3_response(session)

    def sam3_redo(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            session = self._sessions.get(session_id)
        if not session:
            return None

        if not session.sam3_redo_stack:
            return None

        entry, snap = session.sam3_redo_stack.pop()
        session.sam3_prompt_history.append((entry, snap))
        session.sam3_last_masks = self._unpack_masks(snap.masks, snap.mask_shape)
        session.sam3_last_scores = snap.scores
        session.sam3_last_bboxes = snap.bboxes
        return self._sam3_response(session)

    def sam3_remove_instance(
        self,
        session_id: str,
        prompt_index: int,
        instance_index: int,
    ) -> dict[str, Any] | None:
        with self._lock:
            session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        if not session.sam3_prompt_history:
            return None
        if prompt_index < 0 or prompt_index >= len(session.sam3_prompt_history):
            raise ValueError("Invalid prompt index")

        entry, snap = session.sam3_prompt_history[prompt_index]
        masks = self._unpack_masks(snap.masks, snap.mask_shape)
        scores = snap.scores
        bboxes = snap.bboxes
        if masks is None or instance_index < 0 or instance_index >= masks.shape[0]:
            raise ValueError("Invalid instance index")

        masks = np.delete(masks, instance_index, axis=0)
        if scores is not None and scores.shape[0] > instance_index:
            scores = np.delete(scores, instance_index, axis=0)
        if bboxes is not None and bboxes.shape[0] > instance_index:
            bboxes = np.delete(bboxes, instance_index, axis=0)

        with session.lock:
            if masks.shape[0] == 0:
                del session.sam3_prompt_history[prompt_index]
                session.sam3_redo_stack.clear()
            else:
                session.sam3_prompt_history[prompt_index] = (
                    entry,
                    SAM3Snapshot(
                        masks=self._pack_masks(masks),
                        scores=scores,
                        bboxes=bboxes,
                        mask_shape=masks.shape,
                    ),
                )

        if session.sam3_prompt_history:
            _, last = session.sam3_prompt_history[-1]
            session.sam3_last_masks = self._unpack_masks(last.masks, last.mask_shape)
            session.sam3_last_scores = last.scores
            session.sam3_last_bboxes = last.bboxes
        else:
            session.sam3_last_masks = None
            session.sam3_last_scores = None
            session.sam3_last_bboxes = None
        return self._sam3_response(session)

    @staticmethod
    def _pack_masks(masks: np.ndarray | None) -> np.ndarray | None:
        if isinstance(masks, np.ndarray) and masks.dtype == bool:
            return np.packbits(masks)
        return masks

    @staticmethod
    def _unpack_masks(packed: np.ndarray | None, shape: tuple[int, ...] | None) -> np.ndarray | None:
        if packed is not None and shape and len(shape) == 3:
            return np.unpackbits(packed)[: int(np.prod(shape))].reshape(shape).astype(bool)
        return packed

    @staticmethod
    def _sam3_snapshot(result: dict[str, Any]) -> SAM3Snapshot:
        masks = result.get("masks")
        return SAM3Snapshot(
            masks=ModelManager._pack_masks(masks),
            scores=result.get("scores"),
            bboxes=result.get("bboxes"),
            mask_shape=masks.shape if isinstance(masks, np.ndarray) else None,
        )

    @staticmethod
    def _sam3_response(session: Session) -> dict[str, Any]:
        if session.sam3_last_masks is None:
            return {
                "masks": np.array([], dtype=bool),
                "scores": np.array([], dtype=float),
                "bboxes": np.zeros((0, 4), dtype=float),
            }
        return {
            "masks": session.sam3_last_masks,
            "scores": session.sam3_last_scores,
            "bboxes": session.sam3_last_bboxes,
        }

    def set_sam3_settings(
        self,
        keep_loaded: bool | None = None,
        encode_dim: int | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if keep_loaded is not None:
                self._keep_sam3_loaded = bool(keep_loaded)
            if encode_dim is not None:
                if not (SAM3_ENCODE_DIM_MIN <= int(encode_dim) <= SAM3_ENCODE_DIM_MAX):
                    raise ValueError(
                        f"encode_dim must be within {SAM3_ENCODE_DIM_MIN}-{SAM3_ENCODE_DIM_MAX}"
                    )
                self._sam3_encode_dim = int(encode_dim)
            keep_loaded_out = self._keep_sam3_loaded
            backend = self._loaded_models.get(SAM3_MODEL)

        if encode_dim is not None and backend is not None:
            backend.set_encoding_resolution(self._sam3_encode_dim)
            with self._lock:
                sessions = list(self._sessions.values())
            for s in sessions:
                if s.backend is backend and s.image_rgb is not None:
                    self._ensure_full_image_encoded(s)

        encode_dim_out = (
            getattr(backend, "_encode_dim", None)
            if backend is not None
            else self._sam3_encode_dim
        )
        return {
            "keep_loaded": keep_loaded_out,
            "encode_dim": encode_dim_out,
        }

    def _compute_mask(
        self,
        session: Session,
        obj: ObjectState,
    ) -> dict[str, Any]:
        self._ensure_full_image_encoded(session)
        return self._replay_history(session, obj)

    def _ensure_full_image_encoded(self, session: Session) -> None:
        if session.image_rgb is None:
            return
        H, W = session.image_rgb.shape[:2]
        cur = getattr(session.backend, "_src_shape", None)
        if cur != (H, W):
            with self._lock:
                self._last_seg_use = time.time()
            self._ensure_encode_headroom(session.backend)
            session.backend.set_image(session.image_bgr)

    def _replay_history(
        self,
        session: Session,
        obj: ObjectState,
        full_replay: bool = False,
    ) -> dict[str, Any]:
        self._ensure_full_image_encoded(session)
        with session.lock:
            history = list(obj.stroke_history)

        if full_replay:
            obj.last_mask = None
            obj.last_low_res_mask = None
            if not history:
                return {"masks": np.array([]), "scores": np.array([]), "low_res_masks": None}

            result: dict[str, Any] = {"masks": np.array([]), "scores": np.array([]), "low_res_masks": None}
            for idx, entry in enumerate(history):
                is_first = idx == 0
                result = self._apply_stroke(
                    session,
                    obj,
                    entry,
                    first_stroke=is_first,
                    mask_input=None if is_first else obj.last_low_res_mask,
                )
                obj.last_mask = result["masks"]
                obj.last_low_res_mask = result.get("low_res_masks")
            LOGGER.debug(
                "Full replay: %d stroke(s) (%d positive, %d negative in last)",
                len(history),
                history[-1].labels.count(1) if history[-1].labels else 0,
                history[-1].labels.count(0) if history[-1].labels else 0,
            )
            return result

        if len(history) == 0:
            obj.last_mask = None
            obj.last_low_res_mask = None
            return {"masks": np.array([]), "scores": np.array([]), "low_res_masks": None}

        entry = history[-1]
        is_first_prompt = obj.last_low_res_mask is None
        mask_input = (
            obj.last_low_res_mask if (not is_first_prompt and not entry.bboxes) else None
        )
        result = self._apply_stroke(
            session, obj, entry, first_stroke=is_first_prompt, mask_input=mask_input
        )
        obj.last_mask = result["masks"]
        obj.last_low_res_mask = result.get("low_res_masks")
        return result

    def _apply_stroke(
        self,
        session: Session,
        obj: ObjectState,
        entry: StrokeEntry,
        first_stroke: bool,
        mask_input: np.ndarray | None,
    ) -> dict[str, Any]:
        points = entry.points
        labels = entry.labels
        bboxes = entry.bboxes
        pos_count = labels.count(1) if labels else 0
        neg_count = labels.count(0) if labels else 0

        use_multimask = first_stroke and not bboxes and len(points) <= 1

        if first_stroke and not bboxes and pos_count > 0 and neg_count > 0:
            pos_points = [p for p, lb in zip(points, labels) if lb == 1]
            pos_labels = [1] * len(pos_points)
            coarse = session.backend.predict(
                points=pos_points,
                labels=pos_labels,
                bboxes=None,
                mask_input=None,
                multimask_output=False,
            )
            coarse_low = coarse.get("low_res_masks")
            if coarse_low is not None and coarse_low.size > 0:
                LOGGER.debug(
                    "First-stroke refinement: pass1 mask (positives only) -> pass2 refine with %d negatives",
                    neg_count,
                )
                return session.backend.predict(
                    points=points if points else None,
                    labels=labels if labels else None,
                    bboxes=bboxes,
                    mask_input=coarse_low,
                    multimask_output=False,
                )
            return session.backend.predict(
                points=points if points else None,
                labels=labels if labels else None,
                bboxes=bboxes,
                mask_input=mask_input,
                multimask_output=use_multimask,
            )

        return session.backend.predict(
            points=points if points else None,
            labels=labels if labels else None,
            bboxes=bboxes,
            mask_input=mask_input,
            multimask_output=use_multimask,
        )

    def evict_idle_sessions(self) -> int:
        now = time.time()
        evicted = 0
        with self._lock:
            idle_ids = [
                sid for sid, s in self._sessions.items()
                if s is not None and (now - s.last_active) > IDLE_TIMEOUT
            ]
        for sid in idle_ids:
            self.destroy_session(sid)
            evicted += 1
        if evicted:
            LOGGER.debug("Evicted %d idle sessions", evicted)

        detector_idle = now - self._last_detector_use
        if detector_idle > DETECTOR_IDLE_TIMEOUT:
            with self._lock:
                cur_detector = self._detector
                if cur_detector is None or not cur_detector.is_loaded:
                    cur_detector = None
                else:
                    self._detector = None
                    self._current_detector = None
            if cur_detector is not None:
                LOGGER.debug("Residency: unloading idle detector %s (idle for %.0fs)", self._current_detector, detector_idle, extra={"phase":"residency","detector":self._current_detector,"reason":"idle"})
                try:
                    cur_detector.unload_model()
                except Exception:
                    LOGGER.exception("Error unloading idle detector")
                if self._status_callback:
                    self._status_callback("unloading")

        seg_idle = now - self._last_seg_use
        unload_timeout = IDLE_TIMEOUT if self._keep_sam3_loaded else SAM3_IDLE_UNLOAD_TIMEOUT
        unload_model: SegmentationBackend | None = None
        unload_model_name: str | None = None
        if seg_idle > unload_timeout:
            with self._lock:
                current_model = self._current_model
                has_sessions = bool(self._sessions)
                seg_idle_now = time.time() - self._last_seg_use
            if (
                current_model == SAM3_MODEL
                and seg_idle_now > unload_timeout
                and not has_sessions
            ):
                with self._lock:
                    candidate = self._loaded_models.get(current_model)
                    if candidate is not None and self._can_unload(candidate):
                        unload_model_name = current_model
                        unload_model = self._loaded_models.pop(unload_model_name)
                        self._current_model = None
        if unload_model is not None:
            LOGGER.info("Residency: unloading SAM3 (idle)", extra={"phase":"residency","model":SAM3_MODEL,"reason":"idle"})
            self._safe_unload(unload_model)
            if self._status_callback:
                self._status_callback("unloading")

        return evicted

    def get_active_session_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def list_detectors(self) -> list[dict[str, Any]]:
        model_dir = MODEL_WEIGHTS_DIR
        detectors = []
        with self._lock:
            cur = self._current_detector
        for name, display in GROUNDING_DETECTORS.items():
            meta = DETECTOR_METADATA.get(name, {})
            detectors.append({
                "name": name,
                "display_name": display,
                "type": "grounding",
                "loaded": name == cur,
                "downloaded": self._detector_downloaded(name),
                "tier": meta.get("tier", "medium"),
                "perf": meta.get("perf", ""),
            })
        for name, display in YOLOE_MODELS.items():
            meta = YOLOE_METADATA.get(name, {})
            detectors.append({
                "name": name,
                "display_name": display,
                "type": "yoloe",
                "loaded": name == cur,
                "downloaded": (model_dir / f"{name}.pt").exists(),
                "tier": meta.get("tier", "medium"),
                "perf": meta.get("perf", ""),
            })
        return detectors

    def _detector_downloaded(self, detector_name: str) -> bool:
        if detector_name in GROUNDING_DETECTORS:
            hf_id = HF_MODEL_IDS.get(detector_name, detector_name)
            return hf_weights_downloaded(hf_id)
        if detector_name in YOLOE_MODELS:
            return (MODEL_WEIGHTS_DIR / f"{detector_name}.pt").exists()
        return False

    def load_detector(self, detector_name: str) -> None:
        with self._lock:
            if self._current_detector == detector_name and self._detector and self._detector.is_loaded:
                self._last_detector_use = time.time()
                return
            cur_detector = self._detector
            seg_to_unload = None
            if self._no_co_residency and self._current_model and self._current_model in self._loaded_models:
                LOGGER.info("Residency: unloading segmentation model %s (no co-residency)", self._current_model, extra={"phase":"residency","model":self._current_model,"reason":"no-co-residency"})
                seg_to_unload = self._loaded_models.pop(self._current_model)
                self._current_model = None
            keep_seg = (
                self._current_model
                and self._current_model in self._loaded_models
                and os.environ.get("SPECTRA_KEEP_SEG_ON_DETECT", "") == "1"
            )

        if seg_to_unload is not None:
            self._safe_unload(seg_to_unload)

        if cur_detector and cur_detector.is_loaded:
            if not self._safe_unload(cur_detector):
                LOGGER.debug("Residency: previous detector left resident (inference in flight)", extra={"phase":"residency","detector":self._current_detector})

        if detector_name not in GROUNDING_DETECTORS and detector_name not in YOLOE_MODELS:
            with self._lock:
                self._detector = cur_detector
                if cur_detector is not None and cur_detector.is_loaded:
                    self._current_detector = None
            raise ValueError(f"Unknown detector: {detector_name}")

        if detector_name in GROUNDING_DETECTORS:
            new_detector: DetectorBackend = GroundingDetector(device=self._device)
        else:
            new_detector = YOLEDetector(device=self._device)

        t0 = time.time()
        tier = DETECTOR_METADATA.get(detector_name, {}).get("tier", "medium")
        required_mb = {"tiny": 1024, "small": 1024, "medium": 1536, "large": 2560}.get(tier, 2048)
        if keep_seg:
            LOGGER.info("Residency: preserving segmentation model %s (KEEP_SEG_ON_DETECT)", self._current_model, extra={"phase":"residency","model":self._current_model,"reason":"keep-seg-on-detect"})
            self._ensure_vram_headroom(required_mb, keep_backend=self._loaded_models.get(self._current_model))
        else:
            self._ensure_vram_headroom(required_mb)
        downloaded = self._detector_downloaded(detector_name)
        if self._status_callback:
            self._status_callback("detector-downloading" if not downloaded else "detector-loading")
        new_detector.load_model(detector_name)
        load_elapsed = time.time() - t0
        LOGGER.info("Detector %s loaded in %s", detector_name, format_duration(load_elapsed), extra={"phase":"residency","detector":detector_name,"dur_ms":int(load_elapsed*1000)})
        if self._status_callback:
            self._status_callback("detector-loaded")

        with self._lock:
            self._detector = new_detector
            self._current_detector = detector_name
            self._last_detector_name = detector_name
            self._last_detector_use = time.time()

    def detect(
        self,
        session_id: str,
        query: str,
        confidence: float | None = None,
        max_detections: int = 10,
        use_yoloe_masks: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            session = self._sessions.get(session_id)
            self._last_detector_use = time.time()
        if not session:
            raise ValueError(f"Session {session_id} not found")
        if session.image_rgb is None:
            raise ValueError("No image loaded in session")

        detector = self._ensure_detector_loaded()
        if not detector:
            raise ValueError("No detector loaded. Call load_detector() first.")

        seg_to_unload = None
        if os.environ.get("SPECTRA_KEEP_SEG_ON_DETECT") != "1":
            with self._lock:
                if self._current_model and self._current_model in self._loaded_models:
                    candidate = self._loaded_models.get(self._current_model)
                    if candidate is not None and self._can_unload(candidate):
                        seg_to_unload = self._loaded_models.pop(self._current_model)
                        self._current_model = None
        if seg_to_unload is not None:
            LOGGER.debug(
                "Unloading segmentation model %s for detection",
                getattr(seg_to_unload, "_model_path", "?"),
            )
            self._safe_unload(seg_to_unload)
            release_gpu_memory()

        with session.lock:
            image_bgr = session.image_bgr
            image_rgb = session.image_rgb
        if image_bgr is None and image_rgb is not None:
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        if image_bgr is None:
            raise ValueError("No image loaded in session")

        t0 = time.time()
        detections = detector.detect(
            image=image_bgr,
            query=query,
            confidence=confidence,
            max_detections=max_detections,
        )
        detect_elapsed = time.time() - t0
        if hasattr(detector, '_device'):
            LOGGER.debug("Detector %s running on %s", self._current_detector, detector._device, extra={"phase":"detect","detector":self._current_detector})

        det_list = []
        for d in detections:
            entry: dict[str, Any] = {"bbox": d.bbox, "score": d.score, "label": d.label}
            if use_yoloe_masks and d.mask is not None:
                entry["mask"] = mask_to_png_b64(d.mask)
            det_list.append(entry)

        session.last_active = time.time()
        with self._lock:
            self._last_detector_use = time.time()

        release_gpu_memory()

        LOGGER.info(
            "Detected '%s' â€” %d found, %d kept (%s)",
            query, len(detections), len(det_list), format_duration(detect_elapsed),
        )

        return {
            "detections": det_list,
            "query": query,
            "image_id": session.image_id,
        }

    def get_session_info(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            session = self._sessions.get(session_id)
        if not session:
            return None
        return {
            "session_id": session.session_id,
            "model_name": session.model_name,
            "image_id": session.image_id,
            "image_width": session.image_width,
            "image_height": session.image_height,
            "object_count": len(session.objects),
            "last_active": session.last_active,
            "idle_seconds": time.time() - session.last_active,
        }
