# Code reference

The code-level reference for the whole repository. Every module, every route, every
exported symbol and the reasoning behind the non-obvious ones. The topic guides cover
architecture and workflow; this file covers **what the code is and does**:

- `docs/CODE_GUIDE.md` - startup chain, HTTP/env/model catalogue, verification, fork rules
- `docs/BACKEND_GUIDE.md` - working on the backend: endpoints, manager state, recipes
- `docs/FRONTEND.md` - browser side: state, canvas, prompts, undo/redo
- `docs/LOGGING.md` - log format, structured fields, the console panel

## Comment policy

Source files carry **no explanatory comments and no docstrings**. Only tool directives
stay:

- `# type: ignore[...]` for the type checker
- `# noqa: ...` for the linter
- `/// <reference ... />` for TypeScript ambient types

Everything else - why a lock is held, why a fallback exists, what a cache key means -
belongs here, tied to a code reference. `backend/requirements.txt` and
`backend_amd_gpu/requirements.txt` carry no comments either; the version floors and
their reasons are recorded in section 12.

## How entries are written

- Paths are repo-relative. `path#symbol` names the function, method, class or
  module-level constant the text is about. Line numbers are never used.
- `backend/` is the CPU/NVIDIA fork. `backend_amd_gpu/` is the ROCm fork. A fact marked
  **(both forks)** is true of each tree separately; the two trees never import each
  other.
- Section 2 lists exactly which files are identical and which are allowed to differ.
  Read it before assuming a statement about one fork applies to the other.
- Anything quoted from a library is attributed to the version installed in
  `backend_amd_gpu/.venv` (section 3) where the behaviour is version-dependent.

---

## 1. Inventory

Every tracked file, and what it is.

### Root

| Path | Role |
|---|---|
| `install.bat` | CPU/NVIDIA installer: uv, Python 3.13.14, torch, requirements, CLIP, Bun |
| `install_amd.bat` | AMD installer: uv, Python 3.12.13, ROCm torch, requirements, CLIP, Bun, `bun run build` |
| `start.bat` | CPU/NVIDIA launcher: preflight, memory profile, Bun, backend, frontend |
| `start_amd.bat` | AMD launcher: gfx/HIP environment, memory profile, caches, backend, frontend |
| `stop.bat` / `stop_amd.bat` | Stop a running backend/frontend pair |
| `cleanup.bat` / `cleanup_amd.bat` | Remove venv, caches, workspace, toolchain, logs; optional model weights |
| `launcher/setup_memory.ps1` | The only place that decides the memory profile |
| `checks/check_gpu.bat` | Wrapper that runs `check_gpu_health.py` from whichever venv exists |
| `checks/check_gpu_health.py` | GPU/device smoke test; writes a timestamped log next to itself |
| `frontend/` | React 19 + Vite + UnoCSS + jotai UI |
| `frontend/use_web.bat` | Junctions the shared frontend into `<fork>/web` |
| `frontend/bun_env.cmd`, `frontend/bun_fetch.ps1` | Locate or download the project Bun |
| `tools/` | Portable uv/Python/Bun installed by the installers (not tracked) |
| `docs/` | These guides |
| `README.md`, `LICENSE`, `THIRD-PARTY-LICENSES.txt` | Project files |
| `.gitignore` | Ignores the venvs, both uv caches, both `model_weights`, both `logs`, both `web` workspaces, `tools/`, `frontend/node_modules`, `frontend/dist`, `*.pyc` and the health-check logs |

### `backend/` (CUDA/CPU fork)

| Path | Role |
|---|---|
| `main.py` | FastAPI app: env redirects, logging, middleware wiring, request models, every route |
| `models/base.py` | `SegmentationBackend` ABC |
| `models/detector_base.py` | `DetectorBackend` ABC and the `Detection` dataclass |
| `models/manager.py` | Sessions, model residency, subject/stroke history, SAM3 history |
| `models/ultralytics_backend.py` | SAM 2 / 2.1 through Ultralytics `SAM2Predictor` |
| `models/sam3_backend.py` | SAM 3 text-prompt segmentation |
| `models/grounding_detector.py` | Grounding DINO, MM-GDINO, Florence-2, CogFlorence |
| `models/yoloe_detector.py` | YOLOE detectors with a MobileCLIP text encoder |
| `utils/compositing.py` | Mask to PNG, feather, alpha composite, encode |
| `utils/device.py` | Device detection and `release_gpu_memory` |
| `utils/humantime.py` | `format_duration` |
| `utils/logctx.py` | `rid`/`sid` context variables |
| `utils/model_integrity.py` | Pinned URLs, sha256 digests, download, verify, TOFU |
| `utils/net_check.py` | `is_connected()` pre-flight probe |
| `utils/offline.py` | `offline_guard` / `is_offline` |
| `utils/security.py` | Middleware stack and the local token |
| `utils/stderr_progress.py` | `StderrInterceptor` for library progress output |
| `utils/torch_threads.py` | `resolve_threads`, `cpu_threads`, `cudnn_disabled` |
| `run.cmd` | Sets cache redirects and torch/OpenMP knobs, then runs uvicorn |
| `requirements.txt` | Python dependencies (no torch) |

### `backend_amd_gpu/` (ROCm fork)

Everything above, plus:

| Path | Role |
|---|---|
| `amd_gfx.py` | `gfx101*/gfx103*/gfx110*` to `HSA_OVERRIDE_GFX_VERSION` mapping |
| `detect_amd_gpu.py` | Standalone Windows GPU probe: gfx target, best-effort VRAM, override |

`backend_amd_gpu/main.py` adds the AMD preamble and HIP tuning; section 11 lists every
difference.

### `frontend/` (root files and `src/`)

| Path | Role |
|---|---|
| `main.tsx` | Bootstraps React, UnoCSS, global CSS, the grain texture |
| `App.tsx` | Layout switch, global keyboard handling, dialogs, detectors fetch |
| `globals.css` | Imports the two fontsource packages, then `styles/base.css`, `styles/animations.css`, `styles/components.css` in that order |
| `styles/base.css` | Reset (border-box, element defaults, list/heading margins), the `--color-orange-500` token, the dark `body` shell, focus-visible ring and selection colour |
| `styles/animations.css` | Ten keyframes and the utility classes that use them: `animate-fade-in`, `animate-drop-in`, `animate-model-highlight`, `animate-model-highlight-idle`, `animate-pulse-glow`, `animate-shimmer-bar`, `animate-screen-sweep`, plus `grain-shift`/`hero-in`/`shimmer`/`model-load-pulse` consumed elsewhere |
| `styles/components.css` | Range-input styling for both WebKit and Gecko (track, thumb, focus ring, disabled), `custom-scrollbar`, `hero-entrance`, `hero-delay-2`, `smooth-dialog`, `shimmer-overlay`, `animate-model-load`, `grain-bg`/`grain-bg-strong`, the `#splash` screen, and the reduced-motion block |
| `utils/grain.ts` | Generates the 256x256 noise tile behind `--grain-url` |
| `lib/api.ts` | Every HTTP call, token handling, mask decode/encode |
| `lib/types.ts` | Response and view types |
| `lib/mask.ts` | `PackedMask` |
| `lib/sse.ts` | `connectSse` with backoff |
| `lib/utils.ts` | `cn`, image extension/MIME checks, `fitImageFile` |
| `lib/strokePrompt.ts` | Stroke to point/label prompts |
| `lib/detectionEdit.ts` | Shared detection reindexing |
| `store/session.ts` | Session, image, mask, subject and history atoms, palette |
| `store/ui.ts` | Dialog/overlay flags, console state, toasts, callback atoms |
| `store/detection.ts` | Detection mode and detector state |
| `store/layers.ts` | Layers and `nextLayerId` |
| `store/sam3.ts` | SAM 3 mode, prompts, instances, settings |
| `store/logs.ts` | UI log bus and `safeText` redaction |
| `hooks/useSession.ts` | Session lifecycle, upload, predict, undo/redo queue |
| `hooks/useSessionGuard.ts` | `pagehide` release, `pageshow` health check |
| `hooks/useCanvasRenderer.ts` | Four-canvas rendering, mask caches, DPR |
| `hooks/useBrushDrawing.ts` | Brush preview, coordinate mapping, processing ring |
| `hooks/useDetection.ts` | Detector load + run, accumulating detections |
| `hooks/useLayers.ts` | Layer add/remove/sync, server subject clearing |
| `hooks/useSam3.ts` | SAM 3 prompt, undo/redo, instance removal |
| `hooks/usePanZoom.ts` | `fitToView`, `zoomIn`, `zoomOut` |
| `hooks/usePanelDismiss.ts` | Outside-click and Escape dismissal for floating panels |
| `components/Header.tsx` | Title, model selector, transparent toggle, console, settings, upload |
| `components/Canvas.tsx` | Canvas stack, pointer handling, detection hit test |
| `components/Toolbar.tsx` | Brush and detect toolbars, zoom, export entry |
| `components/SubjectsPanel.tsx` | Subject list with per-subject undo/redo/delete |
| `components/LayersPanel.tsx` | Detection groups and SAM 3 segments |
| `components/Sam3Page.tsx` | SAM 3 layout: non-interactive canvas + prompt bar |
| `components/Sam3PromptBar.tsx` | Prompt input, undo/redo, segments, options, zoom |
| `components/ExportDialog.tsx` | Export scope, format, background, feather, ZIP download |
| `components/ConsolePanel.tsx` | Log list, filters, progress rows, model list |
| `components/EmptyState.tsx` | Home screen: mode switch, drop zone, upload |
| `components/ModeDialog.tsx` | Brush/Detect chooser after the first upload |
| `components/ModeSwitchDialog.tsx` | Confirm Brush/Detect <-> SAM 3 switch |
| `components/EndSessionDialog.tsx` | Confirm session teardown |
| `components/UnsupportedFileDialog.tsx` | Rejected upload explanation |
| `components/SettingsDialog.tsx` | Keyboard shortcut list |
| `components/EncodingOverlay.tsx` | Full-screen encode/upload overlay |
| `components/Toast.tsx` | Toast host |
| `components/ui/*` | `Button`, `Select`, `Tooltip`, `TierIcon`, `BoxesToggle`, `DialogFooter` |
| `vite.config.ts` | Port 3000, aliases, CSP/headers, build chunks |
| `uno.config.ts` | Font families and the orange palette |
| `tsconfig.json` | Strict TS, `@/*` path alias |
| `vite-env.d.ts` | Ambient declarations for `vite/client`, `*.css` and `virtual:uno.css` |
| `package.json` | React 19, jotai, clsx; Vite/UnoCSS/TypeScript tooling; `dev`/`build`/`preview`/`typecheck` scripts |
| `index.html` | Shell with the `#splash` element and the `#root` mount point |
| `bun_env.cmd`, `bun_fetch.ps1` | Locate the project Bun, or download it; described in section 19 |
| `use_web.bat` | Wires the shared frontend into a fork's `web` directory; described in section 19 |

---

## 2. The two forks

### Byte-identical files (both trees)

`__init__.py`, `models/__init__.py`, `utils/__init__.py`,
`utils/security.py`, `utils/compositing.py`, `utils/torch_threads.py`,
`utils/stderr_progress.py`, `utils/model_integrity.py`, `utils/net_check.py`,
`utils/offline.py`, `utils/humantime.py`, `utils/logctx.py`,
`models/ultralytics_backend.py` and `requirements.txt`.

`diff` over these files is empty. That list is the whole set: everything else in
the two trees differs at least by an absolute import of its own package.

### Files that differ

| File | The difference |
|---|---|
| `main.py` | AMD-only gfx override, ROCm env defaults, MIOpen cache dirs, `torch` instead of `_torch`, HIP tuning, `_log_gpu_mem`, AMD no-GPU warning, GPU description line, app title |
| `models/base.py`, `models/detector_base.py` | `pass` in AMD abstract stubs, `...` in the CUDA fork |
| `models/sam3_backend.py` | Missing-`clip` hint names `install_amd.bat`; `_encode_image` also wraps the encode in `cudnn_disabled()` |
| `models/manager.py` | Three absolute imports (`net_check.is_connected`, `model_integrity.download_model_file`, `model_integrity.verify_model`) |
| `models/grounding_detector.py` | One absolute import (`net_check.is_connected` in the download path) |
| `models/yoloe_detector.py` | Three absolute imports (`net_check.is_connected`, `model_integrity.download_model_file`, `model_integrity.verify_model`) |
| `utils/device.py` | AMD adds `_detect_rocm()` and checks it before `_detect_nvidia()` |
| `run.cmd` | AMD adds the MIOpen env vars and cache dirs, the tunable-op cache, `OCL_SET_SVM_SIZE`, the HIPBLASLt/addmm switches, `SPECTRA_DETECTOR_FP16=1`, and runs `backend_amd_gpu.main:app` |

`models/sam3_backend.py` is worth being precise about: **both** forks call
`cudnn_disabled()` in `predict_text` and `predict`; only the AMD copy also wraps
`_encode_image` in it, because on ROCm the encode is the expensive MIOpen
kernel-selection step.

To mirror a change: copy it, swap `backend.` for `backend_amd_gpu.` in absolute
imports only, then re-run the checks in section 13 for both trees and confirm the
diff still shows only the rows above.

### Verified stack (installed in `backend_amd_gpu/.venv`)

`torch 2.13.0+rocm10.1.0a20260822`, `torchvision 0.28.0+rocm10.1.0a20260822`,
`transformers 4.57.6`,
`ultralytics 8.4.177`, `huggingface_hub 0.36.2`, `fastapi 0.143.1`,
`starlette 1.8.0`, `numpy 2.4.4`, `opencv-python-headless 5.0.0.93`,
`pillow 12.3.0`, `supervision 0.30.9`, `accelerate 1.15.0`, `timm 1.0.30`,
`einops 0.8.2`. Frontend: React 19.2.8, Vite 8.2.1, UnoCSS 66.7.5, jotai 2.20.2,
TypeScript 7.0.2, clsx 2.1.1.

The `backend/` tree has no venv in this checkout, so its installed versions are not
observable here; the import checks in section 13 are run with the AMD venv.

---

## 3. End-to-end shape

Three editing modes over one API:

| Mode | Model | Path through the code |
|---|---|---|
| Brush | SAM 2 / 2.1 | `Canvas` pointer capture -> `buildStrokePrompt` -> `useSession.predict` -> `/predict` -> `ModelManager.predict` -> `_replay_history` -> `UltralyticsBackend.predict` |
| Detect | Grounding DINO / MM-GDINO / Florence / YOLOE | `Toolbar` query -> `useDetection.detect` -> `/detect` -> `ModelManager.detect` -> detector `detect` -> boxes; then `segment-batch` -> SAM |
| SAM 3 | `sam3.pt` | `Sam3PromptBar` -> `useSam3.prompt` -> `/sam3-prompt` -> `ModelManager.sam3_predict` -> `SAM3Backend.predict_text` |

Brush subjects are server state (`ObjectState`), detections are client state
(`detectionsAtom` + `perDetectionMasksAtom`), SAM 3 prompts are server history with a
client mirror.

---

## 4. `backend/main.py`

### Module import side effects

Before importing `cv2`, `torch` or FastAPI, the module redirects HuggingFace, torch,
Ultralytics, XDG and temp locations into `backend/model_weights/`:

`HF_HOME`, `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE`,
`HF_MODULES_CACHE`, `HF_XET_CACHE`, `HF_HUB_DISABLE_PROGRESS_BARS=1`,
`HF_DATASETS_CACHE`, `SENTENCE_TRANSFORMERS_HOME`, `TORCH_HOME`, `TRITON_CACHE_DIR`,
`NUMBA_CACHE_DIR`, `ULTRALYTICS_HOME`, `YOLO_CONFIG_DIR`, `MPLCONFIGDIR`,
`XDG_CACHE_HOME`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `TMPDIR`, `TEMP`, `TMP`.

The same block exists in `run.cmd` and `start_amd.bat`. They must agree; this copy is
the safety net when the backend is started by hand. Thirteen of those directories are
created immediately.

`sys.path.insert(1, <repo root>)` makes the absolute `backend.*` imports work when the
module is loaded from inside the backend directory.

### `#_file_handler`, `#_buf_handler`, `#_ctx_filter`

`_file_handler` is a `RotatingFileHandler` on `backend/logs/backend.log`, level
`DEBUG`, `maxBytes = SPECTRA_LOG_MAX_MB * 1 MiB` (default 10), `backupCount =
SPECTRA_LOG_BACKUPS` (default 3), format
`%(asctime)s [%(levelname)s] rid=%(rid)s sid=%(sid)s %(name)s: %(message)s`.
The root logger is set to `DEBUG` and owns both handlers.

`_buf_handler` is the `_BufferHandler` feeding the 300-entry ring and SSE. Its level
is `SPECTRA_UI_LOG_LEVEL` (default `INFO`), resolved with `getattr(logging, ...)` and
falling back to `INFO` for an unknown name.

Both handlers carry `_ctx_filter`, because `rid`/`sid` are not record fields until it
stamps them; without it the format string raises.

### `#_BufferHandler.emit`

- A thread-local `in_emit` flag makes the handler drop a record emitted by another
  thread while one emit is in progress. The file handler still receives every record.
- `_clean_message` strips ANSI escapes, literal `[K` and the box-drawing characters,
  then trims; an empty message is dropped.
- The entry always has `level`, `message`, `ts` (`%H:%M:%S`, no date), `rid`, `sid`,
  and copies `model`, `detector`, `phase`, `dur_ms` when present.
- A `dl` dict is flattened into the entry (keys `level`, `message`, `ts` excluded) and
  the level is forced to `DOWNLOAD`.
- With no `dl`, three regexes try to pull `name`/`pct` out of text shaped
  `name: 42% complete`, `name: 42% -` or `name: 42%|`. This is how tqdm output
  captured by `StderrInterceptor` becomes a progress bar.
- Fan-out uses `loop.call_soon_threadsafe(q.put_nowait, entry)` because the ring is
  written from worker threads. A missing or closed loop stops at the ring; a full
  subscriber queue raises inside the loop callback, which is swallowed, so drops are
  silent.

### `#_get_main_loop`, `#_notify_model_status`, `#_broadcast_model_status`

`_main_loop` is captured in `lifespan`. `_notify_model_status(event)` marshals a status
broadcast onto that loop from any thread with `run_coroutine_threadsafe`; failures log
at `DEBUG`. `_broadcast_model_status` recomputes the full status list, serialises it
with the event name and pushes it to every queue, dropping on `QueueFull`.

### Image handling

| Symbol | Behaviour |
|---|---|
| `#MAX_IMAGE_PIXELS` | `4096 * 4096` - the working-pixel ceiling used by `_decode_image` and mask decode |
| `#MAX_UPLOAD_PIXELS` | `32_000_000` - the header ceiling, also assigned to `Image.MAX_IMAGE_PIXELS` |
| `#MAX_IMAGE_DIMENSION` | `8192` - longest side after fit; the header check allows up to 4x this (32768) so an oversized photo can be rejected cleanly instead of decoded |
| `#IMAGE_MAGIC_BYTES` | JPEG and PNG prefixes; WebP is checked separately as `RIFF....WEBP` |
| `#_validate_image_magic` | Requires at least 12 bytes, then one of the three signatures |
| `#_sanitize_filename` | NUL removed, separators and `..` become `_`, anything outside `\w.- ` becomes `_`, max 128 chars, fallback `upload` |
| `#_decode_image` | Pillow probe reads only the header and rejects over the ceiling; then `ImageOps.exif_transpose`, `_fit`, LANCZOS resize, and conversion to BGR (grayscale and RGBA handled); cv2 is only reached if Pillow raises |

`_fit` is `min(1, sqrt(MAX_IMAGE_PIXELS / (w*h)), MAX_IMAGE_DIMENSION / max(w, h))`.
EXIF transposition happens **before** resizing so the mask coordinates match the
pixels the browser shows.

### `#lifespan`

1. Captures the running loop for the log fan-out.
2. If CUDA is available and `SPECTRA_VRAM_HARD_CAP` parses, calls
   `set_per_process_memory_fraction(min(value, 0.95))` when the result is above zero.
3. CUDA-only (not HIP): sets `PYTORCH_CUDA_ALLOC_CONF` (with
   `expandable_segments:True`) if unset, and enables `cudnn.benchmark` with
   `benchmark_limit = 4`.
4. `detect_device()`, logs either "Running on CPU" or the GPU line.
5. Builds `ModelManager(device=...)`, attaches `_status_callback`, mints the token.
6. Starts a task that every 10 s calls `evict_idle_sessions` in a thread and logs the
   evicted count at `DEBUG`.
7. On shutdown: cancels the task, then `destroy_all_sessions()`.

### Middleware

Registered in this order: `TokenAuthMiddleware`, `RequestSizeLimitMiddleware`,
`RateLimitMiddleware`, `HostValidationMiddleware`, `SecurityHeadersMiddleware`,
`RequestLoggingMiddleware`, `CORSMiddleware`.

Starlette inserts each new middleware at position 0, so the effective outer-to-inner
order is the reverse:

```
ServerError -> CORS -> RequestLogging -> SecurityHeaders -> HostValidation
            -> RateLimit -> RequestSizeLimit -> TokenAuth -> Exception -> router
```

CORS allows `http://localhost:3000` and `http://127.0.0.1:3000`, credentials, the
methods `GET/POST/DELETE/OPTIONS` and the headers `Content-Type`, `X-Local-Token`.
Because CORS is outside the inner stack, a preflight `OPTIONS` never reaches host,
rate, size or token checks; it carries no body and no data.

### `#VRAM_OOM_DETAIL`, `#_handle_vram_oom`, `#_maybe_flush_cache`

Any `torch.OutOfMemoryError` that escapes a handler becomes HTTP 507 with the constant
`VRAM_OOM_DETAIL` text. The handler reads `mem_get_info()` when it can and logs free and
total MB under `phase=vram, reason=oom`.

`_maybe_flush_cache` calls `torch.cuda.empty_cache()` in a thread after predict,
segment-batch and SAM 3 prompt, but only when `SPECTRA_FLUSH_EVERY_REQUEST=1`.

### Request models

All validators run before the manager sees the request.

| Model | Fields and limits |
|---|---|
| `CreateSessionRequest` | `model_name` defaults to `sam2.1_b.pt`; must be in `ULTRALYTICS_MODELS` or be `sam3.pt` |
| `CreateSessionResponse` | `session_id`, `model_name` |
| `StrokeRequest` | `object_id` 0..1,000,000; `points` at most 10,000, each `[x, y]` with both coordinates in -10,000..10,000; `labels` at most 10,000, each 0 or 1; `bboxes` exactly 4 numbers in range with `x2>=x1` and `y2>=y1` |
| `PredictResponse` | `scores`, `object_masks` (object id -> base64 PNG), `object_history` (object id -> `{undo, redo, strokes, has_mask}`) |
| `ExportZipFileItem` | `path` 1..500; `mask_b64` non-empty |
| `ExportZipRequest` | `root` up to 200; `files` up to 1000; `include_whole`; `format` png/jpg/jpeg; `background_color` optional `[r,g,b]` 0..255; `feather_radius` 0..50 |
| `ExportImageRequest` | `files`, `format`, `background_color`, `feather_radius`, same rules |
| `DetectRequest` | `query` 1..500, stripped; `confidence` optional 0..1; `max_detections` 1..100, default 10; `use_yoloe_masks` |
| `LoadDetectorRequest` | `detector_name` |
| `SegmentBatchRequest` | 1..200 boxes, each exactly 4 non-negative numbers |
| `Sam3PromptRequest` | `text` non-blank, 1..500, stripped; `confidence` optional 0..1 |
| `Sam3PromptResponse` | `masks`, `scores`, `bboxes` |
| `Sam3RemoveInstanceRequest` | `prompt_index`, `instance_index`, both >= 0 |
| `Sam3SettingsRequest` | `keep_loaded` optional; `encode_dim` optional 384..1500 |

### Routes

Auth: every route requires `X-Local-Token` except the four exempt paths in
`TokenAuthMiddleware` and the release pattern (section 7).

| Route | Behaviour |
|---|---|
| `GET /health` | `{"status": "ok", "active_sessions": n}`; no auth |
| `GET /api/logs?limit=300` | Last `limit` entries of the ring, clamped to the ring length; returns `{entries, total}`; **auth required** |
| `GET /api/logs/stream` | SSE; per-IP cap 5, global cap 100, 1800 s lifetime, 30 s keepalive comment, queue depth 128; sets `Access-Control-Allow-Origin` only for the two dev origins |
| `GET /api/models/status` | `{"models": [...]}` from `_check_model_statuses` |
| `GET /api/models/stream` | SSE, queue depth 64, global subscriber cap shared with the log stream; sends immediately, then on events or every 3 s when the JSON changed |
| `GET /api/token` | `{"token": local_token}` |
| `POST /api/sessions` | `create_session` in a thread; `RuntimeError` becomes **429** |
| `GET /api/sessions/{id}` | Manager session info; 404 when missing |
| `DELETE /api/sessions/{id}` | `destroy_session`; always returns `{"status": "destroyed"}` |
| `POST /api/sessions/{id}/release` | Synchronous `destroy_session` on the event loop; no auth |
| `POST /api/sessions/{id}/image` | Size, extension/MIME, magic-byte, decode and ceiling checks; 404/413/400; returns `{image_id, width, height}` |
| `POST /api/sessions/{id}/predict` | `ValueError` -> 404 if the session is gone, else 400 with the backend's message |
| `POST /api/sessions/{id}/segment-batch` | 400 for `ValueError` (including a missing session), 507 on OOM, 500 otherwise; `{"masks": [b64 or null]}` |
| `POST /api/sessions/{id}/undo`, `/redo` | `?object_id=&steps=`; 404 when the session is gone, 409 when the subject has nothing to move |
| `POST /api/sessions/{id}/clear-object` | `?object_id=`; 404 when the session is gone |
| `POST /api/sessions/{id}/sam3-prompt` | 400 `ValueError`, 507 OOM, 500 otherwise |
| `POST /api/sessions/{id}/sam3-undo`, `/sam3-redo` | **404 for both** "no session" and "empty history" |
| `POST /api/sessions/{id}/sam3-remove-instance` | 400 invalid index, 404 no prompts, 507 OOM |
| `POST /api/settings/sam3` | `{keep_loaded, encode_dim}` echoed from the manager |
| `GET /api/detectors` | Manager detector list |
| `POST /api/detectors/load` | 400 unknown name, 507 OOM, 500 otherwise |
| `POST /api/sessions/{id}/detect` | 400 `ValueError` (including a missing session), 507 OOM, 500 otherwise |
| `POST /api/sessions/{id}/export-zip` | 404 no session, 400 no image or no files; `{data, format: "zip", count}` |
| `POST /api/sessions/{id}/export-image` | 404 when the session or image is missing, 400 no files; raw `image/png` or `image/jpeg` |

Two asymmetries are in the code and worth knowing before debugging a 400:

- `/predict` distinguishes a missing session (**404**) from every other `ValueError`
  (**400**), but `/detect` and `/segment-batch` map *every* `ValueError` to 400,
  including a missing session.
- `/api/sessions` maps every `RuntimeError` from `create_session` to **429**. That
  includes "Maximum sessions reached", but also an integrity failure or "no internet
  connection and model not found" raised by `get_or_load_model`.

### Export helpers

| Symbol | Behaviour |
|---|---|
| `#_decode_mask_b64` | Base64 -> Pillow -> `L`, rejects masks over `MAX_IMAGE_PIXELS`, threshold `>127`, DecompressionBomb/OSError become 400 |
| `#_decode_mask_resized` | Rejects `h*w` over `MAX_IMAGE_PIXELS`; nearest-neighbour resize to the session image, then `>127` |
| `#_safe_arcname` | Normalises backslashes, takes `basename(root)`, `normpath`s the item path, rejects absolute paths, leading `..`, embedded `/..` and any colon, then joins. Colons are rejected outright, so a legitimate timestamp in a filename fails |
| `#_masks_to_png_list` | Skips empty arrays; positional SAM 3/export list |
| `#_masks_positional_png_list` | Keeps position, emitting `None` for an empty mask |
| `#_masks_to_predict_response` | Maps `all_scores`, `object_masks`, `object_history` |

`#export_zip` always encodes each per-item file as **PNG**; `format` only reaches the
optional whole-image entry, which becomes JPEG only when the format is jpg/jpeg **and**
a background colour was given. `include_whole` ORs every mask into one composite and
names it `whole.<ext>`. The archive is built in a `BytesIO` and base64-encoded into the
JSON response, so response memory scales with the archive.

`#export_image` returns one image for the OR of all supplied masks, with the same
JPEG-needs-a-background rule.

---

## 5. `backend/models/base.py`, `detector_base.py`

`SegmentationBackend` declares `load_model(model_path)`, `unload_model()`,
`set_image(image)`, `predict(points, labels, bboxes, mask_input, multimask_output)`,
`reset_image()`, and the `is_loaded` / `has_image` properties. `predict` defaults
`multimask_output=True`; the manager always passes the argument explicitly, so the
default only matters to a direct caller.

`DetectorBackend` declares `load_model(model_path=None)`, `unload_model()`,
`detect(image, query, confidence=None, max_detections=10)` and `is_loaded`.
`Detection` is `(bbox, score, label, mask=None)`.

---

## 6. `backend/models/manager.py`

Module constants: `IDLE_TIMEOUT = 600`, `DETECTOR_IDLE_TIMEOUT = 300`,
`SAM3_IDLE_UNLOAD_TIMEOUT` (env `SAM3_IDLE_UNLOAD_TIMEOUT`, default 300),
`SAM3_ENCODE_DIM_MIN = 384`, `SAM3_ENCODE_DIM_MAX = ENCODE_DIM_MAX` (1500),
`MAX_STROKE_HISTORY` (`SPECTRA_MAX_HISTORY`, 200), `SAM3_MAX_HISTORY` (`SAM3_MAX_HISTORY`, 100),
`MAX_REPLAY_POINTS = 2048`, `MAX_OBJECTS_PER_SESSION` (`SPECTRA_MAX_OBJECTS`, 50),
`MAX_SESSIONS` (`SPECTRA_MAX_SESSIONS`, 100). Every env-backed one is read at import
time, so changing it needs a restart.

### Data model

| Type | Holds |
|---|---|
| `Session` | id, model name, backend, image id/dimensions/RGB/BGR, `objects`, SAM 3 prompt history and redo stack, last SAM 3 masks/scores/bboxes, `has_image`, `last_active`, a per-session `lock` |
| `ObjectState` | `stroke_history`, `redo_stack` (entry + optional snapshot), `last_mask`, `last_low_res_mask`, `mask_snapshots`, `mask_revision`, `png_cache` + `png_revision` |
| `MaskSnapshot` | `low_res` only - the decoder logits one stroke produced |
| `StrokeEntry` | `points`, `labels`, `bboxes` |
| `SAM3PromptEntry` | `text`, `confidence` |
| `SAM3Snapshot` | packed masks, scores, bboxes, mask shape |

`Session` and `ObjectState` use `slots=True`.

### Locks

| Lock | Guards |
|---|---|
| `self._lock` | session table, `_loaded_models`, `_detector`, `_current_*`, `_last_*_use`, `_keep_sam3_loaded`, `_sam3_encode_dim` |
| `session.lock` | that session's `objects`, each `ObjectState`, SAM 3 history/stacks |
| `backend._infer_lock` | inference on one model (`RLock` in `UltralyticsBackend`, `Lock` in `SAM3Backend`) |
| `_model_load_locks[name]` | one lock per model name, created on demand under `self._lock` |

### `#ModelManager.create_session`

`evict_idle_sessions()` runs first. Inside the lock the session id is inserted as
`None` **before** the backend is loaded, so concurrent creations cannot all pass the
`MAX_SESSIONS` check; the slot is replaced with the real `Session` after the load and
removed if the load raises. Every reader treats `None` as missing, which is what the
`# type: ignore[assignment]` marks.

### `#ModelManager.get_or_load_model`

1. Fast path inside the lock: same name, still registered, still loaded -> touch
   `_last_seg_use` and return.
2. Validate the name (`sam3.pt` or a key of `ULTRALYTICS_MODELS`).
3. If the file is missing: `_status_callback("downloading")`, `is_connected()`, then
   `download_model_file`. Offline raises a message that names the directory.
4. If it exists: `verify_model`; an **exception** is logged and loading continues, a
   **False** result raises "integrity check failed".
5. Take the per-name lock and re-check the fast path.
6. `SPECTRA_NO_CO_RESIDENCY`: unload the detector.
7. Unload the previously loaded model (logged as `reason=model-switch`).
8. Construct `SAM3Backend` (with `_encode_dim = _sam3_encode_dim`) or
   `UltralyticsBackend`.
9. Required headroom is `file_size_MB * 2 + 512`, or 4096 when the size cannot be read.
10. `load_model`, register, stamp `_last_seg_use`, notify `loaded`, log
    `Model <name> ready in <duration>` at `phase=residency`.

### `#ModelManager.destroy_session`

Pops the session. If it was the last one, it also pops and remembers the current model
and detector and clears `_current_*`. Outside the lock it resets the backend image when
the inference lock is free, clears image arrays and `objects`, then unloads the model
and detector it took. `destroy_all_sessions` iterates a snapshot of the id list.

### `#ModelManager.load_image`

Resets a previously encoded image, stores id/dimensions/BGR/RGB, clears `objects` and
the SAM 3 history/stacks/last-results, and encodes under `_ensure_encode_headroom`.

An encode failure is **caught**: `session.has_image` becomes `False`, the backend image
is reset, and the exception is logged. The upload route has already built its response
from the decoded array, so the client still receives 200 with the dimensions and no
hint that encoding failed. The next stroke calls `_ensure_model_loaded`, which re-encodes
from the stored BGR array; that retry either succeeds or raises out of `predict` (500,
or 507 for a `torch.OutOfMemoryError`). The clean 400 "No image loaded in session" is
only reachable if the backend still reports no encoded image after that retry.

### `#ModelManager._ensure_model_loaded`

- Loaded and (`has_image` or no image) -> nothing to do.
- Loaded but the image is not encoded -> re-encode under headroom.
- Not loaded -> `status("loading")`, encode headroom with `keep_backend=None`,
  `get_or_load_model`, swap `session.backend`, re-encode when the image is present.
  `status("loaded")` always fires in `finally`.

### `#ModelManager.predict`

Requires a live session and an encoded image; raises `ValueError` for both, plus the
object cap. When points or boxes arrive it appends a `StrokeEntry` and clears the redo
stack under `session.lock`. A stroke longer than `MAX_REPLAY_POINTS` is downsampled with
`np.linspace` over the indices (labels follow the same indices). History past
`MAX_STROKE_HISTORY` drops the oldest stroke **and** its snapshot together, so the two
lists stay aligned.

After `_compute_mask` it logs the decode duration, appends `MaskSnapshot(low_res)` for
the new stroke (or a `None` snapshot when the mask is empty) and stores the mask,
bumping `mask_revision`.

### `#ModelManager._brush_response`

Sets `object_masks` from `_object_masks_png`, `all_scores` to `[1.0] * len(object_masks)`
and `object_history` from `_object_history`. `PredictResponse.scores` is therefore a
list of ones for brush traffic; nothing in the UI reads it.

### `#ModelManager._object_masks_png`

Copies and sorts the subject list under `session.lock`, then encodes each subject
outside it, so a multi-megabyte PNG encode never blocks stroke appends. The list is a
snapshot: a mask computed during the encode appears in the next response.

`_object_mask_png` caches against `mask_revision` and encodes `np.asarray(mask)[0]` -
the first plane of the model's `(1, H, W)` mask.

### Undo and redo

`#ModelManager._restore_object_mask` recomputes a subject's mask for the history as it
stands after the move. A stroke depends only on its own prompt and on the low-res mask
of the stroke before it, and that value is `mask_snapshots[i-1].low_res`; the snapshot
list is index-aligned with the history. So a restore costs **one decoder pass** for the
stroke that is now last, or none when the history emptied.

`undo` and `redo` clamp `steps` to `min(max(steps, 1), len(stack))`, move entries, trim
the redo stack to `MAX_STROKE_HISTORY`, reload the model if it was evicted, then call
the restore. `redo` appends a placeholder `MaskSnapshot(low_res=None)` when the saved
snapshot is missing, so the lists cannot drift; the restore overwrites it with the
low-res mask it just produced.

When the two lists are not the same length, `_restore_object_mask` falls back to
`#_replay_history(full_replay=True)`, which resets the subject and re-applies every
stroke in order. Full replay hands `mask_input` to a bbox stroke; the incremental path
passes `None` for bbox strokes, which is how the stroke was originally computed. The
single-step restore follows the incremental semantics so it reproduces the mask the
user was looking at.

Snapshot cost: each `MaskSnapshot` holds the SAM 2 decoder's 256x256 float32 low-res
plane, about 256 KB per stroke, held for the session's lifetime. `SPECTRA_MAX_HISTORY`
(200) bounds a subject near 50 MB; a stroke lives in the history or the redo stack,
never both, so the two share that budget. SAM 3 returns no low-res logits
(`low_res_masks: None`), so its snapshots - when the brush path is used on a SAM 3
session, which cannot happen through the UI - would hold nothing.

### `#ModelManager.sam3_predict`

Requires a SAM 3 session and an encoded image. `predict_text` runs, the result is
appended as `(SAM3PromptEntry, SAM3Snapshot)`, the redo stack is cleared, the history is
trimmed to `SAM3_MAX_HISTORY`, and the last masks/scores/bboxes are stored for the
undo/redo responses.

`_pack_masks` calls `np.packbits` only for boolean arrays; `_unpack_masks` unpacks only
when a three-dimensional shape was recorded. Packing is an 8x reduction, which is what
keeps a 100-prompt history of 30 instance masks affordable.

`sam3_undo` pops into the redo stack and restores the previous snapshot (unpacking it),
or clears the last result when the history is empty. `sam3_redo` is symmetric.
`sam3_remove_instance` deletes one instance from masks, scores and bboxes; an emptied
prompt is removed and the redo stack cleared; otherwise the prompt is replaced by a new
snapshot. Later `prompt_index` values shift down after a removal, and the client's
indices are not told.

### `#ModelManager.set_sam3_settings`

Stores `keep_loaded`, validates `encode_dim` (384..1500) and stores it. When a SAM 3
backend is loaded and the dimension changed, it calls
`backend.set_encoding_resolution` and re-encodes every session that shares that
backend. The returned `encode_dim` is read from the loaded backend when possible, so the
caller sees the value actually in force.

### `#ModelManager._replay_history` and `#ModelManager._apply_stroke`

`_replay_history` ensures the image is encoded for the current resolution, copies the
history under `session.lock`, then:

- `full_replay=True`: clears the stored mask, replays every stroke chaining
  `last_low_res_mask`, bumps the revision and logs the stroke and label counts.
- otherwise: applies only the newest stroke, with `mask_input = last_low_res_mask`
  unless this is the first prompt or the stroke is a bbox, then stores the result.

`_apply_stroke` is the first-stroke refinement: when the first stroke of a subject
carries both positive and negative points, it runs the positives alone, then re-runs the
full prompt with the coarse low-res mask as `mask_input`. `multimask_output` is true only
for a first stroke with no box and at most one point.

### Residency

| Symbol | Behaviour |
|---|---|
| `#_can_unload` | Returns False when the object is not loaded; tries to acquire `_infer_lock` without blocking and releases it immediately, so a non-blocking acquire is the whole safety check |
| `#_safe_unload` | `_can_unload` then `unload_model()`; logs "unload skipped" at `DEBUG` and exceptions at `ERROR` |
| `#_evict_loaded_models` | No-op without CUDA or on `device == "cpu"`. Under the lock it unloads the detector and every segmentation model except `keep_backend`, leaving anything with inference in flight registered; outside the lock it unloads and then `empty_cache` + `ipc_collect` |
| `#_ensure_vram_headroom` | Reads `mem_get_info`, applies `SPECTRA_VRAM_FRACTION` (capped at 0.95, default 1.0), compares `(total - free) + required` against the budget and evicts with `keep_backend` when over. A fraction of zero disables the check |
| `#_ensure_encode_headroom` | `_evict_loaded_models(keep_backend)` - the encode path wants room and keeps the model being encoded |
| `#_ensure_full_image_encoded` | Re-encodes when `backend._src_shape` differs from the session image, which catches a model swap or a SAM 3 resolution change |

`#evict_idle_sessions` destroys sessions idle beyond `IDLE_TIMEOUT`; unloads a detector
idle beyond `DETECTOR_IDLE_TIMEOUT`; and unloads SAM 3 only when it has been idle beyond
`IDLE_TIMEOUT` (when `keep_loaded`) or `SAM3_IDLE_UNLOAD_TIMEOUT` **and** no sessions
exist at all. It returns the number of sessions destroyed.

### Detection

`#load_detector` first returns early when the requested detector is already current.
It then pops the segmentation model when `SPECTRA_NO_CO_RESIDENCY=1`, notes
`SPECTRA_KEEP_SEG_ON_DETECT`, unloads the previous detector through `_safe_unload`, and
**validates the name after that teardown**. On an unknown name it puts the old detector
object back into `_detector` (it is already unloaded) and clears `_current_detector`
only when that object still reports `is_loaded` - after a successful unload it is
False, so the old name is **left in place**, and the detector list keeps reporting it as
loaded while nothing is. Then it raises `ValueError`. The new detector is built into a
local, sized by tier
(`tiny`/`small` 1024 MB, `medium` 1536, `large` 2560, otherwise 2048), loaded, and only
committed to the registry on success - so a failed *load* leaves the previous name
unselectable but the failed one absent, and a failed *name* leaves `_detector` pointing
at an unloaded object.

`#detect` requires an image and a loaded detector (`_ensure_detector_loaded` reloads
`_last_detector_name`). Unless `SPECTRA_KEEP_SEG_ON_DETECT=1` it pops the segmentation
model when the inference lock is free, unloads it and calls `release_gpu_memory`.
Detections are converted to `{bbox, score, label}` and, when `use_yoloe_masks` is set
and the detector produced a mask, `mask` as base64 PNG. It ends with
`release_gpu_memory()` and an `INFO` line `Detected '<q>' - N found, M kept (...)`
that carries no `phase`.

`#list_detectors` and `_check_model_statuses` both report `loaded` from
`_current_detector`, not from the backend's `is_loaded`. The idle sweep clears both
fields, so it stays consistent; the stale case is the failed unknown-name load above.

---

## 7. `backend/utils/security.py`

| Symbol | Behaviour |
|---|---|
| `#ALLOWED_HOSTS`, `#BACKEND_PORT` | `{127.0.0.1, localhost}`, 8000 |
| `#MAX_UPLOAD_BYTES` | 50 MiB, shared by the size limit and the upload route |
| `#RATE_LIMIT_PER_IP`, `#RATE_LIMIT_WINDOW_IP` | 300 requests / 60 s |
| `#get_or_create_token` | `secrets.token_urlsafe(32)`; despite the name it mints a **new** token on every call and is safe only because `lifespan` calls it once |
| `#HostValidationMiddleware` | Host name must be in `ALLOWED_HOSTS`; when a colon is present the trailing part must be digits and equal 8000. A `Host` with no port passes |
| `#TokenAuthMiddleware` | Exempt: `/health`, `/api/token`, `/api/logs/stream`, `/api/models/stream`, and `^/api/sessions/[^/]+/release$`. Everything else, including `/api/logs`, needs `X-Local-Token`. Comparison is `hmac.compare_digest` over bytes (header latin-1, token utf-8) so a non-ASCII header yields 401 instead of a `TypeError` |
| `#RateLimitMiddleware` | Per-IP deque of timestamps, pruned on each call, stale IPs dropped every 120 s; over the limit -> 429 |
| `#SecurityHeadersMiddleware` | Adds `x-content-type-options: nosniff` and `referrer-policy: no-referrer` with `setdefault` |
| `#RequestSizeLimitMiddleware` | With a `Content-Length`, rejects over 50 MiB and otherwise passes the body through untouched; without one, buffers the stream, rejects past 50 MiB, and stores the body on `request._body` so the endpoint can still read it |
| `#RequestLoggingMiddleware` | Assigns `rid`, extracts the session id from the path, times the request, clears the context, then logs at a level chosen by the ladder below |

Request log level: a path in `_DEBUG_PATHS` (`/health`, `/api/token`, `/api/detectors`,
`/api/logs`, `/api/logs/stream`, `/api/models/status`, `/api/models/stream`) is always
`DEBUG`; then `>=500` -> `ERROR`, `>=400` -> `WARNING`, a path ending in one of
`_ALWAYS_INFO` or a duration over 1000 ms -> `INFO`, everything else `DEBUG`. The check
is `path.endswith(tuple)`, so `/api/sessions/x/predict` matches `/predict`.

---

## 8. `backend/utils/` (remaining modules)

### `compositing.py`

| Symbol | Behaviour |
|---|---|
| `#mask_to_png_b64` | `cv2.imencode(".png", _to_mask_u8(mask))`, base64; raises `ValueError` when the encoder returns a falsy result |
| `#_is_unit_scale` | Ignores non-finite values, returns True when the remainder is inside `[0, 1]`, and treats an all-non-finite array as unit scale. Comparing in the input dtype avoids a float64 copy of a full-resolution mask |
| `#_to_mask_f32` | bool -> float32 0/1; float unit-scale -> cast; other float -> `nan_to_num(nan=0, posinf=255, neginf=0)`, clip 0..255, `/255`; integer -> clip in its own dtype, cast to float32, `/255` (every integer in 0..255 is exact in float32, so no float64 plane is needed) |
| `#_to_mask_u8` | bool -> `*255` directly, skipping the float plane; anything else goes through `_to_mask_f32` and is scaled and clipped |
| `#feather_mask_edge` | Pads with zeros by the radius before `GaussianBlur`, then crops back, so the border does not darken. Returns float 0..1; radius < 1 returns the mask unchanged |
| `#composite_rgba` | Straight-alpha RGBA (not premultiplied), which is what PNG stores |
| `#composite_background` | Straight "over" onto a solid colour or a supplied backdrop image |
| `#composite_and_encode` | Takes the opaque-RGB path when **either** `background_color` or `bg_image` is set, and emits JPEG (quality 95) only when the format is jpg/jpeg **and** a background exists. Otherwise PNG: RGBA without a backdrop, RGB with one |

### `torch_threads.py`

`MAX_THREADS` is `SPECTRA_MAX_THREADS` (default 32). `#resolve_threads(name)` reads the
variable, clamps to 1..`MAX_THREADS`, warns on a non-numeric value and falls back to
`min(cpu_count(), MAX_THREADS)`.

`#cpu_threads(n)` saves `torch.get_num_threads()`, sets the value and restores it. The
setting is process-wide and not atomic, so overlapping contexts can restore a value
nobody asked for.

`#cudnn_disabled()` returns immediately when there is no cuDNN backend or CUDA is
unavailable; otherwise it takes the module-level `_cudnn_toggle_lock` for the whole
block, so every cuDNN-disabled inference in the process is serialised, across both SAM
backends.

### `stderr_progress.py`

`StderrInterceptor` is installed directly on `sys.stderr` by the SAM 2, SAM 3, YOLOE and
both detector loaders, so it implements enough of the text-stream protocol to survive
whatever a library probes:

| Member | Behaviour |
|---|---|
| `write` | Decodes bytes as utf-8/replace, buffers, drains on newline or carriage return, returns `len(data)` |
| `writelines` | Calls `write` per line |
| `read`, `readline` | Return `""` |
| `readable`, `isatty`, `seekable` | `False` (`isatty() == False` makes libraries skip ANSI/progress rendering) |
| `encoding`, `errors` | `utf-8`, `replace` |
| `fileno` | Raises `OSError` |
| `seek`, `tell` | `0`, `0` |
| `flush` | Handles whatever is buffered and clears it |

There is no `close()`; `flush()` is the hook libraries call. The buffer is cleared past
4096 characters. `_handle` looks for `(\d+)%`; a line without one is logged at `DEBUG`,
and a percentage line becomes `"<model>: <desc> <pct>% complete"` (also `DEBUG`-free) at
most every 5 percentage points per description, which the download regex in `main.py`
then turns into a progress entry.

### `model_integrity.py`

`PINNED_MODEL_URLS` and `PUBLISHED_DIGESTS` have the same 18 keys: 8 SAM 2 checkpoints
(four `sam2.1_*`, four `sam2_*`), 2 MobileCLIP encoders, 7 YOLOE checkpoints and
`sam3.pt`. Seventeen URLs point at the
`ultralytics/assets` release `v8.4.0`; `sam3.pt` points at
`https://huggingface.co/bodhicitta/sam3`, a community mirror whose digest cannot be
confirmed from any official source.

`#verify_model` compares against the published digest when there is one. Otherwise it
falls back to `model_weights/.hashes.json` (trust on first use): the first sighting
records the hash, later ones must match. Because every current name has a published
digest, the TOFU branch is unreachable today; if a name loses its digest, a
hand-dropped file becomes the trust anchor and any real file is then rejected.
`_save_hash_db` writes without a lock or an atomic rename.

`#ensure_model_file(name, dir)` returns the absolute path when the file exists,
otherwise checks connectivity and downloads. It raises with the directory name when the
machine is offline and raises "No pinned URL" for a name that has no entry. It does not
verify an existing file; the manager does that.

`#download_model_file` takes a per-name lock, re-verifies a file that already exists,
downloads into `<name>.part` in 64 KiB chunks, logs a progress entry every 10 percentage
points with `done_bytes`, `total_bytes` and `eta`, logs `verify` then `done`, verifies
the part file and only then removes the old file and renames. There is no resume, no
retry, no disk-space check and no `fsync`.

### `offline.py`

`#_apply(offline)` sets or removes `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE`, and also
patches `huggingface_hub.constants.HF_HUB_OFFLINE` and
`transformers.utils.hub._is_offline_mode`, because both libraries read those attributes
at call time rather than re-reading the environment. `#is_offline` reads the hub
constant first, then the environment.

`#offline_guard(offline)` records the previous state under `_lock`, applies the new one,
and restores it in `finally` under the same lock, so overlapping guards cannot restore a
stale value. Cached loads run under `offline_guard(True)`; download attempts run under
`offline_guard(False)` for that call only.

### `net_check.py`

`#is_connected(timeout=1.5)` sends a `HEAD` to `https://huggingface.co` and then
`https://github.com` and returns True on the first success. Those are not the hosts the
checkpoints come from (`github.com/ultralytics/assets`, the HF CDN), so a proxy that
allows one and blocks the other produces a misleading offline verdict.

### `humantime.py`, `logctx.py`

`#format_duration` -> `"412ms"`, `"1.4s"`, `"42s"`, `"3m 07s"`.

`logctx` keeps `rid` and `sid` in `contextvars`; `new_request()` returns 8 hex
characters, `set_session`/`clear_request` set and reset them, and the accessors return
`"-"` when unset. `asyncio.to_thread` copies the context, so worker threads inherit
both.

---

## 9. `backend/models/ultralytics_backend.py`

`ULTRALYTICS_MODELS` holds the eight SAM 2 / 2.1 keys; `MODEL_METADATA` gives each a
`tier` and a `perf` string; `DENYLIST` blocks `FastSAM-s.pt`, `FastSAM-x.pt`,
`sam3_b.pt`, `sam3_l.pt`. `_MODEL_LOAD_LOCK` serialises loads; `_DEFAULT_THREADS` is
`resolve_threads("SAM2_THREADS")`, read at import.

| Symbol | Behaviour |
|---|---|
| `#UltralyticsBackend.load_model` | Rejects denied/unknown names, sets `ULTRALYTICS_HOME`/`YOLO_CONFIG_DIR`, resolves the file with `ensure_model_file`, then under the module lock patches `torch.load` to default `weights_only=True` (the guard against a malicious checkpoint executing code), installs a `StderrInterceptor` on `sys.stderr`, sets `SETTINGS["weights_dir"]` and builds a `SAM2Predictor` with `imgsz=1024` and `quantize` from `SAM2_QUANTIZE` (default 32 on CPU, 16 otherwise). A device failure retries once on CPU. Both swaps are restored in `finally` |
| `#UltralyticsBackend._model_dtype` | `predictor.torch_dtype`, else the first parameter's dtype, else float32; every prompt tensor is built with it |
| `#UltralyticsBackend._encode_image` | Under the inference lock: restore the encoder if offloaded, `empty_cache`, `imgsz=[1024,1024]`, `set_image` inside `cpu_threads` + `cudnn_disabled` + `inference_mode`, cache the dense positional encoding, warm the decoder, synchronise and re-offload |
| `#UltralyticsBackend._warm_decoder` | GPU only: one decoder pass with a centre point, errors swallowed at `DEBUG` |
| `#UltralyticsBackend._offload_encoder`, `#_restore_encoder` | Implement `SAM2_OFFLOAD_ENCODER=1` - the image encoder parks on CPU between images and comes back before the next encode |
| `#UltralyticsBackend.set_image` | Records `_src_shape`, encodes, and on `OutOfMemoryError` or a CUDA "out of memory" `RuntimeError` falls back to `_fallback_cpu_encoder_reencode`, then to `_fallback_full_cpu` |
| `#_cpu_encoder_reencode` | Moves the model, mean and std to CPU under the inference lock, encodes, restores them, moves the features and dense PE back to the device, then offloads and warms. The decoder never leaves the GPU |
| `#_fallback_full_cpu` | Unloads, sets `_device = "cpu"`, reloads the same checkpoint and re-encodes - the whole model, not just the encoder |
| `#UltralyticsBackend.predict` | Scales box corners and points by `r = min(1024/h, 1024/w)`, a box becomes two corner points labelled 2 and 3, `mask_input` is resized to the prompt encoder's `mask_input_size`, prompt encoder and mask decoder run under the inference lock with `inference_mode`, `cudnn_disabled` and thread limits. Output: `masks` `(N, H, W)` bool from the positive logits, `scores`, and `low_res_masks` = the **uncropped 256x256 logits** |
| `#UltralyticsBackend.predict_batch` | Batched decode, falling back to one call per box at `DEBUG` on any exception |
| `#_predict_batch_batched` | Stacks every box into one `(b, 2, 2)` prompt with labels 2/3, expands the image embedding `b` times, decodes once, crops and resizes each mask |

The mask crop is `pm[:, :round(h*r)/4, :round(w*r)/4]`, bilinearly resized to the source
size and thresholded at 0. The single-request contract is that the returned `masks` are
already at image resolution while `low_res_masks` stay in the decoder's own space, which
is what the next stroke chains on.

`unload_model` takes the inference lock, resets the image, drops the predictor and calls
`release_gpu_memory`. `reset_image` clears the predictor image, `_image_pe`, `_image_set`
and `_src_shape`.

---

## 10. `backend/models/sam3_backend.py`

`DEFAULT_ENCODE_DIM = 1024`, `ENCODE_DIM_MAX = 1500`, `MAX_ASPECT_RATIO = 1.6`,
`SAM3_MODEL = "sam3.pt"`, `SAM3_MODEL_DISPLAY = "SAM 3"`. `_SAM3_OVERRIDES` is
`task=segment`, `mode=predict`, `imgsz=1024`, `conf=0.25`, `iou=0.7`,
`agnostic_nms=False`; `_DEFAULT_THREADS` is `resolve_threads("SAM3_THREADS")`.

| Symbol | Behaviour |
|---|---|
| `#SAM3Backend.load_model` | Only accepts `sam3.pt`; requires the `clip` package with a fork-specific install hint; `ensure_model_file`; under the module lock the same `torch.load` and stderr swaps as SAM 2; builds the predictor inside `cpu_threads`. A failure falls back to CPU **unless** it is a `ModuleNotFoundError` or mentions `clip`, so a broken checkpoint still gets treated as "GPU unavailable" once |
| `#SAM3Backend._build_predictor` | `SAM3SemanticPredictor` with the overrides plus model/device/`quantize` (`SAM3_QUANTIZE`, default 32 on CPU and 16 otherwise), `setup_model`, then forces `imgsz = [encode_dim, encode_dim]`. `SAM3_CHANNELS_LAST=1` and `SAM3_COMPILE=1` are applied best-effort and keep the model on failure |
| `#SAM3Backend._prepare_input` | Shrinks so the longest side is at most `encode_dim`, records `_orig_shape`, `_content_shape`, `_pad_scale`; when the aspect ratio exceeds 1.6 it pads to a square with 114 grey and records `_pad_top`/`_pad_left`. A zero side after scaling raises `ValueError` |
| `#SAM3Backend.set_encoding_resolution` | Validates 384..1500, resets the predictor image, clears the cached encode and stores the new dimension; a change to the same value with an encoded image is a no-op |
| `#SAM3Backend.set_image` | Encodes; OOM (or a CUDA "out of memory") falls back to CPU by unloading, switching `_device`, reloading and re-encoding |
| `#SAM3Backend.predict_text` | Reads the cached features and calls `inference_features(features, self._proc_shape or self._src_shape, text=[text])` inside `cpu_threads` (1 thread on GPU), `cudnn_disabled()` (both forks) and `inference_mode()`. Without the inference mode every prompt built a live autograd graph over the prompt encoder, decoder and full-resolution masks |
| `#SAM3Backend.predict` | The visual-prompt path: point prompts raise `NotImplementedError`; boxes/labels go to `inference_features` |
| `#SAM3Backend._restore_outputs` | Crops the content band out of the padded output, interpolates to the original size and thresholds at 0.5; boxes are de-padded, divided by `_pad_scale` and clamped |
| `#SAM3Backend._format_result` | Scores come from box column 4 (zeros when there are no boxes), `bboxes` are the first four columns, `low_res_masks` is always `None`; an optional confidence filters by `score >= confidence` |

`reset_image` clears every shape/pad field as well as the predictor image. `is_loaded`
and `has_image` are the predictor's presence and `_image_set`.

---

## 11. `backend/models/grounding_detector.py`

`GROUNDING_DETECTORS` (7): `grounding-dino-tiny`, `grounding-dino-base`,
`mm-gdino-base-all`, `mm-gdino-large-all`, `florence-2-base`, `florence-2-large`,
`cogflorence-2.2-large`. `DETECTOR_METADATA` carries `tier` and `perf` for each;
`HF_MODEL_IDS` maps each to its repo id; `TRUSTED_HF_IDS` is that value set;
`_FLORENCE_MODELS` is the last three.

### Download state and the snapshot fetch

`#hf_weights_downloaded(hf_id)` walks `model_weights/hf_cache/hub/models--<id>/snapshots`
and reports True when any file with a weights suffix (`.safetensors`, `.bin`, `.pt`,
`.pth`, `.onnx` for the regex, the first three plus `.pth` for the size check) is at
least 50 MB.

`#_snapshot_fetch` lists the repo with `HfApi().model_info(files_metadata=True)`, picks
the largest weight file as the main file, and downloads the main file in a one-thread
pool while up to four threads fetch the rest. It replaces `tqdm` in
`huggingface_hub.file_download` and `huggingface_hub.utils.tqdm` with `_ByteBar`, which
emits the main file's byte progress at most every 0.4 s and a `done` entry on close,
then restores both. When the file listing fails it falls back to a plain
`snapshot_download`; when anything raises it returns False and the caller falls back to
a direct `from_pretrained`.

`#_emit` logs `"<name>: <pct>% - files <done>/<total>"` with a `dl` payload carrying
`name`, `phase="download"`, `pct`, `file`, `done_bytes`, `total_bytes`, `files_done`,
`files_total`.

### Florence handling

`#_is_trusted_local_path` accepts a snapshot directory only when it is under
`model_weights/hf_cache/hub`, has at least three path parts, its parent is `snapshots`
or `.no_exist`, and the repo directory is one of the trusted ids. This is what allows
`trust_remote_code` for a cached snapshot path.

`#_normalize_auto_map` drops the repo prefix from a value like
`microsoft/Florence-2-large--processing_florence2.Florence2Processor`, because
transformers would fetch that module from the other repo instead of using the file this
snapshot already ships.

`#_florence_config_paths` collects `config.json` and `preprocessor_config.json` from
`scan_cache_dir()` revisions. It uses the default cache scan, which is the redirected
`HF_HOME` at runtime.

`#_clean_florence_config` rewrites those files in place: it normalises `auto_map` and
removes the `AutoModelForZeroShotObjectDetection` entry, then writes the JSON back. **The
HuggingFace cache is therefore not read-only for Florence repos** - this is the one place
the app modifies it.

`#_load_florence_model` tries `attn_implementation="sdpa"` then `"eager"`, inside
`offline_guard(local_only)`, with `trust_remote_code`, `low_cpu_mem_usage` and the chosen
dtype. A `transformers` build without sdpa support is remembered in
`_FLORENCE_SDPA_BROKEN` so the next load skips it.

`#_florence_language_generation_config` builds a `GenerationConfig` from the inner
language model's config and overlays the outer config's diff. The inner config is the
base because it carries `decoder_start_token_id` and `bos_token_id`; the outer one has
`num_beams`/`early_stopping` but none of the special-token ids, and copying it wholesale
leaves `generate()` raising.

`#_enable_florence_generation` is what keeps Florence-2 and CogFlorence loadable on
`transformers >= 4.50`. Those releases dropped `GenerationMixin` from `PreTrainedModel`,
but the repos' remote modelling code delegates `generate()` from the outer model to
`self.language_model.generate(...)` and relies on the inherited method. The function
re-bases the loaded model and its language model onto subclasses that also inherit
`GenerationMixin`, and gives the language model a generation config when it has none. It
is idempotent and is called again at the top of `_detect_florence`, so an instance loaded
before the patch still gets fixed.

`#_load_florence_detector` refuses an untrusted id, chooses fp32 unless the device is a
GPU and `SPECTRA_DETECTOR_FP16=1`, loads the processor under a stderr interceptor and
`offline_guard`, loads the model with a `get_imports` patch that strips `flash_attn` from
the Florence imports (retried without the patch on failure), falls back to CPU when
`.to(device)` fails and forces fp32 there, then pre-warms at 1024x1024 with `<OD>` and a
single generated token.

`#_load_grounding_dino` loads processor and model under `offline_guard`, applies the same
dtype and CPU rules, and pre-warms at 800x800 with the text `[["test"]]`, with autocast
enabled only when the model is already fp16.

### `#GroundingDetector.load_model`

Sets `HF_HOME`, repairs a cache where `.no_exist` exists but `snapshots` does not by
renaming, then tries in order: the newest snapshot directory by mtime; `_snapshot_fetch`
plus that directory; a direct `from_pretrained` with `local_files_only=False`. Before any
download it requires `is_connected()`. Failures are logged with `exc_info` and re-raised.

### Detection

`#_detect_florence` joins comma-separated labels into
`<CAPTION_TO_PHRASE_GROUNDING>label1, label2`, uses `<OD>` when the query is empty, and
generates with `max_new_tokens=1024`, `num_beams=3`, no sampling and no cache. It
post-processes with the task tag and the PIL image size, drops degenerate boxes, assigns
`score = 1.0` to every box (Florence reports no per-box confidence), sorts and runs NMS.

`#_detect_grounding_dino` passes `text=[[label, ...]]` and
`post_process_grounded_object_detection` with `threshold=confidence` only when the caller
supplied one. In `transformers 4.57.6` the default is **0.25**, and the frontend never
sends `confidence`, so a detection run through the UI always uses that default. Labels
come from `text_labels` (falling back to `labels`).

`#_nms` uses `supervision` when importable (`with_nms(threshold=0.5,
class_agnostic=True)`) and otherwise a greedy suppression that stops as soon as
`max_detections` boxes are kept, so the kept set differs depending on whether
`supervision` imported.

`unload_model` drops processor and model and releases GPU memory; `is_loaded` is
`_model is not None`.

---

## 12. `backend/models/yoloe_detector.py`, `requirements.txt`

`YOLOE_MODELS` keys have no extension (`yoloe-26l-seg` ...); `_YOLOE_WEIGHTS` maps each to
its `.pt` file. `YOLOE_METADATA` carries `tier` and `perf`.

`#YOLEDetector.load_model`: validates the name, picks `mobileclip2_b.ts` for
`yoloe-26*` and `mobileclip_blt.ts` otherwise, downloads any missing weight or encoder
after a connectivity check, sets `ULTRALYTICS_HOME` and `SETTINGS["weights_dir"]`, and
permanently replaces `ultralytics.nn.text_model.build_text_model` so the two MobileCLIP
variants resolve to the local files. It installs a stderr interceptor and changes the
working directory to the weights folder for the `YOLOE(...)` call only - both are
restored in `finally`. The checkpoint is digest-verified before the load, and
`get_text_pe` is wrapped to pass `cache_clip_model=True`.

`#YOLEDetector.detect` -> `#_detect_standard`: split the query on commas, `set_classes`,
`predict(conf=confidence if given else 0.05)`, then read boxes and classes; the label
comes from `result.names` first and falls back to the requested class list. A detector
mask, when present, is resized to the image with `INTER_LINEAR` and thresholded at 0.5.
Results are sorted by score and capped.

`self._device` is stored but never passed to `YOLOE(...)` or `predict`, so YOLOE runs on
whatever device Ultralytics selects; the manager's device choice does not apply to it.
It also reads no environment variables and never uses `inference_mode`.

Both `requirements.txt` files declare `fastapi`, `uvicorn[standard]`, `starlette`,
`h11`, `python-multipart`, `pillow`, `numpy`, `opencv-python-headless`, `ultralytics`,
`transformers` (`>=4.46,<5.0`), `huggingface_hub[hf_xet]`, `accelerate`, `timm`,
`einops` and `supervision`.

- The Starlette, `h11`, `python-multipart`, Pillow and OpenCV floors are security
  minimums, and each one maps to a published advisory: `starlette>=0.47.2` for
  CVE-2025-54121 / CVE-2024-47874 (multipart DoS), `python-multipart>=0.0.18` for
  CVE-2024-53981 (CPU DoS), `h11>=0.16.0` for CVE-2025-43859 (request smuggling),
  `pillow>=12.3.0` for CVE-2026-55380 / CVE-2026-42309 (decompression-bomb bypass and
  overflow), `opencv-python-headless>=4.12.0` for CVE-2025-53644 (JPEG2000 heap write).
- `ultralytics>=8.3.239` matters because SAM 3 needs 8.3.237 or newer and 8.3.239 is
  the first release that auto-installs the correct CLIP package; PyPI's `clip` package
  breaks it with `SimpleTokenizer' object is not callable`.
- `transformers` is the only package with an upper bound (`<5.0`); everything else is a
  floor.
- torch is **not** in either file. The installers install it explicitly (section 14),
  even though the backend catches `torch.OutOfMemoryError` at import time.
- `accelerate` backs transformers' `low_cpu_mem_usage`; `timm` and `einops` are pulled in
  by the Florence-2 DaViT remote code; `supervision` has exactly one call site,
  `GroundingDetector._nms`.

---

## 13. `backend/utils/device.py` and the AMD-only modules

### `backend/utils/device.py`

`MODEL_WEIGHTS_DIR` is `<backend>/model_weights`. `#_detect_nvidia` requires
`torch.cuda.is_available()` **and** `torch.version.cuda`; `torch.version.cuda` is `None`
on a ROCm build, where `torch.cuda` points at the HIP device, so an AMD card is never
reported as NVIDIA. `#detect_device` returns `cpu` when `SPECTRA_FORCE_CPU=1`, otherwise
`cuda:0` for NVIDIA with a description, otherwise CPU with "no acceleration".
`#release_gpu_memory` synchronises, calls `empty_cache` and `ipc_collect` (each guarded),
then `gc.collect()`.

### `backend_amd_gpu/utils/device.py`

Adds `#_detect_rocm` (CUDA available **and** `torch.version.hip` set), checks ROCm first,
then NVIDIA, then CPU. `release_gpu_memory` is identical.

### `backend_amd_gpu/amd_gfx.py`

`#gfx_override_value` maps `gfx101*` to `10.1.0`, `gfx103*` to `10.3.0` and `gfx110*` to
`11.0.0`; anything else returns `None`, which leaves `HSA_OVERRIDE_GFX_VERSION` unset.
The module has no imports, so it works both as a package module and as a sibling module
imported by `detect_amd_gpu.py` before torch can be imported.

### `backend_amd_gpu/detect_amd_gpu.py`

Runs as a script, not a package: `sys.path` is bootstrapped and
`from amd_gfx import gfx_override_value` follows it, which is what `# noqa: E402` covers.

It queries `Get-CimInstance Win32_VideoController` through PowerShell, falls back to
`wmic`, and keeps names containing `AMD` or `Radeon`. Each GPU is matched by
`DEV_xxxx` from `PNPDeviceID` against `PCI_DEV_TO_GFX` (gfx target, marketing name,
supported) and then by name patterns. Candidates are scored (2 for a PCI hit, 1 for a
retail name) and the best is returned.

`PCI_DEV_TO_VRAM` and `GPU_NAME_TO_VRAM` give best-effort VRAM. Families that have **no**
entry report **0 MB**: Strix Point, Strix Halo, Krackan, Phoenix, Van Gogh, Rembrandt,
Mendocino and every Instinct (gfx908/gfx90a/gfx942/gfx950). Navi 24 *does* have entries
(4096 MB).

Output goes to stdout as exactly three lines - gfx target, VRAM MB, override value (empty
when none) - with human-readable logging on stderr, so callers can parse stdout. Exit 0
when a target was found, 1 otherwise.

### `backend_amd_gpu/main.py` differences

1. `#_configure_amd_gfx_override()` runs at import. It leaves an existing
   `HSA_OVERRIDE_GFX_VERSION` alone; otherwise it takes `AMD_HSA_OVERRIDE`, or maps
   `AMD_GFX`/`AMD_GFX_OVERRIDE` through `gfx_override_value`, or runs
   `detect_amd_gpu.py` with the current interpreter (20 s timeout) and takes its third
   stdout line.
2. Defaults set at import: `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`,
   `OCL_SET_SVM_SIZE=262144`, `MIMALLOC_PURGE_DELAY=0`.
3. `MIOPEN_USER_DB_PATH` and `MIOPEN_CUSTOM_CACHE_DIR` point into
   `model_weights/miopen`, and both directories are created with the other cache dirs.
4. `import torch` instead of `import torch as _torch`; every reference follows.
5. `#_log_gpu_mem(context)` logs allocated/total/reserved at `DEBUG`; called at startup
   and around detection.
6. `lifespan` sets a default `PYTORCH_CUDA_ALLOC_CONF`
   (`garbage_collection_threshold:0.8,max_split_size_mb:512`, no `expandable_segments`)
   before the hard cap; the CUDA fork sets its version later and only when not HIP.
7. HIP tuning: `matmul.allow_fp16_reduced_precision_reduction = True` and
   `cudnn.benchmark = False`. The CUDA fork instead sets `cudnn.benchmark = True` with
   `benchmark_limit = 4`.
8. The no-GPU log is a `WARNING` naming ROCm, and the GPU line includes the gcnArchName
   and the ROCm version instead of compute capability.
9. App title `Spectra Segment (AMD)`; the health route is formatted differently and
   `run_detection` returns inside its `try`.

---

## 14. Installers, launchers, checks

### `install.bat`

`PY_VERSION=3.13.14`. It redirects `UV_CACHE_DIR`, `UV_PYTHON_CACHE_DIR`,
`UV_PYTHON_INSTALL_DIR`, `TEMP`/`TMP` and `UV_NO_CONFIG=1` into the repo; when the repo
path exceeds 50 characters, the uv cache moves to `.uv-cpu` at the project root so
sdist builds stay under the Windows path limit. It then resolves the latest stable uv
through the GitHub API, downloads
`uv-x86_64-pc-windows-msvc.zip` plus its `.sha256`, recomputes the hash and fails on a
mismatch. It installs Python, creates the venv with `--seed`, installs torch and
torchvision from `https://download.pytorch.org/whl/cu130` (falling back to the default
index), then `backend/requirements.txt`, then the Ultralytics CLIP fork from the
`main` branch zip (`codeload.github.com/ultralytics/CLIP/zip/refs/heads/main`, no
checksum), then fetches Bun and runs `bun install`.

### `install_amd.bat`

Same uv handling plus the `.uv-amd` root-level cache fallback, Python `3.12.13`, and
torch from `https://rocm.nightlies.amd.com/whl-multi-arch/` as
`torch[device-<gfx>]`, where the gfx target comes from `detect_amd_gpu.py`
(`.gpu_arch.txt`). TheRock ships the `rocm` meta package as an sdist, so uv builds it
under the cache directory; the short fallback keeps that build below MAX_PATH. If uv
still fails, the torch and CLIP installs are retried with pip using the project
`.tmp` and `backend_amd_gpu/pip_cache`, so no cache leaves the app folder. The extra pulls
`rocm-sdk-core` and `rocm-sdk-libraries`; `rocm-sdk-devel` is deliberately skipped - it
is only needed to compile HIP code. Then `requirements.txt`, the same CLIP fork, Bun,
`bun install` and `bun run build`.

### `launcher/setup_memory.ps1`

Parameters: `-ConfigPath`, `-Backend NVIDIA|AMD`, `-DetectedVramMb`, `-Reconfigure`,
`-ForceCpu`. It reads and writes `<ConfigPath>` (three `KEY=VALUE` lines:
`vram_mode`, `cpu_threads`, `cpu_ram_mb`) and writes `<ConfigPath>.env` in ASCII, which
the launcher imports and deletes.

| Profile | `SPECTRA_VRAM_FRACTION` | `SAM3_IDLE_UNLOAD_TIMEOUT` | `SAM2_OFFLOAD_ENCODER` | `SPECTRA_NO_CO_RESIDENCY` |
|---|---|---|---|---|
| `high` | 0.90 | 86400 | 0 | 0 |
| `balanced` | 0.60 | 300 | 0 | 0 |
| `medium` | 0.50 | 180 | 0 | 0 |
| `low` | 0.30 | 120 | 1 | 1 |
| `cpu` | 0 | 300 | 0 | 0 |

Every profile also writes `SPECTRA_VRAM_MODE`, `SPECTRA_CPU_THREADS` and
`SPECTRA_CPU_RAM_MB`. The GPU profiles write `SPECTRA_VRAM_HARD_CAP=0.90`,
`SPECTRA_FORCE_CPU=0` and the allocator config; `cpu` writes `SPECTRA_FORCE_CPU=1`, a
zero fraction, a zero hard cap, and the three thread variables when a value was
given. GPU allocator config is `PYTORCH_HIP_ALLOC_CONF` plus `PYTORCH_CUDA_ALLOC_CONF`
for AMD and only `PYTORCH_CUDA_ALLOC_CONF=...,expandable_segments:True` for NVIDIA,
with `max_split_size_mb = max(256, floor(vram * fraction))` or the fallback
8192/4096/2048/1024 when VRAM is unknown.

When a valid profile already exists and `-Reconfigure` was not passed, it re-emits the
stored profile without prompting (and forces `cpu` when `-ForceCpu` was passed). Otherwise
it suggests a profile from VRAM (`<4 GB` low, `<6 GB` medium, `<16 GB` balanced,
`>=16 GB` high), prompts for the profile, prompts for CPU threads **only** in `cpu` mode
(`all`/empty -> logical processors, `phys` -> physical cores, `half` -> ceil(logical/2),
or a number), and prompts for a RAM budget. Fractions are formatted with the invariant
culture, because a comma decimal separator used to make Python's `float()` reject the
value and silently disable both the budget and the hard cap.

Nothing reads `SPECTRA_CPU_RAM_MB`, `SPECTRA_VRAM_MODE` or `SPECTRA_CPU_THREADS`; they are
saved and echoed only.

### `start.bat`

1. Preflight: `backend/.venv/Scripts/python.exe` must exist.
2. Memory profile: an NVIDIA check via `Win32_VideoController` decides whether to ask
   GPU or CPU; calls `setup_memory.ps1`, imports the `.env`, deletes it and echoes the
   active values.
3. Bun through `frontend/bun_env.cmd` with `BUN_DIR_OVERRIDE=tools/bun-cpu`.
4. Stops anything listening on 8000 whose command line contains `main:app` and anything
   on 3000 that looks like Vite, writes `backend/.backend_pid` from `netstat`, starts
   `backend/run.cmd` with output to `backend/backend_launch.log`, and polls
   `http://127.0.0.1:8000/health` for up to 30 s (curl when available, a PowerShell
   `DownloadString` otherwise), printing the tail of both logs on failure.
5. Junctions the frontend (`use_web.bat`), runs `bun install` when
   `web/node_modules/vite/package.json` is missing, then runs
   `node_modules/vite/bin/vite.js --host 127.0.0.1 --port 3000` from `frontend`. When the
   dev server exits it kills the backend PID.

### `start_amd.bat`

Six sections. Section 2 sets, before anything else loads: `PYTORCH_HIP_ALLOC_CONF` and
`PYTORCH_CUDA_ALLOC_CONF` (max_split 512 default), `MIOPEN_DEBUG_DISABLE_FIND_DB=0`,
`PYTORCH_TUNABLEOP_CACHE_DIR`, `HSA_ENABLE_SDMA=0`, `TORCH_BLAS_PREFER_HIPBLASLT=1` and
`DISABLE_ADDMM_CUDA_LT=1` (unless already set), `SPECTRA_DETECTOR_FP16=1` (unless already
set) and `MIMALLOC_PURGE_DELAY=0`.

It then resolves the gfx target: `AMD_GFX_OVERRIDE` wins; otherwise `detect_amd_gpu.py`
stdout lines fill `AMD_GFX`, `AMD_VRAM` and an auto override. GPUs whose target starts
`gfx110`, `gfx115` or `gfx12` use AOTriton; everything else sets
`TORCH_BACKENDS_CUDA_MEM_EFF_SDP_ENABLED=0`, `..._FLASH_SDP_ENABLED=0`,
`..._MATH_SDP_ENABLED=1`. `HSA_OVERRIDE_GFX_VERSION` is then taken from the environment,
`AMD_HSA_OVERRIDE`, the auto value, or a local `gfx101/gfx103/gfx110` mapping, with a
warning for `gfx9*`.

Section 4 creates the model cache directories, including `miopen/cache` and
`tunableop_cache`, and exports all cache redirects. Section 5 starts the backend the same
way as `start.bat` (60 s health window). Section 6 starts the frontend with the project
Bun in the background and exits, so the launcher does not own the frontend process.

### `backend/run.cmd`

Sets `PYTHONPATH` to the repo root, every cache redirect, `OMP_NUM_THREADS` from
`SPECTRA_OMP_THREADS` or `min(6, physical cores)` via PowerShell, `KMP_AFFINITY`,
`KMP_BLOCKTIME` and `KMP_DUPLICATE_LIB_OK`, then runs
`python -X pycache_prefix=<model_weights>/pycache -W ignore::DeprecationWarning
-W ignore::FutureWarning -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
--log-level error`. The uvicorn level only controls uvicorn's own handlers; the file and
ring handlers in `main.py` decide what is kept.

The AMD copy adds, all guarded so an existing value wins: `OCL_SET_SVM_SIZE=262144`,
`MIOPEN_USER_DB_PATH=<model_weights>/miopen`, `MIOPEN_CUSTOM_CACHE_DIR=<model_weights>/miopen/cache`,
`MIOPEN_FIND_MODE=2`, `MIOPEN_FIND_ENFORCE=1`, `MIOPEN_SEARCH_CUTOFF=1`,
`MIOPEN_DEBUG_DISABLE_FIND_DB=0`, `TORCH_BLAS_PREFER_HIPBLASLT=1`,
`DISABLE_ADDMM_CUDA_LT=1`, `SPECTRA_DETECTOR_FP16=1` and
`PYTORCH_TUNABLEOP_CACHE_DIR=<model_weights>/tunableop_cache`; it creates the two new
cache directories and runs `backend_amd_gpu.main:app`.

### `stop.bat`, `stop_amd.bat`

`stop.bat` uses `backend/.backend_pid` (ignoring a stale entry whose command line no
longer contains `main:app`) and then kills anything on 8000 matching `*main:app*`. It
stops the frontend only when `frontend/node_modules` is a junction pointing at
`backend/web/node_modules`. `stop_amd.bat` matches `*backend_amd_gpu.main:app*` and checks
the junction target against `backend_amd_gpu/web/node_modules`. Because `start.bat` and
`stop.bat` match any `main:app`, they also stop an AMD backend.

### `cleanup.bat`, `cleanup_amd.bat`

Remove the toolchain (`tools/uv*`, `tools/python`, `tools/bun*`, `tools/clip_src`), the
venv, uv cache (including the root-level `.uv-cpu` / `.uv-amd` long-path fallback),
`backend_amd_gpu/pip_cache`, `.embedded_python`, `.local_token`, `.backend_pid`,
`backend_launch.log`, `spectra_launcher.cfg[.env]`, the web workspace, `frontend/bun.lock`,
`.tmp`, the logs and bytecode caches. Model weights are deleted only after a separate
confirmation. The last step verifies that `.gitignore`, `README.md`,
`THIRD-PARTY-LICENSES.txt`, `LICENSE`, `docs/*.md`, `checks/check_gpu_health.py` and
`.git` still exist and warns if any is missing.

### `checks/check_gpu.bat`, `checks/check_gpu_health.py`

The wrapper picks `backend_amd_gpu/.venv` first, then `backend/.venv`, sets `PYTHONPATH`
to the repo root and runs the script with any extra arguments passed through.

`check_gpu_health.py [fork] [--model NAME] [--runs N] [--download] [--force-cpu]`
defaults to the `backend_amd_gpu` fork, so the wrapper exercises the AMD tree even when
it found the other venv. It writes
`checks/check_gpu_health_<fork>_<YYYYmmdd_HHMMSS>.log`, tees stdout into it, imports the
fork, reports platform/disk/env/device, checks that a model file is present (no download
unless `--download`), runs a cold and a warm encode, asserts the post-encode footprint
stays under 1024 MB on GPU or 2500 MB on CPU, that a second encode grows by at most
64 MB, and times `--runs` predictions (10 on GPU, 2 on CPU by default).

It does not reproduce the launchers' environment - no `PYTORCH_HIP_ALLOC_CONF`,
`HSA_OVERRIDE_GFX_VERSION`, `HSA_ENABLE_SDMA`, attention toggles or the hard cap - so a
regression caused by a launcher-set variable cannot be caught by it.

---

## 15. Frontend: `lib/`

### `lib/api.ts`

`BASE = "http://localhost:8000"` hardcoded. `#ApiError` carries `status` and a message
built from the first 500 characters of the body with control characters replaced.

`#getToken` memoises in `_token` and `_tokenPromise`; a rejected fetch clears the promise
so the next call retries. `#resetToken` clears both. `#apiFetch` sets `X-Local-Token`,
retries **once** after a 401 with a fresh token, and throws `ApiError` for any other
non-OK response.

| Method | Notes |
|---|---|
| `createSession`, `destroySession`, `sessionHealth` | `sessionHealth` returns a boolean and swallows the error |
| `releaseSession` | `navigator.sendBeacon(.../release)` (no headers, hence the backend exemption) and then a `keepalive` `DELETE` with the cached token. It only sends the DELETE when a token is already cached; on 401 it refreshes and retries once. A page teardown therefore sends two destroy requests, both idempotent |
| `uploadImage` | `FormData`, own 401 retry, same body-sanitising error |
| `predict`, `undo`, `redo`, `clearObject` | All go through `toPredictResponse`, which decodes `object_masks` and passes `object_history` through |
| `sam3Prompt` | Decodes every mask |
| `sam3Undo`, `sam3Redo`, `sam3RemoveInstance` | Return `masks: []` - the server's masks are discarded and the local prompt stack stays the authority |
| `setSam3Settings`, `listDetectors`, `loadDetector` | `loadDetector` posts `{detector_name}` |
| `detect` | Sends `query`, `max_detections`, `use_yoloe_masks`. It never sends `confidence` |
| `exportZip` | Posts `root`, `files`, `include_whole`, `format`, `background_color`, `feather_radius` |
| `getLogs`, `getModels` | `getLogs` asks for a limit and returns `{entries, total}` |
| `segmentBatch` | Keeps `null` for an empty mask and decodes the rest in parallel |

`#decodeObjectMasks` keeps `Map<objectId, {source, mask}>`. An entry is reused when the
base64 string is byte-identical, which is why the renderer can compare `PackedMask`
objects by identity. It also deletes cache entries for subjects that are no longer in the
response. `#resetObjectMaskCache` clears the map; `useSession.resetUndoRedoState` calls it
on session, model and image changes so a mask from the previous session cannot be reused.

`#decodeDetectionMask` base64 -> `Blob` -> `Image`, rejects anything over 8192 per side
or over 8192x8192 pixels in total, reads the red channel and stores `1` where it is
above 127, returning `{w, h, data}`. `#encodeMaskPng` writes white where the mask is set, calls `toDataURL`
and strips the data-URL prefix - synchronously, on the main thread.

### `lib/sse.ts`

`#connectSse(url, onMessage, onStateChange)` opens an `EventSource`, resets the attempt
counter on open, and on error closes, reports `false`, and reconnects after
`min(30000, 1000 * 2 ** attempt++)`. `close()` marks the stream closed, clears the timer
and reports `false`.

### `lib/utils.ts`

`#cn` wraps `clsx`. `SUPPORTED_IMAGE_EXTENSIONS` is `jpg, jpeg, png, webp`;
`#isSupportedImage` accepts by extension **or** MIME type. `#fitImageFile` decodes the
file once, reads `naturalWidth`/`naturalHeight` from that image, and only builds a canvas
when `w*h > 4096*4096` or the longest side exceeds 8192. The scale is
`min(1, sqrt(16.7 MP / (w*h)), 8192 / max(w, h))`, the output MIME follows the input
(PNG, WebP, otherwise JPEG at 0.92), and a file that will not decode is returned
unchanged.

### `lib/strokePrompt.ts`

Constants: `MAX_RING_POINTS=32`, `MAX_INTERIOR_POINTS=8`, `MAX_AREA_POINTS=12`,
`AREA_ERODE_FRACTION=0.3`, `RASTER_MAX_DIM=1024`.

`#buildStrokePrompt(stroke, brushSize, mode, imgW, imgH)`:

| Case | Result |
|---|---|
| Empty | no points |
| Fewer than 3 points | One point, label 1 or 0 |
| Negative | `buildAreaPrompt` with label 0 |
| Positive and closed | `buildRingPrompt` |
| Positive and open | `buildAreaPrompt` with label 1 |

Closed means `gap <= max(brushSize * 1.5, 4)` **or** `gap <= 12%` of the stroke's
bounding-box diagonal, and at least 4 points.

`#buildRingPrompt` downsamples with a spacing of `max(6, minSide/96)`, caps at 32 ring
points, computes the centroid and mean radius, and pulls interior points in by
`clamp(brushSize / avgRadius, 0.15, 0.5)` of the radius: the centroid plus every
`ceil(len/8)`-th ring point, capped at 8 positives. The ring itself is emitted as
negative. Fewer than 3 ring points degrades to a single positive point.

`#buildAreaPrompt` rasterises into an offscreen canvas whose longest side is 1024, strokes
with `lineWidth = brushSize*scale - 2*(0.3*brushSize)*scale` (60% of the brush narrower, so
sampled points sit inside the object), then samples on a grid of
`max(4, round(16*scale))` and caps at 12 points by stride. No sample falls back to the
first point.

All emitted coordinates are rounded and clamped to `0..w-1` / `0..h-1`.

### `lib/detectionEdit.ts`

`#removeDetectionMask` drops the removed index and shifts later keys down one.
`#removeDetectionIndexFromLayers` drops the matching layer and decrements every later
`detectionIndex`. `#shiftSelectedDetection` clears the selection when it is the removed
index, decrements it when a later one was removed, and leaves it otherwise. `Canvas.tsx`
and `useLayers.ts` both call these, so the two removal paths cannot drift.

### `lib/types.ts`, `lib/mask.ts`

`PackedMask` is `{w, h, data: Uint8Array}` with one byte per pixel. The response types
mirror the backend: `PredictResponse.objectMasks` is keyed by number, `objectHistory` by
string, `DetectorInfo.type` is `"grounding" | "yoloe"`, `Detection.mask` is an optional
base64 string that the UI clears once decoded.

---

## 16. Frontend: `store/`

Five files, all plain `atom()` on jotai's default store. There is no `Provider`, so every
component shares one store.

### `store/session.ts`

`sessionIdAtom`, `modelNameAtom`, `imageUrlAtom`, `imageWidthAtom`, `imageHeightAtom`,
`imageFileAtom`, `hasImageAtom` (derived from `imageUrl`), `masksAtom`,
`objectMasksAtom`, `perDetectionMasksAtom`, `brushObjectsAtom` (`[0]`),
`activeObjectIdAtom` (0), `objectUndoCountsAtom`, `objectRedoCountsAtom`,
`objectHistoryAtom`, `subjectMetaAtom`, `sam3ReadyAtom`, `modelsAtom`.

The three mask atoms mean different things:

| Atom | Keyed by | Meaning |
|---|---|---|
| `masksAtom` | position | Brush: `Object.values(objectMasks)` in id order. SAM 3: every instance, flattened. Also the "is there anything to export / show transparently" test |
| `objectMasksAtom` | subject id | The per-subject breakdown the colour overlay is built from |
| `perDetectionMasksAtom` | detection index | One mask per detection, which is why indices must stay stable |

`#SUBJECT_COLORS` is ten muted colours at hues 29.5, 54.2, 90.3, 125.8, 161.9, 198.1,
234.2, 269.7, 305.8 and 341.9 degrees: roughly 36-degree steps with a 24.7-degree minimum
between the first two and 47.6 between the last and the wrap.

`#subjectColor(id)` picks `SUBJECT_COLORS[id % 10]`, then shifts a repeat of the cycle up
or down in lightness by `0.13 * ceil(cycle / 2)` (alternating sign) via
`shiftLightness`, which converts to HSL, clamps lightness to 0.15..0.85 and converts back.
Results are memoised per id. Lightness clamping means two ids far apart in the cycle can
converge on the same swatch; the palette is not guaranteed unique beyond the first ten.

`#subjectName(id)` is `Subject <id+1>`.

### `store/ui.ts`

Dialog and overlay flags: `settingsOpenAtom`, `exportOpenAtom`, `showTransparentAtom`,
`hideBboxesAtom`, `endSessionOpenAtom`, `unsupportedFileAtom`, `modeLockAtom`
(`"brush" | "detect" | null`), `modeDialogOpenAtom`, `modeSwitchTargetAtom`,
`isExportingAtom`, `imageEncodingAtom`, `encodingMessageAtom`, `fitToViewAtom`,
`zoomInAtom`, `zoomOutAtom`, `uploadHoveredAtom`, `consoleOpenAtom`.

Four atoms persist through `atomWithStorage`: `consoleFilterAtom`,
`consoleSourceAtom`, `consoleCategoryAtom`, `consoleShowTimeAtom`.

`#pushToast` keeps at most the last three toasts (it slices to the last two before
appending), emits a `console` category entry and removes the toast after 4 seconds.

### `store/detection.ts`

`detectModeAtom`, `toolModeAtom` (`positive`/`negative`), `brushSizeAtom` (20),
`featherRadiusAtom` (0), `detectorsAtom`, `selectedDetectorAtom` (`""`),
`loadedDetectorAtom`, `detectorLoadingAtom`, `detectQueryAtom`, `detectionsAtom`,
`selectedDetectionAtom`, `isDetectingAtom`, `yoloeMasksEnabledAtom`.

### `store/layers.ts`

`Layer` is `{id, type: brush|detection|sam3, label, preview, objectId, detectionIndex?,
promptIndex?, instanceIndex?, createdAt}`. `#nextLayerId(existing)` scans the current ids
for the highest `layer-<n>`, advances a module-level counter past it and returns the next
id. Taking the id from a value captured once made every `addLayer` call in one synchronous
loop produce the same id, so `selectedLayers` addressed them all and `removeLayer(id)`
deleted them all.

### `store/sam3.ts`

`sam3ModeAtom`, `sam3PromptsAtom`, `sam3RedoStackAtom`, `sam3PromptInputAtom`,
`sam3PromptingAtom`, `sam3InstancesAtom`, `selectedSam3InstanceAtom`,
`sam3KeepLoadedAtom` (true), `sam3EncodeDimAtom` (1024; type `512 | 1024 | 1500`).

### `store/logs.ts`

`#emitUi(category, level, message, detail?)` builds `{id, source: "ui", level,
category, message, ts}` (`ts` is `HH:MM:SS` from `toTimeString`), applies `safeText`
(300 characters for the message, 200 for the detail) and notifies every subscriber
inside a `try/catch`. `#logErr(category, err)` is `emitUi(category, "ERROR", safeText(err))`.
`#subscribeLogs` returns an unsubscribe function. `uiCategoryLabels` is the twelve
categories: `session`, `upload`, `mode`, `detect`, `segment`, `brush`, `sam3`, `subject`,
`layer`, `export`, `settings`, `console`.

`safeText` coerces (`Error` -> `Name: message`, else `JSON.stringify` with a `String`
fallback), replaces `PATH_RE` (`X:\...` and `/home|Users|root|tmp|var|opt|srv|workspace|data/...`)
with `[path]`, `SECRET_RE` (`token|bearer|x-local-token` followed by `:` or `=` and a
value) with `[redacted]`, and `LONGHEX_RE` (32+ hex characters) with `[hash]`, then
truncates with `...`. The backend has no equivalent filter.

---

## 17. Frontend: `hooks/`

### `useSession.ts`

Module-level state, because `useSession()` is instantiated in eight components (App,
Canvas, EmptyState, Header, ModeSwitchDialog, EndSessionDialog, SubjectsPanel, Toolbar)
and a ref inside the hook belongs to one instance:

| Name | Purpose |
|---|---|
| `uploadGuardRef` | One upload at a time |
| `endGuardRef` | One teardown at a time |
| `blobUrlRef` | The single live object URL, revoked exactly once (a per-instance ref leaked one per mode switch and end-session) |
| `sessionQueueMap` | One promise chain per session id |
| `deletedSubjects` | Ids the user deleted; `syncSubjects` refuses to recreate them |
| `pendingStrokesRef` | In-flight stroke count per subject |
| `historyRef` | The authoritative ledger copy the buttons read |
| `historyQueueRef`, `historyScheduledRef` | The coalescing undo/redo queue and its drain flag |

`#enqueue(sessionId, task)` appends to that session's chain and stores the caught promise,
so every mutation - predict, undo/redo burst, clear-object - is FIFO per session and
responses cannot land out of order.

`#applyHistory(history)` replaces the ledger: it rebuilds `objectHistoryAtom`,
`objectUndoCountsAtom` and `objectRedoCountsAtom` from the `object_history` the server
sent. Keys absent from the payload are dropped, which is the signal that the subject was
cleared. `#syncSubjects` creates metadata for ids it has not seen unless they are in
`deletedSubjects`, so a late response cannot resurrect a deleted subject.
`#patchHistory(oid, patch)` merges one subject and is what the 409 paths use - calling
`applyHistory({[oid]: ...})` there would replace the whole ledger and zero every other
subject's depth.

`#uploadImage` refuses a second concurrent upload, rejects an unsupported file into
`unsupportedFileAtom`, creates a session first when there is none (aborting if the model
changed mid-flight), revokes the previous blob URL, runs `fitImageFile`, sets the image
atom and resets masks/layers/detections/SAM 3 state, then uploads. On success it stores
the dimensions, logs the resize, and opens the mode dialog unless SAM 3 mode is active.
The guard is released when the upload settles.

`#switchModel` resets the undo/redo state and SAM 3 state, destroys the current session,
clears image-derived state and creates a session for the new model, re-uploading the
stored `File`. `#endSession` destroys, revokes the blob URL, resets every atom and
`resetUndoRedoState`. `#recoverSession` returns early when the health check passes;
otherwise it resets undo/redo, drops `brush` layers and recreates the session.
Detection masks and layers survive because they are client-side.

`#resetUndoRedoState` clears `deletedSubjects`, the object-mask cache, pending strokes, the
ledger, the queue and the counts, resets `brushObjects` to `[0]`, `activeObjectId` to 0 and
`subjectMeta` to a single subject 0.

`#predict` adjudicates the active subject id, counts the stroke as pending, enqueues the
call, applies the response and decrements the counter in `finally`. **404** triggers
`recoverSession`; **400** shows the backend's message as a toast (object cap, no image);
anything else shows "Prediction failed". It deliberately does not rebuild on 400 - doing
so used to destroy a live session whenever the object cap was hit.

`#requestHistory(oid, kind)` does nothing when the adjusted depth is zero, pushes the
intent and schedules a drain. `#runHistoryOps` takes the whole queue, clamps each op's
steps to the depth at that moment, calls undo or redo, applies the response and logs
"Undid N stroke(s) on Subject X". A **409** patches only that subject ("nothing left to
undo" also resets `strokes` and `has_mask`); a **404** recovers the session. Within one
drain every undo runs before every redo, so a mixed burst resolves undo-first rather than
in click order.

`#historyDepth` uses `availableSteps(depth, pending, kind)` from `lib/historyQueue.ts`:
undo depth plus pending strokes, redo depth minus pending. A pending stroke was enqueued
before the click, lands in the history first, and clears the redo stack, so a queued redo
would ask for an entry the server no longer has.

`#clearObjectHistory(oid)` runs `clear-object` through the same queue and falls back to
patching the ledger to zero when the call fails.

Returned: `sessionId`, `imageUrl`, `uploadImage`, `switchModel`, `endSession`,
`recoverSession`, `predict`, `undo`, `redo`, `undoSubject`, `redoSubject`, `canUndo`,
`canRedo` (both from the count atoms for the active subject), `clearObjectHistory`,
`applyHistory`, `forgetSubject`.

### `lib/historyQueue.ts`

`#availableSteps(depth, pendingStrokes, kind)` returns the usable depth, floored at 0.
`#HistoryQueue` accumulates steps per `(objectId, kind)` in two maps, so five clicks on one
subject become one op with `steps: 5`; `drain()` emits every undo then every redo and
clears both maps.

### `useSessionGuard.ts`

On `pagehide`, releases the session once per session id. On `pageshow` with `persisted`, it
re-arms and health-checks the session, recovering when it is gone. `recoverSession` is held
in a ref so the listeners never capture a stale closure.

### `useCanvasRenderer.ts`

Constants: `MASK_OVERLAY_ALPHA=0.42`, `ACTIVE_OVERLAY_ALPHA=0.56`,
`SUBJECT_LAYER_BUDGET_BYTES=192 MB`, `REBUILD_MIN_INTERVAL=80`, `REBUILD_TRAILING_DELAY=90`.

`currentDpr()` caches `devicePixelRatio`; `supportsCtxFilter()` probes a 16x16 canvas once
and caches the answer.

`#boxBlurFloat(src, w, h, r)` is a separable box blur over **interleaved RGBA** float data,
3 passes, clamping indices at the edges. `#blurPadded(source, radius, crop)` pads by the
radius so the border does not darken, uses `ctx.filter = blur(...)` when supported, else
premultiplies, runs the box blur three times and un-premultiplies. `#blurMaskCanvas` is
`blurPadded(..., crop=true)`.

`#computeState(zoom, x, y)` returns `baseScale = min(cw/iw, ch/ih, 1)`,
`scale = baseScale*zoom`, and offsets `floor((c - floor(i*scale))/2) + pan`, with the inner
`floor` matching the wheel handler's rounding so the pixel under the cursor stays put.

`#ensureSubjectLayer(id, mask, feather, img)` returns a cached full-resolution tinted
canvas when `mask` (identity), image (identity), feather radius and colour all match,
otherwise rebuilds: draw the mask white into the reused scratch canvas via a
`Uint32Array` view, feather it, `source-in` the subject colour **on that layer**, then
scale it onto an image-sized canvas. Layers above the 192 MB budget are not cached at all
and are rebuilt every time; the cache evicts the least recently inserted subject first
(the `Map` iteration order is refreshed on a hit) and never evicts the entry it is
currently building. The scratch canvas is reused, so the cached canvas is always a
separate allocation.

Tinting the accumulator instead would recolour every subject already composited and
multiply alpha by 0.42 per pass, leaving every mask one colour at `0.42^N` opacity.

`#rebuildSubjectMaskCache` drops layers whose subject disappeared or whose image changed,
orders subjects so the active one is drawn last, composites at 42%/56%, and caches under
`#subjectMaskKey(ids, activeId, featherRadius, revision)` - the same helper the caller uses
to compare. When nothing could be drawn it stores an **empty** key, which can never match,
so the next frame retries instead of painting a cached blank.

`#rebuildMaskCache` is the no-subject fallback: one union overlay tinted `0x783475d4`,
keyed on `f<feather>|r<revision>`, and it skips any mask whose size differs from the image.
`#rebuildTransparentComposite` draws the image then punches it with `destination-in`
through the feathered union mask, with the same key rule.

`#maybeRebuild(name, fn)` runs immediately when 80 ms have passed since the last rebuild;
otherwise it stores the job under its name (`subject`, `masks`, `transparent`), schedules
one 90 ms trailing timer and runs every pending job plus a redraw. The per-name map is
what stops a deferred subject rebuild from being overwritten by a mask rebuild.

`renderMasks` clears the mask canvas with the DPR transform, then either draws the
transparent composite (or the plain image when there are no masks), the per-subject overlay
when `objectMasks` is non-empty, or the union overlay.

`renderDetections` and `renderSam3Boxes` share the detection canvas and draw green
`#00ff88` boxes with a `#<n> <label> <score>%` / `#<n> <text>` chip, orange when
selected; both return early when `hideBboxes` is on in transparent mode.

`#noteMaskScale` emits one `subject` warning per `maskSize->imageSize` pair when a mask's
resolution differs; the mask is still stretched onto the image rect, because skipping it
left a silently empty composite cached forever.

`#redraw` sizes every backing store to `floor(css * dpr)` (skipping the assignment when it
is unchanged), resets each context transform, paints the checkerboard in transparent mode
(cached pattern per container size), and calls the three renderers. `#invalidateCache`
empties the three cache keys, bumps `maskRevisionRef` and drops the checker pattern; it
deliberately keeps the cached canvases so a deferred rebuild shows the previous overlay
rather than a blank one.

### `useBrushDrawing.ts`

Holds `isDrawingRef`, `strokePointsRef`, `activeStrokeModeRef` and the processing-ring
state. `#drawBrushPreview` clears the draw canvas and strokes the whole point list in
image space, using the active subject colour for positive strokes and `#ff4444` for
negative, at `brushSize * scale` width and 60% alpha. `#canvasToImage` maps a CSS point to
image coordinates and clamps to the image. `#isInsideImage` tests the unclamped mapping.
`#startProcessingRing` animates a pulsing ring at a point until `#stopProcessingRing`
clears it; the animation frame is cancelled on unmount.

### `useDetection.ts`

`#detect(sessionId)` requires a query, defaults the detector to `grounding-dino-tiny`,
loads it when it is not the loaded one, posts `max_detections: 20` with
`use_yoloe_masks`, logs the count and duration, appends the new detections and **keeps only
the existing mask entries**, so keys stay aligned with the array. It clears the selection,
returns `{detections, baseOffset}` and clears `loadedDetector` on failure.

### `useLayers.ts`

`#addLayer` mints an id from `nextLayerId(layers)` and appends. `#removeLayer` on a
detection layer reindexes masks, detections, layer indices and the selection through
`lib/detectionEdit.ts`; on a brush layer it drops the layer, and when no brush layer
remains it clears masks and calls `clear-object` for every brush object id. `#clearAllLayers`
does both lists and clears every server brush subject. `#selectLayer` replaces the set or
toggles with Ctrl/Meta. `#syncDetectionLayers(dets)` replaces all detection layers at
offset 0; `#appendDetectionLayers(dets, offset)` keeps existing detection layers and appends
with `offset` (or 0 when none exist).

### `useSam3.ts`

`#prompt(text)` posts the trimmed text, appends `{text, masks, scores, bboxes}`, clears the
redo stack and reconciles; it extracts a `"detail"` string from an error body for display.
`#syncFromPrompts` is the single reconciliation point: it flattens every prompt's masks
into `masksAtom`, builds `sam3InstancesAtom` as
`{promptIndex, instanceIndex, text, bbox, score}` and calls `#reconcileLayers`, which
reuses layers by the `${promptIndex}:${instanceIndex}` key (updating only the label) and
mints new ids from `nextLayerId`.

`#undo` and `#redo` are optimistic: they change the local stack, call the endpoint, and
restore the previous state if it fails. Because `api.sam3Undo`/`sam3Redo` discard the
server's masks, the local stack is the pixel authority; after a `recoverSession`, which
starts an empty server history, the two can diverge. `#removeInstance` filters the
instance out of its prompt's masks/scores/bboxes, drops the prompt when it becomes empty
(which shifts every later `promptIndex` on both sides) and clears the redo stack.

`canUndo`/`canRedo` are `prompts.length > 0` and `redoStack.length > 0`.

### `usePanZoom.ts`

`fitToView` resets zoom to 1 and pan to 0. `zoomIn` multiplies by 1.25 capped at 50,
`zoomOut` by 0.8 floored at 0.1. Each schedules a redraw.

### `usePanelDismiss.ts`

While open, listens on `document` for `mousedown` outside the panel and outside
`ignoredRef`, and for `keydown` Escape. `ignoredRef` is the panel's own toggle button:
without it the button's `mousedown` closes the panel and the following `click` reopens it,
so the button could never close anything.

---

## 18. Frontend: `components/`

### `App.tsx`

Renders `Header`, then either `EmptyState`, `Sam3Page` or `Canvas` + `Toolbar` based on
`hasImageAtom` and `sam3ModeAtom`, then the dialogs, `EncodingOverlay`, `ToastHost` and the
upload vignette. `SettingsDialog` and `ExportDialog` are lazy-loaded.

Global keys: Escape closes the mode-switch dialog, blurs a focused input, then closes
mode/end/settings/export or clears the detection selection in that order. Ctrl+Z /
Ctrl+Y are prevented and routed to SAM 3 or brush undo/redo (ignoring key repeat; Shift is
not considered). Delete/Backspace removes the selected SAM 3 instance or the last one, and
in non-detect mode removes the last layer. `beforeunload` prompts while a session or
unsaved masks/layers exist. On mount it fetches the detector list and fades out the
`#splash` element.

### `Canvas.tsx`

Four stacked canvases: `imageCanvas`, `maskCanvas`, `detectCanvas`, `drawCanvas`, the last
one owning the pointer handlers. `allMasks` is `perDetectionMasks + masks`, which is what
the union and transparent caches consume.

Pointer behaviour: left button in brush mode starts a stroke and takes pointer capture;
right button starts a **negative** stroke (also handled through `onMouseDown`, because a
right press does not always produce a pointerdown on the same target); middle button or
Shift+left pans; wheel zooms around the cursor with a 0.9/1.1 delta clamped to 0.1..50 and
recomputes pan from the image point under the cursor. Points are collected at 2 px
(`STROKE_COLLECT_SPACING`) minimum spacing and clamped to the image.

`#handlePointerUp` is bound to both `onPointerUp` and `onPointerLeave`. A left drag holds
pointer capture, so leaving the canvas keeps collecting; a right drag does not, so it
commits mid-gesture. On commit it builds the prompt, logs the sample count, clears the draw
canvas, starts the processing ring after 150 ms, awaits `predict`, and stops the ring only
if no newer stroke has started.

`#handleDetectClick` hit-tests detections from the topmost down, toggles the selection and
the matching layer, and segments on first click: the detector's own mask when YOLOE masks
are enabled and present (cleared from the detection afterwards), otherwise
`segmentBatch` for that single box. It adds a detection layer when none exists for that
index. A failure in the YOLOE-decode fallback path is not caught.

Delete/Backspace removes the selected detection, or the last one in detect mode, using the
same `detectionEdit` helpers. A `ResizeObserver` and the image-load effect both invalidate
the caches and schedule a redraw; mask, detection, feather and SAM 3 changes invalidate
too, while selection/transparent changes only redraw.

### `Toolbar.tsx`

Floating bar with two personalities. Detect: query input (Enter runs), detector select
grouped into Grounding DINO / Florence-2 / YOLOE with tier icons, hints and a "Loaded"
chip; the existing detector is disabled in the list; a YOLOE toggle appears only for YOLOE
detectors; the Detect button; when results exist, an "All" batch segment button, a result
count and the boxes toggle in transparent mode. Brush: undo/redo, subject switcher
(previous/next/open/add/delete), brush size 2..80, feather 0..20, and clear-all-masks when
there is anything to clear. Both modes: export when masks exist, and zoom in/fit/out.

`#handleDetect` remembers `offset = detections.length` and uses `syncDetectionLayers` for
the first batch or `appendDetectionLayers(dets, offset)` afterwards. `#handleSegmentAll`
sends every detection box in one `segmentBatch`, stores the non-null masks by index and
adds layers for detections that have none. `#addSubject` uses `max(ids) + 1`, so ids are
never reused - `clear_object` deletes server state for an id, and a reused id would inherit
the deleted subject's history. `#deleteSubject` refuses the last subject, moves the active
id to a neighbour, drops the metadata, marks the id deleted and clears it on the server.
`#clearAllBrushObjects` clears every subject on the server and resets to subject 0.

### `SubjectsPanel.tsx`

Lists `brushObjects` with colour swatch, name, stroke count, a mask dot, and per-subject
undo/redo/delete buttons that target that id. Selecting a row sets the active subject.
Missing metadata is filled on render. Delete is the same logic as the toolbar's, including
`forgetSubject` and the server clear.

### `LayersPanel.tsx`

Used by the detect toolbar and by the SAM 3 bar. Detection groups are keyed by the layer
label with the trailing score removed; each row shows `#<index+1> <label> - <score>%`, a
mask dot and a remove button. SAM 3 layers get their own section, select/toggle the SAM 3
selection and call `onRemoveSam3Instance`. Dismissal uses `usePanelDismiss`.

### `Header.tsx`

Owns the model selector (segmentation models, or only SAM 3 models in SAM 3 mode), the
image size chip, the Transparent toggle, `ConsolePanel`, Settings, End Session and
Upload/Replace. It subscribes to `/api/models/stream` and applies each payload to
`modelsAtom` and the detector list, deriving `loadedDetectorAtom` from the `loaded` flag.
Stream events drive the overlay label: `downloading` (10 min timeout), `loading` (30 s),
`unloading` (3 s), `loaded` (clears). The model list is refetched when a local
load/download finishes. Switching a model shows "Downloading model..." when the target is
not on disk. Residency transitions are not logged by the client because the backend
already logs them.

### `Sam3Page.tsx`, `Sam3PromptBar.tsx`

`Sam3Page` renders `Canvas interactive={false}` plus the prompt bar. The bar has the prompt
input (Enter submits, clearing the input immediately), Segment (spinner while prompting),
undo/redo, a segment-count button opening `LayersPanel`, Export (disabled with no
instances), the boxes toggle in transparent mode, and an options popover with the
keep-loaded switch and the Fast/Balanced/High encode buttons (512/1024/1500) that call
`/api/settings/sam3`; changing the dimension shows a re-encoding spinner. Errors from
`useSam3` render inline. The popover closes on outside mousedown.

### `ExportDialog.tsx`

Three scopes: everything, one group, or picked sub-layers. Groups are SAM 3 prompts (by
text), one group per brush subject, or detections grouped by layer label. `includeWhole` is
true only for "everything". Paths are `<root>/<folder>/<slug>_<n>.png` for detections and
SAM 3, and `<root>_subj<n>.png` for brush subjects; the request `root` is `""` in brush
mode, so brush exports land at the archive top level.

`#slugify` lowercases and replaces non-alphanumerics with `_`; `#safeFolder` strips
separators, reserved characters and control characters and rejects `.`/`..`. Format is
PNG or JPG (`jpg` forces a solid background; a solid background disables transparent), the
colour comes from a colour input, and feather is the shared `featherRadiusAtom`. On export
it encodes every selected mask with `encodeMaskPng` (the brush lookup is
`objectMasks[oid] ?? masks[oid]`), posts them, converts the returned base64 into a Blob,
triggers a download and shows "Downloaded!" for two seconds. Selection is filtered to items
that actually have a mask.

### `ConsolePanel.tsx`

`MAX_LOG_ENTRIES=500`, `MIN_UI_ENTRIES=150`, `MAX_MESSAGE=400`, `RENDER_LIMIT=120`.
`LEVEL_RANK` maps `DEBUG`/`DOWNLOAD` to -1, `INFO` 0, `WARNING` 1, `ERROR` 2,
`CRITICAL` 3. `FILTER_OPTIONS` is the level ladder; `FILTER_COMBOS` is the six presets
(Everything, App only, Server only, Warnings & errors, Errors only, Debug & above) that set
a level **and** a source together.

`#trim` keeps the last 150 UI entries, fills the remaining slots with the newest other
entries and re-sorts by the numeric `seq`, so ordering stays chronological across the
merge. `normalise` builds the search haystack from message, level, category, source and
`name`; only entries that arrive before the initial `/api/logs` fetch settles are queued in
`pendingRef`.

Filtering is level rank, source, category (UI categories by name, all server entries under
`backend`) and a lowercase substring search. Download rows are always excluded from the
list and shown as progress bars instead: grouped by `name`, merging fields with the
previous row, removed on a `done` phase or on 100% without a file count, with the phase
chip (`Starting`/`Downloading`/`Verifying`), byte text, file name, ETA and a second bar for
files done/total. Category chips are listed **only for categories with a non-zero count**,
sorted by count. The model section lists downloaded/loaded state. Copy writes the filtered
rows plus a `[downloads]` block through the clipboard API when the context is secure,
falling back to a hidden textarea + `execCommand`. Unread count, error/warning badge and
the "Latest" jump button are all local state; stick-to-bottom is `pinnedRef` with a 30 px
threshold and a hover check.

### The smaller dialogs and panes

| Component | Behaviour |
|---|---|
| `EmptyState.tsx` | The home screen. Whole area is a drop target; the hidden file input accepts `image/jpeg,image/png,image/webp`; the Upload button is disabled until a model is chosen (Brush/Detect) or `sam3Ready` is true (SAM 3). A drop is ignored unless the active mode is ready and the file has an `image/` MIME type. The two mode buttons set `modeSwitchTargetAtom`, which is what opens `ModeSwitchDialog`. The caption switches between "Choose a mode...", "Loading SAM 3 model...", the SAM 3 prompt hint, "Release to upload" while dragging, and "Drop an image here or click Upload". `uploadHoveredAtom` is set only while no mode/model is chosen, and paints a ring in `Sam3Page`/`App`. |
| `ModeDialog.tsx` | The Brush/Detect chooser shown after the first upload. Picking one writes `modeLockAtom`, closes itself, and sets `detectModeAtom` plus `toolModeAtom` (brush forces `positive`). There is no close button and the backdrop has no click handler, so one of the two must be picked. |
| `ModeSwitchDialog.tsx` | Confirms a Brush/Detect <-> SAM 3 switch. If a session exists it awaits `endSession()` first, then sets `sam3ModeAtom` from the target; the footer is the `accent` tone with a busy state, and Cancel is ignored while busy. |
| `EndSessionDialog.tsx` | Confirms teardown; awaits `endSession()` and closes in `finally`. `danger` footer tone, Cancel disabled while busy. |
| `UnsupportedFileDialog.tsx` | Reports the rejected file name and extension and lists the formats built from `SUPPORTED_IMAGE_EXTENSIONS`. Dismissed by the backdrop, the X, or "Got it". |
| `SettingsDialog.tsx` | Shortcut list only. What it shows depends on `hasImageAtom` and `modeLockAtom`: no image -> "Open an image to see shortcuts", brush -> Esc/Del/Ctrl+Z/Ctrl+Y/Scroll/Shift+Drag/Right click, detect -> Esc/Enter/Del/Ctrl+Z/Ctrl+Y/Scroll/Shift+Drag. Lazy-loaded by `App`. |
| `EncodingOverlay.tsx` | Driven by `imageEncodingAtom` plus `encodingMessageAtom`. `SWEEP_MS = 850`, `BACKDROP_FADE_MS = 350`. While encoding it blocks input with a spinner and shimmer bar; when encoding ends it plays one screen-sweep, then holds `blocking` for another 100 ms before releasing pointer events. Title falls back to "Encoding image...". |
| `Toast.tsx` | `ToastHost` renders `toastAtom`; error toasts are red with a warning icon, info toasts orange. `role="status"`, bottom-centre, `pointer-events-none` container with interactive toasts. |
| `ui/Button.tsx` | `variant: default \| ghost \| accent`, `size: sm` only, disabled styling, passes the rest through to `<button>`. |
| `ui/Select.tsx` | Custom listbox (not a native `<select>`): `value`, `onChange`, `options` (`value`, `label`, `icon`, `hint`, `section`, `disabled`, `loaded`), `position: auto \| up \| down`, `disabled`. Auto mode flips up when the button plus 200 px would overflow the viewport. Arrow keys open and move focus, Enter/Space select, Escape closes. The current value is rendered as a disabled row, sections render a heading, and a `loaded` option gets a green "Loaded" chip. Closes on outside `mousedown`. |
| `ui/Tooltip.tsx` | Wraps a child and renders a fixed-position tip on hover or focus, `side: top \| bottom`, measures itself to clamp the x position into the viewport. Cancels its show timer on unmount. |
| `ui/TierIcon.tsx` | Maps `tiny` to a yellow bolt, `small` to a green feather, `large` to a purple server and anything else (including `medium`) to a blue chip. Used in the detector list. |
| `ui/BoxesToggle.tsx` | One chip labelled "Boxes" or "Boxes off"; orange when boxes are hidden. Used by the detect toolbar and the SAM 3 bar in transparent mode. |
| `ui/DialogFooter.tsx` | Shared Cancel + confirm pair with a busy spinner and a `danger`/`accent` gradient. Both buttons are disabled while `busy`. |

---

## 19. Frontend: configuration

`vite.config.ts`: alias `@` -> `src`, `dedupe` React, dev server on port 3000 with
`strictPort`, `host: "localhost"` (both launchers override it with `--host 127.0.0.1`),
`fs.allow` covering the frontend directory, `frontend`, `backend/web` and
`backend_amd_gpu/web` (the junction targets), and headers: a CSP whose `connect-src`
allows `127.0.0.1:8000`, `localhost:8000` and the 3000 websocket, `X-Frame-Options: DENY`,
`nosniff`, `Referrer-Policy: no-referrer` and a `Permissions-Policy` disabling camera,
microphone, geolocation and USB. `preview` repeats them. Build: `dist`, no sourcemaps,
ES2020 targets, `react-vendor` and `jotai` chunks.

`uno.config.ts`: `presetUno`, the `transition-interactive` shortcut, Inter/JetBrains Mono
families and the orange scale `400 #d47534`, `500 #b85c2a`, `600 #96461d`, `950 #221005`.

`tsconfig.json`: ES2020, bundler resolution, `strict`, `noUnusedLocals`,
`noUnusedParameters`, `noFallthroughCasesInSwitch`, `noEmit`, and the `@/*` path.

`index.html` carries the splash element and the base colours; `main.tsx` imports
`virtual:uno.css`, `globals.css`, calls `initGrain()` and renders `App` in
`React.StrictMode`. `initGrain` paints a 256x256 random-noise PNG into `--grain-url`, which
`grain-bg` uses as a background layer.

`use_web.bat <WebDir>` creates junctions at `frontend/node_modules` and `frontend/dist`
pointing at the fork's `web` directory, hard-links `web/package.json` to the shared
manifest and moves a stray `frontend/bun.lock` into the web directory. It replaces a real
(non-reparse-point) `node_modules` without asking, and resolves an existing junction that
points elsewhere by deleting and recreating it.

`bun_env.cmd` picks `BUN_DIR_OVERRIDE` or `tools/bun`, asks `bun_fetch.ps1` to populate it,
and fails with instructions when no executable exists. It sets `DO_NOT_TRACK=1`,
`BUN_INSTALL_CACHE_DIR`, `BUN_RUNTIME_TRANSPILER_CACHE_PATH` and `TMPDIR` inside that
directory and prepends it to `PATH`.

---

## 20. Old reference entries, kept and re-verified

These are the explanations that used to be source comments. Each was checked against the
current code; the text is the same fact, now with the surrounding behaviour.

| Reference | What it explains |
|---|---|
| `main.py#run_prediction` | 404 only for a missing session, 400 for every other `ValueError`; the client depends on the split |
| `manager.py#ModelManager.create_session` | The `None` slot under the lock; the `type: ignore` marks it |
| `manager.py#ModelManager._object_masks_png` | Snapshot under the lock, encode outside it |
| `manager.py#ModelManager._restore_object_mask` | One decoder pass for the stroke that is now last; full replay when snapshots are misaligned |
| `grounding_detector.py#_normalize_auto_map` | A repo-prefixed `auto_map` value is trimmed to the module path |
| `grounding_detector.py#_florence_config_paths` | Snapshot paths, not `hf_hub_download`, whose resolution differs across hub versions |
| `grounding_detector.py#_florence_language_generation_config` | Inner config is the base because it carries the special-token ids |
| `grounding_detector.py#_enable_florence_generation` | Re-basing onto `GenerationMixin` for `transformers >= 4.50` |
| `compositing.py#_is_unit_scale` | Range test that ignores non-finite values; NaN/inf folding happens in `_to_mask_f32` |
| `compositing.py#_to_mask_u8` | A boolean mask skips the float plane |
| `device.py#_detect_nvidia` | `torch.version.cuda` is `None` on ROCm, so an AMD card is never reported as NVIDIA |
| `model_integrity.py#ensure_model_file` | Path when present, download when missing, directory named when offline |
| `offline.py#_apply` | The module globals are patched because the libraries read them, not the environment |
| `offline.py#offline_guard` | Cached loads run forced-offline, downloads run permissive, state restored in `finally` |
| `amd_gfx.py` | The three gfx families and the `None` fallback; importable before torch |
| `detect_amd_gpu.py` | `# noqa: E402` on the import after the `sys.path` bootstrap |
| `Header.tsx` model stream | The status stream drives the header only; residency is logged by the backend |
| `useDetection.ts#detect` | Masks are trimmed to the old length so keys stay aligned |
| `useSession.ts#enqueue` | One FIFO chain per session id |
| `useSession.ts#syncSubjects` | Deleted ids are skipped |
| `api.ts#resetObjectMaskCache` | Cleared on session/image change |
| `api.ts#decodeObjectMasks` | Byte-identical base64 is reused; changed masks decode in parallel |
| `api.ts#resetToken` | Both the token and the in-flight promise are cleared |
| `api.ts#releaseSession` | A stale token is refreshed and the DELETE retried once |
| `lib/detectionEdit.ts` | The three shared reindex helpers |
| `lib/utils.ts#decodeImage` | One decode returns the image and the object URL; failure is `null` |
| `store/session.ts#SUBJECT_COLORS` | Ten muted hues, 24.7 degrees minimum gap |
| `store/session.ts#subjectColor` | Stable colour per id, repeats shifted in lightness |
| `usePanelDismiss.ts` | `ignoredRef` stops the toggle button from reopening the panel |
| `ui/BoxesToggle.tsx`, `ui/DialogFooter.tsx` | The "Boxes" chip and the shared cancel/confirm pair |
| `useCanvasRenderer.ts#SUBJECT_LAYER_BUDGET_BYTES` | 192 MB, evicting the least recently used subject |
| `useCanvasRenderer.ts#SubjectLayer` | Full-resolution tinted layer; `mask` compared by identity |
| `useCanvasRenderer.ts#noteMaskScale` | A mismatched mask is stretched and reported once per size pair |
| `useCanvasRenderer.ts#subjectIds`, `#subjectMaskKey`, `#ensureSubjectLayer` | Ascending ids, one key format for both sides, cache-or-rebuild |

---

## 21. Limitations that are visible in the code

These are behaviours, not opinions; each one is reachable today.

1. `/detect` and `/segment-batch` return **400** for a missing session, while `/predict`
   returns 404. A client that "recovers on 404" will not recover from those two.
2. `POST /api/sessions` maps any `RuntimeError` to **429**, including a failed download or
   an integrity failure.
3. `load_detector` unloads the current detector before validating the new name; an unknown
   name leaves `_detector` registered but unloaded.
4. A failed encode during upload is logged and swallowed, so the client sees 200 and the
   next stroke fails with 400 "No image loaded".
5. The frontend never sends `confidence`; Grounding DINO therefore always runs at the
   transformers default (0.25 with `transformers 4.57.6`) and YOLOE at its own 0.05 default.
   There is no confidence control in the UI.
6. `export-zip` encodes per-layer files as PNG regardless of `format`; only the optional
   whole image can be JPEG, and only with a background colour.
7. Export memory is unbounded: the ZIP is built in memory and base64-encoded into JSON.
8. Session memory is unbounded; `MAX_SESSIONS` limits the count, not the bytes.
9. `_clean_florence_config` rewrites `config.json`/`preprocessor_config.json` inside the
   HuggingFace cache.
10. `start.bat`/`stop.bat` match any `main:app`, so they also stop the AMD backend;
    `stop_amd.bat` matches only the AMD one.
11. `detect_amd_gpu.py` reports 0 MB for Strix Point, Strix Halo, Krackan, Phoenix,
    Van Gogh, Rembrandt, Mendocino and every Instinct, and `setup_memory.ps1` then uses a
    fallback budget without warning.
12. `get_or_create_token()` mints a new token per call; safe only because `lifespan` calls
    it once.
13. `SPECTRA_CPU_RAM_MB`, `SPECTRA_VRAM_MODE` and `SPECTRA_CPU_THREADS` are written and
    echoed but never read by code.
14. `cudnn_disabled` serialises every cuDNN-disabled inference process-wide, and
    `cpu_threads` saves/restores a process-wide torch setting non-atomically.
15. The stderr and `torch.load` swaps use a per-module lock in the two SAM backends and no
    lock in the detectors, so concurrent loads can interleave them.
16. `_BufferHandler` drops a record emitted by a second thread during another thread's
    emit; the file handler still gets it.
17. SAM 3 undo/redo answer 404 for both "no session" and "empty history".
18. Removing a SAM 3 instance shifts every later `prompt_index`; the client's cached
    indices go stale silently.
19. The frontend has no `AbortController`; a hung request leaves a spinner latched, and
    `imageEncodingAtom` drives a full-screen overlay with no escape.
20. `SPECTRA_MAX_HISTORY` does not change the cost of an undo, only how deep it goes.
21. `torch` is not declared in `requirements.txt` even though the module catches
    `torch.OutOfMemoryError` at import time.
22. `sam3.pt` comes from an unvetted community mirror whose digest cannot be confirmed;
    `pin` changes there break every install.
23. Nothing except Python is pinned: AMD torch comes from a nightly index and the CLIP fork
    from a mutable `main` branch with no checksum.
