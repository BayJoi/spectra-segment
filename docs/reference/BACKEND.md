# Backend code reference

The code-level reference for the whole repository. Every module, every route, every
exported symbol and the reasoning behind the non-obvious ones. The topic guides cover
architecture and workflow; this file covers **what the code is and does**:

- `docs/CODE_GUIDE.md` - startup chain, HTTP/env/model catalogue, verification, fork rules
- `docs/BACKEND_GUIDE.md` - working on the backend: endpoints, manager state, recipes
- `docs/FRONTEND.md` - browser side: state, canvas, prompts, undo/redo
- `docs/LOGGING.md` - log format, structured fields, the console panel
- `docs/modes/MATTE.md` - matte mode: models, measured timings, guidance semantics, licensing

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
| `install.bat` | CPU/NVIDIA installer: uv, Python 3.13.14, capability-matched CUDA torch, requirements, CLIP, Bun |
| `install_amd.bat` | AMD installer: uv, Python 3.12.13, ROCm torch, requirements, CLIP, Bun, `bun run build` |
| `start.bat` | CPU/NVIDIA launcher: preflight, memory profile, Bun, backend, frontend |
| `start_amd.bat` | AMD launcher: gfx/HIP environment, memory profile, caches, backend, frontend |
| `stop.bat` / `stop_amd.bat` | Stop a running backend/frontend pair |
| `cleanup.bat` / `cleanup_amd.bat` | Remove venv, caches, workspace, toolchain, logs; optional model weights |
| `launcher/setup_memory.ps1` | The only place that decides the memory profile |
| `checks/check_gpu.bat` | Auto-selects AMD, NVIDIA or CPU and calls the matching wrapper |
| `checks/check_gpu_amd.bat` | AMD ROCm health check using `backend_amd_gpu/.venv` |
| `checks/check_gpu_nvidia.bat` | CPU/NVIDIA health check using `backend/.venv`; NVIDIA is labelled untested |
| `checks/check_gpu_health.py` | Universal phased checker: device, catalogue, smoke model, optional 33-model sweep and 4K pass |
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
| `utils/cuda_profile.py` | NVIDIA-only capability, wheel-support, precision and TF32 policy |
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
| `amd_gfx.py` | Windows TheRock target registry: family, support level, FP16, detector cap |
| `detect_amd_gpu.py` | Standalone Windows GPU probe that writes the hardware/launcher config |

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
| `lib/matte.ts` | Matte types, layered alpha, `decodeAlphaPng`, local Edge/Feather/Invert with a shift cache |
| `lib/sse.ts` | `connectSse` with backoff |
| `lib/utils.ts` | `cn`, image extension/MIME checks, `fitImageFile` |
| `lib/strokePrompt.ts` | Stroke to point/label prompts |
| `lib/detectionEdit.ts` | Shared detection reindexing |
| `store/session.ts` | Session, image, mask, subject and history atoms, palette |
| `store/ui.ts` | Dialog/overlay flags, console state, toasts, callback atoms |
| `store/detection.ts` | Detection mode and detector state |
| `store/layers.ts` | Layers and `nextLayerId` |
| `store/sam3.ts` | SAM 3 mode, prompts, instances, settings |
| `store/matte.ts` | Matte models, subjects, active subject, guide, strokes, busy/history state, edge controls, export flag |
| `store/logs.ts` | UI log bus and `safeText` redaction |
| `hooks/useSession.ts` | Session lifecycle, upload, predict, undo/redo queue |
| `hooks/useSessionGuard.ts` | `pagehide` release, `pageshow` health check |
| `hooks/useSessionBusy.ts` | Running upload, matte, detection, detector load and batch segmentation |
| `hooks/useCanvasRenderer.ts` | Four-canvas rendering, mask caches, DPR |
| `hooks/useBrushDrawing.ts` | Brush preview, coordinate mapping, processing ring |
| `hooks/useDetection.ts` | Detector load + run, accumulating detections |
| `hooks/useLayers.ts` | Layer add/remove/sync, server subject clearing |
| `hooks/useSam3.ts` | SAM 3 prompt, undo/redo, instance removal |
| `hooks/useMatte.ts` | Layer runs, compressed-alpha snapshots, undo/redo, add/remove/select, reset |
| `hooks/usePanZoom.ts` | `fitToView`, `zoomIn`, `zoomOut` |
| `hooks/usePanelDismiss.ts` | Outside-click and Escape dismissal for floating panels |
| `components/Header.tsx` | Title, model selector, transparent toggle, console, settings, upload/replace |
| `components/Canvas.tsx` | Canvas stack, pointer handling, detection hit test |
| `components/Toolbar.tsx` | Brush and detect toolbars, zoom, export entry |
| `components/MatteBar.tsx` | Two-row matte toolbar: model, Auto/Stroke, layers, undo/redo, local edge controls, export |
| `components/SubjectsPanel.tsx` | Subject list with per-subject undo/redo/delete |
| `components/LayersPanel.tsx` | Detection groups and SAM 3 segments |
| `components/Sam3Page.tsx` | SAM 3 layout: non-interactive canvas + prompt bar |
| `components/Sam3PromptBar.tsx` | Prompt input, undo/redo, segments, options, zoom |
| `components/ExportDialog.tsx` | Export scope, format, background, feather; ZIP, combined image or separate direct downloads |
| `components/MatteExportDialog.tsx` | Combined/active/separate downloads of visible subjects: PNG or JPG with an optional solid background |
| `components/MatteLayersPanel.tsx` | Subject picker with per-subject cutout state, visibility and add/remove |
| `components/ConsolePanel.tsx` | Simple activity feed with Developer toggle, progress rows, model list |
| `components/EmptyState.tsx` | Home screen: mode switch, drop zone, upload |
| `components/ModeDialog.tsx` | Brush/Detect chooser after the first upload; Matte is chosen on the home screen |
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
`models/matte_backend.py`,
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

`torch 2.16.0a0+rocm10.3.0a20261110`, `torchvision 0.30.0a0+rocm10.3.0a20261110`,
`torchaudio 2.11.1+rocm10.3.0a20261110`, `transformers 4.57.6`,
`ultralytics 8.4.201`, `huggingface_hub 0.36.2`, `fastapi 0.162.0`,
`starlette 1.10.0`, `numpy 2.5.3`, `opencv-python-headless 5.0.0.93`,
`pillow 13.0.0`, `supervision 0.31.4`, `accelerate 1.15.0`, `timm 1.0.31`,
`einops 0.8.2`, `filelock 4.2.10`, `setuptools 84.2.1`. Frontend: React 19.2.8,
Vite 8.2.4, UnoCSS 66.7.5, jotai 2.20.2, TypeScript 7.0.2, clsx 2.1.1.

The `backend/` tree has no venv in this checkout, so its installed versions are not
observable here; the import checks in section 13 are run with the AMD venv.

---

## 3. End-to-end shape

Four editing modes over one API:

| Mode | Model | Path through the code |
|---|---|---|
| Brush | SAM 2 / 2.1 | `Canvas` pointer capture -> `buildStrokePrompt` -> `useSession.predict` -> `/predict` -> `ModelManager.predict` -> `_replay_history` -> `UltralyticsBackend.predict` |
| Detect | Grounding DINO / MM-GDINO / Florence / YOLOE | `Toolbar` query -> `useDetection.detect` -> `/detect` -> `ModelManager.detect` -> detector `detect` -> boxes; then `segment-batch` -> SAM |
| Matte | BiRefNet | `MatteBar` -> `useMatte.run` -> `/matte` -> `ModelManager.run_matte` -> `BiRefNetBackend.matte` -> soft alpha; layers, undo/redo and Edge/Feather are client-side |
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

`#_launch_log_path` picks `SPECTRA_LOG_FILE` when set, otherwise a new
`backend/logs/backend_<YYYYmmdd_HHMMSS>.log` for this process. Relative paths are
resolved under `backend/logs/`. `_file_handler` is a `RotatingFileHandler` on that path,
level `DEBUG`, `maxBytes = SPECTRA_LOG_MAX_MB * 1 MiB` (default 10), `backupCount =
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
2. Applies `SPECTRA_CPU_RAM_MB` as a process memory cap when set (Windows Job Object,
   POSIX `RLIMIT_AS`), logging success or a warning.
3. Calls `detect_device()` and `configure_nvidia_runtime(profile)`. The latter enables
   TF32 on Ampere+ and `cudnn.benchmark` with `benchmark_limit = 4`; unsupported cards
   and CPU mode skip it.
4. If the selected device is CUDA and `SPECTRA_VRAM_HARD_CAP` parses, calls
   `set_per_process_memory_fraction(min(value, 0.95))` when the result is above zero.
   A rejected cap logs a warning; the allocator is then uncapped.
5. Logs the `NVIDIA:` profile lines (name, family, capability, fp16/TF32 and the
   wheel/arch reason) or "Running on CPU". The CUDA allocator variable is set at
   module import, before `import torch`, not here.
6. Builds `ModelManager(device=...)`, attaches `_status_callback`, mints the token.
7. Starts a task that every 10 s calls `evict_idle_sessions` in a thread and logs the
   evicted count at `DEBUG`.
8. On shutdown: cancels the task, then `destroy_all_sessions()`.

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
| `CreateSessionRequest` | `model_name` is optional and defaults to `None`; when present it must be in `ULTRALYTICS_MODELS` or be `sam3.pt`. `None` creates a matte-only session with no segmentation backend |
| `CreateSessionResponse` | `session_id`, `model_name` |
| `StrokeRequest` | `object_id` 0..1,000,000; `points` at most 10,000, each `[x, y]` with both coordinates in -10,000..10,000; `labels` at most 10,000, each 0 or 1; `bboxes` exactly 4 numbers in range with `x2>=x1` and `y2>=y1` |
| `PredictResponse` | `scores`, `object_masks` (object id -> base64 PNG), `object_history` (object id -> `{undo, redo, strokes, has_mask}`) |
| `ExportZipFileItem` | `path` 1..500; `mask_b64` non-empty |
| `ExportZipRequest` | `root` up to 200; `files` up to 1000; `include_whole`; `format` png/jpg/jpeg; `background_color` optional `[r,g,b]` 0..255; `feather_radius` 0..50 |
| `ExportImageFileItem` | `mask_b64` non-empty; no `path`, which is ZIP-only |
| `ExportImageRequest` | `files` up to 1000; `format` png/jpg/jpeg; `background_color` optional `[r,g,b]` 0..255; `feather_radius` 0..50 |
| `DetectRequest` | `query` 1..500, stripped; `confidence` optional 0..1; `max_detections` 1..100, default 10; `use_yoloe_masks` |
| `LoadDetectorRequest` | `detector_name` |
| `SegmentBatchRequest` | 1..200 boxes, each exactly 4 non-negative numbers |
| `Sam3PromptRequest` | `text` non-blank, 1..500, stripped; `confidence` optional 0..1 |
| `Sam3PromptResponse` | `masks`, `scores`, `bboxes` |
| `Sam3RemoveInstanceRequest` | `prompt_index`, `instance_index`, both >= 0 |
| `Sam3SettingsRequest` | `keep_loaded` optional; `encode_dim` optional 384..1500 |

### Routes

Auth: every route requires the local token except `GET /health` and
`GET /api/token`. `GET /api/logs/stream`, `GET /api/models/stream`, and
`POST /api/sessions/{id}/release` also accept `?token=...` because browser
SSE/beacon APIs cannot set headers.

| Route | Behaviour |
|---|---|
| `GET /health` | `{"status": "ok", "active_sessions": n}`; no auth |
| `GET /api/logs?limit=300` | Last `limit` entries of the ring, clamped to the ring length; returns `{entries, total}`; **auth required** |
| `GET /api/logs/stream` | SSE; per-IP cap 5, global cap 100, 1800 s lifetime, 30 s keepalive comment, queue depth 128; sets `Access-Control-Allow-Origin` only for the two dev origins |
| `GET /api/models/status` | `{"models": [...]}` from `_check_model_statuses` |
| `GET /api/models/stream` | SSE; per-IP cap 5, global cap 100 shared with the log stream, 1800 s lifetime, queue depth 64; sends immediately, then on events or every 3 s when the JSON changed |
| `GET /api/token` | `{"token": local_token}` with `Cache-Control: no-store` |
| `POST /api/sessions` | `create_session` in a thread; `RuntimeError` becomes **429**. `model_name: null` skips the model load entirely |
| `GET /api/sessions/{id}` | Manager session info; 404 when missing |
| `DELETE /api/sessions/{id}` | `destroy_session`; always returns `{"status": "destroyed"}` |
| `POST /api/sessions/{id}/release` | Synchronous `destroy_session` on the event loop; token required, accepts `?token=` |
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
| `POST /api/sessions/{id}/export-matte` | 404 when the session is missing, 400 no image/matte; raw PNG or JPEG for the latest stored alpha |

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
| `#_decode_mask_b64` | Rejects base64 over `MAX_MASK_B64_CHARS` and decoded bytes over `MAX_UPLOAD_BYTES`, then Base64 -> Pillow -> `L`; rejects masks over `MAX_IMAGE_PIXELS`, thresholds `>127`, DecompressionBomb/OSError become 400 |
| `#_decode_mask_resized` | Rejects `h*w` over `MAX_IMAGE_PIXELS`; nearest-neighbour resize to the session image, then `>127` |
| `#_safe_arcname` | Normalises backslashes, takes `basename(root)`, `normpath`s the item path, rejects empty/`.`/`..`, absolute paths, leading `..`, embedded `/..`, any colon and NUL, then joins. Colons are rejected outright, so a legitimate timestamp in a filename fails |
| `#_arcname_with_extension` | Replaces the final extension with the encoded `.png` or `.jpg`; a path with no stem becomes `image.<ext>` |
| `#_masks_to_png_list` | Skips empty arrays; positional SAM 3/export list |
| `#_masks_positional_png_list` | Keeps position, emitting `None` for an empty mask |
| `#_masks_to_predict_response` | Maps `object_masks` and `object_history`; `scores` is always empty |

`#export_zip` applies `format` to every per-item file and to the optional whole image.
`#_arcname_with_extension` rewrites each entry to `.png` or `.jpg`, so names match the
bytes. A jpg/jpeg request without `background_color` uses white, because JPEG has no
alpha channel. `include_whole` ORs every mask into one composite and names it
`whole.<ext>`. The archive is built in a `BytesIO` and base64-encoded into the JSON
response, so response memory scales with the archive.

`#export_image` returns one image for the OR of all supplied masks. It applies the same
format rule and also uses a white background for JPEG when none was supplied.

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

`GroundingDetector` sets the processor image size to
`SPECTRA_DETECT_MAX_SIDE` (1280 on the 8 GB AMD profile) instead of leaving the
processor at its 1333 px default. On OOM it changes the actual processor input to
1024 and then 800 px, releasing the cache between attempts; `_active_side` keeps the
successful lower size. CPU fallback runs only after every GPU size fails. A lock
serialises processor-size changes and inference.

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
`# type: ignore[assignment]` marks. A `None` model name skips `get_or_load_model`
entirely: the session keeps `backend=None`, `_ensure_model_loaded` answers "This
session has no segmentation model", and only upload, matte, detection and session
info work.

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

Sets `object_masks` from `_object_masks_png` and `object_history` from
`_object_history`. It emits no scores, so `PredictResponse.scores` is empty for brush
traffic; nothing in the UI reads it.

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

`_apply_stroke` separates prompts by type. A mixed prompt (the ring generated from
a closed positive stroke, or a box) sends all of its points and labels to SAM, so
the ring's negative outline points are not mistaken for an eraser. An all-negative
stroke is an automatic selection: the points are re-labelled positive,
`_selection_box` builds a stroke-sized box, SAM predicts the region under the
stroke, `_subtract_selection` dilates that selection slightly and subtracts it from
`obj.last_mask`, and `_mask_to_low_res` rebuilds the 256x256 prompt so a later
positive stroke cannot resurrect the removed region. `StrokeEntry.radius` comes
from the request's `stroke_radius` (1..400), or a size-relative fallback, and sizes
the selection box and edge cleanup. A benchmark on a 1920x1075 frame removed
99.8% of the stroke with 0.2% collateral, while a negative-label prompt lost 99.4%
of the subject. `multimask_output` is true only for a first stroke with no box and
at most one point.

### Residency

| Symbol | Behaviour |
|---|---|
| `#_can_unload` | Returns False when the object is not loaded; tries to acquire `_infer_lock` without blocking and releases it immediately, so a non-blocking acquire is the whole safety check |
| `#_safe_unload` | `_can_unload` then `unload_model()`; logs "unload skipped" at `DEBUG` and exceptions at `ERROR` |
| `#_evict_loaded_models` | No-op without CUDA or on `device == "cpu"`. Under the lock it unloads the detector and every segmentation model except `keep_backend`, leaving anything with inference in flight registered; outside the lock it unloads and then `empty_cache` + `ipc_collect` |
| `#_ensure_vram_headroom` | Reads `mem_get_info` and evicts with `keep_backend` when free VRAM is below `required_mb`. The hard cap lives in `main.py` |
| `#_ensure_encode_headroom` | Calls `_ensure_vram_headroom(2048, keep_backend)` - the encode path wants room and keeps the model being encoded |
| `#_ensure_full_image_encoded` | Re-encodes when `backend._src_shape` differs from the session image, which catches a model swap or a SAM 3 resolution change |

`#evict_idle_sessions` destroys sessions idle beyond `IDLE_TIMEOUT`; unloads a detector
idle beyond `DETECTOR_IDLE_TIMEOUT`; and unloads SAM 3 only when it has been idle beyond
`IDLE_TIMEOUT` (when `keep_loaded`) or `SAM3_IDLE_UNLOAD_TIMEOUT` **and** no sessions
exist at all. It returns the number of sessions destroyed.

### Detection flow

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
| `#ALLOWED_HOSTS`, `#BACKEND_PORT` | `{127.0.0.1, localhost, ::1}`, `SPECTRA_BACKEND_PORT` or 8000 |
| `#MAX_UPLOAD_BYTES` | 50 MiB, shared by the size limit and the upload route |
| `#RATE_LIMIT_PER_IP`, `#RATE_LIMIT_WINDOW_IP` | 300 requests / 60 s |
| `#get_or_create_token` | `secrets.token_urlsafe(32)`; despite the name it mints a **new** token on every call and is safe only because `lifespan` calls it once |
| `#HostValidationMiddleware` | `_split_host` separates name and port, including bracketed IPv6. The name must be in `ALLOWED_HOSTS`; a present port must be digits and equal `BACKEND_PORT`. A `Host` with no port passes |
| `#TokenAuthMiddleware` | Required on every path except `/health` and `/api/token`. `/api/logs/stream`, `/api/models/stream` and `^/api/sessions/[^/]+/release$` also accept `?token=`; no other path does. Comparison is `hmac.compare_digest` over bytes (header latin-1, token utf-8) so a non-ASCII header yields 401 instead of a `TypeError` |
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
`tier` and a `perf` string. `ULTRALYTICS_MODELS` is also the only allowlist; any name
outside it is rejected. `_MODEL_LOAD_LOCK` serialises loads; `_DEFAULT_THREADS` is
`resolve_threads("SAM2_THREADS")`, read at import.

| Symbol | Behaviour |
|---|---|
| `#UltralyticsBackend.load_model` | Rejects unknown names, sets `ULTRALYTICS_HOME`/`YOLO_CONFIG_DIR`, resolves the file with `ensure_model_file`, then under the module lock patches `torch.load` to default `weights_only=True` (the guard against a malicious checkpoint executing code), installs a `StderrInterceptor` on `sys.stderr`, sets `SETTINGS["weights_dir"]` and builds a `SAM2Predictor` with `imgsz=1024` and `quantize` from `SAM2_QUANTIZE` (default 32 on CPU, 16 otherwise). A GPU load failure retries on CPU unless `SPECTRA_ALLOW_CPU_FALLBACK=0`; both swaps are restored in `finally` |
| `#UltralyticsBackend._model_dtype` | `predictor.torch_dtype`, else the first parameter's dtype, else float32; every prompt tensor is built with it |
| `#UltralyticsBackend._encode_image` | Under the inference lock: restore the encoder if offloaded, `empty_cache`, `imgsz=[1024,1024]`, `set_image` inside `cpu_threads` + `cudnn_disabled` + `inference_mode`, cache the dense positional encoding, warm the decoder, synchronise and re-offload |
| `#UltralyticsBackend._warm_decoder` | GPU only: one decoder pass with a centre point, errors swallowed at `DEBUG` |
| `#UltralyticsBackend._offload_encoder`, `#_restore_encoder` | Implement `SAM2_OFFLOAD_ENCODER=1` - the image encoder parks on CPU between images and comes back before the next encode |
| `#UltralyticsBackend.set_image` | Records `_src_shape` and encodes. On OOM the automatic CPU fallback runs unless `SPECTRA_ALLOW_CPU_FALLBACK=0`; the encoder path runs first, then `_fallback_full_cpu` |
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
| `#SAM3Backend.set_image` | Encodes; on OOM (or a CUDA "out of memory") retries at 768 and then 512 px on the GPU. Only after those fail does it unload, switch `_device` to CPU, reload and re-encode |
| `#SAM3Backend._retry_lower_encode_dim` | Walks the lower encode dimensions, changes the predictor resolution and re-encodes; returns True on the first success. A non-OOM exception propagates |
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
`HF_MODEL_IDS` maps each to its repo id; `HF_MODEL_REVISIONS` maps each model name to
the pinned commit; `TRUSTED_HF_IDS` is the repo-id set; `_FLORENCE_MODELS` is the last
three.

### Download state and the snapshot fetch

`#hf_weights_downloaded(hf_id, revision=...)` walks
`model_weights/hf_cache/hub/models--<id>/snapshots/<revision>` and reports True when
any file with a weights suffix (`.safetensors`, `.bin`, `.pt`, `.pth`, `.onnx` for the
regex, the first three plus `.pth` for the size check) is at least 50 MB.

`#_snapshot_fetch` lists the repo with `HfApi().model_info(revision=..., files_metadata=True)`, picks
the largest weight file as the main file, and downloads the main file in a one-thread
pool while up to four threads fetch the rest. It replaces `tqdm` in
`huggingface_hub.file_download` and `huggingface_hub.utils.tqdm` with `_ByteBar`, which
emits the main file's byte progress at most every 0.4 s and a `done` entry on close,
then restores both. When the file listing fails it falls back to a plain
`snapshot_download` at the same pinned revision; when anything raises it returns False
and the caller falls back to a direct `from_pretrained` at that revision.

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

`#_florence_auto_map` reads those files without writing: it normalises `auto_map`, drops
the `AutoModelForZeroShotObjectDetection` entry and returns the result. `_load_florence_model`
fetches the config with `trust_remote_code=True`, overwrites `auto_map` in memory and passes
the object as `config=` to `from_pretrained`. The HuggingFace cache stays read-only.

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
enabled only when the model is already fp16. The pinned transformers build reports
`_supports_sdpa = False` for Grounding DINO, so the load path deliberately leaves eager
attention in place and does not request SDPA.

### `#GroundingDetector.load_model`

Sets `HF_HOME`, repairs a cache where `.no_exist` exists but `snapshots` does not by
renaming, then loads only `snapshots/<pinned revision>` with `local_files_only=True`.
If that snapshot is missing, it downloads the same revision with `_snapshot_fetch`, then
falls back to a direct `from_pretrained` with that revision and
`local_files_only=False`. Before any download it requires `is_connected()`. Failures are
logged with `exc_info` and re-raised.

### Detector internals

`#_detect_florence` joins comma-separated labels into
`<CAPTION_TO_PHRASE_GROUNDING>label1, label2`, uses `<OD>` when the query is empty, and
generates with `max_new_tokens=1024`, `num_beams=3`, no sampling and no cache. It
post-processes with the task tag and the PIL image size, drops degenerate boxes, assigns
`score = 1.0` to every box (Florence reports no per-box confidence), sorts and runs NMS.

`#_detect_grounding_dino` passes `text=[[label, ...]]` and
`post_process_grounded_object_detection` with `threshold=confidence` only when the caller
supplied one. In `transformers 4.57.6` the default is **0.25**, and the frontend never
sends `confidence`, so a detection run through the UI always uses that default. Labels
come from `text_labels` (falling back to `labels`). On an OOM retry, `detect` clears the
failed forward's traceback frames before calling `release_gpu_memory`, so the failed
attempt's input/output tensors do not stay resident while the smaller size is tried.

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
`transformers` (`>=4.57.6,<5.0`), `huggingface_hub[hf_xet]` (`>=0.36.2,<1.0`),
`filelock` (`>=4.2.1,<5.0`), `setuptools` (`>=84.1.0`), `accelerate`, `timm`,
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
- `transformers` stays on 4.x because the pinned Florence-2 remote config fails on
  both tested 5.x releases; 5.10 is also yanked. 4.57.6 still has 11 OSV advisories, so
  the app allow-lists fixed remote-code revisions and does not use Trainer,
  `save_pretrained` or LightGlue. See `SECURITY.md` for the exception and review
  trigger. `filelock>=4.2.1` and `setuptools>=84.1.0` are security floors; the
  installed audit returns zero findings for both.
- torch is **not** in either file. The installers install it explicitly (section 14),
  even though the backend catches `torch.OutOfMemoryError` at import time.
- `accelerate` backs transformers' `low_cpu_mem_usage`; `timm` and `einops` are pulled in
  by the Florence-2 DaViT remote code; `supervision` has exactly one call site,
  `GroundingDetector._nms`.

---

## 13. `backend/utils/device.py`, `cuda_profile.py` and the AMD-only modules

### `backend/utils/device.py`

`MODEL_WEIGHTS_DIR` is `<backend>/model_weights`. `#_detect_nvidia` asks
`cuda_profile.get_cuda_profile()`, which still requires `torch.cuda.is_available()`
**and** `torch.version.cuda`; `torch.version.cuda` is `None` on a ROCm build, where
`torch.cuda` points at the HIP device, so an AMD card is never reported as NVIDIA.
`#detect_device` returns `cpu` when `SPECTRA_FORCE_CPU=1`, otherwise `cuda:0` when the
GPU is both available and covered by `torch.cuda.get_arch_list()`, otherwise CPU. An
available but unsupported GPU logs the architecture and wheel reason instead of
silently pretending there is no GPU. `#release_gpu_memory` runs `gc.collect()` first,
then synchronises and calls `empty_cache` and `ipc_collect` (each guarded).
Collecting first matters for OOM recovery: a live traceback can
keep the failed forward's input/output tensors referenced, so a cache flush before
that collection cannot return their blocks.

### `backend/utils/cuda_profile.py`

NVIDIA-only, with no AMD counterpart. `#get_cuda_profile` caches one `CudaProfile`:
GPU name, compute capability, family, VRAM, driver, torch/CUDA versions, arch list,
and whether the wheel actually contains a kernel for the card. `#arch_supported`
accepts exact SASS/PTX entries, forward-compatible PTX, and forward-compatible SASS
within a major family (except Turing does not reuse Volta SASS); Blackwell 10.x,
Blackwell 12.x and Rubin 10.7 are treated as separate families. `#prepare_nvidia_environment` sets `SAM2_QUANTIZE`,
`SAM3_QUANTIZE`, `SPECTRA_DETECTOR_FP16` and `SPECTRA_MATTE_FP16` before model modules
are imported, using fp32 on Maxwell/Pascal and fp16 on Volta and newer.
`#configure_nvidia_runtime` enables TF32 on Ampere and newer (including a
`torch.backends.cuda.matmul.fp32_precision` attempt for PyTorch 2.9+) and enables the
cuDNN benchmark; `SPECTRA_TF32=0` opts out. Both functions are no-ops without an
NVIDIA CUDA build.

### `backend_amd_gpu/utils/device.py`

Adds `#_detect_rocm` (CUDA available **and** `torch.version.hip` set), checks ROCm first,
then NVIDIA, then CPU. `release_gpu_memory` is identical.

### `backend_amd_gpu/amd_gfx.py`

Single source of truth for Windows TheRock targets. Each `TargetInfo` records family,
support level (`release`, `build`, `linux-only`), integrated/discrete, FP16 policy,
default detector cap and AOTriton availability. `gfx942`/`gfx950` are marked Linux-only
and are never selected by the Windows installer; `gfx1170`-`gfx1172` and GCN/CDNA are
marked build-only. No HSA override is derived here.

### `backend_amd_gpu/detect_amd_gpu.py`

Runs as a script, not a package: `sys.path` is bootstrapped before importing `amd_gfx`.

It queries `Get-CimInstance Win32_VideoController` through PowerShell, falls back to
`wmic`, and keeps AMD adapters. Each GPU is matched by `DEV_xxxx` from `PNPDeviceID`
against `PCI_DEV_TO_GFX`, then by product-name patterns. Candidates score discrete over
integrated, add VRAM, release over build, and prefer PCI-backed matches. The highest
supported target is selected; an unsupported-only machine exits 1 with the reason.

`--write-config` emits target, product, VRAM, family, support, FP16/quantization,
detector cap, matte input cap, AOTriton and explicit override to
`spectra_amd_hardware.cfg`.
`--write-launcher-config` seeds `spectra_launcher.cfg` only when missing. Default stdout
keeps the legacy gfx/VRAM/override lines; `--json` returns the full record. Exit 0 when
a supported target was found, 1 otherwise.

### `backend_amd_gpu/main.py` differences

1. `_load_amd_profile()` reads `spectra_amd_hardware.cfg` before torch is imported and
   sets FP16, quantization, detector cap and allocator defaults. It never probes for a
   GPU. Automatic CPU fallback is enabled unless explicitly disabled.
2. Explicit `AMD_HSA_OVERRIDE` is the only source of `HSA_OVERRIDE_GFX_VERSION`.
   `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` is set only when the generated
   `AMD_AOTRITON=1` record says the target supports it. `OCL_SET_SVM_SIZE=262144` and
   `MIMALLOC_PURGE_DELAY=0` remain defaults.
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
mismatch. It installs Python, creates the venv with `--seed`, then reads
`nvidia-smi` and installs capability-matched torch/torchvision from `cu126`,
`cu132` or `cu134` (see `launcher/select_nvidia_runtime.ps1`). A failed CUDA install
retries the helper-provided `cu126` fallback for Turing-Hopper, then CPU; Blackwell and
Rubin have no older usable wheel and go straight to CPU. The installed torch is
verified and its arch list printed.
It then installs `backend/requirements.txt`, then the Ultralytics CLIP fork from the
`main` branch zip (`codeload.github.com/ultralytics/CLIP/zip/refs/heads/main`, no
checksum), then fetches Bun and runs `bun install`.

### `install_amd.bat`

Same uv handling plus the `.uv-amd` root-level cache fallback, Python `3.12.13`, and
torch from TheRock's nightly index
(`https://nightly.repo.amd.com/rocm/whl-next/`, override with `TORCH_INDEX_URL`) as
`torch[device-<gfx>]`, where the gfx target comes from `detect_amd_gpu.py` and is
stored in `backend_amd_gpu/spectra_amd_hardware.cfg` (with the launch profile in
`spectra_launcher.cfg`). It runs `--refresh --upgrade --prerelease=allow`, so every run
resolves the newest nightly and replaces the installed torch/torchvision/torchaudio
wheels. The fallback is `pip --pre --upgrade`, and `uv pip check` must pass before
the install proceeds. The verified RX 6600 stack is documented in `docs/AMD.md`.
TheRock ships the `rocm` meta package as an sdist, so uv builds it under the cache
directory; the short fallback keeps that build below MAX_PATH. If uv still fails,
the torch and CLIP installs are retried with pip using the project `.tmp` and
`backend_amd_gpu/pip_cache`, so no cache leaves the app folder. The extra pulls
`rocm-sdk-core` and `rocm-sdk-libraries`; `rocm-sdk-devel` is deliberately skipped - it
is only needed to compile HIP code. Then `requirements.txt`, the same CLIP fork, Bun,
`bun install` and `bun run build`.

### `launcher/setup_memory.ps1`

Parameters: `-ConfigPath`, `-Backend NVIDIA|AMD`, `-DetectedVramMb`, `-Reconfigure`,
`-ForceCpu`, `-ForceGpu`. It reads and writes `<ConfigPath>` (three `KEY=VALUE` lines:
`vram_mode`, `cpu_threads`, `cpu_ram_mb`) and writes `<ConfigPath>.env` in ASCII, which
the launcher imports and deletes.

| Profile | `SAM3_IDLE_UNLOAD_TIMEOUT` | `SAM2_OFFLOAD_ENCODER` | `SPECTRA_NO_CO_RESIDENCY` |
|---|---|---|---|
| `high` | 86400 | 0 | 0 |
| `balanced` | 300 | 0 | 0 |
| `medium` | 180 | 0 | 0 |
| `low` | 120 | 1 | 1 |
| `cpu` | 300 | 0 | 0 |

Every profile also writes `SPECTRA_VRAM_MODE`, `SPECTRA_CPU_THREADS` and
`SPECTRA_CPU_RAM_MB`. The GPU profiles write `SPECTRA_VRAM_HARD_CAP=0.90`,
`SPECTRA_FORCE_CPU=0` and the allocator config; `cpu` writes `SPECTRA_FORCE_CPU=1`, a
zero hard cap, and the three thread variables when a value was given. GPU allocator config is `PYTORCH_HIP_ALLOC_CONF` plus `PYTORCH_CUDA_ALLOC_CONF`
for AMD and only `PYTORCH_CUDA_ALLOC_CONF=...,expandable_segments:True` for NVIDIA,
with `max_split_size_mb = max(256, floor(vram * fraction))` or the fallback
8192/4096/2048/1024 when VRAM is unknown.

When a valid profile already exists and `-Reconfigure` was not passed, it re-emits the
stored profile without prompting unless the chosen mode conflicts with it: `-ForceCpu`
rejects a stored GPU profile, and `-ForceGpu` rejects a stored `cpu` profile. Otherwise
it suggests a profile from VRAM (`<4 GB` low, `<6 GB` medium, `<16 GB` balanced,
`>=16 GB` high), prompts for the profile, prompts for CPU threads **only** in `cpu` mode
(`all`/empty -> logical processors, `phys` -> physical cores, `half` -> ceil(logical/2),
or a number), and prompts for a RAM budget. The allocator split and hard cap are
formatted with the invariant culture, because a comma decimal separator used to make
Python's `float()` reject the value and silently disable the hard cap.

`SPECTRA_CPU_RAM_MB` is applied by `main.py` at startup as a process memory cap.
`SPECTRA_VRAM_MODE` and `SPECTRA_CPU_THREADS` are saved and echoed only.

### `start.bat`

Accepts `--reconfigure` in any order; the single-dash `-reconfigure` and
`/reconfigure` spellings are kept for compatibility.

1. Preflight: `backend/.venv/Scripts/python.exe` must exist and start; the launcher
   prints `[ OK ]` after the import check.
2. Memory profile: an NVIDIA check via `Win32_VideoController` decides whether to ask
   GPU or CPU; before offering GPU it runs `backend.utils.cuda_profile` in the venv and
   falls back to CPU with a warning when the installed wheel has no kernel for the card.
   It then calls `setup_memory.ps1`, imports the `.env`, deletes it and echoes the
   active values, including the RAM and VRAM hard caps.
3. Runs `launcher/prepare_caches.ps1` for `backend/model_weights` and prints `[ OK ]`.
4. Bun through `frontend/bun_env.cmd` with `BUN_DIR_OVERRIDE=tools/bun-cpu`.
5. Stops anything listening on 8000 whose command line contains `main:app` and anything
   on 3000 that looks like Vite, writes `backend/.backend_pid` from `netstat`, starts
   `backend/run.cmd` with output to
   `backend/logs/launcher_<YYYYmmdd_HHMMSS>.log`, and polls
   `http://127.0.0.1:8000/health` for up to 30 s (curl when available, a PowerShell
   `DownloadString` otherwise), printing the tail of both logs on failure.
6. Junctions the frontend (`use_web.bat`), runs `bun install` when
   `web/node_modules/vite/package.json` is missing, then runs
   `node_modules/vite/bin/vite.js --host 127.0.0.1 --port 3000` from `frontend`. When the
   dev server exits it kills the backend PID.

### `start_amd.bat`

Six sections. It accepts `--reconfigure`, `--cpu` and `--gpu` in any order; the
single-dash `-reconfigure`, `-cpu`, `-gpu` and slash-prefixed forms are kept for
compatibility. `--cpu`/`--gpu` override the saved profile for that launch. Section 2 sets, before anything else loads: `PYTORCH_HIP_ALLOC_CONF` and
`PYTORCH_CUDA_ALLOC_CONF` (max_split 512 default), `MIOPEN_DEBUG_DISABLE_FIND_DB=0`,
`PYTORCH_TUNABLEOP_CACHE_DIR`, `HSA_ENABLE_SDMA=0`, `TORCH_BLAS_PREFER_HIPBLASLT=1` and
`DISABLE_ADDMM_CUDA_LT=1` (unless already set), `SPECTRA_DETECTOR_FP16=1` (unless already
set) and `MIMALLOC_PURGE_DELAY=0`.

It then reads `spectra_amd_hardware.cfg` from `detect_amd_gpu.py`; the target,
VRAM, family, support level, FP16/quantization policy, detector cap and AOTriton flag
all come from `amd_gfx.py`. `AMD_GFX_OVERRIDE` is passed to the detector when set.
AOTriton is enabled only for targets whose registry record says it is available.
`HSA_OVERRIDE_GFX_VERSION` comes only from an explicit `AMD_HSA_OVERRIDE`; there is no
automatic family mapping. On cards below 12 GB, `SPECTRA_MATTE_INPUT_SIZE=1024` caps
BiRefNet HR input; ROI-guided mattes run on the requested crops instead of allocating
a 4K activation. A 4K image can therefore use ROI mattes without entering shared GPU
memory.

Section 4 runs `launcher/prepare_caches.ps1` for the model cache directories,
including `miopen/cache` and `tunableop_cache`, then exports all cache redirects. Section
5 starts the backend the same way as `start.bat`, with a 60 s health window, and writes
`logs/backend_<stamp>.log` plus `logs/launcher_<stamp>.log`. Section 6 starts the frontend
with the project Bun in the background and exits, so the launcher does not own the
frontend process.

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
`stop.bat` match any `main:app`, they also stop an AMD backend. Both stoppers print an
explicit `RESULT: STOPPED` or `RESULT: STOP INCOMPLETE` panel, and return nonzero when a
port is still held. They pause on the final panel unless `SPECTRA_NO_PAUSE=1`.

### `cleanup.bat`, `cleanup_amd.bat`

Remove the toolchain (`tools/uv*`, `tools/python`, `tools/bun*`, `tools/clip_src`), the
venv, uv cache (including the root-level `.uv-cpu` / `.uv-amd` long-path fallback),
`backend_amd_gpu/pip_cache`, `.embedded_python`, `.local_token`, `.backend_pid`, the
legacy root `backend_launch.log`, the per-launch `logs/`,
`spectra_amd_hardware.cfg`, `spectra_launcher.cfg[.env]`, the web workspace,
`frontend/bun.lock`, `.tmp` and bytecode caches. They never remove `docs/`,
`checks/`, backend source, or `frontend/src/`. Model weights are deleted only after a
separate confirmation. The last step verifies that `.gitignore`, `README.md`,
`THIRD-PARTY-LICENSES.txt`, `LICENSE`, `docs/*.md`, the backend entry points, all four
checker files (`check_gpu.bat`, `check_gpu_amd.bat`, `check_gpu_nvidia.bat`,
`check_gpu_health.py`), `frontend/src/App.tsx` and `.git` still exist and warns if any
is missing.

### `checks/check_gpu.bat`, `checks/check_gpu_amd.bat`, `checks/check_gpu_nvidia.bat`, `checks/check_gpu_health.py`

`check_gpu.bat` inspects the installed venvs and display adapters. It chooses AMD when
only AMD is present, NVIDIA when only NVIDIA is present, and asks the user to use an
explicit wrapper when both vendors are present. `check_gpu_amd.bat` uses
`backend_amd_gpu/.venv`; `check_gpu_nvidia.bat` uses `backend/.venv` and prints the
project-untested warning. All wrappers set `PYTHONPATH`, UTF-8 output, pass arguments
through, and support `SPECTRA_NO_PAUSE=1`.

`check_gpu_health.py` supports `auto`, `amd`, `nvidia` and `cpu` backends, plus
`--model`, `--runs`, `--download`, `--image`, `--all-models`, `--skip-smoke` and
`--skip-4k`. It runs six phases: preflight, device, catalogue, smoke, optional
33-model sweep and 4K/memory. Each sweep entry runs in a fresh child process; the
checker records load, encode, prompt/inference and peak allocator memory. It applies
the backend cache paths and the 8 GB `SPECTRA_MATTE_INPUT_SIZE=1024` cap, but does not
reproduce every launcher variable (`PYTORCH_HIP_ALLOC_CONF`, `HSA_OVERRIDE_GFX_VERSION`,
`HSA_ENABLE_SDMA`, attention toggles or the hard cap), so a launcher-only regression
still needs a real launch. On 2026-11-12 the AMD RX 6600 run passed 33/33 models
(41 checks passed, 2 skipped, 315.0 s) and the separate 4K pass peaked at 1,383 MB
of 8,176 MB. NVIDIA remains untested on physical hardware.

---
