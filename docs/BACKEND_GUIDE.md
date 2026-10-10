# Backend Guide

How to work on the Spectra Segment backend — **both forks at once**, the CPU/NVIDIA
one in `backend\` and the AMD ROCm one in `backend_amd_gpu\`. This is the practical
companion to `docs\CODE_GUIDE.md`, which has the deep architectural narrative.

Read this when you are changing a route, a model backend, a utility or an
environment variable. Read `CODE_GUIDE.md` when you need the full reasoning behind
a design choice. The browser side is `docs\FRONTEND.md` and the logging story is
`docs\LOGGING.md`.

---

## 1. The two forks

The backend ships twice. Both expose the **same API surface, the same request and
response models, and the same `models/manager.py` logic**. They differ only in
hardware-specific loading and tuning. Treat them as one codebase with two thin
hardware shims.

| | `backend\` | `backend_amd_gpu\` |
|---|---|---|
| Hardware | CPU / NVIDIA CUDA | AMD ROCm (TheRock wheels) |
| Virtualenv | `backend\.venv` | `backend_amd_gpu\.venv` |
| Package cache | `backend\.uv` | `backend_amd_gpu\.uv_cache_amd` |
| Model weights | `backend\model_weights` | `backend_amd_gpu\model_weights` |
| Frontend workspace | `backend\web` | `backend_amd_gpu\web` |
| Launcher | `start.bat` | `start_amd.bat` |
| Import prefix | `backend.` | `backend_amd_gpu.` |

**Rule zero: every change to `backend\` must be mirrored to `backend_amd_gpu\`.**
Section 9 is the exact, current list of what is allowed to differ.

---

## 2. Module map

```
backend\main.py                      FastAPI app: routes, request models, image
                                     decode, SSE fan-out, logging setup, OOM guard
backend\models\manager.py            Session + model-residency + mask-history state
backend\models\base.py               SegmentationBackend ABC
backend\models\detector_base.py      DetectorBackend ABC + Detection dataclass
backend\models\ultralytics_backend.py  SAM 2.1 / SAM 2 (Ultralytics SAM2Predictor)
backend\models\sam3_backend.py       SAM 3 text-prompt segmentation
backend\models\grounding_detector.py Grounding DINO + Florence-2 + CogFlorence
backend\models\yoloe_detector.py     YOLOE + MobileCLIP text encoder
backend\utils\security.py            The middleware stack and the auth token
backend\utils\device.py              Device detection, release_gpu_memory
backend\utils\model_integrity.py     Pinned URLs, sha256 digests, download + verify
backend\utils\compositing.py         Mask -> PNG, feather, alpha composite, encode
backend\utils\torch_threads.py       Thread resolution, cpu_threads, cudnn_disabled
backend\utils\stderr_progress.py     StderrInterceptor (tqdm / HF progress capture)
backend\utils\offline.py             offline_guard / is_offline (force HF offline)
backend\utils\net_check.py           is_connected() pre-flight probe
backend\utils\humantime.py           format_duration
backend\utils\logctx.py              contextvars for rid / sid
```

`backend_amd_gpu\` mirrors this exactly. Only the files in section 9 differ.

---

## 3. Running and verifying

Start the whole app (backend + frontend) with the launcher, not uvicorn directly,
because the launcher is what sets the memory profile and the ROCm environment:

```
start.bat          # CPU / NVIDIA, backend :8000, frontend :3000
start_amd.bat      # AMD ROCm
```

Verify a change — run all of it for both forks:

```
# syntax
backend_amd_gpu\.venv\Scripts\python.exe -m py_compile backend\main.py backend\models\manager.py
backend_amd_gpu\.venv\Scripts\python.exe -m py_compile backend_amd_gpu\main.py backend_amd_gpu\models\manager.py

# import (catches import-time and wiring errors a compile check misses)
set PYTHONPATH=<repo root>
backend_amd_gpu\.venv\Scripts\python.exe -X utf8 -c "import importlib; importlib.import_module('backend.main'); importlib.import_module('backend_amd_gpu.main')"

# GPU smoke test after any wheel change
checks\check_gpu.bat
```

`check_gpu.bat` does **not** reproduce the launchers' environment (no
`PYTORCH_HIP_ALLOC_CONF`, `HSA_OVERRIDE_GFX_VERSION`, attention toggles or hard cap),
so it cannot catch a regression caused by a launcher-set variable.

---

## 4. Request pipeline

Middleware is registered bottom-up: Starlette inserts each new middleware at
position 0, so **the last one registered ends up outermost**. The effective order
is:

```
ServerErrorMiddleware
  -> CORS
    -> RequestLogging
      -> SecurityHeaders
        -> HostValidation      (host must be 127.0.0.1/localhost AND port 8000)
          -> RateLimit         (300 req / 60 s / IP)
            -> RequestSizeLimit (50 MiB)
              -> TokenAuth      (X-Local-Token, except the exempt paths)
                -> ExceptionMiddleware
                  -> router
```

Auth is a `secrets.token_urlsafe(32)` generated once at startup, never written to
disk, and handed to the frontend by `GET /api/token`. Comparison runs on **bytes**,
because headers arrive latin-1 and a non-ASCII token used to raise `TypeError`.

Auth-exempt paths: `/api/token`, `/health`, the SSE endpoints, `/api/logs`,
`/api/models/stream`, and `POST /api/sessions/{id}/release` (called by
`navigator.sendBeacon`, which cannot set headers). SSE carries its own caps instead:
5 connections per IP, 100 subscribers total, 1800 s max lifetime.

Every handler runs on the threadpool through `asyncio.to_thread`. The rid/sid
context is captured in `contextvars` and copied into the worker thread
automatically.

---

## 5. Endpoint reference

Auth column: **Y** = token required, **–** = exempt.

| Method | Path | Auth | Body / notes |
|---|---|---|---|
| GET | `/health` | – | liveness + active session count |
| GET | `/api/token` | – | bootstrap token |
| GET | `/api/logs` | – | tail of the 300-entry ring |
| GET | `/api/logs/stream` | – | SSE log stream |
| GET | `/api/models/status` | Y | one-shot model + detector list |
| GET | `/api/models/stream` | – | SSE model status, 3 s poll |
| POST | `/api/sessions` | Y | `CreateSessionRequest` -> `CreateSessionResponse` |
| GET | `/api/sessions/{id}` | Y | session info |
| DELETE | `/api/sessions/{id}` | Y | destroy |
| POST | `/api/sessions/{id}/release` | – | teardown from `sendBeacon` |
| POST | `/api/sessions/{id}/image` | Y | upload, decode, resize, encode |
| POST | `/api/sessions/{id}/predict` | Y | `StrokeRequest` -> `PredictResponse` |
| POST | `/api/sessions/{id}/segment-batch` | Y | `SegmentBatchRequest` |
| POST | `/api/sessions/{id}/undo` \| `/redo` | Y | -> `PredictResponse` |
| POST | `/api/sessions/{id}/clear-object` | Y | -> `PredictResponse` |
| POST | `/api/sessions/{id}/sam3-prompt` | Y | `Sam3PromptRequest` -> `Sam3PromptResponse` |
| POST | `/api/sessions/{id}/sam3-undo` \| `/redo` | Y | -> `Sam3PromptResponse` |
| POST | `/api/sessions/{id}/sam3-remove-instance` | Y | `Sam3RemoveInstanceRequest` |
| POST | `/api/settings/sam3` | Y | `Sam3SettingsRequest` |
| GET | `/api/detectors` | Y | detector catalog + download/load state |
| POST | `/api/detectors/load` | Y | `LoadDetectorRequest` |
| POST | `/api/sessions/{id}/detect` | Y | `DetectRequest` |
| POST | `/api/sessions/{id}/export-zip` | Y | `ExportZipRequest` -> `{data, format, count}` (base64 zip) |
| POST | `/api/sessions/{id}/export-image` | Y | `ExportImageRequest` -> raw image bytes (`image/png` / `image/jpeg`) |

### Request models

All of these are defined in `main.py` with `field_validator`s that reject
out-of-range input **before** the manager sees it.

```
CreateSessionRequest   model_name: str = "sam2.1_b.pt"   (must be in the allowlist)
StrokeRequest          object_id: int = 0, points: [[x,y]] (<=10000, |c|<=10000),
                       labels: [0|1] (<=10000), bboxes: [x1,y1,x2,y2]
SegmentBatchRequest    bboxes: [[x1,y1,x2,y2]] (1..200, all coordinates >= 0)
DetectRequest          query: str (1..500), confidence: float | None,
                       # response: {detections: [{bbox, score, label, mask?}]}
                       max_detections: int = 10 (1..100), use_yoloe_masks: bool
LoadDetectorRequest    detector_name: str
Sam3PromptRequest      text: str (1..500, non-blank), confidence: float | None
Sam3RemoveInstanceRequest prompt_index: int >= 0, instance_index: int >= 0
Sam3SettingsRequest    keep_loaded: bool | None, encode_dim: int | None (384..1500)
ExportZipRequest       root: str (<=200), files: [{path, mask_b64}] (<=1000),
                       include_whole: bool, format: "png"|"jpg"|"jpeg",
                       background_color: [r,g,b] | None, feather_radius: int (0..50)
ExportImageRequest     files, format, background_color, feather_radius
```

### Response models

```
CreateSessionResponse  session_id: str, model_name: str
PredictResponse        masks: [b64 png], scores: [float],
                       object_masks: {oid: b64}, object_history: {oid: {undo,redo,strokes,has_mask}}
Sam3PromptResponse     masks: [b64 png], scores: [float], bboxes: [[x1,y1,x2,y2]]
```

### The history error contract

`/undo` and `/redo` distinguish two failures that used to share one status code:

- **404** — the session is gone. The client rebuilds it.
- **409** — the session exists but this subject has no strokes. The client
  resynchronises its counters and stops retrying.

`object_history` on every mutating response is the authority: the client adopts it
instead of guessing, which is what makes repeated undo-failure loops impossible.

---

## 6. `models/manager.py`

The single stateful part of the backend. It owns sessions, which models are loaded,
and each subject's mask history.

| Type | Holds |
|---|---|
| `Session` | id, model name, backend, image RGB/BGR, `objects`, SAM3 prompt history + redo stack, `has_image`, `last_active`, per-session lock |
| `ObjectState` | one **subject**: `stroke_history`, `redo_stack`, `last_mask`, `last_low_res_mask`, `mask_snapshots` |
| `MaskSnapshot` | low-res decoder logits for one stroke (or `None`) |
| `StrokeEntry` / `SAM3PromptEntry` / `SAM3Snapshot` | prompt inputs and results |

### Locks

| Lock | Guards |
|---|---|
| `self._lock` | session table, `_loaded_models`, `_detector`, `_last_*_use` |
| `session.lock` | one session's `objects` and each `ObjectState` |
| `backend._infer_lock` | inference on one model (`RLock` in UltralyticsBackend, plain `Lock` in SAM3Backend) |
| `_model_load_locks[name]` | one per model name, so two loads of different models cannot both win VRAM |

### The three jobs

1. **Residency.** `_evict_loaded_models`, `_can_unload`, `_safe_unload`,
   `_ensure_vram_headroom`, `get_or_load_model`. A model with inference in flight is
   left in the registry — removing it while it holds VRAM would leak.
2. **History.** `_replay_history` runs in two modes: **incremental** (one decoder
   pass, the live drawing path) for `predict`, and **full replay** for `undo`/`redo`
   so every stroke runs exactly once from the state before it. `mask_snapshots` is
   index-aligned with `stroke_history`.
3. **Lifetime.** `create_session` reserves its slot under the lock; `evict_idle_sessions`
   sweeps every 10 s and reclaims idle SAM3 models; `last_active` is refreshed by
   every working endpoint so an active session is not evicted mid-edit.

### Adding a session-scoped feature

Do **not** keep per-subject counters on the client. Return `object_history` from the
mutating endpoint and let the client adopt it. That is the one rule that keeps undo
and redo correct across deletes, recoveries and id reuse.

---

## 7. Model backends

| File | Class | Loads |
|---|---|---|
| `base.py` | `SegmentationBackend` | abstract: `load_model`, `unload_model`, `set_image`, `predict`, `reset_image`, `is_loaded`, `has_image` |
| `detector_base.py` | `DetectorBackend`, `Detection` | abstract: `load_model`, `unload_model`, `detect`, `is_loaded` |
| `ultralytics_backend.py` | `UltralyticsBackend` | SAM 2 / SAM 2.1 via `SAM2Predictor` |
| `sam3_backend.py` | `SAM3Backend` | SAM 3 text prompts |
| `grounding_detector.py` | `GroundingDetector` | Grounding DINO + Florence-2 + CogFlorence |
| `yoloe_detector.py` | `YOLEDetector` | YOLOE + MobileCLIP |

Key facts to remember while editing them (full detail in `CODE_GUIDE.md` §6):

- SAM 2 / SAM 3 wrappers do all their prompt work under `torch.inference_mode()`.
  Without it, every prompt builds a live autograd graph over the prompt encoder and
  decoder; with it, that memory is never allocated. Keep it.
- Precision is decided at runtime by `_model_dtype()` / `*_QUANTIZE`; the AMD fork
  sets `SPECTRA_DETECTOR_FP16=1` by default, the CUDA fork does not.
- `torch.load` is patched to default `weights_only=True` in `UltralyticsBackend` —
  that is the guard against arbitrary code execution from a malicious checkpoint.
  Keep it.
- Florence-2 / CogFlorence need `_enable_florence_generation` on `transformers`
  >= 4.50: those releases removed `GenerationMixin` from `PreTrainedModel` while the
  repos' remote code still relies on inheriting it. The loader re-bases the model and
  gives its language model a valid generation config.
- Florence returns no per-box score, so `_detect_florence` hardcodes `score = 1.0`;
  the confidence slider cannot filter Florence results.
- `_nms` prefers `supervision` when importable, otherwise a hand-written greedy pass.

---

## 8. Environment variables

Read by the backend (defaults come from the launcher's memory profile):

| Variable | Read by | Effect |
|---|---|---|
| `SPECTRA_FORCE_CPU` | `utils/device.py` | ignore all GPUs |
| `SPECTRA_VRAM_FRACTION` | `manager.py` | soft VRAM budget before proactive eviction |
| `SPECTRA_VRAM_HARD_CAP` | `main.py` | `set_per_process_memory_fraction`, default 0.90 |
| `SPECTRA_KEEP_SEG_ON_DETECT` | `manager.py` | keep the SAM model resident during detection |
| `SPECTRA_NO_CO_RESIDENCY` | `manager.py` | unload the segmentation model when a detector loads |
| `SPECTRA_DETECTOR_FP16` | `grounding_detector.py` | fp16 detectors on GPU. Ignored on CPU |
| `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` | `utils/offline.py` | forced on while a cached HF model loads |
| `SAM2_OFFLOAD_ENCODER` | `ultralytics_backend.py` | park the SAM2 encoder on CPU between images |
| `SAM2_QUANTIZE`, `SAM3_QUANTIZE` | SAM backends | working precision |
| `SAM2_THREADS`, `SAM3_THREADS`, `SPECTRA_MAX_THREADS` | `torch_threads.py` | torch thread count / clamp |
| `SAM2_CHANNELS_LAST`, `SAM3_CHANNELS_LAST` | SAM backends | channels-last format |
| `SAM2_COMPILE`, `SAM3_COMPILE` | SAM backends | `torch.compile` after load (GPU only) |
| `SAM3_IDLE_UNLOAD_TIMEOUT` | `manager.py` | seconds before an idle SAM3 unloads |
| `SPECTRA_MAX_HISTORY`, `SAM3_MAX_HISTORY`, `SPECTRA_MAX_OBJECTS`, `SPECTRA_MAX_SESSIONS` | `manager.py` | 200 / 100 / 50 / 100 |
| `SPECTRA_UI_LOG_LEVEL` | `main.py` | SSE log filter, default `INFO` |
| `SPECTRA_FLUSH_EVERY_REQUEST` | `main.py` | `torch.cuda.empty_cache()` after every request |
| `SPECTRA_OMP_THREADS` | `run.cmd` | becomes `OMP_NUM_THREADS` |
| `SPECTRA_LOG_MAX_MB`, `SPECTRA_LOG_BACKUPS` | `main.py` | log rotation, 10 MB / 3 backups |

Portability cache redirects (`HF_HOME`, `HF_HUB_CACHE`, `TORCH_HOME`, `TRITON_CACHE_DIR`,
`MPLCONFIGDIR`, `XDG_*`, `TEMP`/`TMP`, ...) are set in three places that must stay in
agreement: `start_amd.bat`, `backend\run.cmd`, and `main.py` at import time. The copy
in `main.py` is the safety net for a hand-started backend.

Saved but **never enforced**: `SPECTRA_VRAM_MODE`, `SPECTRA_CPU_THREADS` (launcher
echo only) and `SPECTRA_CPU_RAM_MB` (saved and shown, read by nothing).

---

## 9. What is allowed to differ between the forks

These files are **byte-identical** and must stay that way:
`security.py`, `compositing.py`, `torch_threads.py`, `stderr_progress.py`,
`model_integrity.py`, `net_check.py`, `offline.py`, `humantime.py`, `logctx.py`.

Files that legitimately differ:

| File | Allowed difference |
|---|---|
| `main.py` | AMD-only gfx/HIP preamble, MIOpen cache dirs, `torch` vs `_torch` aliasing, the `torch.version.hip` branch, `cudnn.benchmark=False` on HIP, `matmul.allow_fp16_reduced_precision_reduction` on HIP, `_log_gpu_mem()` around detect, the AMD no-GPU warning, and `backend_amd_gpu.*` import paths |
| `utils/device.py` | AMD adds `_detect_rocm()` and only claims NVIDIA when `torch.version.cuda` is set |
| `models/sam3_backend.py` | AMD also applies `cudnn_disabled` around the SAM3 encode and decode paths |
| `models/grounding_detector.py` | AMD already had the per-detect `empty_cache` removals and the 1024-square Florence pre-warm |
| `models/ultralytics_backend.py` | AMD already had the `RLock` and the locked `unload_model()` |
| `models/base.py`, `models/detector_base.py` | `...` vs `pass` in the abstract stubs |
| `models/manager.py` | import paths only |

Every other shared file must be byte-identical. To mirror a change:

1. Copy the edit across.
2. Swap `backend.` for `backend_amd_gpu.` in **absolute** imports; leave relative
   imports alone.
3. Run the section-3 verification for both forks.
4. Confirm only the differences in the table above remain.

---

## 10. Recipes

### Add a route

1. Add the request model in `main.py` next to the others, with validators that reject
   bad input before the manager.
2. Add the route with `@app.post(...)`, get the session through the manager (never
   touch a `Session` directly from the route), and run the work via the manager so it
   goes through `asyncio.to_thread`.
3. If it mutates a subject, include `object_history` in the response.
4. Mirror to the AMD fork (import-path swap only) and verify both.

### Add a segmentation model

1. Add its file name to the allowlist in `ultralytics_backend.py` (`ULTRALYTICS_MODELS`)
   or `sam3_backend.py`.
2. If it has a single-file checkpoint, add a pinned URL + sha256 digest to
   `utils/model_integrity.py`. `PINNED_MODEL_URLS` and `PUBLISHED_DIGESTS` must have
   the same key set; a key with no digest silently falls back to a trust-on-first-use
   branch that is otherwise unreachable.
3. If it is HuggingFace-hosted, confirm `is_downloaded` reads the HF cache and that
   the load goes through `offline_guard(True)`.

### Add a detector

1. Register it in `GroundingDetector` or `YOLEDetector` with its `tier` and `perf`
   strings (the model selector shows them).
2. Give it a pre-warm that matches its real input size. A dummy image that does not
   allocate the real shapes leaves the allocator and autotuner cold, and the first
   real detection can then OOM right after a "successful" load.
3. Verify `detect` end to end on the target device.

### Force offline / allow a download

Wrap cached loads in `offline_guard(True)`. Call with `local_files_only=False` to make
the guard permissive for a download. Never edit the read-only HF model cache.

---

## 11. Gotchas

- **`get_or_create_token()` mints a new token on every call.** It is safe only
  because `lifespan` calls it exactly once. A second call would invalidate every
  token already issued.
- **`_collect_all_masks` reads the id list under one lock and each `last_mask` under
  a second.** A `clear_object` landing between the two yields a partial result.
- **Export memory is unbounded.** `export-zip` builds the whole archive in a
  `BytesIO` and base64-encodes it into one JSON field.
- **Session memory is unbounded.** Worst case ~3.5 GB for 50 brush subjects, or
  ~6–13 GB for a 100-prompt SAM3 history at 30 instances each. `MAX_SESSIONS = 100`
  limits the count only.
- **`cpu_threads` saves/sets/restores a process-wide torch setting** and is not
  atomic; overlapping contexts can restore a value nobody asked for.
- **`cudnn_disabled` takes a module-level lock that spans both backends**, so it
  serialises all cuDNN-disabled inference process-wide.
- **The stderr/`torch.load` swaps use different locks in different modules**, and the
  detectors use none, so concurrent loads can leave the real stderr replaced.
- **`start.bat` / `stop.bat` kill any process whose command line contains `main:app`**,
  which includes the AMD backend.
- **`detect_amd_gpu.py` reports 0 MB VRAM for whole GPU families** (Strix Point,
  Strix Halo, Krackan, Navi 24, Rembrandt, Mendocino, Instinct). `setup_memory.ps1`
  then silently substitutes a fallback budget.
- **`sam3.pt` is fetched from a community mirror** with a digest that cannot be
  confirmed against any official source.

---

## 12. License obligations that touch the backend

`ultralytics` is **AGPL-3.0**. Because the backend links it, the combined,
distributed work is AGPL-3.0, and running it as a network service triggers AGPL-3.0
§13. The Meta SAM 3 weights are governed by the Meta SAM License, which is separate
from and in addition to the code license. See `THIRD-PARTY-LICENSES.txt` before
distributing anything.
