# Backend Guide

How to work on the Spectra Segment backend - **both forks at once**, the CPU/NVIDIA
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

The AMD fork adds two files with no CPU/CUDA counterpart: `amd_gfx.py` (the complete
Windows TheRock target registry and support levels) and `detect_amd_gpu.py` (the
installer probe that scores adapters and writes the hardware/launcher config before
torch is importable). `HSA_OVERRIDE_GFX_VERSION` is never derived automatically;
only an explicit `AMD_HSA_OVERRIDE` is applied. Everything else mirrors this map;
section 9 lists the files allowed to differ.

---

## 3. Running and verifying

Start the whole app (backend + frontend) with the launcher, not uvicorn directly,
because the launcher is what sets the memory profile and the ROCm environment:

```
start.bat          # CPU / NVIDIA, backend :8000, frontend :3000
start_amd.bat      # AMD ROCm
```

Verify a change - run all of it for both forks:

```
# syntax
backend_amd_gpu\.venv\Scripts\python.exe -m py_compile backend\main.py backend\models\manager.py
backend_amd_gpu\.venv\Scripts\python.exe -m py_compile backend_amd_gpu\main.py backend_amd_gpu\models\manager.py

# import (catches import-time and wiring errors a compile check misses)
set PYTHONPATH=<repo root>
backend_amd_gpu\.venv\Scripts\python.exe -X utf8 -c "import importlib; importlib.import_module('backend.main'); importlib.import_module('backend_amd_gpu.main')"

# GPU and catalogue checks after any wheel change
checks\check_gpu.bat
checks\check_gpu_amd.bat --all-models
checks\check_gpu_nvidia.bat
```

`check_gpu.bat` auto-selects AMD, NVIDIA or CPU. The explicit wrappers avoid that
choice on mixed machines. The checker applies the backend cache paths and the
8 GB matte cap, but it does **not** reproduce every launcher setting
(`PYTORCH_HIP_ALLOC_CONF`, generated per-card settings, attention toggles or the
hard cap). It therefore complements, rather than replaces, a launcher smoke test.
NVIDIA remains untested on physical hardware even when its code path runs.

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
        -> HostValidation      (host must be 127.0.0.1/localhost; a port, if sent, must be 8000)
          -> RateLimit         (300 req / 60 s / IP)
            -> RequestSizeLimit (50 MiB)
              -> TokenAuth      (token except health/token; SSE/release may use ?token=)
                -> ExceptionMiddleware
                  -> router
```

Auth is a `secrets.token_urlsafe(32)` generated once at startup, never written to
disk, and handed to the frontend by `GET /api/token`. Comparison runs on **bytes**,
because headers arrive latin-1 and a non-ASCII token used to raise `TypeError`.

Auth is required for every endpoint except `/api/token` and `/health`.
`EventSource` and `navigator.sendBeacon` cannot set request headers, so the two SSE
endpoints and `POST /api/sessions/{id}/release` also accept `?token=...`; no other
path accepts a query token. `/api/logs` itself requires the token header.

SSE keeps its own resource caps: each stream allows 5 connections per IP with a
1800 s maximum lifetime, and both share one global cap of 100 subscribers (503
when full). `/api/models/stream` also stops after the same lifetime.

Every handler that touches the manager runs on the threadpool through
`asyncio.to_thread`; the token route and the SSE streams stay on the event loop. The
rid/sid context is captured in `contextvars` and copied into the worker thread
automatically.

---

## 5. Endpoint reference

Auth column: **Y** = token required, **-** = exempt.

| Method | Path | Auth | Body / notes |
|---|---|---|---|
| GET | `/health` | - | liveness + active session count |
| GET | `/api/token` | - | bootstrap token |
| GET | `/api/logs` | Y | tail of the 300-entry ring |
| GET | `/api/logs/stream` | Y | SSE log stream; accepts `?token=` |
| GET | `/api/models/status` | Y | one-shot model + detector list |
| GET | `/api/models/stream` | Y | SSE model status, 3 s poll; accepts `?token=` |
| POST | `/api/sessions` | Y | `CreateSessionRequest` -> `CreateSessionResponse` |
| GET | `/api/sessions/{id}` | Y | session info |
| DELETE | `/api/sessions/{id}` | Y | destroy |
| POST | `/api/sessions/{id}/release` | Y | teardown from `sendBeacon`; accepts `?token=` |
| POST | `/api/sessions/{id}/image` | Y | upload, decode, resize, encode |
| POST | `/api/sessions/{id}/predict` | Y | `StrokeRequest` -> `PredictResponse` |
| POST | `/api/sessions/{id}/segment-batch` | Y | `SegmentBatchRequest` |
| POST | `/api/sessions/{id}/undo` \| `/redo` | Y | `?object_id=<int>&steps=<int>` -> `PredictResponse` (404 gone, 409 nothing to move) |
| POST | `/api/sessions/{id}/clear-object` | Y | `?object_id=<int>` -> `PredictResponse` |
| POST | `/api/sessions/{id}/sam3-prompt` | Y | `Sam3PromptRequest` -> `Sam3PromptResponse` |
| POST | `/api/sessions/{id}/sam3-undo` \| `/redo` | Y | -> `Sam3PromptResponse` |
| POST | `/api/sessions/{id}/sam3-remove-instance` | Y | `Sam3RemoveInstanceRequest` |
| POST | `/api/settings/sam3` | Y | `Sam3SettingsRequest` |
| GET | `/api/matte/models` | Y | matte model catalog + download/load state |
| POST | `/api/sessions/{id}/matte` | Y | `MatteRequest` -> `{alpha, width, height, model, guide, box, ms}` |
| POST | `/api/sessions/{id}/matte/clear` | Y | drop the stored alpha |
| GET | `/api/detectors` | Y | detector catalog + download/load state |
| POST | `/api/detectors/load` | Y | `LoadDetectorRequest` |
| POST | `/api/sessions/{id}/detect` | Y | `DetectRequest` |
| POST | `/api/sessions/{id}/export-zip` | Y | `ExportZipRequest` -> `{data, format, count}` (base64 zip) |
| POST | `/api/sessions/{id}/export-image` | Y | `ExportImageRequest` -> raw image bytes (`image/png` / `image/jpeg`) |
| POST | `/api/sessions/{id}/export-matte` | Y | `MatteExportRequest` -> raw cutout bytes |

### Request models

All of these are defined in `main.py` with `field_validator`s that reject
out-of-range input **before** the manager sees it.

```
CreateSessionRequest   model_name: str | None = None    (None = matte-only session,
                       otherwise must be in the allowlist)
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
ExportImageRequest     files: [{mask_b64}] (<=1000), format, background_color,
                       feather_radius
MatteRequest           model: str (must be in MATTE_MODELS),
                       guide: "auto"|"stroke"|"text"|"box",
                       points/labels/bbox (same bounds as StrokeRequest),
                       text: str (<=500), confidence: float | None,
                       max_detections: 1..4, crop_margin: 0.0..0.5,
                       stroke_radius: 1..400 | None,
                       edge_shift: -50..50, edge_feather: 0..50, invert: bool
MatteExportRequest     format: "png"|"jpg"|"jpeg", background_color: [r,g,b] | None
```

### Response models

```
CreateSessionResponse  session_id: str, model_name: str
PredictResponse        scores: [float], object_masks: {oid: b64 png},
                       object_history: {oid: {undo,redo,strokes,has_mask}}
Sam3PromptResponse     masks: [b64 png], scores: [float], bboxes: [[x1,y1,x2,y2]]
MatteResponse          alpha: b64 png, width, height, model, guide, box, ms
```

`object_masks` is the only place a brush response carries masks: each subject
appears once, keyed by object id, and an unchanged subject is served from its
cached PNG encode (`ObjectState.png_cache`, invalidated by `mask_revision`).
Mask data is not repeated in a second positional list. `masks` remains the
positional field of `SegmentBatch` and `Sam3PromptResponse`, which have no
object ids to key on.

### The history error contract

`/undo` and `/redo` distinguish two failures that used to share one status code:

- **404** - the session is gone. The client rebuilds it.
- **409** - the session exists but this subject has no strokes to pop (undo) or
  none to push back (redo). The client corrects that subject's counters and stops.

`steps` (default 1) pops or pushes up to that many entries and **clamps to what the
subject actually has**, so a burst of clicks or key repeats never turns into a 409:
one undo with `steps=5` on a three-stroke subject pops three and succeeds. A caller
that asks for more than exists gets the truthful end state, not an error.

`/predict` answers by the same rule: **404** only when the session is really gone,
and **400** for every other `ValueError` - no image loaded, object cap reached - with
the reason in the detail. It used to answer 400 for a missing session as well, and
since the client rebuilds the session on 404 it had been taught to rebuild on 400
too; a stroke refused by `SPECTRA_MAX_OBJECTS` therefore discarded the live session
and everything in it.

`object_history` on every mutating response is the authority: the client adopts it
instead of guessing, which is what makes repeated undo-failure loops impossible.
Only `object_history` keys are authoritative - a subject absent from it has been
cleared, so the client drops it rather than keeping stale counters.

---

## 6. `models/manager.py`

The single stateful part of the backend. It owns sessions, which models are loaded,
and each subject's mask history.

| Type | Holds |
|---|---|
| `Session` | id, model name, backend, image RGB/BGR, `objects`, SAM3 prompt history + redo stack, `has_image`, `last_active`, per-session lock |
| `ObjectState` | one **subject**: `stroke_history`, `redo_stack`, `last_mask`, `last_low_res_mask`, `mask_snapshots`, `mask_revision` + `png_cache`/`png_revision` (per-subject encode cache) |
| `MaskSnapshot` | low-res decoder logits for one stroke (or `None`); normally a fixed 256x256 float32 plane, so roughly 256 KB per stroke per subject, bounded by `SPECTRA_MAX_HISTORY`. SAM 3 returns no low-res logits, so its snapshots carry `None` and cost far less |
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
   left in the registry - removing it while it holds VRAM would leak.
2. **History.** A stroke's mask depends only on its own points and on the low-res
   mask of the stroke before it, and `mask_snapshots[i].low_res` is the low-res mask
   that stroke `i + 1` chained on. So:
   - `predict` -> `_replay_history` incremental: one decoder pass for the new stroke.
   - `undo`/`redo` -> `_restore_object_mask`: one decoder pass for the stroke that is
     now last (`None` when the history emptied). Cost is one pass regardless of how
     long the history is; the old path replayed every remaining stroke.

   The snapshot per stroke is what buys that: at 256 KB each, `SPECTRA_MAX_HISTORY`
   (200) caps a subject at about 50 MB and eight subjects at about 400 MB. A stroke
   sits in `stroke_history` **or** `redo_stack`, never both, so the two share that
   budget rather than doubling it. Caching full-resolution masks instead would make
   undo a copy, at 4 bytes per image pixel per stroke - an order of magnitude more
   than the low-res chain.
   - `_replay_history(full_replay=True)` remains as the correctness fallback, used
     when `mask_snapshots` is not index-aligned with `stroke_history`. It also
     differs in one detail: it feeds `mask_input` to a bbox stroke, while the
     incremental path deliberately passes `None` for bbox strokes (matching how the
     stroke was first computed). The single-step restore reproduces the incremental
     semantics exactly, which is the state the user was looking at.
3. **Lifetime.** `create_session` reserves its slot under the lock; `evict_idle_sessions`
   sweeps every 10 s and reclaims idle SAM3 models; `last_active` is refreshed by
   every working endpoint so an active session is not evicted mid-edit. A `None`
   model name skips `get_or_load_model`, so a matte-only session uploads and mats
   without a segmentation backend; predict, segment-batch, detect and SAM 3 answer 400
   until one is loaded.

### Adding a session-scoped feature

Do **not** keep per-subject counters on the client. Return `object_history` from the
mutating endpoint and let the client adopt it. That is the one rule that keeps undo
and redo correct across deletes, recoveries and id reuse.

---

### Matte state and residency

`Session.matte` is a `MatteState` (`model`, `alpha` as uint8, cached `alpha_png`,
`guide`, `box`) or `None`. The server keeps one latest alpha per session,
so a new run replaces the old one. The editor's multi-layer model and its per-layer
undo/redo live in the frontend (`lib/matte.ts`, `store/matte.ts`, `useMatte.ts`);
layer alphas are not stored server-side. `run_matte` resolves the ROI, calls the
backend, applies `edge_shift` / `edge_feather` / `invert` when the API asks for them,
and stores the result under `session.lock`. The UI sends those three as neutral and
applies them locally.

Matte models live in `_matte_backends` with `_current_matte`, separate from
`_loaded_models`. `_evict_loaded_models` evicts them too (it takes `keep_matte`), so
loading a matte model first unloads the segmentation model and detector, then attempts
the load. It applies the generated `SPECTRA_MATTE_INPUT_SIZE` cap (1024 px on cards below
12 GB) so heavy HR checkpoints stay below the card limit. Switching back to Brush or
Detect reloads SAM on demand; the matte model
stays until another load needs the VRAM or the last session is destroyed.

The first uncached load emits `matte-downloading` instead of `matte-loading`, then
`matte-loaded`; the header turns those into "Downloading matte model..." and refreshes
the matte model list when the load finishes. The matte backend routes stderr through
`StderrInterceptor` with `dl` metadata (name, phase, percent, file), so both the
simple activity feed and the Developer view show real download progress rows. The
console's Models section fetches `/api/models/status` and `/api/matte/models`, lists
matte checkpoints separately, labels heavy entries, and never disables one based on
GPU size.

Each matte layer stores its own model, so switching the model on one subject does not
change another. Every matte checkpoint is selectable and heavy entries are labelled
`Heavy` in the picker. Loading evicts the segmentation model and any detector first,
then attempts the load. The per-card input cap prevents an HR checkpoint from
requesting a 2048² activation on an 8 GB card; the full 2048² path is reserved for
cards with at least 12 GB. `list_matte_models` returns the catalog and loaded state only.

Text guidance calls `manager.detect()`, loading Grounding DINO Tiny when no detector
has ever been loaded. Detection unloads the segmentation model, which is already gone
in matte mode, so the call costs the detector load and nothing else.

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
| `matte_backend.py` | `BiRefNetBackend`, `MatteModelSpec` | BiRefNet soft-alpha matting |

Key facts to remember while editing them (full detail in `CODE_GUIDE.md` section 6):

- SAM 2 / SAM 3 wrappers do all their prompt work under `torch.inference_mode()`.
  Without it, every prompt builds a live autograd graph over the prompt encoder and
  decoder; with it, that memory is never allocated. Keep it.
- Precision is decided at runtime by `_model_dtype()` / `*_QUANTIZE`; the AMD fork
  sets `SPECTRA_DETECTOR_FP16=1` in `run.cmd`, while the CUDA fork sets it from the
  detected capability (on for Volta+, off for Maxwell/Pascal).
- `torch.load` is patched to default `weights_only=True` in `UltralyticsBackend` and
  in `SAM3Backend` - that is the guard against arbitrary code execution from a
  malicious checkpoint. Keep it, and restore the original in `finally`.
- Florence-2 / CogFlorence need `_enable_florence_generation` on `transformers`
  >= 4.50: those releases removed `GenerationMixin` from `PreTrainedModel` while the
  repos' remote code still relies on inheriting it. The loader re-bases the model and
  gives its language model a valid generation config.
- Florence returns no per-box score, so `_detect_florence` hardcodes `score = 1.0`;
  the confidence slider cannot filter Florence results.
- `_nms` prefers `supervision` when importable, otherwise a hand-written greedy pass.
- `matte_backend.py` runs the full image when no ROI is supplied. On a large image
  with ROIs, and when the card cap is below the model's native input size, it runs
  `_crop_matte` directly so the selected subject gets the available detail. On
  smaller images, or at native HR input, it keeps the full-image alpha and filters it
  with `_roi_component_mask`. Crops still use context margins to avoid empty alpha
  inside uniform objects.
- Grounding DINO and Florence snapshots are downloaded at the commit revisions in
  `HF_MODEL_REVISIONS`. The revision is passed to `HfApi.model_info`,
  `snapshot_download`, `hf_hub_download` and the direct `from_pretrained` fallback;
  cached detector loads use `local_files_only=True` and the pinned snapshot directory.
- BiRefNet loads with `trust_remote_code=True` at a pinned commit revision per model.
  Never drop the revision: the remote code would then change under an installed copy.
  Cached matte snapshots load with `local_files_only=True`; only a missing or unreadable
  snapshot falls back to a pinned download. These weights are not in
  `PUBLISHED_DIGESTS`; the Hub's own file hashes apply.

---

## 8. Environment variables

Read by the backend (defaults come from the launcher's memory profile):

| Variable | Read by | Effect |
|---|---|---|
| `SPECTRA_FORCE_CPU` | `utils/device.py` | ignore all GPUs |
| `SPECTRA_VRAM_HARD_CAP` | `main.py` | `set_per_process_memory_fraction`, default 0.90; applied only when CUDA is used |
| `SPECTRA_TF32` | `utils/cuda_profile.py` | set to `0` to disable TF32 on Ampere and newer |
| `SPECTRA_MATTE_FP16` | `matte_backend.py` | fp16 autocast for BiRefNet on Volta+ (default on); off on Maxwell/Pascal; CPU stays fp32 |
| `SPECTRA_CPU_RAM_MB` | `main.py` | process memory hard cap applied at startup |
| `SPECTRA_KEEP_SEG_ON_DETECT` | `manager.py` | keep the SAM model resident during detection |
| `SPECTRA_NO_CO_RESIDENCY` | `manager.py` | unload the segmentation model when a detector loads |
| `SPECTRA_DETECTOR_FP16` | `grounding_detector.py` | fp16 detectors on Volta+ (default on); off on Maxwell/Pascal. Ignored on CPU |
| `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` | `utils/offline.py` | forced on while a cached HF model loads |
| `YOLO_OFFLINE` | `main.py` | set to `1` before ultralytics loads, so it skips its DNS probe and PyPI update check |
| `SAM2_OFFLOAD_ENCODER` | `ultralytics_backend.py` | park the SAM2 encoder on CPU between images |
| `SAM2_QUANTIZE`, `SAM3_QUANTIZE` | SAM backends | working precision; fp32 on CPU and Maxwell/Pascal, fp16 on Volta+ |
| `SAM2_THREADS`, `SAM3_THREADS`, `SPECTRA_MAX_THREADS` | `torch_threads.py` | torch thread count / clamp |
| `SAM2_CHANNELS_LAST`, `SAM3_CHANNELS_LAST` | SAM backends | channels-last format |
| `SAM2_COMPILE`, `SAM3_COMPILE` | SAM backends | Optional `torch.compile` after load. Default Inductor needs Triton, which the Windows ROCm wheel does not ship; the guarded path logs the failure and keeps eager execution. |
| `SAM3_IDLE_UNLOAD_TIMEOUT` | `manager.py` | seconds before an idle SAM3 unloads |
| `SPECTRA_MAX_HISTORY`, `SAM3_MAX_HISTORY`, `SPECTRA_MAX_OBJECTS`, `SPECTRA_MAX_SESSIONS` | `manager.py` | 200 / 100 / 50 / 100 |
| `SPECTRA_UI_LOG_LEVEL` | `main.py` | SSE log filter, default `INFO` |
| `SPECTRA_BACKEND_PORT` | `security.py` | port the Host check expects, default `8000` |
| `SPECTRA_FLUSH_EVERY_REQUEST` | `main.py` | `torch.cuda.empty_cache()` after every request |
| `SPECTRA_OMP_THREADS` | `run.cmd` | becomes `OMP_NUM_THREADS` |
| `SPECTRA_LOG_FILE` | `main.py` | explicit path for this launch; default is a new timestamped file |
| `SPECTRA_LOG_MAX_MB`, `SPECTRA_LOG_BACKUPS` | `main.py` | rotation safeguard, 10 MB / 3 backups |
| `SPECTRA_ALLOW_CPU_FALLBACK` | both backends | automatic CPU retry on placement/OOM, default enabled; `0` disables |
| `SPECTRA_MATTE_INPUT_SIZE` | both backends | BiRefNet input cap, 1024 below 12 GB / 2048 at 12 GB+ on NVIDIA |

On the CUDA fork, `utils/cuda_profile.py` fills only unset precision variables before
model imports: fp32 on Maxwell/Pascal, fp16 on Volta and newer, and TF32 on Ampere and
newer (`SPECTRA_TF32=0` opts out). It also checks `torch.cuda.get_arch_list()` against
the detected compute capability and falls back to CPU when the wheel has no kernel for
the card. The AMD fork keeps its own `SPECTRA_DETECTOR_FP16=1` and does not use this
module. See [`docs/NVIDIA.md`](NVIDIA.md) for the full matrix.

Both forks enable automatic CPU fallback on model placement failure or OOM unless
`SPECTRA_ALLOW_CPU_FALLBACK` is set to `0`/`false`/`no`/`off`/`disabled`. The retry is
per model, not a silent permanent device switch, and explicit CPU selections still work.
Each launch writes a new `logs/backend_<YYYYmmdd_HHMMSS>.log`; `SPECTRA_LOG_FILE`
overrides that path.

Portability cache redirects (`HF_HOME`, `HF_HUB_CACHE`, `TORCH_HOME`, `TRITON_CACHE_DIR`,
`MPLCONFIGDIR`, `XDG_*`, `TEMP`/`TMP`, ...) are set in three places that must stay in
agreement: `start_amd.bat`, `backend\run.cmd`, and `main.py` at import time. The copy
in `main.py` is the safety net for a hand-started backend.

VRAM eviction uses actual free memory, not a per-profile budget: 2048 MB of headroom
before an encode and twice the estimated weights plus 512 MB before a load. The hard
caps are `SPECTRA_VRAM_HARD_CAP` and `SPECTRA_CPU_RAM_MB`.

`SPECTRA_VRAM_MODE` and `SPECTRA_CPU_THREADS` are launcher echo only.
`SPECTRA_CPU_RAM_MB` is enforced by `main.py` at startup; a rejected limit is logged
as a warning.

---

## 9. What is allowed to differ between the forks

These files are **byte-identical** and must stay that way:
`utils/security.py`, `utils/compositing.py`, `utils/torch_threads.py`,
`utils/stderr_progress.py`, `utils/model_integrity.py`, `utils/net_check.py`,
`utils/offline.py`, `utils/humantime.py`, `utils/logctx.py`,
`models/matte_backend.py`,
`utils/__init__.py`, `models/__init__.py`, `models/ultralytics_backend.py` and
`requirements.txt`.

Files that legitimately differ:

| File | Allowed difference |
|---|---|
| `main.py` | AMD-only gfx/HIP preamble, MIOpen cache dirs, `torch` vs `_torch` aliasing, the `torch.version.hip` branch, `cudnn.benchmark=False` on HIP, `matmul.allow_fp16_reduced_precision_reduction` on HIP, `_log_gpu_mem()` around detect, the AMD no-GPU warning, and `backend_amd_gpu.*` import paths |
| `utils/device.py` | AMD adds `_detect_rocm()`. **Both** forks only claim NVIDIA when `torch.version.cuda` is set, so a ROCm build is never mislabelled on the CUDA/CPU side either |
| `models/sam3_backend.py` | AMD also wraps the SAM3 **encode** in `cudnn_disabled` (both forks already wrap the decode paths), and its missing-`clip` hint names `install_amd.bat`. Nothing else: model resolution and download live in `utils/model_integrity.ensure_model_file()` for both forks |
| `models/grounding_detector.py` | One absolute import path (the `net_check` import inside the download path). The per-detect `empty_cache`, the OOM traceback clearing before it, and the 1024-square Florence pre-warm are in both forks |
| `models/yoloe_detector.py` | Three absolute import paths (`net_check`, `model_integrity` twice) |
| `models/manager.py` | Three absolute import paths (`net_check`, `model_integrity` twice) |
| `models/base.py`, `models/detector_base.py` | `...` vs `pass` in the abstract stubs |
| `run.cmd` | AMD adds the MIOpen env vars and cache dirs, `PYTORCH_TUNABLEOP_CACHE_DIR`, `OCL_SET_SVM_SIZE`, `TORCH_BLAS_PREFER_HIPBLASLT`/`DISABLE_ADDMM_CUDA_LT`, `SPECTRA_DETECTOR_FP16=1` (all only when unset), and runs `backend_amd_gpu.main:app` |
| `amd_gfx.py` | AMD only; no CPU/CUDA counterpart |
| `detect_amd_gpu.py` | AMD only; no CPU/CUDA counterpart |

Every other shared file must be byte-identical. To mirror a change:

1. Copy the edit across.
2. Swap `backend.` for `backend_amd_gpu.` in **absolute** imports; leave relative
   imports alone.
3. Run the section-3 checks for both forks.
4. Confirm only the differences in the table above remain.

The duplication the forks carry is deliberate: each tree is a self-contained
deployment that its own installer and venv own, so nothing is imported across
the `backend/` boundary. The only close call inside a tree is
`_fallback_cpu_and_reencode` in `sam3_backend.py` versus `_fallback_full_cpu` in
`ultralytics_backend.py` - eight near-identical lines that stay separate because
the two classes reload and re-encode differently.

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
   Resolution and download are shared: call `ensure_model_file(model_path, model_dir)`
   and it returns the absolute path, downloading (digest-checked) only when the file
   is missing, and raising a message that names `model_dir` when the machine is
   offline.
3. If it is HuggingFace-hosted, confirm `hf_weights_downloaded` reads the HF cache and
   that the load goes through `offline_guard(True)`.

### Add a matte model

1. Add a `MatteModelSpec` to `MATTE_MODELS` in `models/matte_backend.py` with the
   repo id, **pinned commit revision**, input size, tier, perf string and weight
   estimate. Put a clear Heavy/Medium/Light label in `perf`; do not add a VRAM
   compatibility gate or a per-model hardware limit.
2. Confirm `AutoModelForImageSegmentation.from_pretrained(repo_id, revision=...,
   trust_remote_code=True)` loads it.
3. Mirror `matte_backend.py` to the other fork; it is byte-identical.
4. Run it once on the target card and record the warm time and peak VRAM in
   `docs/modes/MATTE.md`.

### Add a detector

1. Register it in `GroundingDetector` or `YOLEDetector` with its `tier` and `perf`
   strings (the model selector shows them).
2. Give it a pre-warm that matches its real input size. A dummy image that does not
   allocate the real shapes leaves the allocator and autotuner cold, and the first
   real detection can then OOM right after a "successful" load.
3. Verify `detect` end to end on the target device.

### Force offline / allow a download

Wrap cached detector loads in `offline_guard(True)`. Call with
`local_files_only=False` to make the guard permissive for a download. Matte uses
`local_files_only=True` for a cached snapshot and only falls back to a pinned download
when that snapshot cannot be loaded. Never edit the read-only HF model cache.

---

## 11. Gotchas

- **`get_or_create_token()` mints a new token on every call.** It is safe only
  because `lifespan` calls it exactly once. A second call would invalidate every
  token already issued.
- **`_object_masks_png` reads the object list under the session lock but encodes
  outside it**, on purpose: a full-size PNG encode must not stall stroke appends.
  The list therefore reflects the moment of the read, and a mask computed during the
  encode is picked up by the next response.
- **Export memory is unbounded for ZIP.** `export-zip` builds the whole archive in a
  `BytesIO` and base64-encodes it into one JSON field. The direct `export-image`
  path returns one encoded image and is the preferred path for multi-layer work.
  `format` applies to every ZIP entry and extensions are rewritten to match; JPEG
  without `background_color` uses white because it cannot store alpha.
- **Session memory is unbounded.** Worst case ~3.5 GB for 50 brush subjects, or
  ~6-13 GB for a 100-prompt SAM3 history at 30 instances each. `MAX_SESSIONS = 100`
  limits the count only.
- **`cpu_threads` saves/sets/restores a process-wide torch setting** and is not
  atomic; overlapping contexts can restore a value nobody asked for.
- **`cudnn_disabled` takes a module-level lock that spans both backends**, so it
  serialises all cuDNN-disabled inference process-wide.
- **The stderr/`torch.load` swaps use different locks in different modules**, and the
  detectors use none, so concurrent loads can leave the real stderr replaced.
- **`start.bat` / `stop.bat` kill any process whose command line contains `main:app`**,
  which includes the AMD backend.
- **Integrated GPUs have no dedicated VRAM figure.** The detector reports 0 MB for
  iGPUs and selects a conservative low/balanced profile from the target registry;
  it does not claim a dedicated VRAM size that does not exist.
- **`sam3.pt` is fetched from a community mirror** with a digest that cannot be
  confirmed against any official source.

---

## 12. License obligations that touch the backend

`ultralytics` is **AGPL-3.0**. Because the backend links it, the combined,
distributed work is AGPL-3.0, and running it as a network service triggers AGPL-3.0
section 13. The Meta SAM 3 weights are governed by the Meta SAM License, which is separate
from and in addition to the code license.See `THIRD-PARTY-LICENSES.txt` before distributing anything.

The BiRefNet code and the ten catalogued matte checkpoints are MIT-licensed, which is
why matte mode can bundle them. All ten checkpoints have measured performance on
the tested RX 6600. SAM2Matting was evaluated and rejected: its weights are
CC BY-NC 4.0 and its requirements pin NVIDIA-only `torch-tensorrt`. The full
reasoning is in `docs/modes/MATTE.md` section 9.
