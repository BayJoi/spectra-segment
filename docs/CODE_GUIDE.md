# Spectra Segment — Code Guide

A single, complete reference for the project: how it is shaped, how it starts, the
HTTP layer, the model manager, the model backends, the utilities, the model
catalogue, the frontend, the known limitations, how to verify a change and how to
keep the two forks in sync.

This file exists so the source stays free of prose. Anything that would otherwise
be a comment in the code lives here instead. The companion files are
`docs\BACKEND_GUIDE.md` (working on the backend), `docs\FRONTEND.md` (browser side)
and `docs\LOGGING.md` (logging).

---


## 1. Shape of the project

Windows only. Portable. Local first — nothing leaves the machine. Three editing
modes on top of one FastAPI backend:

| Mode | What it does |
|---|---|
| **Brush** | Draw strokes around what to keep. Multiple subjects, each its own mask |
| **Detection** | Type what to find. The detector finds boxes, then a model masks them |
| **SAM 3** | Type a prompt. One model does detection, segmentation and masking in one step |

The backend exists in **two forks**:

| | `backend/` | `backend_amd_gpu/` |
|---|---|---|
| Hardware | CPU / NVIDIA CUDA | AMD ROCm (TheRock wheels) |
| Virtualenv / package cache | `backend\.venv`, `backend\.uv` | `backend_amd_gpu\.venv`, `.uv_cache_amd` |
| Model weights | `backend\model_weights` | `backend_amd_gpu\model_weights` |
| Frontend workspace | `backend\web` | `backend_amd_gpu\web` |
| Launcher | `start.bat` | `start_amd.bat` |

Both forks share the same API surface, the same request/response models, and the
same `models/manager.py` logic. They differ only in hardware-specific loading and
tuning. **Every change to `backend/` must be mirrored to `backend_amd_gpu/`** —
see part 2, section 12.

### Directory map

```
install.bat / install_amd.bat   fetch uv, Python, Bun; create the venv; install wheels
start.bat / start_amd.bat       GPU detect -> memory profile -> backend (8000) -> frontend (3000)
stop*.bat                       stop processes only
cleanup*.bat                    remove venv, caches, workspace, toolchain
launcher\setup_memory.ps1       memory profile -> env vars written to <cfg>.env
checks\check_gpu_health.py      GPU smoke test (run after any wheel change)
backend\                        FastAPI app (CUDA/CPU fork)
backend_amd_gpu\                FastAPI app (ROCm fork)
frontend\                       React 19 + Vite + UnoCSS + jotai
tools\                          portable uv / Python / Bun, installed by the installers
```

---

## 2. Startup chain

### Installers

`install.bat` and `install_amd.bat` both:

1. Point `UV_CACHE_DIR`, `UV_PYTHON_CACHE_DIR` and `UV_PYTHON_INSTALL_DIR` inside
   the repo, set `UV_NO_CONFIG=1`, and redirect `TEMP`/`TMP` into the repo.
2. Resolve **uv** through the GitHub API (latest stable release), download the zip
   plus its `.sha256`, recompute the hash, and fail on mismatch.
   `install.bat` pins Python `3.13.14`. `install_amd.bat` runs
   `uv python install 3.12.13`.
3. Create the virtualenv with `uv venv --python <ver> --seed`, install torch, then
   install `requirements.txt`.

AMD only: torch comes from TheRock's **nightly** multi-arch index,
`https://rocm.nightlies.amd.com/whl-multi-arch/`, with the `torch[device-gfxNNNN]`
extra. The `gfxNNNN` target comes from `detect_amd_gpu.py`, from
`AMD_GFX_OVERRIDE`, or from a small hard-coded PCI-ID table. `rocm-sdk-devel` is
deliberately skipped — it is only needed to compile HIP code, and it adds about
1.4 GB.

Nothing except the Python version is pinned. That means installs are not
reproducible, and it is the largest supply-chain exposure in the project.

### Launchers

`start_amd.bat` step 2 sets the ROCm environment before anything else loads:

| Variable | Purpose |
|---|---|
| `PYTORCH_HIP_ALLOC_CONF`, `PYTORCH_CUDA_ALLOC_CONF` | Allocator block budget, set by the memory profile |
| `MIOPEN_DEBUG_DISABLE_FIND_DB`, `MIOPEN_USER_DB_PATH`, `MIOPEN_CUSTOM_CACHE_DIR` | Keep the MIOpen kernel database under `model_weights\miopen` |
| `HSA_ENABLE_SDMA=0` | Works around DMA copy stalls on Windows |
| `TORCH_BLAS_PREFER_HIPBLASLT=1`, `DISABLE_ADDMM_CUDA_LT=1` | Prefer hipBLASLt for speed, but its addmm path crashes on some RDNA2 wheels |
| `SPECTRA_DETECTOR_FP16=1` | AMD default. The CUDA fork does not set it |
| `MIMALLOC_PURGE_DELAY=0` | Return memory eagerly |
| `TORCH_BACKENDS_CUDA_*_SDP_ENABLED` | Math attention on RDNA2 and older, AOTriton elsewhere |

`start.bat` sets none of these. It only checks for NVIDIA with
`Win32_VideoController | ? Name -match 'NVIDIA'`, and passes `-ForceCpu` if there
is none.

Both launchers then call `launcher\setup_memory.ps1`, which is the only thing that
decides memory policy.

### `launcher\setup_memory.ps1`

Profiles: `high`, `balanced`, `medium`, `low`, `cpu`. They are read and written in
`<backend>\spectra_launcher.cfg` as `vram_mode` / `cpu_threads` / `cpu_ram_mb`,
then written as `KEY=VALUE` lines to `<cfg>.env`, which the `.bat` file imports.

| Profile | VRAM fraction | Idle unload | Encoder offload | No co-residency |
|---|---|---|---|---|
| `high` | 0.90 | 86400 s | 0 | 0 |
| `balanced` | 0.60 | 300 s | 0 | 0 |
| `medium` | 0.50 | 180 s | 0 | 0 |
| `low` | 0.30 | 120 s | 1 | 1 |
| `cpu` | 0 | 300 s | 0 | 0 |

A separate hard ceiling of **0.90** is applied through
`torch.cuda.set_per_process_memory_fraction()`. Because of that, the process can
never use more than 90% of the user's VRAM. For the `high` profile the soft
fraction and the hard ceiling are the same number, so they are identical there.

`max_split_size_mb` works out to `vram x fraction`, floored at 256 MB. If VRAM
cannot be detected, the fallbacks 8192 / 4096 / 2048 / 1024 MB are used and
nothing warns the user.

Two things to know:

- `cpu_threads` is only asked for and emitted in `cpu` mode. Going GPU, then CPU,
  then GPU loses the thread setting.
- Fractions are written with an invariant decimal separator. On a comma-decimal
  locale they used to come out as `0,9`, which Python's `float()` rejects. Both
  the budget and the hard cap were then silently switched off.

`SPECTRA_CPU_RAM_MB` is asked for and saved to the config, but **no backend code
reads it**.

### Frontend serving

There is no production serving path. The launchers always run the **Vite dev
server** on `127.0.0.1:3000`. `frontend\use_web.bat` junctions
`frontend\node_modules` to `<web>\node_modules` and `frontend\dist` to
`<web>\dist`, so each backend's frontend dependencies stay separately cleanable.
Nothing ever reads `bun run build` output. See `docs\FRONTEND.md` for the details.

---

## 3. Environment variable reference

### Read by the backend

| Variable | Read by | Effect |
|---|---|---|
| `SPECTRA_FORCE_CPU` | `utils/device.py` | Ignore all GPUs |
| `SPECTRA_VRAM_FRACTION` | `models/manager.py` | Soft VRAM budget before proactive eviction |
| `SPECTRA_VRAM_HARD_CAP` | `main.py` | `set_per_process_memory_fraction`, default 0.90 |
| `SPECTRA_KEEP_SEG_ON_DETECT` | `models/manager.py` | Keep the SAM model loaded during detection |
| `SPECTRA_NO_CO_RESIDENCY` | `models/manager.py` | Unload the segmentation model when a detector loads |
| `SPECTRA_DETECTOR_FP16` | `models/grounding_detector.py` | fp16 detectors on GPU. Ignored on CPU |
| `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` | `utils/offline.py` | Forced to `1` while a cached HuggingFace model loads; cleared only for a download that is actually needed |
| `SAM2_OFFLOAD_ENCODER` | `models/ultralytics_backend.py` | Park the SAM2 image encoder on CPU between images |
| `SAM2_QUANTIZE`, `SAM3_QUANTIZE` | both SAM backends | Working precision. Default fp32 on CPU, fp16 on GPU |
| `SAM2_THREADS`, `SAM3_THREADS` | `utils/torch_threads.py` | Torch thread count, clamped to `SPECTRA_MAX_THREADS` (32) |
| `SAM2_CHANNELS_LAST`, `SAM3_CHANNELS_LAST` | both SAM backends | Channels-last memory format |
| `SAM2_COMPILE`, `SAM3_COMPILE` | both SAM backends | `torch.compile` after load. GPU only |
| `SAM3_IDLE_UNLOAD_TIMEOUT` | `models/manager.py` | Seconds before an idle SAM3 model unloads |
| `SPECTRA_MAX_HISTORY`, `SAM3_MAX_HISTORY`, `SPECTRA_MAX_OBJECTS`, `SPECTRA_MAX_SESSIONS` | `models/manager.py` | 200 / 100 / 50 / 100 |
| `SPECTRA_UI_LOG_LEVEL` | `main.py` | SSE log filter, default `INFO` |
| `SPECTRA_FLUSH_EVERY_REQUEST` | `main.py` | `torch.cuda.empty_cache()` after every request |
| `SPECTRA_OMP_THREADS` | `backend\run.cmd` | Becomes `OMP_NUM_THREADS` |
| `SPECTRA_MAX_THREADS` | `utils/torch_threads.py` | Upper clamp for the `*_THREADS` variables, default 32 |
| `SPECTRA_LOG_MAX_MB`, `SPECTRA_LOG_BACKUPS` | `main.py` | Log rotation, default 10 MB and 3 backups |

### Set for portability

These are all redirected into `model_weights\` so the app stays self-contained:

`HF_HOME`, `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE`,
`HF_MODULES_CACHE`, `HF_XET_CACHE`, `HF_DATASETS_CACHE`,
`SENTENCE_TRANSFORMERS_HOME`, `TORCH_HOME`, `TRITON_CACHE_DIR`, `NUMBA_CACHE_DIR`,
`ULTRALYTICS_HOME`, `YOLO_CONFIG_DIR`, `MPLCONFIGDIR`, `XDG_CACHE_HOME`,
`XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `TMPDIR` / `TEMP` / `TMP`, plus
`PYTORCH_TUNABLEOP_CACHE_DIR` and `PYTORCH_HIP_ALLOC_CONF` on the AMD side.

The same block is applied in three places: `start_amd.bat`, `backend\run.cmd`, and
`main.py` at import time. They have to stay in agreement. The copy in `main.py` is
the safety net for when the backend is started by hand.

### Set but never read

`SPECTRA_VRAM_MODE` and `SPECTRA_CPU_THREADS` are launcher echo only.
`SPECTRA_CPU_RAM_MB` is saved and displayed but never enforced.

---

## 4. Backend — the HTTP layer

`main.py` builds the app, picks the device, and creates the `ModelManager`.

Middleware is registered in the order below. Starlette inserts each new middleware
at position 0, so **the last one registered ends up outermost**:

```
ServerErrorMiddleware
  -> CORSMiddleware
    -> RequestLoggingMiddleware
      -> SecurityHeadersMiddleware
        -> HostValidationMiddleware        (stops DNS rebinding)
          -> RateLimitMiddleware           (300 requests / 60 s / IP)
            -> RequestSizeLimitMiddleware  (50 MiB)
              -> TokenAuthMiddleware       (X-Local-Token header)
                -> ExceptionMiddleware
                  -> router
```

`HostValidationMiddleware` requires the host to be `127.0.0.1` or `localhost`
**and** the port to be 8000. That combination is what makes DNS rebinding useless
against this app. `CORSMiddleware` sits outside everything, so preflight `OPTIONS`
requests skip the whole inner stack including host validation. They carry no body,
so nothing is disclosed.

The token is `secrets.token_urlsafe(32)`. It is generated once during startup, is
never written to disk, and the frontend fetches it from the auth-exempt
`/api/token`. The comparison runs on **bytes**, because Starlette decodes headers
as latin-1 and a non-ASCII `X-Local-Token` used to raise `TypeError` — producing a
500 and a traceback instead of a clean 401.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness plus active session count. No auth needed |
| GET | `/api/token` | Bootstrap token. No auth needed |
| GET | `/api/logs` | Tail of the 300-entry in-memory log ring |
| GET | `/api/logs/stream` | SSE log stream. No auth needed |
| GET | `/api/models/status` | One-shot model and detector list |
| GET | `/api/models/stream` | SSE model status, polled every 3 s. No auth needed |
| POST | `/api/sessions` | Create a session for a model |
| GET | `/api/sessions/{id}` | Session info |
| DELETE | `/api/sessions/{id}` | Destroy a session |
| POST | `/api/sessions/{id}/release` | Teardown from `sendBeacon`. No auth needed |
| POST | `/api/sessions/{id}/image` | Upload, decode, resize, encode |
| POST | `/api/sessions/{id}/predict` | Stroke or box prompt, returns a mask |
| POST | `/api/sessions/{id}/segment-batch` | One mask per box, positions preserved |
| POST | `/api/sessions/{id}/undo` or `/redo` | Move through stroke history |
| POST | `/api/sessions/{id}/clear-object` | Drop one subject |
| POST | `/api/sessions/{id}/sam3-prompt` | Text prompt for SAM3 |
| POST | `/api/sessions/{id}/sam3-undo` or `/redo` | Move through prompt history |
| POST | `/api/sessions/{id}/sam3-remove-instance` | Drop one instance |
| POST | `/api/settings/sam3` | Keep-loaded flag and encode dimension |
| GET | `/api/detectors` | Detector list with download and load state |
| POST | `/api/detectors/load` | Load a detector |
| POST | `/api/sessions/{id}/detect` | Open-vocabulary detection |
| POST | `/api/sessions/{id}/export-zip` | Build an archive of composited images |
| POST | `/api/sessions/{id}/export-image` | Build a single composited image |

### Error contract for the history endpoints

`/undo` and `/redo` now tell apart two cases that used to share one 404:

- **404** — the session is gone. The client should rebuild it.
- **409** — the session is there, but this subject has no strokes. The client
  should resynchronise its counters and stop retrying.

Reading the server's ledger instead of guessing is what makes the loop of repeated
undo failures impossible.

### Image upload

`_decode_image` looks at the file header first. It rejects anything above 32
megapixels or with a side above 32768 pixels. It then applies
`ImageOps.exif_transpose` **before** resizing, so photo orientation is baked in and
the mask coordinates line up with the pixels. It downscales to at most 4096x4096
and converts to BGR for OpenCV. The cv2 path is only reached if Pillow raises
during the probe.

`_safe_arcname` is a real zip-slip check. It normalises backslashes first, then
`normpath` collapses `..`, and it rejects absolute paths, anything starting with
`..`, and anything containing a colon. That last rule also rejects legitimate
filenames with a colon in them, such as timestamps.

### SSE bookkeeping

The log ring is a `deque(maxlen=300)` written to from any thread. Fan-out uses
`loop.call_soon_threadsafe(q.put_nowait, entry)`, which is the correct way to touch
an `asyncio.Queue` from a worker thread. Both queues are bounded at 128 and 64, and
drops are silent by design.

The SSE endpoints cannot use auth (`EventSource` cannot send headers), so they
carry a per-IP cap of 5, a global cap of 100, and a maximum connection lifetime of
1800 seconds. Without those, any local client could hold every slot open forever.

See `docs\LOGGING.md` for the full logging story.

---

## 5. Backend — `models/manager.py`

This is the single stateful part of the backend, and the most intricate file in the
project. It owns sessions, which models are loaded, and each subject's mask
history.

### Data model

| Type | Holds |
|---|---|
| `Session` | Session id, model name, backend, image RGB and BGR, `objects`, the SAM3 prompt history and redo stack, `has_image`, `last_active`, and a per-session lock |
| `ObjectState` | One **subject**: `stroke_history`, `redo_stack`, `last_mask`, `last_low_res_mask`, `mask_snapshots` |
| `MaskSnapshot` | The low-resolution decoder logits for one stroke, or `None` |
| `StrokeEntry` | `points`, `labels`, `bboxes` |
| `SAM3PromptEntry` | The prompt text and its confidence |
| `SAM3Snapshot` | Packed masks, scores, boxes and the mask shape |

### Threading

| Lock | Guards |
|---|---|
| `self._lock` | The session table, `_loaded_models`, `_detector`, and the `_last_*_use` timestamps. Short critical sections only |
| `session.lock` | One session's `objects` dict and each `ObjectState` |
| `backend._infer_lock` | Inference on one model. `UltralyticsBackend` uses an `RLock`, because `unload_model()` takes it and then calls `reset_image()`, which takes it again. `SAM3Backend` uses a plain `Lock` |
| `_model_load_locks[name]` | One per model name, so two requests for *different* models cannot both end up loaded in VRAM |

Every manager entry point runs on the threadpool through `asyncio.to_thread`, and a
background loop sweeps for idle sessions every 10 seconds. The concurrency is real,
not theoretical.

### The server owns the history

The client used to keep its own per-subject undo counter. It drifted whenever a
subject was deleted mid-flight, a session was recovered with fresh server state, or
an object id was reused after a delete. The result was repeated `404 Nothing to
undo` responses, and the client retried each one in a loop.

Now every mutating response carries `object_history`, keyed by subject id, holding
the real `undo` / `redo` / `strokes` / `has_mask` depth. The client adopts it
directly. `_object_history()` reads the real lists under `session.lock`.

### Subject identity

A **subject** is an `ObjectState` addressed by an integer object id. The UI keeps
`brushObjectsAtom` — the ordered ids — and `subjectMetaAtom` — the name and colour
for each id.

Ids only ever increase. That matters because `clear_object` deletes the
`ObjectState` on the server, so reusing an id would make a re-added subject inherit
a deleted one's stroke history.

`subjectColor(id)` picks a stable entry from a colour palette. The same colour
appears in the subjects panel, the toolbar chip, the brush preview and the mask
overlay. That shared colour is what makes "which subject is which" answerable
everywhere at once.

Deleting a subject calls `clear_object`, removes the id from `brushObjects`,
deletes its metadata, and moves `activeObjectId` to a neighbouring id.

### Undo and redo

`_replay_history(session, obj, full_replay=...)` has two modes:

- **Incremental**, used by `predict`. It applies only the newest stroke, chaining
  on the cached low-resolution mask. That is one decoder pass, and it is the live
  drawing path.
- **Full replay**, used by `undo` and `redo`. It starts from a clean state and
  re-applies every stroke in order, so each stroke runs exactly once from the state
  that existed before it. The image encoder is already warm, so the extra cost is
  one cheap decoder pass per stroke.

The old code only ever replayed the *last* stroke while setting
`last_low_res_mask` to the *previous* stroke's output. That applied a stroke twice
and pushed the remaining first stroke onto a different code path, so undo never
restored the mask you actually had.

`mask_snapshots` runs parallel to `stroke_history` and has to stay index-aligned.
`redo` therefore appends a placeholder `MaskSnapshot(low_res=None)` when the saved
one is missing, so the two lists cannot drift apart.

### Model residency

| Piece | What it does |
|---|---|
| `_evict_loaded_models(keep_backend)` | Unloads the detector and every segmentation model except `keep_backend`. A model with inference in flight is **left in the registry**, because removing it while it still holds VRAM would leak. `_last_detector_name` is kept, so the detector can reload itself |
| `_can_unload(obj)` | Probes the inference lock in one step. This is the safe check behind every unload decision |
| `_safe_unload(obj)` | Wraps `_can_unload` and calls `unload_model()` only when it is safe |
| `_ensure_vram_headroom(required_mb, keep_backend)` | Evicts early when the projected footprint would cross the profile's budget. It takes `keep_backend`, which is how `SPECTRA_KEEP_SEG_ON_DETECT` gets honoured instead of defeated |
| `get_or_load_model` | Validates the model name, serialises loads per name, and leaves `_loaded_models` consistent when a load fails |
| `create_session` | Reserves its slot under the lock, so N concurrent creations cannot all pass the `MAX_SESSIONS` check |
| `evict_idle_sessions` | Re-reads the timestamps under the lock, uses `_can_unload`, and only unloads idle **SAM3** models. An idle non-SAM3 model is reclaimed by `destroy_session` or by an encode-headroom eviction |

### Session lifetime

`last_active` is refreshed by `get_session`, `load_image`, `predict`, `undo`,
`redo`, `clear_object`, `segment_batch`, `detect` and `sam3_predict`. It used to be
touched by only two of those, so a session you were actively working on could be
evicted at `IDLE_TIMEOUT = 600` seconds and your work destroyed.

`_ensure_model_loaded` is the self-heal path. It reloads a backend that was evicted
or re-encodes when the cached features were dropped.

`load_image` records `session.has_image` on both success and failure. It used to
return HTTP 200 with nothing encoded.

### Detection flow

`load_detector` validates the name **before** it tears anything down. It builds
the new detector into local variables and only commits on success. A failed load
therefore leaves the previous detector loaded and selectable, instead of leaving a
dead one reported as current. It honours `SPECTRA_KEEP_SEG_ON_DETECT` by passing
`keep_backend=self._loaded_models.get(self._current_model)`.

`detect` unloads the segmentation model through `_can_unload` and `_safe_unload`, so
a `predict` running on another session is not killed in the middle of a forward
pass.

### Memory accounting

`_pack_masks` and `_unpack_masks` shrink SAM3 masks by a factor of 8. The SAM2 path
stores plain float32 and bool arrays.

Worst case per session is about 3.5 GB for 50 brush subjects, or about 6 to 13 GB
for a SAM3 history of 100 prompts with 30 instances each. Nothing tracks total
session bytes, so `MAX_SESSIONS = 100` limits the count only.

---


## 6. Backend — the model backends

| File | Class | Notes |
|---|---|---|
| `models/base.py` | `SegmentationBackend` | Abstract: `load_model`, `unload_model`, `set_image`, `predict`, `reset_image`, `is_loaded`, `has_image` |
| `models/detector_base.py` | `DetectorBackend`, `Detection` | Abstract: `load_model`, `unload_model`, `detect`, `is_loaded` |
| `models/ultralytics_backend.py` | `UltralyticsBackend` | SAM 2 / Hiera, through `SAM2Predictor` |
| `models/sam3_backend.py` | `SAM3Backend` | SAM 3 semantic text prompts |
| `models/grounding_detector.py` | `GroundingDetector` | Grounding DINO and Florence-2 |
| `models/yoloe_detector.py` | `YOLEDetector` | YOLOE with a MobileCLIP text encoder |

`base.predict` defaults `multimask_output` to `True`. `UltralyticsBackend`
overrides it to `False`, and `SAM3Backend` leaves it at `True`. The manager always
passes it explicitly, so the mismatch is dormant.

### UltralyticsBackend (SAM 2)

`DENYLIST` blocks `FastSAM-*`, `sam3_b.pt` and `sam3_l.pt`. `ULTRALYTICS_MODELS` is
the allowlist. Loading happens under a module-level lock, with `torch.load`
patched to default `weights_only=True` — that is protection against arbitrary code
execution from a malicious checkpoint — and with `sys.stderr` swapped for
`StderrInterceptor`. Any failure other than a missing `clip` package on a GPU
device falls back to rebuilding on CPU. That fallback is too broad: a corrupt
checkpoint or a plain bug also gets treated as "GPU unavailable".

`_model_dtype()` works out the working precision at runtime, and every prompt
tensor is built with it. `_offload_encoder` and `_restore_encoder` implement
`SAM2_OFFLOAD_ENCODE`. `_image_pe`, the dense positional encoding, is cached per
image and read under the inference lock.

`predict` letterboxes the image into 1024x1024. It turns a box into two corner
points with labels 2 and 3, and feeds the previous low-resolution mask in as
`mask_input`. It returns the **uncropped** 256x256 logits as `low_res_masks`. Those
logits are in the same coordinate space the encoder expects, so refinement stays
consistent.

`predict_batch` stacks all boxes into one decoder call. If anything goes wrong it
falls back to one call per box, which hides the original error.

### SAM3Backend

`DEFAULT_ENCODE_DIM = 1024`, `ENCODE_DIM_MAX = 1500`, `MAX_ASPECT_RATIO = 1.6`.

`_prepare_input` shrinks the image so its longest side matches the encode size. It
pads it to a square only when the aspect ratio is above 1.6. It records
`_orig_shape`, `_proc_shape`, `_content_shape`, `_pad_top`, `_pad_left` and
`_pad_scale`. `_restore_outputs` crops the content band, resizes back, and
thresholds at 0.5.

`predict_text` is the main path. It reads the features cached by `set_image`, calls
`inference_features(features, self._proc_shape or self._src_shape, text=[text])`
inside `cpu_threads` + `cudnn_disabled` + `torch.inference_mode()`, and
`_format_result` filters by confidence. `predict` is the visual-prompt path and
raises `NotImplementedError` for point prompts.

The `inference_mode()` call matters. Without it, every prompt built a live autograd
graph over the prompt encoder, the mask decoder and the full-resolution masks.
That was hundreds of megabytes of avoidable VRAM per prompt, freed only when
`_format_result` detached.

`_build_predictor` forces `predictor.imgsz = [encode_dim, encode_dim]` and can
optionally apply `SAM3_CHANNELS_LAST` and `SAM3_COMPILE`. Both of those swallow
their errors and keep the working model.

### GroundingDetector

Seven detectors: Grounding DINO tiny, Grounding DINO base, `mm-gdino-base-all`,
`mm-gdino-large-all`, `florence-2-base`, `florence-2-large`, and
`cogflorence-2.2-large`.

Loading tries a local snapshot first, then `snapshot_download`, then a direct
`from_pretrained`. `_clean_florence_config` removes conflicting `auto_map` entries,
and `trust_remote_code` is gated behind a trusted-id check.

Florence pre-warms at **1024x1024**, because its vision encoder runs at 768x768. A
64x64 dummy image never allocated the real shapes, so the first real detection
paid for kernel autotuning and could run out of memory right after a "successful"
load. Grounding DINO pre-warms at **800x800**, with autocast **on** when the model
is fp16. It used to always warm in fp32, which left the fp16 allocator cold.

`_enable_florence_generation` is what keeps Florence-2 and CogFlorence loadable on
current `transformers` (>= 4.50). Those releases dropped `GenerationMixin` from
`PreTrainedModel`, but the repos' remote modeling code predates the change: the
outer `Florence2ForConditionalGeneration.generate()` merges image and text
embeddings and then delegates to `self.language_model.generate(...)`, and the
language model relied on inheriting `generate`. Without help the delegated call
raises `AttributeError` and every detection returns 500. The loader re-bases the
loaded model and its `language_model` onto subclasses that also inherit
`GenerationMixin`, and gives the language model a generation config built from its
own config (which carries `decoder_start_token_id` and `bos_token_id`) overlaid
with the outer repo's settings. Copying the outer `generation_config` straight
across does **not** work — it has no special-token ids and `generate()` then dies
with "`decoder_start_token_id` or `bos_token_id` has to be defined". The patch is
idempotent and is applied again at the top of `_detect_florence`, so a detector
loaded before the patch existed still gets fixed.

Florence does not report a confidence per box, so `_detect_florence` assigns every
box `score = 1.0` and ignores its `confidence` argument. The confidence slider in
the UI therefore has **no effect** on Florence results. Grounding DINO and YOLOE
do report real scores.

`_nms` uses `supervision` when it is importable — class-agnostic, IoU 0.5 — and
otherwise a hand-written greedy pass. That fallback stops as soon as it has enough
boxes instead of finishing suppression, so the kept set differs depending on
whether `supervision` imported.

`SPECTRA_DETECTOR_FP16` only applies on GPU. On CPU the model loads in fp32 and the
flag is ignored. **The AMD fork sets it to 1 by default and the CUDA fork does not
set it at all** — that is a real difference in behaviour between the forks.

### YOLEDetector

Seven YOLOE variants. Each one also needs the matching MobileCLIP text encoder:
`mobileclip2_b.ts` for `yoloe-26*`, `mobileclip_blt.ts` for `yoloe-11*`.

It permanently patches `ultralytics.nn.text_model.build_text_model` to force local
encoder paths, and it changes the working directory to the weights folder during
load. Neither change is undone afterwards. It reads no environment variables and
uses no explicit `inference_mode`, relying on ultralytics internals.

Its default confidence floor is **0.05** when the caller passes nothing. Grounding
DINO's library default is 0.1. So detection density varies a lot depending on which
detector you pick.

---

## 7. Backend — the utilities

| Module | Responsibility |
|---|---|
| `utils/model_integrity.py` | Pinned URLs and published sha256 digests, download, verify, TOFU |
| `utils/security.py` | The six middlewares and the auth token |
| `utils/device.py` | Device detection and `release_gpu_memory` |
| `utils/compositing.py` | Mask to PNG, feathering, alpha compositing, encoding |
| `utils/torch_threads.py` | Thread resolution, `cpu_threads`, `cudnn_disabled` |
| `utils/stderr_progress.py` | `StderrInterceptor`, captures HuggingFace download progress |
| `utils/net_check.py` | `is_connected()`, a pre-flight connectivity probe |
| `utils/offline.py` | `offline_guard()` / `is_offline()`, forces HF offline mode for cached loads |
| `utils/humantime.py` | `format_duration` |

### `model_integrity.py`

There are 18 entries. Each has both a pinned URL and a published digest. 17 of them
point at the `ultralytics/assets` release `v8.4.0`, and those digests were checked
character for character against the upstream release page.

The exception is `sam3.pt`, at `huggingface.co/bodhicitta/sam3`. That is a
community mirror, and its digest cannot be confirmed from any official source —
yet `verify_model` treats it as absolute truth. A single wrong constant there
permanently breaks SAM 3 for every user.

The download itself is atomic. It downloads to `<name>.part`, verifies the hash
**before** publishing, then removes the old file and renames. A per-model-name lock
with the existence check **inside** it means two sessions racing for the same
missing model line up correctly.

`is_connected()` may run first, but it probes `huggingface.co` and `github.com`.
Neither of those is where the weights actually come from, so a proxy that allows
one and blocks the other produces a misleading "no internet connection" error.

There is no resume, no retry, no disk-space check, and no `fsync` before the
rename. The TOFU branch of `verify_model` — remember the hash of whatever file is
already on disk when no digest is published — is unreachable today, because every
name has a digest. If a name ever loses its digest, a hand-dropped file becomes the
trust anchor and the real file is then rejected forever. `.hashes.json` is written
without a lock and without an atomic rename.

### `compositing.py`

`mask_to_png_b64` enforces the dtype rules instead of trusting callers: bool
becomes 0 or 255, floats are normalised (handling both 0..1 and 0..255), and
anything else is clamped. The cv2 return value is checked rather than assumed.

`feather_mask_edge` pads with zeros before blurring so the image border does not
darken, and always returns a float in 0..1. `composite_background` is ordinary
straight-alpha "over" compositing. The RGBA handed to PIL is not premultiplied,
which is what PNG expects.

### `torch_threads.py`

`resolve_threads` prefers the environment variable but clamps it to
`SPECTRA_MAX_THREADS`, which is 32.

`cpu_threads` saves, sets and restores `torch.set_num_threads`. That setting is
process-wide and not atomic, so overlapping contexts can restore a value nobody
asked for. The two `load_model` call sites are also guarded by a different lock
than the inference ones.

`cudnn_disabled` takes a module-level lock that spans both backends, which
serialises all cuDNN-disabled inference across the whole process.

### `stderr_progress.py`

`StderrInterceptor` is installed directly onto `sys.stderr` by four backends. So it
implements enough of the text-stream protocol to survive whatever a library pokes
at it:

| Member | Behaviour |
|---|---|
| `write` | Returns the number of characters written |
| `writelines` | Writes each line in turn |
| `readable` | False |
| `isatty` | False |
| `encoding`, `errors` | `utf-8`, `replace` |
| `fileno` | Raises `OSError` |
| `seek`, `tell` | `0`, `0` |
| `seekable` | False |
| `close` | Flushes the buffer |

Returning False from `isatty` makes libraries skip ANSI and progress rendering.

The `sys.stderr` and `torch.load` swaps are guarded by **different** module-local
locks in `sam3_backend` and `ultralytics_backend`, and by no lock at all in the
detectors. Concurrent loads can therefore leave the real stderr permanently
replaced.

### `offline.py`

`offline_guard(offline)` forces or restores HuggingFace offline mode for the
duration of a block. It flips `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` in the
environment **and** patches `huggingface_hub.constants.HF_HUB_OFFLINE` and
`transformers.utils.hub._is_offline_mode`, because both libraries read those at
call time instead of re-reading the environment.

Every cached HuggingFace load (`AutoProcessor` / `AutoModelForCausalLM` /
`AutoModelForZeroShotObjectDetection` with `local_files_only=True`) runs inside
`offline_guard(True)`, so a downloaded model can never contact the Hub again. The
download paths call with `local_files_only=False`, which makes the guard
permissive for that call only. `is_offline()` reports the current state. The module
re-uses the same `_lock` for every toggle, so overlapping guards cannot restore a
stale value.

---

## 8. Model catalogue

Segmentation models: `sam2.1_{t,s,b,l}.pt`, `sam2_{t,s,b,l}.pt`, `sam3.pt`.
Detectors: 7 Grounding DINO / Florence models and 7 YOLOE models. Each has a
`tier` and a `perf` string that the model selector shows.

Whether a model is downloaded is read from the HuggingFace cache for grounding
models, and from `model_weights\<name>.pt` for YOLOE.

`list_detectors` reports `loaded` from `_current_detector`. So a detector that the
idle sweep unloaded still reads as loaded if the name was not cleared.

---

## 9. Frontend architecture

React 19, Vite, UnoCSS and jotai. There is no router — `App.tsx` switches between
`EmptyState`, `Canvas` (which covers Brush and Detection) and `Sam3Page` based on
`sam3ModeAtom`.

Full detail lives in `docs\FRONTEND.md`. The short version:

### Build config

`vite.config.ts` runs on port 3000. `server.fs.allow` covers `import.meta.dirname`,
`../frontend`, `../backend/web` and `../backend_amd_gpu\web` — that list is what
makes the junction indirection work. It also sets a dev CSP, `X-Frame-Options:
DENY`, `X-Content-Type-Options: nosniff` and `Referrer-Policy: no-referrer`.

`server.host: "localhost"` in the config is dead. Both launchers override it with
`--host 127.0.0.1`. `connect-src` allows both `127.0.0.1:8000` and
`localhost:8000`, which matches `BASE`. There is no proxy — the frontend talks to
the backend directly.

### State

Five jotai stores on the default module store. There is no custom `Provider`, so
every `useAtom` call in every component shares one store.

| Store | Atoms |
|---|---|
| `session.ts` | `sessionIdAtom`, `modelNameAtom`, `imageUrlAtom`, `imageWidthAtom`, `imageHeightAtom`, `imageFileAtom`, `masksAtom`, `objectMasksAtom`, `perDetectionMasksAtom`, `brushObjectsAtom`, `activeObjectIdAtom`, `objectUndoCountsAtom`, `objectRedoCountsAtom`, `objectHistoryAtom`, `subjectMetaAtom`, `brushPredictInFlightAtom`, `modelsAtom` |
| `ui.ts` | Dialog and overlay flags, the console state, toasts, imperative callback atoms |
| `detection.ts` | `detectModeAtom`, `toolModeAtom`, `brushSizeAtom`, `featherRadiusAtom`, `detectorsAtom`, `selectedDetectorAtom`, `loadedDetectorAtom`, `detectorLoadingAtom`, `detectQueryAtom`, `detectionsAtom`, `selectedDetectionAtom`, `isDetectingAtom`, `yoloeMasksEnabledAtom` |
| `layers.ts` | `layersAtom`, `selectedLayersAtom`, and the `nextLayerId()` counter that mints every layer id |
| `sam3.ts` | `sam3ModeAtom`, `sam3PromptsAtom`, `sam3RedoStackAtom`, `sam3PromptInputAtom`, `sam3PromptingAtom`, `sam3InstancesAtom`, `selectedSam3InstanceAtom`, `sam3KeepLoadedAtom`, `sam3EncodeDimAtom` |

Four atoms persist to `localStorage` through `atomWithStorage`: `consoleFilterAtom`,
`consoleSourceAtom`, `consoleCategoryAtom` and `consoleShowTimeAtom`.

`masksAtom` carries three meanings at once. In brush mode it is the union of every
subject's mask. In SAM3 mode it is the flat list of every instance. It is also the
"is there anything to export or show transparently" check. `objectMasksAtom` is the
per-subject breakdown, and it is what the per-subject colour overlay is built from.
`perDetectionMasksAtom` is keyed by **detection array index**, which is why
detection indices have to stay stable.

### `useSession` and its module-level guards

`useSession()` is instantiated in **eight** components: App, Canvas, EmptyState,
Header, ModeSwitchDialog, EndSessionDialog, SubjectsPanel and Toolbar. A ref
created inside the hook belongs to one instance, so it is not a guard. These four
are module-level instead:

| Guard | What it does |
|---|---|
| `uploadGuardRef` | One upload at a time |
| `endGuardRef` | One session teardown at a time |
| `blobUrlRef` | The single live object URL, revoked exactly once. The old per-instance ref leaked a blob URL on every mode switch and every End Session |
| `predictQueueMap` | One promise chain per session id, which serialises predict calls |

`applyHistory(history)` is the only writer of the undo and redo counters. It adopts
the server's `object_history` as-is, and it also creates subject metadata for any
new id.

### Undo and redo flow

`predict` calls `applyHistory(res.objectHistory)`. `undo` and `redo` do the same on
success. On **409** the local counter is zeroed and a toast is shown — no session
teardown, no retry. On **404** the session is recovered. `clearObjectHistory`
follows the same pattern.

### Canvas pipeline

Four stacked canvases: `imageCanvas`, `maskCanvas`, `detectCanvas`, `drawCanvas`.

`computeState` returns CSS-pixel layout: `baseScale = min(cw/iw, ch/ih, 1)`,
`scale = baseScale x zoom`, and integer offsets centred in the container. The wheel
handler reproduces that rounding exactly so the cursor stays anchored.

`redraw` sizes every backing store in **device pixels** — `clientWidth x dpr` — and
scales each context by the DPR. Coordinates stay in CSS space, but the rendering is
crisp on high-density displays. There was no `devicePixelRatio` handling at all
before, so everything was soft on any HiDPI screen.

Mask rendering prefers the per-subject overlay. `rebuildSubjectMaskCache` builds
one offscreen canvas per subject, blurs it if feathering is on, and tints it with
that subject's colour using `source-in`. They are drawn back to front, with the
active subject last. It falls back to the single union overlay when there are no
per-subject masks. Transparent mode builds a separate composite using
`destination-in`.

`blurMaskCanvas` calls `blurPadded`, which pads by the radius so the border does
not darken, then either uses `ctx.filter = blur(...)` or a three-pass separable box
blur. The box-blur fallback works on **interleaved RGBA float** data. The old version
allocated a single-channel destination while being handed a `w x h x 4` plane. So the
horizontal pass only saw the first quarter of the image, and the rest came out as
`NaN`. Most of the feather ended up fully transparent.

The rebuild cache key includes the mask ids, the active subject and the feather
radius. `Canvas.tsx` calls `invalidateCache()` whenever the relevant arrays change
identity.

### Brush drawing

`useBrushDrawing` captures points in image space with a 2-pixel minimum spacing and
clamps them to the image bounds. Right-drag sets `activeStrokeModeRef` to
`"negative"`. The preview uses the active subject's colour for positive strokes and
red for negative ones.

`handlePointerUp` is bound to both `onPointerUp` and `onPointerLeave`. Left drags
have pointer capture, so leaving the canvas keeps collecting points. Right drags
have none, so a right-drag that leaves the canvas commits in the middle of the
gesture.

`buildStrokePrompt` classifies the stroke:

| Case | Result |
|---|---|
| Fewer than 3 points | One point, label +1 or 0 |
| Negative | Always an area prompt |
| Positive and closed | A ring prompt — a donut of up to 8 interior positive points plus the ring as negative |
| Positive and open | An area prompt |

`buildAreaPrompt` rasterises into an offscreen canvas at most 1024 px wide. It
deliberately **narrows the stroke by 60% of the brush on each side**, so the
sampled points sit inside the object rather than straddling the boundary. It then
samples a grid and caps the result at 12 points.

### Detection mode

`useDetection.detect` **appends** to `detections` instead of replacing them. Before,
every query cleared both the detections and the masks, so searching for a second
object made the first one disappear. New detections land at the end of the array,
which keeps the mask keys stable. `handleDetect` uses `syncDetectionLayers` for the
first batch and `appendDetectionLayers(dets, offset)` after that.

`segment-batch` returns one entry per box, with `null` for an empty mask. The
backend used to drop the empty entries, which shifted every later mask onto the
wrong detection. Indexing by position is only correct now because positions are
preserved.

`addLayer` makes ids from a module-level counter that is also advanced past any id
already in the store. The old version derived the id from a closure value of
`layerIdCounter`. That meant every `addLayer` call inside one synchronous loop
produced the **same** id. So `selectedLayers` addressed them all at once, and
`removeLayer(id)` deleted them all.

Removing a detection has to reindex `perDetectionMasks`, `detections`,
`layers[].detectionIndex` and `selectedDetection` together. That logic used to be
copy-pasted into `Canvas.removeDetectionAt` and `useLayers.removeLayer`, which is
exactly how two copies drift apart. It now lives in one place,
`frontend\src\lib\detectionEdit.ts` (`removeDetectionMask`,
`removeDetectionIndexFromLayers`, `shiftSelectedDetection`), and both call sites
use it.

### SAM3 mode

`useSam3.prompt` appends `{text, masks, scores, bboxes}` and clears the redo stack.
`syncFromPrompts` is the single reconciliation point: it flattens `prompts[].masks`
into `masksAtom` and builds `sam3InstancesAtom` as
`{promptIndex, instanceIndex, text, bbox, score}`. `reconcileLayers` reuses layers
by the composite key `${promptIndex}:${instanceIndex}` and takes ids from a local
counter, so it is immune to the duplicate-id bug.

`api.sam3Undo` and `api.sam3Redo` throw away the masks the server returns, so the
**local** prompt stack is the authority for the mask pixels. After a
`recoverSession` — which recreates an empty server history — the two diverge.

`removeInstance` drops any prompt whose mask list becomes empty. The backend does
the same, which means every later `prompt_index` shifts down. The client's cached
indices go stale, and there is no way for it to notice.

### Export

`ExportDialog` encodes every mask to a base64 PNG synchronously on the main thread
— one `w x h` loop plus a `toDataURL` per mask — and posts it to `/export-zip`. 20
masks at 4096x4096 is a multi-second freeze. `/export-image` exists but is never
called.

`feather` is bound straight to `featherRadiusAtom`, so the export slider changes
the on-screen feathering live. Brush mask lookup uses
`objectMasks[f.brushOid] ?? masks[f.brushOid]`, and that fallback is only correct
while object ids are contiguous `0..n-1`.

### ConsolePanel

The panel consumes `/api/logs/stream` through `EventSource`, and it also
subscribes to the frontend's own log store so both render in one list with a
`source: "ui" | "backend"` tag. It buffers entries that arrive before the initial
`/api/logs` fetch settles.

`progressRows` folds the `DOWNLOAD` entries into a `Map<name, Row>`. Filtering is
by level, by source, and by category, with live counts per category. Search matches
message, level, category, source and name.

`MAX_LOG_ENTRIES` is 500, and 150 of those slots are reserved for UI entries so a
large model download cannot evict every action the user just took. Only the newest
120 rows render, with a "show older" button for the rest. See `docs\LOGGING.md`.

There is no `AbortController` anywhere, and `imageEncodingAtom` drives a
full-screen overlay with no way out if a request hangs.

### `api.ts`

`BASE = "http://localhost:8000"`, hardcoded. `ApiError` carries the HTTP status, so
callers can tell 404 (recover) from 409 (resync).

The token is memoised in `_token` and `_tokenPromise`. A 401 clears both and
retries exactly once with a fresh token.

`releaseSession` fires `sendBeacon` — which cannot set headers, so the backend
exempts that path from auth — **and** a token-bearing `DELETE`. A page teardown
therefore sends two destroy requests. Both are idempotent.

---

## 10. Known limitations still open

Ranked, with the reason each one is still open.

1. **Nothing is pinned except Python.** The AMD path installs torch from a nightly
   index. `clip` is installed from a mutable GitHub `main` branch with no checksum.
   The `uv` download in the same script is sha256-verified.
2. **`sam3.pt` comes from an unvetted community mirror** under a digest nobody can
   confirm, and `PUBLISHED_DIGESTS` carries no record of where it came from. A
   re-upload under the same name breaks every existing install.
3. ~~`request._body` replay is almost certainly a no-op.~~ **Resolved / not a bug.**
   This was a suspicion, not a finding. It was tested against the installed Starlette
   (1.7.0): a chunked request with no `Content-Length` is fully readable by the
   endpoint, and an oversized chunked request is correctly rejected with 413, so the
   `request._body` replay in `RequestSizeLimitMiddleware` does work. The initial
   `Content-Length` check covers the common case; the buffered path is the fallback
   for chunked bodies.
4. **Export memory is unbounded.** `export-zip` builds the whole archive in a
   `BytesIO` and base64-encodes it into one JSON field. A 50 MB request can push
   RAM into the gigabyte range, with no cap on the output side.
5. **Session memory is unbounded.** Worst case is about 6-13 GB per session times
   100 sessions, tracked by count only.
6. **`_collect_all_masks` snapshots the id list under `session.lock`**, but each
   object's `last_mask` is read in a second critical section. A `clear_object`
   landing between the two still yields a partial result.
7. **One Escape press can close several UI layers at once.** App, ConsolePanel,
   LayersPanel, SubjectsPanel and Select each register their own listener.
8. **`start.bat` and `stop.bat` interfere across forks.** `start.bat` kills any
   process whose command line contains `main:app`, which includes the AMD backend.
9. **`install_amd.bat` still installs from a nightly index**, and there is no
   rollback path.
10. **`detect_amd_gpu.py` reports 0 MB of VRAM for whole GPU families** — Strix
    Point, Strix Halo, Krackan, Navi 24, Rembrandt, Mendocino and all Instinct.
    `setup_memory.ps1` then quietly substitutes a fallback budget with no warning.
11. **`get_or_create_token()` makes a fresh token on every call.** It is safe only
    because it is called exactly once. A second call would invalidate every token
    already issued.
12. **`SPECTRA_CPU_RAM_MB`, `SPECTRA_VRAM_MODE` and `SPECTRA_CPU_THREADS` are saved
    and shown but never enforced.**
13. **`accelerate`, `timm` and `einops` are declared but never imported by our
    code.** They are not dead: `accelerate` backs transformers'
    `low_cpu_mem_usage`, and `timm` / `einops` are pulled in by the Florence-2
    DaViT remote code. `supervision` has exactly one optional call site
    (`GroundingDetector._nms`).
14. **`torch` is a transitive dependency** in both `requirements.txt` files, even
    though the code uses `torch.OutOfMemoryError` at import time.
15. **Florence-2 / CogFlorence ignore the confidence slider.** Those models return
    no per-box score, so `_detect_florence` hardcodes `score = 1.0`; the threshold
    the user sets cannot filter its boxes.

---

## 11. Verifying a change

Backend, both forks:

```
& "backend_amd_gpu\.venv\Scripts\python.exe" -m py_compile backend\main.py backend\models\manager.py
& "backend_amd_gpu\.venv\Scripts\python.exe" -m py_compile backend_amd_gpu\main.py backend_amd_gpu\models\manager.py
$env:PYTHONPATH = "D:\Spectra Group\Spectra Segment"
& "backend_amd_gpu\.venv\Scripts\python.exe" -X utf8 -c "import importlib,sys; sys.path.insert(0,'D:/Spectra Group/Spectra Segment'); importlib.import_module('backend.main'); importlib.import_module('backend_amd_gpu.main')"
```

Frontend:

```
cd frontend
& "D:\Spectra Group\Spectra Segment\tools\bun-amd\bun.exe" x tsc --noEmit
& "D:\Spectra Group\Spectra Segment\tools\bun-amd\bun.exe" run build
```

GPU health, after any wheel change:

```
checks\check_gpu.bat
```

That script does not reproduce the launchers' environment —
`PYTORCH_HIP_ALLOC_CONF`, `HSA_OVERRIDE_GFX_VERSION`, `HSA_ENABLE_SDMA`, the
attention toggles and the hard cap are all absent. So a regression caused by a
launcher-set variable cannot be caught by it.

---

## 12. Keeping the two forks in sync

These utility files are **byte-identical** between the forks and must stay that
way: `security.py`, `compositing.py`, `torch_threads.py`, `stderr_progress.py`,
`model_integrity.py`, `net_check.py`, `offline.py`, `humantime.py`, and `logctx.py`.

Files that legitimately differ, and how:

| File | Allowed difference |
|---|---|
| `main.py` | The AMD-only gfx and HIP preamble, `miopen` cache directories, `torch` versus `_torch` aliasing, the `torch.version.hip` branch, `cudnn.benchmark=False` on HIP, `matmul.allow_fp16_reduced_precision_reduction` on HIP, `_log_gpu_mem()` around detect, the AMD no-GPU warning, and `backend_amd_gpu.*` import paths |
| `utils/device.py` | AMD adds `_detect_rocm()` and only claims NVIDIA when `torch.version.cuda` is set |
| `models/sam3_backend.py` | AMD imports and applies `cudnn_disabled` in the SAM3 encode and decode paths |
| `models/grounding_detector.py` | AMD already had the per-detect `empty_cache` removals and the 1024-square Florence pre-warm |
| `models/ultralytics_backend.py` | AMD already had the `RLock` and the locked `unload_model()` |
| `models/base.py`, `models/detector_base.py` | `...` versus `pass` in the abstract stubs |
| `models/manager.py` | Import paths only |

Every other shared file must be byte-identical, with no exceptions.

How to mirror a change:

1. Copy the change across.
2. Swap `backend.` for `backend_amd_gpu.` in absolute imports. Leave relative
   imports alone.
3. Run the verification in section 11 for **both** forks.
4. Check that only the differences in the table above remain.
