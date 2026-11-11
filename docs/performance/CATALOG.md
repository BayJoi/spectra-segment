# Performance catalogue

> **Research catalogue, not a current implementation report.** These are proposed
experiments from 2026-10-26. Some may since have been implemented; verify the code
before treating a "Current" line as present behavior. The measured results that do
exist are labelled with a date.

Sections 4-14 of the performance notes: orchestration, SAM 3, SAM 2, CUDA,
ROCm, CPU, post-processing, serving, model loading and alternatives.

## 4. Catalogue: orchestration, caching, and residency

### A1. Cache SAM 3 text-encoder outputs

- **Change:** memoize `model.backbone.forward_text(text)` keyed by the exact prompt
  string (and, if it ever varies, the tokenizer context length). The official
  `VETextEncoder.forward` accepts an already-encoded tuple
  `(text_attention_mask, text_memory_resized, tokenized)` and explicitly documents
  "the text is already encoded, use as is". A small wrapper around
  `forward_text` can return the cached tuple.
- **Why:** `SAM3SemanticPredictor.inference_features()` encodes text on every
  prompt. The encoder is 24 layers × 1024 width × 16 heads with a 32-token context.
  Repeating the same prompt—after undo/redo, after switching subject, or after a
  failed request—pays that cost again even though the image features are cached.
- **Where:** `backend/models/sam3_backend.py:253-296` calls
  `inference_features(features, ..., text=[text])`.
- **Platforms:** CUDA, ROCm, CPU.
- **Tier/effort:** T1 / S.
- **Expected:** removes one text-encoder pass per repeated prompt. Likely small on a
  fast GPU and material on CPU; measure the text phase separately.
- **Verify:** run the same prompt twice with and without the cache; assert the mask
  arrays, scores, and boxes are bit-identical. Check that the cache is cleared on
  model unload and bounded in size.
- **Risks:** prompt normalization (whitespace/case) changes the key; a stale cache
  after a model swap. Clear on `unload_model()` and on tokenizer changes.

### A2. Reuse image features for repeat uploads

- **Change:** key image features by `(sha256(image), model, encode_dim, device)` and
  reuse them when the same image is set again. At minimum, avoid re-encoding when
  the backend already holds features for the same image.
- **Why:** the encoder is the dominant cost for a new image (SAM 3 external
  reference: 78% of per-class latency; DART 87/112 ms on an RTX 4080). Reloading a
  page or re-uploading the same file re-encodes from scratch.
- **Where:** `backend/models/manager.py:402-444` always calls `set_image()`;
  `_ensure_model_loaded()` re-encodes after an eviction.
- **Platforms:** all.
- **Tier/effort:** T1 / M.
- **Expected:** large for repeat uploads, zero for a genuinely new image.
- **Verify:** time `set_image` cold vs cached; assert masks identical.
- **Risks and caveats:** the backend keeps **one** feature set per model, not per
  session (`SAM3Backend._image_set` / `_src_shape`, `UltralyticsBackend._src_shape`).
  The manager shares a backend instance across sessions. Do not add a cross-session
  cache until session/feature ownership is verified; otherwise a cache could return
  features for the wrong image. This is a correctness prerequisite, not a
  performance detail.

### A3. Keep models resident; evict only when VRAM actually requires it

- **Change:** make `_ensure_encode_headroom()` a no-op unless free VRAM is below a
  threshold (or the new model genuinely cannot fit). Stop unloading the segmentation
  model on every detection by default; make `SPECTRA_KEEP_SEG_ON_DETECT=1` the
  automatic behavior when memory allows. If eviction is needed, evict the least
  recently used backend, not all of them.
- **Why:** a reload costs a checkpoint hash (A4), a weight load, and a full image
  re-encode. Alternating Detect → Brush → Detect currently pays that repeatedly.
- **Where:** `backend/models/manager.py:141-215` and `:1215-1282`.
- **Platforms:** CUDA, ROCm; CPU residency is cheap but still avoids reloads.
- **Tier/effort:** T1 / S–M.
- **Expected:** removes multi-second stalls from mixed workflows. The exact win is
  the reload frequency; measure it.
- **Verify:** instrument `load_model`, `verify_model`, and `set_image` calls per
  workflow; assert zero unintended reloads and identical masks.
- **Risks:** higher steady-state VRAM; an OOM is much worse than a reload. Keep a
  hard-cap/available-memory check and a fallback eviction path.

### A4. Cache the checkpoint digest by size and mtime

- **Change:** store the last verified digest keyed by `(path, size, mtime_ns)` in
  the existing hash database, and skip the SHA-256 when those match. Re-verify
  fully when the file changes, on first use, and on an explicit "verify" action.
- **Why:** `verify_model()` hashes the entire checkpoint every time
  `ModelManager.get_or_load_model()` loads or reloads a model. `sam3.pt` is
  multi-GB; on a hard drive that is seconds to tens of seconds, and it is repeated
  after every eviction. Note that `ensure_model_file()` (used when a backend is
  constructed directly) skips verification, so the cost appears only on the product
  path.
- **Where:** `backend/utils/model_integrity.py:87-145` and
  `backend/models/manager.py:252-262`.
- **Platforms:** all.
- **Tier/effort:** T1 / S.
- **Expected:** removes the hash from the reload path; the first load still pays it
  once.
- **Verify:** time `verify_model` cold/warm; corrupt a copy and confirm the cache
  invalidates on size/mtime change; keep the published-digest check.
- **Risks:** a file replaced with the same size and mtime would evade the check.
  That is a deliberate trade-off; document it, and keep full verification available.

### A5. Remove per-request `empty_cache` / `ipc_collect` / `gc.collect`

- **Change:** move `release_gpu_memory()` to explicit unload, OOM recovery, and
  settings changes. Do not call it at the end of every `detect()`. In
  `_evict_loaded_models()`, drop `ipc_collect()` unless multi-process IPC is
  actually in use.
- **Why:** `empty_cache()` returns cached blocks to the driver; the next allocation
  has to acquire them again. `ipc_collect()` walks IPC handles, and `gc.collect()`
  walks the whole Python heap. None of that helps a steady-state single process.
- **Where:** `backend/models/manager.py:1281`, `backend/utils/device.py:53-67`,
  `backend/models/manager.py:176-190`.
- **Platforms:** CUDA, ROCm; CPU only pays `gc.collect`, which is still per-detect.
- **Tier/effort:** T1 / S.
- **Expected:** removes allocator churn and GC pauses; may improve p95 more than p50.
- **Verify:** run 100 detects/predicts and compare p50/p95 and peak memory; confirm
  no OOM with the same workload.
- **Risks:** higher resident memory. Measure the watermark, not just latency.

### A6. Warm up at load, per model and per shape

- **Change:** after `load_model()`, run one dummy `set_image()` of the configured
  encode size and one decode. For SAM 2 this also triggers the existing decoder
  warm-up; for SAM 3 it triggers lazy module loading and any compiled graph. If
  MIOpen/cuDNN tuning is enabled, this is where the one-time search happens.
- **Why:** the first real request otherwise pays kernel/module initialization,
  algorithm search, and (if enabled) `torch.compile`. That spike is easy to
  misattribute to the model.
- **Where:** `backend/models/sam3_backend.py:104-137`,
  `backend/models/ultralytics_backend.py:215-290`.
- **Platforms:** all.
- **Tier/effort:** T1 / S.
- **Expected:** moves cost from the first user request to startup; total work is
  unchanged, first-request latency drops.
- **Verify:** report cold start, first request, and steady-state separately.
- **Risks:** startup time and memory; do not warm up every model at once on a
  VRAM-constrained device.

### A7. Batch multiple pending prompts

- **Change:** when several prompts are queued for the same image, call
  `inference_features` with `text=[prompt1, prompt2, ...]` in one pass instead of
  one call per prompt, then split the returned masks per prompt. Check that the SAM 3
  wrapper preserves per-prompt attribution: the stock Ultralytics
  `Predictor.inference_features` has no `text` argument, and the SAM 3 subclass may
  treat a list as synonyms for one concept rather than as separate prompts.
- **Why:** DART (2026) shows that text conditioning enters through cross-attention,
  which is batch-independent, and that batching class prompts converts N sequential
  encoder-decoder runs into one. The backbone is already shared. The same principle
  applies to multiple text prompts in SAM 3.
- **Where:** `backend/models/sam3_backend.py:253-296`; the UI currently sends one
  prompt per request, so this needs a queue/debounce or a multi-prompt request.
- **Platforms:** all.
- **Tier/effort:** T2 (batched kernels can round differently) / M–L.
- **Expected:** near-linear reduction in decoder passes for multi-prompt workflows;
  no benefit for a single prompt.
- **Verify:** compare each prompt's masks against a sequential run; check batch-size
  memory. Do not exceed a batch size that causes an OOM.
- **Risks:** per-prompt history/snapshot ordering, response size, and memory.

### A8. Memoize identical prompt results (short TTL)

- **Change:** cache the result for `(image revision, text, confidence, model,
  encode_dim)` and return it without a model call. Invalidate on image change,
  model change, settings change, or a short TTL.
- **Why:** duplicate submissions happen (double Enter, retry after a client error,
  undo followed by redo in some flows). The result is deterministic.
- **Where:** `backend/models/manager.py:695-732`.
- **Platforms:** all.
- **Tier/effort:** T1 / S.
- **Expected:** only duplicate calls; low overall, but removes a full prompt for
  those calls.
- **Verify:** identical result on cache hit; cache cleared on every invalidation.
- **Risks:** stale results if the image buffer is mutated in place; memory.

### A9. Store one full-resolution image buffer, not two

- **Change:** keep RGB (or BGR) only and convert at the rare detection/export
  boundary. The upload path already has RGB from PIL, converts to BGR, then
  `load_image` converts back to RGB at full resolution.
- **Why:** one full-resolution `cvtColor` and one extra buffer per upload; at 4K
  that is tens of MB and a few milliseconds, plus cache pressure.
- **Where:** `backend/models/manager.py:412-414`; `_decode_image` in
  `backend/main.py`.
- **Platforms:** all.
- **Tier/effort:** T1 / S.
- **Expected:** small per upload; memory win.
- **Verify:** pixel equality of the buffer used by the encoder/detector; time upload.
- **Risks:** every consumer must agree on channel order. Add an assertion rather
  than a silent conversion.

### A10. Avoid re-serializing unchanged object masks

This is the application-level half of G3; kept here because it is an orchestration
decision. See §10 for the wire-format details.

---
## 5. Catalogue: SAM 3 execution

### B1. Compile the right submodules, not the whole model

- **Change:** compile `model.backbone.visual.trunk` (the ViT) and the pixel decoder
  separately, with `fullgraph=True` and a static shape, instead of wrapping the
  entire `predictor.model` with `dynamic=True`. This mirrors
  `build_sam3_image_model(compile=True)` in the official repository, which sets
  `compile_mode="default"` on the ViT and `PixelDecoder`.
- **Why:** the official model was written to be compiled in those two places. The
  whole-model wrapper crosses Python-heavy detection/mask code, which produces
  graph breaks; `dynamic=True` then adds guard overhead and prevents some
  specializations. The official code also compiles SAM 2's encoder only, with
  `mode="max-autotune"`.
- **Where:** `backend/models/sam3_backend.py:130-137`.
- **Platforms:** CUDA ✅; ROCm Linux ⚠️ (Inductor/Triton support exists; verify the
  target gfx); CPU Windows ⚠️ (Inductor CPU needs MSVC); Windows GPU ⚠️
  (Triton/Inductor on Windows is not a safe assumption in 2026).
- **Tier/effort:** T2 / M.
- **Expected:** medium to large on transformer-heavy paths. Published
  `torch.compile` gains are workload-dependent; measure the encode and decode
  phases separately.
- **Verify:** `TORCH_LOGS=graph_breaks,recompiles` must show no graph breaks for the
  compiled regions; compare masks against the golden set; report compile time and
  steady-state p50/p95 separately.
- **Risks:** compile time, recompiles when `encode_dim` changes, Windows toolchain,
  and subtle numerical changes. Keep an environment switch and a fallback to eager.

### B2. Use static shapes where the shape is actually static

- **Change:** for the default 1024 encode size, use `dynamic=False`. If the user can
  change `encode_dim` (384–1500), either recompile per selected size or compile a
  small set of buckets and accept a one-time compile when the setting changes.
- **Why:** `dynamic=True` is currently forced. Static shapes let Inductor
  specialize, fuse, and use CUDA graphs where applicable. PyTorch's default
  recompilation limit is 8; a handful of encode sizes is acceptable, but unbounded
  dynamic shapes can hit guard failures and fall back to eager.
- **Where:** `backend/models/sam3_backend.py:135`.
- **Platforms:** all; strongest on CUDA.
- **Tier/effort:** T2 / S.
- **Expected:** better codegen; avoid recompilation storms.
- **Verify:** count compilations with `TORCH_LOGS=recompiles`; sweep encode sizes
  384/512/768/1024/1500.
- **Risks:** a new compile on every slider move if the UI sends intermediate values;
  debounce the setting.

### B3. Warm the compiled graph and persist compiler caches

- **Change:** run a dummy forward after compiling (A6). Set
  `TORCHINDUCTOR_CACHE_DIR` under `model_weights/` (Triton already has a cache
  directory in `run.cmd`) and consider `TORCHINDUCTOR_FX_GRAPH_CACHE=1`.
- **Why:** compilation is a one-time cost, but it is paid again on every process
  start unless the cache persists. A warm cache turns a multi-second startup into a
  load from disk.
- **Where:** `backend/run.cmd`, `backend_amd_gpu/run.cmd`; `TRITON_CACHE_DIR` is
  already set.
- **Platforms:** CUDA, ROCm (where Inductor works), CPU (where Inductor works).
- **Tier/effort:** T1 (cache only) / S.
- **Expected:** startup and first-request improvement.
- **Verify:** cold-cache vs warm-cache startup and first-prompt latency.
- **Risks:** stale cache after a torch/driver update; clear it when versions change.

### B4. `channels_last`: scope it to convolutions

- **Change:** apply `channels_last` to the conv-heavy neck/pixel decoder, not the
  whole model. Keep the ViT in its native layout. Benchmark both.
- **Why:** `channels_last` helps NHWC convolution kernels (cuDNN/MIOpen/oneDNN) but
  is neutral for Linear layers, while forcing a layout change on the whole model can
  insert transposes. Ultralytics explicitly warns that its `channels_last` argument
  is unsupported for SAM predictors; the manual `.to(memory_format=...)` is a
  different mechanism, but it still needs measurement.
- **Where:** `backend/models/sam3_backend.py:120-129`,
  `backend/models/ultralytics_backend.py:110`.
- **Platforms:** CUDA ✅, ROCm ✅ (MIOpen supports NHWC), CPU ⚠️ (oneDNN may prefer
  it for conv, but ViT is matmul-heavy).
- **Tier/effort:** T2 / S.
- **Expected:** small, positive for conv-heavy stages; possibly neutral or negative
  for the ViT.
- **Verify:** per-phase timings; mask equality; check for transposes in a profiler.
- **Risks:** layout conversions can cost more than they save; keep it opt-in.

### B5. Remove or scope `cudnn_disabled()` around decode

- **Change:** benchmark the decoder with cuDNN/MIOpen enabled. If the call exists to
  work around a crash or a bad kernel, scope it to the encoder or the specific op,
  rather than disabling the backend for every prompt.
- **Why:** this is the most suspicious line for GPU decode latency. It disables
  cuDNN on CUDA and MIOpen on ROCm for the prompt encoder and mask decoder, forcing
  fallback convolution implementations. It may also remove the cuDNN SDPA backend,
  depending on the PyTorch version. The decoder is small, but it runs on every
  stroke, and its convolutions are exactly what cuDNN/MIOpen optimize.
- **Where:** `backend/models/sam3_backend.py:263,290`;
  `backend/models/ultralytics_backend.py:370,450`.
- **Platforms:** CUDA ✅, ROCm ✅; no effect on CPU (the context manager returns
  early).
- **Tier/effort:** T2 / S.
- **Expected:** unknown until measured; potentially the largest single decode win.
- **Verify:** A/B with the context manager removed, on both CUDA and ROCm; compare
  masks, scores, and boxes; profile the conv kernels.
- **Risks:** if it is a workaround, removing it may reintroduce a crash on a
  specific driver/GPU. Keep the fallback and document the reason.

### B6. Attention backend selection

- **Change:** do not force math SDPA. On CUDA, leave flash/mem-efficient/cuDNN SDPA
  enabled and verify `torch.backends.cuda.can_use_flash_attention()` for the actual
  shapes. On ROCm, AOTriton is the default flash backend; test
  `torch.backends.cuda.preferred_rocm_fa_library("ck")` as an alternative. On CPU,
  test the default SDPA path against `SDPBackend.MATH`.
- **Why:** SAM 3's ViT uses `F.scaled_dot_product_attention`. Four global-attention
  blocks operate on 72×72 = 5,184 tokens; the attention matrix is large, and a math
  fallback materializes it. A flash/memory-efficient kernel can change both speed
  and memory dramatically.
- **Where:** model internals (`sam3/model/vitdet.py` in the official code); this app
  does not override SDPA.
- **Platforms:** CUDA ✅, ROCm ✅, CPU ⚠️ (CPU SDPA support varies by version/shape).
- **Tier/effort:** T2 / S.
- **Expected:** potentially large for the global-attention blocks; measure.
- **Verify:** `torch.backends.cuda.sdp_kernel` diagnostics, `can_use_*` checks,
  profiler kernel names, mask equality.
- **Risks:** different backends round differently; the mask gate decides.

### B7. TF32 and reduced-precision reduction flags (CUDA)

- **Change:** on sm80+, consider `torch.backends.cuda.matmul.allow_tf32 = True` and
  `torch.backends.cudnn.allow_tf32 = True` (the official SAM 3 builder enables both
  on compute capability ≥ 8.0). Test
  `torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction`.
- **Why:** the model is fp16 on GPU, so most GEMMs already use Tensor Cores; TF32
  mainly affects fp32 upcasts and some normalization/softmax paths. The gain is
  likely small here, but it is a one-line experiment.
- **Where:** `backend/main.py:339-365` (CUDA-only block).
- **Platforms:** CUDA only. ROCm documents that fp16/bf16 reduced-precision
  reductions are not supported; TF32 is not the same knob.
- **Tier/effort:** T2 / S.
- **Expected:** small; measure before keeping.
- **Verify:** golden masks and scores; check for any change in borderline masks.
- **Risks:** numerically different; do not enable blindly on a quality-sensitive
  product.

### B8. `flash-attn-3` and `cc_torch` (CUDA-only, not exposed by Ultralytics)

- **Change:** the official SAM 3 README lists `flash-attn-3` and
  `git+https://github.com/ronghanghu/cc_torch.git` as optional dependencies for
  faster inference. They are used through the model's `use_fa3` path and fused
  kernels, which Ultralytics does not expose.
- **Why:** potentially faster attention and fused MLP, but a heavy build, CUDA-only,
  and not reachable without patching the Ultralytics model.
- **Where:** SAM 3 model internals; not this app's code.
- **Platforms:** CUDA only. No ROCm or CPU equivalent in this path.
- **Tier/effort:** T2 (attention kernel changes numerics; golden gate mandatory) / L,
  with high maintenance/compatibility risk.
- **Expected:** unknown here; official optional path.
- **Verify:** only after a controlled patch; not a first-line experiment.
- **Risks:** build fragility, version coupling, and a second attention
  implementation that must be quality-checked.

### B9. Do not bother disabling activation checkpointing

The official SAM 3 code gates `use_act_checkpoint` on `self.training` for both the
ViT and the transformer encoder. In `eval()` there is no recomputation overhead, so
there is nothing to disable. This is recorded to prevent a wasted experiment.

### B10. SAM 3.1 is not an image-speed upgrade

SAM 3.1 (2026-03-27) adds Object Multiplex for **video** tracking: it groups tracked
objects into buckets and processes them together, with a reported ~7× speedup at
128 objects on an H100. The image detector and the interactive point-prompt head
keep the SAM 3 architecture. Upgrading the checkpoint will not speed up
single-image background removal in this app, and `sam3.1_multiplex.pt` is not in the
allowed model list. Do not switch for image speed.

---
## 6. Catalogue: SAM 2 execution

### C1. Compile the image encoder only, with `max-autotune`

- **Change:** follow the official `vos_optimized` path:
  `image_encoder.forward = torch.compile(..., mode="max-autotune",
  fullgraph=True, dynamic=False)`. Do not compile the prompt encoder or the mask
  decoder by default; they are small and have dynamic prompt shapes.
- **Why:** the official SAM 2 `build_sam2_video_predictor(vos_optimized=True)` sets
  `compile_image_encoder=True`, and `SAM2Base` compiles exactly the encoder forward
  with those options. The encoder is the dominant cost for a new image. Compiling
  the decoder adds recompiles for every point count and boxes little.
- **Where:** `backend/models/ultralytics_backend.py:114-123` currently compiles the
  whole model, and only when `SAM2_COMPILE=1`.
- **Platforms:** CUDA ✅; ROCm ⚠️; CPU ⚠️ (MSVC on Windows).
- **Tier/effort:** T2 / M.
- **Expected:** medium; the encoder is where the time is.
- **Verify:** phase timings, graph-break log, mask equality; report compile time.
- **Risks:** `max-autotune` has a long first compile; cache it (B3).

### C2. Keep the decoder warm; avoid compiling it

`_warm_decoder()` already runs a dummy prompt after `set_image()`, which is good. If
B1/C1 is adopted, keep that warm-up after the compiled encoder is installed.
Compiling the decoder is usually counterproductive because point counts vary.

### C3. Use batched box decoding (already present)

`predict_batch()` performs one decoder call for many boxes, expanding the image
embedding. The app exposes `/segment-batch`. Verify the UI uses it where multiple
detections are converted to masks; if it loops one request per box, batching is a
free win. This is already implemented, not a new optimization.

### C4. Low-resolution mask feedback (already present)

`predict()` passes the previous low-res mask as `mask_input`, which is the intended
SAM 2 refinement path. Do not replace it with a full-resolution mask; that would be
slower and numerically different.

### C5. `cudnn_disabled()` and `channels_last` apply here too

See B5 and B4. The same context manager wraps `predict()` and `predict_batch()`, and
the same whole-model `channels_last` switch exists.

### C6. CPU compile is opt-in

`SAM2_COMPILE` is deliberately disabled on CPU (`self._device != "cpu"`). If CPU
compile is explored, it needs MSVC and a static 1024 shape; keep it opt-in and
benchmark the compile time against the session length.

---
## 7. Catalogue: CUDA

Ordered by expected value per effort. These are platform-specific complements to
the model-level items above.

### D1. Re-check cuDNN benchmarking settings

- Current: `cudnn.benchmark=True`, `benchmark_limit=4` at startup.
- Experiment: `benchmark_limit=0` (try every algorithm) for the fixed 1008/1024
  shapes. The one-time search is paid at warm-up (A6), then cached.
- Tier T2, effort S. Verify mask equality and steady-state p50.

### D2. cuBLASLt for fp16 GEMMs

- Current: default cuBLAS. Try
  `torch.backends.cuda.preferred_blas_library("cublaslt")` (or
  `TORCH_BLAS_PREFER_CUBLASLT=1`) for the model's fp16 Linear layers.
- Tier T2, effort S. Not guaranteed to help; the heuristic may already choose well.

### D3. Allocator configuration

- Current: `garbage_collection_threshold:0.8,max_split_size_mb:512,expandable_segments:True`.
- `garbage_collection_threshold` can trigger collections while the model is running;
  `max_split_size_mb` can prevent reuse of large blocks. Test the defaults and
  `expandable_segments:True` alone. Keep the setting that minimizes p95 and peak
  memory.
- Set the environment variable in the launcher, before Python starts, not inside
  `lifespan` (the allocator may already be initialized).

### D4. `CUDA_DEVICE_MAX_CONNECTIONS=1`

- Sometimes improves small-kernel launch ordering; sometimes hurts. Test with the
  decoder-only and full-pipeline workloads. Tier T2, effort S.

### D5. Remove `torch.cuda.synchronize()` from production paths

- `sam3_backend._encode_image()` and the SAM 2 encode path synchronize to make the
  log timing exact and to keep an OOM inside `set_image()`'s CPU-fallback `try`
  block. The next decode is already ordered on the same stream, so the sync is not
  needed for normal execution and it blocks host/device overlap. Removing it can
  move an OOM exception into the first `predict()` call, which has no fallback:
  either keep the trailing sync or extend the fallback to cover `predict()`.
  Measure the successful path separately from the OOM path.
- Tier T1 for successful runs, effort S.

### D6. cuDNN SDPA

- Do not disable cuDNN SDPA unless B5 proves it is necessary. On newer GPUs the
  cuDNN attention backend can be the fastest option for the global blocks.

### D7. Memory cap

- `torch.cuda.set_per_process_memory_fraction(0.90)` is a hard cap. If the model
  fits comfortably, the cap only reduces the memory available to the allocator and
  can trigger the CPU fallback (which is dramatically slower). Measure with and
  without; keep the cap only if it prevents a worse failure mode.

### D8. Driver, power, and environment

These are not code changes but can dominate a laptop benchmark:

- NVIDIA persistence mode (Linux), maximum performance power plan (Windows);
- disable battery saver and "Game Mode" during measurement;
- exclude `model_weights/` and the venv from real-time antivirus scanning
  (security trade-off; at minimum, measure the effect);
- keep the driver and CUDA runtime matched to the installed torch wheel.

---
## 8. Catalogue: ROCm

ROCm is where the current configuration leaves the most obvious headroom, partly
because the launcher deliberately prefers fast startup over tuned kernels.

### E1. Tune MIOpen once, persist the database, then use fast find

- Current: `MIOPEN_FIND_MODE=2` (FAST), `MIOPEN_FIND_ENFORCE=1` (NONE),
  `MIOPEN_SEARCH_CUTOFF=1`, and `torch.backends.cudnn.benchmark=False`.
- What that means: MIOpen uses the find database or immediate-mode fallback; if a
  shape is not in the database, it picks a workable kernel rather than benchmarking
  candidates. At the default sizes (1008 for SAM 3, 1024 for SAM 2) the shapes
  repeat, which is the ideal case for a one-time search. If the user changes
  `encode_dim` (384–1500), each new size introduces new convolution shapes; tune
  the default and let FAST/immediate handle the rest, or tune each supported bucket.
- Experiment: run one warm-up with `MIOPEN_FIND_MODE=NORMAL` (or
  `MIOPEN_FIND_ENFORCE=SEARCH_DB_UPDATE`) and a persistent
  `MIOPEN_USER_DB_PATH`/`MIOPEN_CUSTOM_CACHE_DIR`; then switch back to FAST and
  confirm the database is hit. Alternatively, use `HYBRID`/`DYNAMIC_HYBRID` and
  measure.
- Result (2026-11-10, torch 2.16, RX 6600): setting
  `torch.backends.cudnn.benchmark=True` was still searching the 9-dimensional
  convolution space after 60 s on a 1024² matte load. The guarded warm path was
  about 0.21 s per pass. Keep benchmark mode off; the FAST path is the safe default.
- Tier T2, effort M. Verify with `MIOPEN_ENABLE_LOGGING=1` and
  `MIOPEN_ENABLE_LOGGING_ELAPSED_TIME=1`; compare conv kernel names and timings.
- Risk: a one-time tuning run can take minutes and use significant memory; do it
  offline or during install, not on the user's first request.

### E2. Enable TunableOp (the cache directory is already configured)

- Current: `PYTORCH_TUNABLEOP_CACHE_DIR` is set, but `PYTORCH_TUNABLEOP_ENABLED`
  is not. `TORCH_BLAS_PREFER_HIPBLASLT=1` is set.
- Experiment: `PYTORCH_TUNABLEOP_ENABLED=1` with a tuning run, then reuse the cache
  with tuning disabled. Compare against hipBLASLt-only. TunableOp tunes GEMM
  shapes, which are exactly the ViT/decoder Linear layers.
- Tier T2, effort M. Risk: tuning time, memory, and a known history of OOM reports
  on some ROCm versions; pin versions and measure.

### E3. hipBLASLt / rocBLAS selection

- `TORCH_BLAS_PREFER_HIPBLASLT=1` is already set. Test
  `torch.backends.cuda.preferred_blas_library("hipblaslt")` explicitly, and test
  `DISABLE_ADDMM_CUDA_LT=1` on/off. The ROCm 7.x notes say hipBLASLt is the
  preferred backend on newer architectures, with ongoing improvements for gfx1100
  and gfx1200.
- Tier T2, effort S. Verify with a profiler, not just wall time.

### E4. Flash attention backend: AOTriton vs CK

- PyTorch on ROCm defaults to AOTriton for flash attention; ROCm 7.1/7.2 upgraded
  the AOTriton version and made SDPA call the optimized kernel automatically.
  `torch.backends.cuda.preferred_rocm_fa_library("ck")` (or
  `TORCH_ROCM_FA_PREFER_CK=1`) selects Composable Kernel instead.
- Test both on the global-attention blocks; check
  `torch.backends.cuda.can_use_flash_attention()` for the actual shapes. RDNA3
  support has historically lagged CDNA; do not assume it works on every gfx.
- Tier T2, effort S.

### E5. Allocator configuration

- Current: `PYTORCH_CUDA_ALLOC_CONF` is set in `backend_amd_gpu/main.py` without
  `expandable_segments`. On ROCm, test `PYTORCH_HIP_ALLOC_CONF` (or
  `PYTORCH_ALLOC_CONF` in newer torch) with `expandable_segments:True`, and measure
  fragmentation/peak memory. Set it in the launcher before torch starts.
- Tier T1 (allocation behavior), effort S. Risk: version-dependent support; verify
  it is actually read.

### E6. Stop disabling MIOpen for decode

This is B5 on ROCm: `cudnn_disabled()` is a shim over MIOpen, so every prompt
currently runs decoder convolutions without MIOpen's tuned kernels. A/B it.

### E7. OpenCV OpenCL

- If OpenCV initializes an OpenCL context on the AMD GPU, it can compete with ROCm
  and add startup cost. Call `cv2.ocl.setUseOpenCL(False)` in the AMD backend and
  measure resize/encode. Tier T1 for the CPU-side ops, effort S.

### E8. Windows-specific ROCm notes

- The installer uses the ROCm nightly multi-arch wheels
  (`torch[device-gfxXXXX]`), so the exact torch/ROCm combination is a moving target.
  Record `torch.__version__`, `torch.version.hip`, and `gcnArchName` in every
  benchmark.
- AOTriton on Windows works for some gfx targets (community reports for gfx1151);
  gfx1100/gfx1200 support varies. Verify before relying on flash attention.
- `torch.compile` on Windows ROCm depends on a Triton port that was still being
  upstreamed in early 2026; treat it as experimental and keep the eager fallback.
- `HSA_OVERRIDE_GFX_VERSION` is never set automatically. Use an explicit
  `AMD_HSA_OVERRIDE` only for a known workaround.

---
## 9. Catalogue: CPU

CPU is the platform where the smallest configuration mistakes have the largest
effect, because every phase is exposed.

### F1. Align the thread knobs

Current state:

- `backend/run.cmd` sets `OMP_NUM_THREADS=min(6, physical cores)` (unless
  `SPECTRA_OMP_THREADS` is set);
- `torch_threads.resolve_threads()` returns `min(os.cpu_count(), 32)`, i.e.
  **logical** CPUs, and `cpu_threads()` calls `torch.set_num_threads()` per phase;
- `KMP_AFFINITY=granularity=fine,compact,1,0` and `KMP_BLOCKTIME=1` are set.

Experiment: set `OMP_NUM_THREADS`, `torch.set_num_threads`, and (once, at startup)
`torch.set_num_interop_threads(1)` to the same value. Sweep 1, 2, 4, 6, physical
cores, logical cores. Benchmark encode and decode separately; the decoder may be
faster with fewer threads because its matrices are small.

- Tier T2 (reduction order can change), effort S.
- Verify: phase timings and mask equality. On Windows, set the environment before
  Python starts; OpenMP reads it at initialization.
- Risk: `torch.set_num_threads` is documented to be called before running eager,
  JIT, or autograd code; changing it mid-process for a benchmark is not
  representative.

### F2. Set inter-op threads to 1

`torch.set_num_interop_threads(1)` prevents a second pool from competing with the
intra-op pool. It must be called before parallel work begins. Tier T2, effort S.

### F3. Stop OpenCV from oversubscribing

`cv2.setNumThreads(1)` (or a small number) in the serving process. OpenCV otherwise
creates its own pool, which competes with torch's OpenMP pool during resize,
color conversion, and PNG encode. Zero (`cv2.setNumThreads(0)`) disables OpenCV
threading entirely; one is the safer starting point.
Tier T2 (scheduling only), effort S. Verify total wall time, not just the resize.

### F4. Per-phase thread counts

Use all threads for the image encoder, fewer for the decoder/text encoder, and one
for PNG encode. The current code uses `_DEFAULT_THREADS` for every phase and
switches to 1 only on GPU. On CPU, a 256×256 decoder GEMM with 32 threads can be
slower than with 4. This is a cheap experiment with a potentially visible win.

### F5. `KMP_AFFINITY` / `KMP_BLOCKTIME`

`KMP_BLOCKTIME=1` (ms) makes worker threads sleep quickly; for a bursty server,
a longer block time (e.g. 200) can reduce wake-up latency at the cost of idle CPU.
`granularity=fine,compact,1,0` pins threads; on hybrid P/E-core CPUs it can pin work
to E-cores. Test `KMP_AFFINITY=disabled` and explicit core sets. Tier T2, effort S.

### F6. `torch.compile` on CPU (Windows: MSVC required)

- PyTorch's Windows Inductor tutorial requires MSVC (or `clang-cl`/`icx-cl`) and
  supports CPU (and XPU), not CUDA GPU. `torch.compile` on CPU can fuse pointwise
  ops and use C++/OpenMP kernels.
- Start with the ViT/pixel decoder, static 1024 shape, `fullgraph=True`; keep
  `dynamic=False`. Measure compile time and cache it (B3).
- Tier T2, effort M. Risk: Windows toolchain, compile time, and CPU-specific
  numerical differences.

### F7. oneDNN / MKLDNN

`torch.backends.mkldnn` is enabled by default. Verify it is active; test
`channels_last` for the conv neck on CPU. Do not disable oneDNN. `DNNL_VERBOSE=1`
shows primitive selection, but it destroys performance—use it only for diagnosis.

### F8. bfloat16 autocast on AVX512-BF16 / AMX

- If the CPU supports AVX512-BF16 or AMX, `torch.autocast("cpu",
  dtype=torch.bfloat16)` can speed up Linear/convolution. It changes numerics.
- Tier T2; if the golden gate fails, reject it rather than downgrading the tier.
  Effort M. Keep it opt-in and validate masks; do not enable it on CPUs without
  BF16 support.

### F9. Intel Extension for PyTorch (IPEX)

`ipex.optimize(model, dtype=torch.bfloat16)` can help on Intel CPUs, but it is a
large dependency, version-coupled to torch, and bf16 changes numerics. Treat it as
an experiment behind a flag, not a default. Windows support and wheel availability
must be checked for the exact torch version.

### F10. OpenVINO

- There is an official OpenVINO notebook for SAM 3 image segmentation (2026). It
  converts the HuggingFace model to OpenVINO IR and runs it with the OpenVINO
  runtime; the app uses the Ultralytics checkpoint, so a conversion/export step is
  required.
- Runtime knobs: `hint.performance_mode("LATENCY")` (default) or `"THROUGHPUT"`,
  `inference_num_threads`, `streams`, `AFFINITY`, and `CACHE_DIR` for compiled
  blobs. The async API can overlap preprocessing and inference.
- Tier T2 (different kernels; fp32 IR should be close), effort XL. Quality gate is
  mandatory. CPU only (or Intel GPU); no ROCm.

### F11. ONNX Runtime

- Export the encoder/decoder to ONNX and run with `CPUExecutionProvider` (or
  `CUDAExecutionProvider` / `ROCMExecutionProvider` on Linux / `MIGraphX`).
- Set `intra_op_num_threads` to physical cores, `execution_mode=ORT_SEQUENTIAL`
  (or `ORT_PARALLEL` only if the graph branches), `graph_optimization_level=
  ORT_ENABLE_ALL`, and tune `session.intra_op.allow_spinning` / `spin_duration_us`.
- Tier T2, effort XL. Export of the full SAM 3 pipeline is the hard part; validate
  masks against the PyTorch baseline.

### F12. Flush denormals (tiny, numerically different)

`torch.set_flush_denormal(True)` can speed up CPU floating point by avoiding
denormal handling. It changes very small values. Tier T2; if the golden gate fails,
reject it. Effort S; measure before keeping.

### F13. NUMA and affinity

On multi-socket CPUs, pin the process to one NUMA node (`start /affinity` on
Windows, `numactl` on Linux) so the model's threads do not cross the interconnect.
Tier T2, effort S. Measure with the same thread count.

### F14. Do not use fp16 on CPU

`SAM3_QUANTIZE`/`SAM2_QUANTIZE` default to 32 on CPU for a reason: CPU fp16 is slow
and Ultralytics notes it can return empty masks. Keep fp32 on CPU; use bf16 only
with the F8 validation.

### F15. Persist oneDNN/OpenMP settings

Set thread and affinity variables in `run.cmd` (before Python), not inside the app.
This makes runs reproducible and avoids the "it was fast once" problem.

---
## 10. Catalogue: post-processing, serialization, export

This is where the cheapest exact wins are, especially for large images and many
subjects.

### G1. Encode binary masks as 1-bit PNG

- **Change:** `mask_to_png_b64()` currently converts a boolean mask to 8-bit
  grayscale (0/255) and PNG-encodes it. Encode a 1-bit PNG instead (Pillow mode
  `"1"`, or `packbits` + `zlib`), or send a packed/RLE binary payload.
- **Why:** a binary mask has one bit of information per pixel. An 8-bit PNG stores
  eight. The 1-bit PNG is exact, roughly 8× smaller before compression, and faster
  to encode. The browser's PNG decoder handles 1-bit grayscale natively.
- **Where:** `backend/utils/compositing.py:11-15`.
- **Platforms:** all.
- **Tier/effort:** T1 / S.
- **Expected:** large for 4K masks; the exact win depends on mask content and PNG
  overhead. Measure encode time and bytes, not just file size.
- **Verify:** decode both PNGs and assert `np.array_equal`; compare base64 length
  and encode time.
- **Risks:** frontend decoders that assume 8-bit; test canvas/ImageBitmap paths.

### G2. Set PNG compression explicitly

- OpenCV's `imencode` default is not a promise; set
  `[cv2.IMWRITE_PNG_COMPRESSION, 1]` (or 0) for mask transport. Pillow's default is
  6; `compress_level=1` is documented as "best speed" and remains lossless.
- Tier T1, effort S. Trade-off: larger payload, faster encode. Measure the total
  (encode + base64 + JSON + transfer + decode), not just encode.
- For masks, G1 usually dominates; do both.

### G3. Send only the mask that changed

- **Change:** return the changed object's mask plus a revision per object, instead
  of all object masks on every predict. The client already has the others.
- **Why:** `_brush_response()` serializes every object's base64 PNG on every
  stroke. For 20 subjects at 4K, the response can be tens of MB even though one
  mask changed. JSON serialization and base64 copying then dominate the decoder.
- **Where:** `backend/models/manager.py:575-612`; frontend `PredictResponse`
  handling.
- **Platforms:** all.
- **Tier/effort:** T1 / M (requires a protocol change).
- **Expected:** large for multi-object sessions; zero for one object.
- **Verify:** decoded masks are identical to the current all-masks response; measure
  response bytes and server CPU.
- **Risks:** client state can drift if a revision is missed; include a full-state
  fallback on reconnect.

### G4. Avoid base64 for large payloads

- Base64 adds 33% to every mask. For a single mask, return `image/png` directly
  (`Response(content=..., media_type="image/png")`); for many, use multipart or a
  small binary framing. This is exact.
- Tier T1, effort M. The existing `export-image` endpoint already returns raw
  bytes, so the pattern exists.
- Risk: frontend fetches must stop assuming JSON.

### G5. Decode export masks with the faster decoder

- `_decode_mask_b64()` uses Pillow (`Image.open` + `convert("L")` + numpy).
  `cv2.imdecode` is usually faster for large PNGs, and PNG decoding is
  pixel-exact. Use it for the export mask path, then keep the threshold identical.
- Where: `backend/main.py:1301-1315`. Tier T1, effort S.
- Verify: decoded boolean arrays equal on a corpus of masks, including odd sizes.

### G6. Export encoding settings

- **PNG:** `Image.save(..., format="PNG", compress_level=1)` is still lossless.
  Default 6 spends CPU for smaller files; for an interactive export, 1 is usually
  the better trade-off.
- **ZIP:** `ZIP_DEFLATED` re-compresses already-compressed PNGs. Use `ZIP_STORED`
  (or `compresslevel=1`) and measure the archive size/time. The image bytes are
  unchanged.
- Where: `backend/utils/compositing.py:76-98`, `backend/main.py:1318-1375`.
- Tier T1, effort S. Verify decoded pixels, not ZIP bytes.

### G7. Remove full-array scans in mask normalization

`_to_mask_f32()` calls `_is_unit_scale()`, which does `np.all(np.isfinite(...))`,
`min()`, and `max()` over the whole array even for `uint8` and `bool`, where the
range is known. Short-circuit those dtypes. This is pure CPU work on every export.
Where: `backend/utils/compositing.py:18-42`. Tier T1, effort S.

### G8. Reduce allocations in feather/composite

`feather_mask_edge()` pads a float32 full-resolution mask and runs a Gaussian blur;
`composite_background()` creates float32 temporaries for both image and background.
At 4K this is hundreds of MB of transient allocation.

Options, in order of exactness:

1. Use `cv2.copyMakeBorder` instead of `np.pad` (same result, fewer copies).
2. Process in tiles with a halo equal to the blur radius; the Gaussian result is
   identical because the kernel support is bounded.
3. Use a fixed-point uint16 intermediate (T2: alpha may differ by 1/255).

Tier T1 for (1)/(2), T2 for (3); effort M. Measure peak RSS and export wall time.
Verify decoded pixels for (1)/(2), PSNR for (3).

### G9. Crop transport masks to their bounding box

Send a mask plus its bbox instead of a full-resolution mask when the subject is
small. The client places it at the bbox. This is exact if the bbox and crop are
included and the outside is known to be zero. It dramatically reduces bytes for
small subjects. Tier T1, effort M; protocol change. Verify by reconstructing the
full mask and comparing.

### G10. Do not compress PNG payloads with gzip

The app does not currently add GZip middleware; keep it that way for mask
responses. Base64-of-PNG is already compressed; gzip spends CPU for a few percent.
If compression is ever added, exclude the mask endpoints.

### G11. Serialize mask JSON with `orjson`

FastAPI's default `JSONResponse` uses the standard-library `json` encoder. For
responses dominated by multi-megabyte base64 strings, `orjson` (or
`ORJSONResponse` for the mask routes) reduces encode time and peak memory. The
result is exact: only whitespace/escaping differs, and the decoded object is
identical. Tier T1, effort S, all platforms. This adds a dependency, so benchmark
the full response path rather than `json.dumps` alone; base64 encoding may still
dominate. Where: the predict/brush routes in `backend/main.py`.

---
## 11. Catalogue: serving, concurrency, and startup

### H1. One worker, no per-request model reloads

The in-process model state (features, locks, history) requires a single uvicorn
worker. Do not add `--workers N` to "speed up" inference; it multiplies memory and
splits sessions. Throughput should come from batching/streams, not workers.

### H2. The upload request blocks on encoding

`load_image()` encodes before the upload response returns, so the user waits for the
encoder before seeing the canvas. Options:

- return after decode and encode asynchronously, with the first prompt waiting on a
  readiness event (perceived latency win, same total work);
- show a progress state keyed on the existing status callback.

Tier T1, effort M. Measure time-to-canvas and time-to-first-mask separately.

### H3. Request coalescing

If two identical prompts arrive while one is running, coalesce them (A8). This is
especially useful for double-Enter and retry storms.

### H4. Keep-alive and client reuse

The frontend should use a persistent HTTP connection; a new TCP connection per
prompt adds a small but avoidable cost on Windows. Tier T1, effort S.

### H5. Concurrency and model ownership

The manager shares one backend per model across sessions. `_infer_lock` serializes
inference, which is correct but means concurrent sessions queue. Before adding
feature caches or parallel streams, verify that per-session image features are not
being overwritten by another session (see A2). This is a correctness prerequisite.

### H6. Thread-pool sizing

`asyncio.to_thread` uses the default executor (`min(32, cpu+4)`). GPU inference is
serialized anyway; a large pool only queues work. Keep the default unless profiling
shows thread creation overhead.

### H7. Startup

- Move allocator environment variables to the launcher, before Python.
- Cache the model digest (A4), persist compiler caches (B3), and warm up (A6).
- Avoid loading every model at startup; load the selected one and keep it resident.

---
## 12. Catalogue: model loading and environment

### I1. `mmap` checkpoint loading

`torch.load(..., mmap=True)` can reduce peak RAM and speed up cold loads for large
checkpoints. The SAM 3 loader already wraps `torch.load` with `weights_only=True`;
adding `mmap=True` there is a small, testable change. The Ultralytics SAM 2 loader
has its own load path and needs a separate patch or an upstream option. `mmap` does
not change weights or outputs, but on Windows it can be slower for a checkpoint that
is read in full. Tier T1, effort S. Verify load time and RSS; some checkpoint
formats do not benefit.

### I2. Keep the checkpoint on fast storage

A multi-GB checkpoint on a spinning disk dominates cold start. Keep
`model_weights/` on the same SSD as the venv, and exclude it from real-time
antivirus scanning during measurement.

### I3. Do not re-verify on every load

See A4. The digest cache is the single highest-value startup change if models are
evicted often.

### I4. `low_cpu_mem_usage` for detectors

The HuggingFace-based detectors already use `accelerate`/`low_cpu_mem_usage`.
Verify it is enabled for every `from_pretrained` call; it reduces peak RAM and can
speed up loading. Do not change the model weights.

### I5. Persist compile caches

See B3. `TRITON_CACHE_DIR` is already set; add the Inductor cache and clear both
when torch/driver versions change.

---
## 13. Alternative runtimes (larger projects)

These are not first-line experiments. They require an export pipeline, a new
dependency, and a full quality gate. They are listed because the user asked for
thorough coverage, not because they are recommended today.

### J1. ONNX Runtime

- **What:** export the encoder and decoder to ONNX; run with CPU, CUDA, ROCm
  (Linux), MIGraphX (Linux), or DirectML (Windows AMD).
- **Why:** graph optimization, operator fusion, and a runtime with mature thread
  control. ONNX Runtime's CPU provider defaults to one intra-op thread per physical
  core and enables graph optimizations.
- **Platforms:** CPU ✅, CUDA ✅, ROCm ⚠️ (Linux ROCm/MIGraphX; Windows DirectML),
  Windows ✅ for CPU/DirectML.
- **Tier:** T2 at best (different kernels). Verify masks.
- **Effort:** XL. The SAM 3 graph has dynamic text prompts and mask heads; the
  export is the hard part.
- **Expected:** unknown; ONNX Runtime is often faster than eager PyTorch on CPU for
  transformer encoders, but not guaranteed for this model.

### J2. OpenVINO

- **What:** convert to OpenVINO IR (an official SAM 3 notebook exists) and run on
  CPU or Intel GPU.
- **Why:** OpenVINO is tuned for Intel CPU instruction sets (AVX-512, AMX) and
  supports latency/throughput hints, async inference, and compiled-blob caching.
- **Platforms:** CPU ✅ (Intel especially), Intel GPU ✅, AMD/NVIDIA ❌.
- **Tier:** T2. Use fp32 IR for the first validation; any int8 quantization is T3.
- **Effort:** XL. The app uses Ultralytics, not the HuggingFace checkpoint the
  notebook targets, so a separate export is needed.
- **Expected:** potentially large on Intel CPU; unknown on AMD CPU.

### J3. TensorRT (CUDA only)

- **What:** export a split engine: vision backbone + encoder-decoder, with the text
  encoder left in PyTorch so prompts remain open-vocabulary. DART reports 112 ms →
  53 ms per class on an RTX 4080 at 1008² with FP16 and graph restructuring.
- **Why:** kernel fusion, autotuning, and FP16 Tensor Cores.
- **Platforms:** CUDA only. No ROCm; no CPU.
- **Tier:** T2 at best, and only with a quality gate. DART documents that naive FP16
  export of the ViT destroys quality: complex-valued RoPE and fused SDPA break
  TensorRT's attention pattern matching, and generic FP16 matmuls compound rounding
  through 32 blocks (cosine similarity 0.058 vs FP32). The fix is to rewrite RoPE as
  real cos/sin buffers and attention as explicit QKᵀ → scale → softmax → V before
  export.
- **Effort:** XL. Build/deploy pipeline, engine caching per GPU, and a new runtime.
- **Expected:** large on paper; the engineering and validation cost is large too.
- **Verdict:** a valid research direction, not a quick win. Do not ship a naive
  FP16 engine.

### J4. `torch.export` + AOTInductor

- **What:** export the model to a stable graph and compile ahead-of-time with
  Inductor. Removes Python overhead and can use CUDA graphs.
- **Platforms:** CUDA/Linux best; Windows support is limited; ROCm depends on
  Inductor.
- **Tier:** T2. Effort: L–XL.
- **Why not now:** TorchScript is deprecated in PyTorch 2.10; `torch.export` is the
  forward path, but SAM 3's dynamic prompt code makes a full export nontrivial.

### J5. DirectML / MIGraphX

- **DirectML:** ONNX Runtime on Windows AMD GPUs; useful when ROCm is unavailable.
  It is a different runtime with its own kernel set; validate quality and measure
  against ROCm before adopting.
- **MIGraphX:** ROCm's graph compiler for ONNX models, Linux-oriented. Not a
  Windows path.

---
## 14. Rejected because they change quality

Listed so they are not proposed as "free" wins:

| Change | Why it is out of scope |
|---|---|
| INT8/FP8 quantization (bitsandbytes, torchao, NNCF, TensorRT INT8) | Changes weights/activations; needs calibration and a quality budget. |
| Smaller backbone (SAM 2.1 tiny/small, EfficientSAM3, RepViT/TinyViT distillation) | Different model; quality and behavior change. |
| Lower `encode_dim` (384–768) | Fewer pixels/tokens; boundary quality changes. The app already exposes this as a user setting. |
| Pruning / token merging / approximate attention | Removes computation that affects masks. |
| JPEG masks or lossy WebP | Lossy mask boundaries. PNG/1-bit/RLE are the lossless options. |
| Downscaling the uploaded image | Changes the pixels being segmented and exported. |
| `torch.set_float32_matmul_precision("medium")` | Uses bf16 for fp32 matmuls; numerically lossy. `"high"` (TF32) is T2 and still needs validation. |
| CPU fp16 | Slow and can produce empty masks; the app correctly uses fp32. |
| Disabling activation checkpointing | No effect in eval; a wasted change. |

---
