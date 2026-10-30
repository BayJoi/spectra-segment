# Performance research: faster background removal at identical output quality

> **Research status: not implemented, not measured.**
>
> This document is a plan, a code review, and a literature review. **None of the
> optimizations below has been applied or benchmarked in this repository.** At the
> time of writing, `cleanup_amd.bat` had removed the AMD backend's `.venv` and
> `model_weights/`; no backend virtual environment or checkpoints existed anywhere
> in the checkout, so no model could be loaded and no timing could be taken. Every
> "expected effect" is either a hypothesis about this codebase or an external number
> from a cited source; the two are labelled separately. The priority order is a
> starting point for experiments, not a result.

**Scope:** text-prompted and click-prompted background removal in this application:
SAM 3 (`sam3.pt`) through Ultralytics, SAM 2.1 through Ultralytics, the optional
detectors, mask post-processing, export, and the HTTP layer.

**Targets:** CUDA (Windows and Linux), ROCm (Windows and Linux), and CPU (Windows and
Linux). The application is Windows-first, so Windows caveats are called out
separately.

**Repository note:** `backend/` and `backend_amd_gpu/` are near-identical trees. The
AMD `main.py` diverges for ROCm-specific setup; `models/manager.py` and
`models/sam3_backend.py` differ by a handful of lines. Code references below use
`backend/...` paths and line numbers; apply each change to both trees unless the
item is explicitly platform-specific, and re-check the AMD line numbers first.

**Research date:** 2026-10-26. Sources include 2026 releases (PyTorch 2.10–2.14,
ROCm 7.2, SAM 3.1, DART) and older foundational material (SAM 2, MIOpen, TorchInductor).

---

## 1. Short version

The largest safe wins in this codebase are probably **not** in the model
mathematics. They are in four places:

1. **Do not repeat work.** Cache text embeddings, keep models resident, cache the
   model-file digest, and stop re-encoding masks that did not change.
2. **Stop paying per-request cleanup costs.** `empty_cache()`, `ipc_collect()`,
   `gc.collect()`, a full SHA-256 of a multi-GB checkpoint, and a model reload are
   all currently on paths a user can hit repeatedly.
3. **Let the compiler and kernel libraries do their job.** The official SAM 3 and
   SAM 2 implementations compile specific submodules with `fullgraph=True`; this
   code compiles the whole model with `dynamic=True`. ROCm runs MIOpen in fast-find
   mode with benchmarking disabled, which may be leaving convolution performance on
   the table.
4. **Serialization can be the bottleneck.** Binary masks are sent as 8-bit PNGs,
   base64-encoded, for every object, on every brush response. A 1-bit PNG is
   lossless and roughly eight times smaller before compression; sending only the
   changed object is exact and can dominate the win for multi-object 4K sessions.

The top candidates, in rough priority order:

| ID | Change | Output tier | CUDA | ROCm | CPU | Effort | Expected |
|---|---|---|---|---|---|---|---|
| A3 | Keep models resident; evict only under real VRAM pressure | T1 exact | ✅ | ✅ | ✅ | S | Avoids multi-second reload + re-encode |
| A4 | Cache checkpoint SHA-256 by size + mtime | T1 exact | ✅ | ✅ | ✅ | S | Removes seconds per reload |
| A5 | No `empty_cache`/`ipc_collect`/`gc.collect` per request | T1 exact | ✅ | ✅ | ✅ | S | Removes allocator and GC churn |
| A1 | Cache SAM 3 text-encoder outputs | T1 exact | ✅ | ✅ | ✅ | S | Removes one 24-layer transformer pass per repeated prompt |
| G1 | Encode binary masks as 1-bit PNG | T1 exact | ✅ | ✅ | ✅ | S | ~8× smaller raw mask; faster encode |
| G3 | Send only changed masks / revisions | T1 exact | ✅ | ✅ | ✅ | M | Payload stops scaling with object count |
| B1 | Compile the ViT + pixel decoder, not the whole model | T2 numeric | ✅ | ⚠️ | ⚠️ | M | Compiler path designed for SAM 3 |
| B5 | Remove or scope `cudnn_disabled()` around decode | T2 numeric | ✅ | ✅ | n/a | S | Restores cuDNN/MIOpen kernels and cuDNN SDPA |
| E1 | Tune MIOpen once, persist the DB, then fast-find | T2 numeric | ❌ | ✅ | ❌ | M | Better convolution kernels at fixed shapes |
| F1 | Align `OMP_NUM_THREADS`, `torch.set_num_threads`, inter-op, OpenCV | T2 numeric | ✅ | ✅ | ✅ | S | CPU thread oversubscription/underuse |
| J3 | TensorRT split engine (CUDA only) | T2 | ✅ | ❌ | ❌ | XL | Large on paper; quality gate mandatory |
| J2 | OpenVINO encoder/decoder (CPU/Intel GPU) | T2 numeric | ❌ | ❌ | ✅ | XL | Promising on Intel CPU; needs export + validation |

Legend for output tiers: **T1** = bit-identical outputs; **T2** = numerically
equivalent, must pass the golden-mask gate; **T3** = lossy and therefore out of
scope. Platform cells: ✅ native/expected to work, ⚠️ experimental or needs a
specific configuration, ❌ not available, n/a = no effect on that platform.
"Expected" is qualitative; it is not a measurement.

---

## 2. What "without any loss in quality" means here

A speedup is only useful if it preserves the thing the user sees: the mask, the
alpha matte, and the exported image. Three tiers are used throughout:

### T1 — exact

The computation is unchanged. Only caching, scheduling, serialization, allocation,
or I/O changes. Acceptance:

- binary masks: `np.array_equal(actual, golden)` is true;
- scores and boxes: bit-identical, or identical after the same `float64` conversion;
- exported PNG: decoded pixels identical (file bytes may differ because compression
  settings changed);
- alpha: identical after the same 8-bit quantization.

PNG compression level, 1-bit vs 8-bit PNG, base64 vs binary transport, hash caching,
model residency, and text-embedding reuse are all T1.

### T2 — numerically equivalent

Same algorithm, but a different kernel, precision, reduction order, or library.
Examples: `torch.compile`, cuDNN/MIOpen algorithm selection, FlashAttention vs math
SDPA, thread-count changes, TF32, fp32 OpenVINO/TensorRT engines. These are not
automatically quality-preserving. Acceptance:

- **Preferred:** zero differing mask pixels across the golden set. This is often
  achievable because masks are thresholded and the differences are far below 0.5.
- **If not zero:** per-mask IoU ≥ 0.99995, total symmetric-difference pixels ≤ 0.005%
  of image area, score delta ≤ 1e-4, box delta ≤ 0.5 px, and a visual review of the
  changed pixels. A single-pixel change on a 4K image is usually invisible; a
  systematic boundary shift is not.
- Exported alpha: max delta ≤ 1/255 and PSNR ≥ 50 dB against the baseline.

If a T2 change cannot meet that bar, it is T3 and is not acceptable for this work.

### T3 — lossy or model-changing

Quantization (INT8/FP8), smaller backbones, lower encode resolution, pruning,
distillation, approximate attention, JPEG masks, and downscaled uploads. These may be
good product decisions, but they violate the constraint and are only listed in
§14 so they are not accidentally proposed as "free" wins.

> **Important:** the application already runs fp16 on GPU (`SAM3_QUANTIZE=16`,
> `SAM2_QUANTIZE=16`). That is an existing precision choice, not a new optimization.
> A speedup that changes fp16 to fp8 or int8 is a different quality tier.

---

## 3. Where the time goes today

### 3.1 Request flow

```
upload
  main._decode_image            PIL/cv2 decode, EXIF, cap to 4096² (LANCZOS/cv2)
  manager.load_image            cv2.cvtColor BGR→RGB at full resolution
  backend.set_image
    SAM 3: _prepare_input       resize longest side to encode_dim, pad if aspect > 1.6
           predictor.set_image  LetterBox to square, normalize, ViT, neck
    SAM 2: predictor.set_image  LetterBox 1024, Hiera encoder, high-res features
  (upload response returns after encoding)

prompt / stroke
  manager.sam3_predict          text encoder → detector → mask head
  or manager.predict            prompt encoder → mask decoder
  backend._restore_outputs      crop padding, bilinear to full resolution, > 0.5
  manager._brush_response       PNG-encode every object mask (cached per revision),
                                include all objects + full history entry
  FastAPI                        JSON with base64 PNG strings
  frontend                       base64 decode, image decode, canvas upload
```

### 3.2 Dominant costs (external evidence, not this app's measurements)

- **New image, SAM 3:** the vision backbone dominates. DART (2026) measures a
  439M-parameter ViT-H/14 at **78% of per-class latency (87 of 112 ms on an RTX
  4080 at 1008²)** for the full detector. Meta reports **~30 ms per image on an
  H200 with 100+ objects**, which is a different class of GPU.
- **Repeated prompt, SAM 3:** the image backbone is already skipped because
  `predictor.features` is reused. What remains is the 24-layer text encoder, the
  6-layer fusion encoder, the 6-layer/200-query decoder, the mask head, and the
  full-resolution upsample. DART measures the encoder-decoder at ~25 ms for three
  classes on an RTX 4080, i.e. roughly a quarter of the first-image cost.
- **SAM 2 stroke:** the Hiera encoder runs once per image; each stroke is the prompt
  encoder + a small mask decoder. The decoder is cheap, but the full-resolution
  bilinear upsample and mask PNG encoding are not free at 4K.
- **CPU:** the encoder dominates even more, but the text encoder, decoder, and PNG
  encode become material because they are no longer hidden behind a GPU.
- **Multi-object sessions:** the response includes **every** object's mask on every
  stroke (`_brush_response`), so JSON size and base64 cost grow with subject count,
  not with the one mask that changed.

These numbers are from the cited external sources. The actual split in this app is
unknown until the harness in §17 is run; it may be very different on a laptop CPU or
an RDNA3 GPU.

### 3.3 What is already optimized

Worth preserving, and worth not "optimizing" again:

- image features are cached between prompts (`predictor.features`);
- fp16 on GPU, fp32 on CPU (`quantize`), with CPU fp16 explicitly avoided;
- `torch.inference_mode()` around encode and decode;
- SAM 2 low-resolution mask feedback (`mask_input`), decoder warm-up, encoder
  offload, and batched box decoding (`predict_batch`);
- SAM 3 image resize/pad done once in `_prepare_input`, not per prompt;
- per-object PNG cache keyed by `mask_revision`;
- `cudnn.benchmark=True` on CUDA at startup, allocator configuration with
  `expandable_segments:True`;
- model residency with idle unload and OOM fallback;
- optional `channels_last`, `SAM2_COMPILE`, `SAM3_COMPILE`, encoder offload;
- single uvicorn worker (required for in-process model state), phase-tagged logs.

### 3.4 Concrete gaps found in the source

| Location | Observation |
|---|---|
| `backend/models/manager.py:141-215` | `_ensure_encode_headroom()` calls `_evict_loaded_models()` unconditionally, before every `set_image`/re-encode. On CUDA it evicts **all other segmentation models and the detector**, then calls `empty_cache()` and `ipc_collect()`. |
| `backend/models/manager.py:1215-1282` | `detect()` unloads the segmentation model by default (`SPECTRA_KEEP_SEG_ON_DETECT != "1"`) and ends with `release_gpu_memory()`, which calls `empty_cache()`, `ipc_collect()`, and `gc.collect()`. |
| `backend/utils/model_integrity.py:87-145` + `backend/models/manager.py:252-262` | `verify_model()` computes a full SHA-256 every time `ModelManager.get_or_load_model()` has to load or reload a model. There is no size/mtime shortcut. `sam3.pt` is multi-GB. |
| `backend/models/sam3_backend.py:130-137` | `SAM3_COMPILE=1` compiles the whole predictor model with `dynamic=True`. The official SAM 3 builder compiles only the ViT and the pixel decoder with `fullgraph=True`. |
| `backend/models/sam3_backend.py:263,290` | `cudnn_disabled()` wraps every text and box decode. This disables cuDNN (and likely the cuDNN SDPA backend) on CUDA, and MIOpen on ROCm, for the entire decoder. |
| `backend/models/ultralytics_backend.py:370,450` | Same `cudnn_disabled()` pattern for SAM 2 decode and batch decode. |
| `backend/models/sam3_backend.py:120-129` | `channels_last` converts the whole model. ViT is Linear-heavy and NCHW; only the conv neck/pixel decoder can benefit. |
| `backend/models/manager.py:575-612` | `_brush_response()` includes every object's PNG on every predict. The PNG encoding is cached, but base64 + JSON serialization is not. |
| `backend/utils/compositing.py:11-15` | Binary masks are converted to 8-bit grayscale (`0/255`) and encoded as a normal PNG. A 1-bit PNG is exact and much smaller. |
| `backend/utils/compositing.py:44-54` | Feathering pads and blurs a full-resolution float32 mask. At 4K this is a large allocation and a large convolution. |
| `backend/utils/compositing.py:76-98` + `backend/main.py:1318-1375` | Export re-encodes PNGs with Pillow's default compression level (6) and puts them in a `ZIP_DEFLATED` archive (`main.py:1338`), even though PNG is already compressed. |
| `backend/run.cmd:35` | `OMP_NUM_THREADS` is capped at `min(6, physical cores)`, while `torch_threads.resolve_threads()` can set up to 32. The two knobs disagree. |
| `backend_amd_gpu/run.cmd:27-31` | MIOpen runs with `FIND_MODE=2` (fast), `FIND_ENFORCE=1` (none), `SEARCH_CUTOFF=1`; `backend_amd_gpu/main.py:440` sets `cudnn.benchmark=False`. Fixed 1008/1024 shapes are never exhaustively tuned. |
| `backend_amd_gpu/run.cmd:33-34` | `TORCH_BLAS_PREFER_HIPBLASLT=1` and `DISABLE_ADDMM_CUDA_LT=1` are set, but `PYTORCH_TUNABLEOP_CACHE_DIR` is set without `PYTORCH_TUNABLEOP_ENABLED=1`. |
| `backend/models/yoloe_detector.py:49,120,174` | The detector stores `_device` but never passes it to `YOLOE(...)` or `predict()`. Ultralytics chooses the device itself. |
| `backend/main.py:339-365` | CUDA allocator configuration is set inside `lifespan`, after `torch` import. `PYTORCH_CUDA_ALLOC_CONF` is read at allocator initialization, so this is fragile and should move to the launcher. |

---

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
- `HSA_OVERRIDE_GFX_VERSION` is already applied for gfx101x/103x/110x by
  `amd_gfx.py`; do not stack a second override manually.

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

## 15. Platform summary

Every catalogue entry is listed here so the cross-platform requirement can be
checked without reading all 85 catalogue entries. Legend: ✅ = expected to work with the
normal install, ⚠️ = experimental, needs a specific torch/build/driver, or only
partially applicable, ❌ = not available/irrelevant on that platform, — =
informational entry with no per-platform change (or no device-specific behavior).
"Windows note" applies to all three device columns.

### 15.1 Orchestration, serving, loading (A, H, I)

| Change | CUDA | ROCm | CPU | Windows note |
|---|---|---|---|---|
| A1 text cache | ✅ | ✅ | ✅ | None |
| A2 image-feature reuse | ✅ | ✅ | ✅ | Verify session ownership first |
| A3 model residency | ✅ | ✅ | ✅ | Biggest win when alternating Detect/Brush; CPU also unloads on `detect()` |
| A4 hash cache | ✅ | ✅ | ✅ | Antivirus can amplify the win |
| A5 no per-request cleanup | ✅ | ✅ | ✅ | `ipc_collect` is Windows-relevant; CPU only pays `gc.collect` |
| A6 warm-up | ✅ | ✅ | ✅ | MIOpen/cuDNN tuning happens here |
| A7 batch pending prompts | ✅ | ✅ | ✅ | Same-host requests only; holds `_infer_lock` longer |
| A8 prompt memoization | ✅ | ✅ | ✅ | TTL must be short; invalidate on image/model change |
| A9 one image buffer | ✅ | ✅ | ✅ | Pure RAM win; large images only |
| A10 mask re-serialization | ✅ | ✅ | ✅ | Overlaps G3 |
| H1 one worker | ✅ | ✅ | ✅ | In-process state is not multi-worker safe |
| H2 async encode | ✅ | ✅ | ✅ | Perceived-latency win, not throughput |
| H3 request coalescing | ✅ | ✅ | ✅ | Depends on A8 |
| H4 keep-alive | ✅ | ✅ | ✅ | Windows TCP setup cost is noticeable |
| H5 model ownership | ✅ | ✅ | ✅ | Correctness prerequisite for A2 |
| H6 thread-pool sizing | ✅ | ✅ | ✅ | Default executor is usually fine |
| H7 startup order | ✅ | ✅ | ✅ | Env vars must precede Python |
| I1 `mmap` load | ✅ | ✅ | ✅ | Benefits large `.pt`; verify RSS |
| I2 fast storage | ✅ | ✅ | ✅ | Exclude weights from antivirus during tests |
| I3 digest cache | ✅ | ✅ | ✅ | Same as A4 |
| I4 `low_cpu_mem_usage` | ✅ | ✅ | ✅ | Detector load path only |
| I5 compiler caches | ✅ | ⚠️ | ⚠️ | Inductor/Triton cache format is version-specific |

### 15.2 Model execution (B, C, D, E, F)

| Change | CUDA | ROCm | CPU | Windows note |
|---|---|---|---|---|
| B1 compile submodules | ✅ | ⚠️ | ⚠️ | MSVC for CPU; ROCm Triton on Windows uncertain |
| B2 static shapes | ✅ | ⚠️ | ⚠️ | Recompiles when `encode_dim` changes |
| B3 warm compiled graph | ✅ | ⚠️ | ⚠️ | Cache the first compile; never on the request path |
| B4 scope `channels_last` | ✅ | ✅ | ⚠️ | CPU oneDNN may be neutral or slower |
| B5 remove `cudnn_disabled` | ✅ | ✅ | — | May be a driver workaround; A/B it |
| B6 SDPA backend | ✅ | ✅ | ⚠️ | CPU has math only, not flash |
| B7 TF32/reduced reduction | ✅ | ❌ | ❌ | TF32 is sm80+ only; reduction flag is ROCm-specific (see E) |
| B8 `flash-attn-3`/`cc_torch` | ✅ | ❌ | ❌ | Not exposed by Ultralytics; integration required |
| B9 activation checkpointing | — | — | — | Already disabled; no change recommended |
| B10 SAM 3.1 | — | — | — | Informational: not an image-speed upgrade |
| C1 compile encoder only | ✅ | ⚠️ | ⚠️ | `max-autotune` first compile is long |
| C2 keep decoder eager | — | — | — | Guidance, no code change |
| C3 batched box decode | ✅ | ✅ | ✅ | Already present; ensure UI uses it |
| C4 low-res mask feedback | ✅ | ✅ | ✅ | Already present; do not replace |
| C5 `cudnn_disabled`/`channels_last` | ✅ | ✅ | ⚠️ | Same as B4/B5 |
| C6 CPU compile opt-in | ❌ | ❌ | ✅ | MSVC/clang-cl required |
| D1 cuDNN benchmark sweep | ✅ | ❌ | ❌ | Standard backend only; AMD backend disables benchmark |
| D2 cuBLASLt | ✅ | ❌ | ❌ | Test on the actual GPU/driver |
| D3 allocator config | ✅ | ❌ | — | Set before Python starts |
| D4 `CUDA_DEVICE_MAX_CONNECTIONS` | ✅ | ❌ | ❌ | CUDA-only variable |
| D5 remove `cuda.synchronize` | ✅ | ✅ | — | Keep it in benchmark mode |
| D6 cuDNN SDPA | ✅ | ❌ | ❌ | Do not disable it pre-emptively |
| D7 memory cap | ✅ | ✅ | — | Cap can force a slow fallback; measure |
| D8 driver/power/AV | ✅ | ✅ | ✅ | Largest laptop source of run-to-run noise |
| E1 MIOpen tuning | ❌ | ✅ | ❌ | Windows ROCm: verify the user DB path |
| E2 TunableOp | ❌ | ✅ | ❌ | Cache dir already set; enable it explicitly |
| E3 hipBLASLt/rocBLAS | ❌ | ✅ | ❌ | Already preferred; verify with a profiler |
| E4 AOTriton vs CK | ❌ | ✅ | ❌ | gfx support varies; check `can_use_flash_attention` |
| E5 HIP allocator config | ❌ | ✅ | ❌ | Version-dependent variable name |
| E6 MIOpen decode | ❌ | ✅ | — | Same as B5 on ROCm |
| E7 OpenCV OpenCL | ❌ | ✅ | ❌ | Disable to avoid competing with ROCm |
| E8 Windows ROCm notes | — | ⚠️ | — | Checklist, not an optimization |
| F1 align thread knobs | ✅ | ✅ | ✅ | Env before Python; sweep, do not assume |
| F2 inter-op = 1 | ✅ | ✅ | ✅ | Call before parallel work starts |
| F3 OpenCV threads | ✅ | ✅ | ✅ | Affects host preprocessing on GPU too |
| F4 per-phase threads | ✅ | ✅ | ✅ | Biggest effect on CPU; harmless on GPU |
| F5 `KMP_AFFINITY`/`KMP_BLOCKTIME` | ✅ | ✅ | ✅ | Host threads only; P/E-core pinning risk |
| F6 CPU Inductor | ❌ | ❌ | ✅ | MSVC/clang-cl required; long compile |
| F7 oneDNN/MKLDNN | ❌ | ❌ | ✅ | Enabled by default; verify, do not disable |
| F8 CPU bf16 autocast | ❌ | ❌ | ⚠️ | Only AVX512-BF16/AMX; T2 |
| F9 IPEX | ❌ | ❌ | ⚠️ | Intel CPU; wheel/torch version matching is fragile |
| F10 OpenVINO | ❌ | ❌ | ✅ | Intel CPU/GPU; AMD CPU works but is not tuned |
| F11 ONNX Runtime | ✅ | ⚠️ | ✅ | DirectML for AMD Windows; ROCm provider is Linux |
| F12 flush denormals | ❌ | ❌ | ✅ | Tiny, numerically different; T2 |
| F13 NUMA/affinity | ❌ | ❌ | ✅ | Server CPUs only |
| F14 no fp16 on CPU | — | — | — | Informational guard against a common mistake |
| F15 persist oneDNN/OpenMP settings | ❌ | ❌ | ✅ | Set in the launcher, not at runtime |

### 15.3 Masks, serialization, export (G)

| Change | CUDA | ROCm | CPU | Windows note |
|---|---|---|---|---|
| G1 1-bit PNG | ✅ | ✅ | ✅ | Verify the frontend decoder |
| G2 explicit PNG compression | ✅ | ✅ | ✅ | Lossless; measure total transfer, not encode alone |
| G3 changed masks only | ✅ | ✅ | ✅ | Protocol change; needs revisions |
| G4 binary response | ✅ | ✅ | ✅ | Frontend fetch change |
| G5 faster mask decode | ✅ | ✅ | ✅ | `cv2.imdecode`; keep threshold identical |
| G6 export compression | ✅ | ✅ | ✅ | Lossless; `ZIP_STORED` likely better |
| G7 drop full-array scans | ✅ | ✅ | ✅ | Pure CPU win on every export |
| G8 fewer feather allocations | ✅ | ✅ | ✅ | Large-image RAM/time win |
| G9 bbox-cropped masks | ✅ | ✅ | ✅ | Protocol change; exact reconstruction required |
| G10 no gzip on PNG | ✅ | ✅ | ✅ | Keep it that way if middleware is added |
| G11 `orjson` responses | ✅ | ✅ | ✅ | New dependency; exact JSON, different bytes |

### 15.4 Alternative runtimes (J)

| Change | CUDA | ROCm | CPU | Windows note |
|---|---|---|---|---|
| J1 ONNX Runtime | ✅ | ⚠️ | ✅ | DirectML for AMD Windows; MIGraphX is Linux |
| J2 OpenVINO | ❌ | ❌ | ✅ | Intel CPU/GPU; separate export |
| J3 TensorRT | ✅ | ❌ | ❌ | CUDA only; quality gate mandatory |
| J4 `torch.export` + AOTInductor | ✅ | ⚠️ | ⚠️ | Windows support limited; ROCm depends on Inductor |
| J5 DirectML/MIGraphX | ❌ | ⚠️ | ❌ | DirectML: AMD Windows; MIGraphX: ROCm Linux |

**Reading the table:** an item can be ✅ on a platform and still be a poor first
experiment. Priority and expected magnitude are in §1 and §16; this table answers
only "is it applicable here?".

---

## 16. Recommended experiment order

1. **Build the harness and golden set (§17) before changing anything.** Without a
   baseline and a mask-equality gate, every later result is a guess.
2. **Measure the baseline phase breakdown** on one CUDA machine, one ROCm machine,
   and one CPU machine: decode, upload color conversion, encode, text, detector,
   mask head, upsample, PNG encode, JSON, export.
3. **Apply T1 wins first**: A3 residency, A4 hash cache, A5 cleanup removal, A1 text
   cache, G1 1-bit PNG, G3 changed-masks response, G6 export settings, G7 scan
   removal, D5 sync removal. These should not change a single mask pixel.
4. **Then the T2 experiments**: B5 `cudnn_disabled`, B1 compile submodules, B6 SDPA,
   E1 MIOpen tuning, F1 thread sweep. Validate masks after each, one variable at a
   time.
5. **Only then consider runtime alternatives** (J1/J2/J3), and only if the measured
   model time is still the dominant cost. If serialization is 60% of latency, a 2×
   TensorRT engine will not fix the product.
6. **Re-run the full matrix after each change**; keep a results table with medians,
   p95, peak memory, and a quality verdict.

---

## 17. How to write a proper performance test

This section is a methodology, not a test suite. **The code below is a template;
it has not been executed in this checkout.** The goal is a test that (a)
measures the right thing, (b) proves the output did not change, and (c) produces a
number someone can trust next month.

### 17.1 Principles

1. **Measure quality before speed.** A fast wrong mask is worthless. Run the
   equality gate first, then the timing.
2. **Separate cold, warm, and steady state.** Model load, checkpoint hash, compile,
   and first-request autotune are one-time costs. Report them separately from p50.
3. **Synchronize GPU work.** A wall-clock timer around an asynchronous launch
   measures the launch, not the kernel. Use CUDA events or synchronize at phase
   boundaries.
4. **Do not measure allocation or GC as if it were model time.** Keep the allocator
   warm; do not call `empty_cache()` between iterations; disable debug logging.
5. **Use fixed inputs.** Same image bytes, same prompt, same model hash, same
   `encode_dim`, same thread count, same device. Record all of them.
6. **Repeat and report distributions.** A single run is noise. Report median, p95,
   and a confidence interval; compare A/B in the same session, interleaved.
7. **One variable per process.** `torch.set_num_threads`, `MIOPEN_FIND_MODE`,
   compile mode, and dtype changes should each get a fresh process. Allocator and
   compiler state leaks across configurations.
8. **Do not assert absolute milliseconds in CI.** Assert quality always; assert
   relative improvement only on dedicated hardware, with a tolerance.
9. **Benchmark the product path, not just the model.** The HTTP JSON/PNG layer can
   dominate; measure both the backend call and the full request.
10. **Record the environment.** GPU, driver, ROCm/CUDA version, torch version,
    Python version, CPU model, OS build, power plan.

### 17.2 Test matrix

At minimum:

| Dimension | Values |
|---|---|
| Device | `cuda` (NVIDIA), `cuda` (ROCm), `cpu` |
| Model | `sam3.pt`, `sam2.1_b.pt` (and one small SAM 2 for CPU) |
| Image | 1024², 2048², 4032×3024 (12 MP), 6000×4000 (24 MP, gets capped) |
| Prompt | one object, three objects, a negative/empty prompt, a long phrase |
| State | cold load, first encode, repeated prompt, new prompt, undo/redo |
| Objects | 1, 5, 20 subjects (for response-size effects) |
| Export | 1 mask, whole image, 4K feather radius 0/3/10, PNG and JPEG |
| Concurrency | 1, 2, 4 clients (for throughput/p95) |

Do not benchmark only a 1024² image. The mask PNG and full-resolution upsample
effects only appear at 4K, and those are the phases most likely to dominate a real
background-removal session.

### 17.3 Suggested file layout

```
tests/
  perf/
    conftest.py              # CLI options, environment capture, fixtures
    test_pipeline_perf.py    # phase timings + quality gate
    test_http_perf.py        # optional: full request path
    quality_gate.py          # mask equality / IoU helpers
    report.py                # JSON + markdown summary
    golden/                  # generated by the baseline commit, not by the test
      sam3_1024_person.npz
      sam2_2048_person.npz
```

Keep the golden files out of the normal unit-test path; they are large and
model-specific. Generate them once from the baseline commit and store the model
hash, image hash, prompt, device, and torch version in the `.npz` metadata.

### 17.4 CLI and environment capture

`conftest.py` should add options rather than read ad-hoc environment variables:

```python
# tests/perf/conftest.py
import json
import os
import platform
import subprocess
import sys

import pytest
import torch


def pytest_addoption(parser):
    parser.addoption("--device", default="cuda", choices=["cuda", "cpu"])
    parser.addoption("--model", default="sam3.pt")
    parser.addoption("--image", required=True)
    parser.addoption("--prompt", default="person")
    parser.addoption("--repeats", type=int, default=30)
    parser.addoption("--warmup", type=int, default=3)
    parser.addoption("--golden", default=None)
    parser.addoption("--out", default="perf-results.json")


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


@pytest.fixture(scope="session")
def env_report() -> dict:
    return {
        "git": git_revision(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "hip": torch.version.hip,
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        ),
        "os": platform.platform(),
        "cpu_count": os.cpu_count(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "miopen_find_mode": os.environ.get("MIOPEN_FIND_MODE"),
        "torch_threads": torch.get_num_threads(),
    }
```

**Important:** `OMP_NUM_THREADS` and allocator variables must be set before Python
starts. Do not set them inside the test and expect OpenMP to honor them. Run the
matrix with a wrapper that sets the environment per process, or use
`subprocess.run(..., env=...)` for each configuration.

Run pytest from the repository root with the root on `PYTHONPATH` (the backend uses
package-qualified imports such as `backend.models.manager`):

```bash
PYTHONPATH=. backend/.venv/Scripts/python.exe -m pytest tests/perf -m perf \
  --device cuda --model sam3.pt --image samples/portrait.jpg --repeats 30
```

Register the marker in `pytest.ini` (`markers = perf: performance benchmarks`) so
`-m perf` does not warn.

### 17.5 A phase timer that is correct on GPU and CPU

```python
# tests/perf/timing.py
import time
import torch


class PhaseTimer:
    """Measures one phase, synchronizing CUDA/HIP at both boundaries."""

    def __init__(self, device: torch.device):
        self.device = device
        self.on_gpu = device.type == "cuda" and torch.cuda.is_available()
        self.elapsed_ms = float("nan")

    def __enter__(self):
        if self.on_gpu:
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(self.device)
            self.start.record()
        else:
            self.t0 = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.on_gpu:
            self.end.record()
            torch.cuda.synchronize(self.device)
            self.elapsed_ms = self.start.elapsed_time(self.end)
        else:
            self.elapsed_ms = (time.perf_counter_ns() - self.t0) / 1e6
        return False
```

`perf_counter_ns` avoids the resolution and wall-clock issues of `time.time()`.
`torch.cuda.Event.elapsed_time` returns milliseconds. Synchronizing at phase
boundaries adds a few microseconds but makes the number meaningful.

### 17.6 Model-level benchmark

This measures the backend directly, without HTTP. It is the right first test
because it isolates encode, text, and decode.

```python
# tests/perf/test_pipeline_perf.py
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from backend.models.sam3_backend import SAM3Backend
from backend.models.ultralytics_backend import UltralyticsBackend
from tests.perf.timing import PhaseTimer


def _load_backend(model: str, device: str):
    if model == "sam3.pt":
        backend = SAM3Backend(device=device)
    else:
        backend = UltralyticsBackend(device=device)
    backend.load_model(model)
    return backend


@pytest.mark.perf
def test_encode_and_prompt(pytestconfig):
    device = pytestconfig.getoption("--device")
    model = pytestconfig.getoption("--model")
    image_path = pytestconfig.getoption("--image")
    prompt = pytestconfig.getoption("--prompt")
    repeats = pytestconfig.getoption("--repeats")
    warmup = pytestconfig.getoption("--warmup")

    torch_device = torch.device("cuda" if device == "cuda" else "cpu")
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    assert image is not None

    backend = _load_backend(model, device)

    encode_ms = []
    prompt_ms = []
    for i in range(warmup + repeats):
        with PhaseTimer(torch_device) as t:
            backend.set_image(image)
        if i >= warmup:
            encode_ms.append(t.elapsed_ms)

        with PhaseTimer(torch_device) as t:
            if model == "sam3.pt":
                result = backend.predict_text(prompt, confidence=0.5)
            else:
                result = backend.predict(points=[[image.shape[1] / 2, image.shape[0] / 2]],
                                         labels=[1])
        if i >= warmup:
            prompt_ms.append(t.elapsed_ms)

    def summarize(xs):
        a = np.asarray(xs, dtype=np.float64)
        return {
            "n": int(a.size),
            "median_ms": float(np.median(a)),
            "p95_ms": float(np.quantile(a, 0.95)),
            "min_ms": float(a.min()),
            "max_ms": float(a.max()),
        }

    report = {
        "encode": summarize(encode_ms),
        "prompt": summarize(prompt_ms),
        "shapes": {"masks": list(result["masks"].shape)},
    }
    Path(pytestconfig.getoption("--out")).write_text(json.dumps(report, indent=2))

    if pytestconfig.getoption("--golden"):
        golden = np.load(pytestconfig.getoption("--golden"))
        assert result["masks"].shape == golden["masks"].shape
        assert np.array_equal(result["masks"], golden["masks"]), "mask changed"
        assert np.allclose(result["scores"], golden["scores"], atol=1e-6)
        assert np.allclose(result["bboxes"], golden["bboxes"], atol=1e-3)
```

Notes:

- `set_image` includes preprocessing and the encoder. If you want to separate
  preprocessing from the encoder, call the internal `_prepare_input` and
  `_encode_image` separately, or use `torch.profiler` (17.11).
- **This test calls `backend.load_model()` directly, which uses
  `ensure_model_file()` and therefore skips the SHA-256 verification that
  `ModelManager.get_or_load_model()` performs.** Measure the manager path (17.7) to
  see the A4 cost; do not conclude from this test that model loading is cheap.
- The prompt loop intentionally reuses the features. That is the interactive case.
  Measure a new-image prompt separately.
- For SAM 2, use a point inside the object rather than the image center; the test
  above is only a template.
- Save the golden file from the baseline commit. Do not generate it from the same
  code you are testing:

  ```python
  # tests/perf/make_golden.py — run once on the baseline commit
  np.savez_compressed(
      "golden/sam3_1024_person.npz",
      masks=result["masks"], scores=result["scores"], bboxes=result["bboxes"],
      meta=np.array([model_hash, image_sha256, prompt, torch.__version__, str(device)]),
  )
  ```

### 17.7 Manager-level benchmark (A3, A4, A5)

The product path goes through `ModelManager`, and that is where residency,
checkpoint hashing, per-request cleanup, and mask serialization live. The
backend-level test above cannot see any of them.

```python
# tests/perf/test_manager_perf.py
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.models.manager import ModelManager


@pytest.mark.perf
def test_manager_cold_load(pytestconfig):
    device = "cuda:0" if pytestconfig.getoption("--device") == "cuda" else "cpu"
    model = pytestconfig.getoption("--model")
    image = cv2.imread(pytestconfig.getoption("--image"), cv2.IMREAD_COLOR)
    assert image is not None

    def timed(fn):
        t0 = time.perf_counter()
        result = fn()
        return (time.perf_counter() - t0) * 1000.0, result

    manager = ModelManager(device=device)

    # A4: cold load includes the full SHA-256 when the checkpoint is already on
    # disk (get_or_load_model -> verify_model). The second call hits the residency
    # fast path and must be near zero.
    cold_ms, _ = timed(lambda: manager.get_or_load_model(model))
    resident_ms, _ = timed(lambda: manager.get_or_load_model(model))

    session = manager.create_session(model)
    encode_ms, _ = timed(
        lambda: manager.load_image(session.session_id, image, "perf-image-1")
    )

    # A3: on CUDA, _ensure_encode_headroom() evicts every other loaded model. With
    # only one segmentation model resident there is nothing to evict; load the
    # detector first (see the notes) to expose that cost.
    reupload_ms, _ = timed(
        lambda: manager.load_image(session.session_id, image, "perf-image-2")
    )

    # Prompt latency includes the text encoder, decoder, and mask packing. A10/G3
    # (PNG/base64 serialization) is added by the HTTP layer; measure it in 17.10.
    prompt = pytestconfig.getoption("--prompt")
    for _ in range(pytestconfig.getoption("--warmup")):
        manager.sam3_predict(session.session_id, prompt, confidence=0.5)
    prompt_ms = []
    for _ in range(pytestconfig.getoption("--repeats")):
        ms, response = timed(
            lambda: manager.sam3_predict(session.session_id, prompt, confidence=0.5)
        )
        prompt_ms.append(ms)

    manager.destroy_session(session.session_id)

    samples = np.asarray(prompt_ms)
    report = {
        "cold_load_ms": cold_ms,
        "resident_load_ms": resident_ms,
        "first_encode_ms": encode_ms,
        "reupload_ms": reupload_ms,
        "prompt": {
            "n": int(samples.size),
            "median_ms": float(np.median(samples)),
            "p95_ms": float(np.quantile(samples, 0.95)),
        },
        "response_keys": sorted(response.keys()),
        "masks_shape": list(response["masks"].shape),
    }
    Path(pytestconfig.getoption("--out")).write_text(json.dumps(report, indent=2))
```

Notes:

- The example assumes `--model sam3.pt`; for SAM 2, replace the `sam3_predict`
  loop with the `manager.predict(...)` call described below.
- `cold_ms` includes SHA-256, `torch.load`, predictor setup, and any compile/warm-up.
  A4 removes only the hashing portion; time `verify_model()` directly (cold vs
  cached) to size that piece. If the checkpoint was downloaded in this process, the
  hash was already paid once and the cold number will look smaller; run against a
  pre-populated `model_weights/`.
- To measure A3 directly, load a detector with `manager.load_detector(...)` before
  the second upload, or alternate two model names (for example `sam3.pt` then
  `sam2.1_b.pt`), so a load triggers eviction. That test needs the extra checkpoints
  and is CUDA-only by design.
- For SAM 2 strokes, `manager.predict(session_id, object_id, points=..., labels=[1])`
  appends a history entry per call. Call `manager.clear_object(session_id,
  object_id)` between iterations, outside the timer, or history will grow and the
  later iterations will measure replay instead of decode.
- `sam3_predict()` returns Python structures, not the HTTP JSON. Response bytes and
  base64 cost must be measured at the HTTP layer (17.10).
- The call to `manager.detect()` is deliberately absent: it unloads the segmentation
  model by default (`SPECTRA_KEEP_SEG_ON_DETECT`), so it belongs in a separate test
  that measures the switch, not in the prompt benchmark.

### 17.8 Quality gate

```python
# tests/perf/quality_gate.py
import numpy as np


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def assert_equivalent(actual: dict, golden: dict, tier: str = "T1") -> None:
    am, gm = np.asarray(actual["masks"]), np.asarray(golden["masks"])
    assert am.shape == gm.shape, (am.shape, gm.shape)

    if tier == "T1":
        assert np.array_equal(am, gm), "T1 requires bit-identical masks"
    else:
        for i, (a, g) in enumerate(zip(am, gm)):
            iou = mask_iou(a, g)
            diff = np.logical_xor(a, g).sum()
            assert iou >= 0.99995, f"mask {i}: IoU {iou:.6f}"
            assert diff <= 0.00005 * a.size, f"mask {i}: {diff} differing pixels"

    assert np.allclose(actual["scores"], golden["scores"], atol=1e-4, rtol=0)
    assert np.allclose(actual["bboxes"], golden["bboxes"], atol=0.5, rtol=0)
```

For exported images, compare decoded arrays; do not compare PNG bytes. For alpha,
allow at most one 8-bit level of difference in T2:

```python
def assert_alpha_close(actual: np.ndarray, golden: np.ndarray) -> None:
    delta = np.abs(actual.astype(np.int16) - golden.astype(np.int16))
    assert delta.max() <= 1
    mse = np.mean((actual.astype(np.float64) - golden.astype(np.float64)) ** 2)
    psnr = 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
    assert psnr >= 50.0
```

### 17.9 Statistics that do not lie

```python
import numpy as np


def summarize(samples_ms: list[float], seed: int = 0) -> dict:
    a = np.asarray(samples_ms, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = np.median(
        rng.choice(a, size=(4000, a.size), replace=True), axis=1
    )
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {
        "n": int(a.size),
        "median_ms": float(np.median(a)),
        "p95_ms": float(np.quantile(a, 0.95)),
        "median_ci95_ms": [float(lo), float(hi)],
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
    }


def relative(a: dict, b: dict) -> dict:
    ratio = b["median_ms"] / a["median_ms"]
    return {"ratio": ratio, "speedup": 1.0 / ratio}
```

Rules:

- Use the median for "typical"; use p95 for "interactive worst case".
- Require the 95% CI of the **ratio** to exclude 1.0 before claiming an improvement.
  A simple way is to bootstrap the two samples together, or run A/B in alternating
  blocks and bootstrap the paired differences.
- Discard a run only if you can name the cause (thermal throttle, driver reset,
  another process). Do not discard outliers to make a result look better.
- Report variance. A 30% improvement with a ±25% interval is not a result.

### 17.10 HTTP-level benchmark

The model benchmark omits JSON and base64, which can dominate at 4K. Add a second
test that starts the real server in a subprocess and measures the request:

```python
import subprocess
import time

import httpx


def wait_for(url: str, timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=1.0).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise TimeoutError(url)


with subprocess.Popen(
    ["backend/.venv/Scripts/python.exe", "-m", "uvicorn",
     "backend.main:app", "--host", "127.0.0.1", "--port", "8765",
     "--log-level", "warning"],
) as server:
    wait_for("http://127.0.0.1:8765/health")
    with httpx.Client(base_url="http://127.0.0.1:8765", timeout=120) as client:
        # create session, upload, prompt; time each request with perf_counter_ns
        ...
    server.terminate()
```

Measure: upload time, encode wait, prompt time, response bytes, JSON decode time,
and client-side base64/PNG decode time. The last two are often missing from backend
benchmarks and are exactly what makes a 4K session feel slow.

### 17.11 Profiling, not guessing

- **PyTorch profiler:**

  ```python
  from torch.profiler import ProfilerActivity, profile

  activities = [ProfilerActivity.CPU]
  if torch.cuda.is_available():
      activities.append(ProfilerActivity.CUDA)
  with profile(activities=activities, record_shapes=True) as prof:
      backend.predict_text("person")
  print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))
  prof.export_chrome_trace("trace.json")
  ```

  On ROCm, `ProfilerActivity.CUDA` maps to HIP kernels. On CPU-only, use
  `self_cpu_time_total`.

- **Compile diagnostics:** `TORCH_LOGS=graph_breaks,recompiles` and
  `torch._dynamo.explain(model)(...)`. A graph break in the detector's Python
  post-processing means the compile only covers part of the model.

- **CUDA:** Nsight Systems (`nsys profile`) and Nsight Compute (`ncu`) for kernel
  time and occupancy. Windows is supported by Nsight Systems.

- **ROCm:** `rocprofv3` for kernel traces; `MIOPEN_ENABLE_LOGGING=1` and
  `MIOPEN_ENABLE_LOGGING_ELAPSED_TIME=1` to see which convolution kernels were
  selected; `AMD_LOG_LEVEL` for HIP API traces (very verbose).

- **CPU:** `torch.profiler` for op-level time; on Windows, ETW/WPA or Intel VTune
  for thread/affinity analysis.

- **Memory:** `torch.cuda.reset_peak_memory_stats()`,
  `torch.cuda.max_memory_allocated()`, and `psutil.Process().memory_info().rss`.
  Report peak, not just steady state.

### 17.12 Benchmark hygiene checklist

- [ ] Same checkpoint hash and same image bytes for A and B.
- [ ] `model.eval()`; no dropout; no training-mode checkpointing.
- [ ] Warm-up ≥ 3 iterations; for `torch.compile`, until compile/recompile settles.
- [ ] Report compile time and first-request time separately.
- [ ] Synchronize at phase boundaries; no bare `time.time()` around GPU work.
- [ ] No `empty_cache()` between iterations.
- [ ] No debug/profiler logging during the timed run.
- [ ] Fixed clocks/power plan; not on battery; no background GPU work.
- [ ] Environment variables set before Python starts.
- [ ] One configuration per process for thread/compile/MIOpen changes.
- [ ] A/B/A ordering or randomized blocks to detect drift.
- [ ] At least 3 independent process runs; report the spread.
- [ ] Quality gate runs before the timing comparison.
- [ ] Store the raw samples, not only the summary.

### 17.13 Acceptance criteria for a change

A change is accepted only if all of these hold:

1. **Quality:** T1 passes bit-identical masks; T2 passes the golden gate with zero
   or negligible differing pixels and reviewed score/box deltas.
2. **Speed:** the median improves by at least 5% on the target workload, with a 95%
   confidence interval that excludes no improvement. p95 must not regress by more
   than 2%.
3. **Memory:** peak VRAM/RSS does not increase by more than 10% unless the change
   is specifically a memory trade-off, in which case it must be documented.
4. **Cold start:** no more than a 10% regression, or the compile/load cost is moved
   to an explicit warm-up and clearly reported.
5. **Reproducibility:** two independent runs agree within 3% on the median.
6. **Cross-platform:** the change is tested on at least one CUDA, one ROCm, and one
   CPU configuration before it becomes a default. If a platform is skipped, say so
   in the results table and keep the change opt-in.

### 17.14 What would invalidate the benchmark

- Comparing a compiled/warm run against an eager/cold run.
- Measuring a cached prompt against an uncached one.
- Changing the image, prompt, model, or `encode_dim` between A and B.
- Running on a laptop on battery, or with a game/other GPU workload in the
  background.
- Reporting the mean of a bimodal distribution (e.g. first call + steady state).
- Comparing PNG bytes when only compression changed, or comparing masks after a
  different resize.
- Using `torch.cuda.synchronize()` only in one arm.
- Letting MIOpen/Inductor caches differ between arms without saying so.
- Running A and B in the same process when one changes thread count or compile mode.
- Ignoring the HTTP/JSON/base64 layer when the product is interactive.

---

## 18. Sources and further reading

### SAM 3 / SAM 3.1

- **SAM 3.1 release blog**, Meta, 2026-03-27. Object Multiplex for video; ~7× at 128
  objects on an H100; image detector unchanged.
  <https://ai.meta.com/blog/segment-anything-model-3/>
- **SAM 3 README and code**, facebookresearch/sam3. `build_sam3_image_model` with
  `compile=True` compiles the ViT and pixel decoder; `_setup_tf32()` enables TF32 on
  sm80+; `VETextEncoder.forward` accepts pre-encoded text; `Sam3Processor` uses
  1008² and a 32-token text context; optional `flash-attn-3` and `cc_torch`.
  <https://github.com/facebookresearch/sam3>
- **SAM 3.1 release notes**, 2026-03-27. Multiplex, reduced CPU-GPU sync, improved
  `torch.compile` fusion, batched postprocessing.
  <https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md>
- **SAM 3: Segment Anything with Concepts**, arXiv:2511.16719.
  <https://arxiv.org/abs/2511.16719>
- **Ultralytics SAM 3 docs.** `quantize=16` for FP16; `inference_features` for
  feature reuse; SAM 3.1 image support; the image detector keeps the SAM 3
  architecture.
  <https://docs.ultralytics.com/models/sam-3>

### 2026 acceleration research

- **DART: Detect Anything in Real Time**, arXiv:2603.11441, 2026-03-12. Measures
  SAM 3's ViT-H/14 backbone at 439M parameters and 78% of per-class compute (87 of
  112 ms on an RTX 4080 at 1008²); caches text embeddings; batches class prompts;
  documents TensorRT FP16 graph restructuring and the quality failure of naive
  export (cosine 0.058).
  <https://arxiv.org/html/2603.11441v1>
- **PyTorch 2.10 release blog**, 2026-01-21. Combo-kernel horizontal fusion in
  Inductor, Python 3.14 support, TorchScript deprecation, deterministic compile.
  <https://pytorch.org/blog/pytorch-2-10-release-blog/>
- **PyTorch 2.14 docs**: `torch.compile` modes, `fullgraph`, recompilation limit,
  `torch.backends.cuda.preferred_blas_library`, `preferred_rocm_fa_library`,
  `torch.backends.miopen.immediate`, `cudnn.benchmark`.
  <https://docs.pytorch.org/docs/2.14/generated/torch.compile.html>
  <https://docs.pytorch.org/docs/2.14/backends.html>
- **How to use torch.compile on Windows CPU/XPU**, PyTorch tutorial. MSVC (or
  clang-cl/icx-cl) is required; CPU/XPU, not CUDA GPU.
  <https://docs.pytorch.org/tutorials/unstable/inductor_windows.html>

### SAM 2

- **SAM 2 build script and base model**, facebookresearch/sam2.
  `vos_optimized=True` compiles the image encoder with
  `mode="max-autotune", fullgraph=True, dynamic=False`.
  <https://github.com/facebookresearch/sam2/blob/main/sam2/build_sam.py>
  <https://github.com/facebookresearch/sam2/blob/main/sam2/modeling/sam2_base.py>

### ROCm / AMD

- **ROCm 7.2.3 PyTorch compatibility**, 2026-03-13. SDPA uses AOTriton; flash
  attention is automatic since PyTorch 2.8/ROCm 7.1; hipBLASLt defaults on
  gfx908+; AOTriton 0.10b supports gfx950/gfx1201 and experimentally
  gfx1101/gfx1151/gfx1150/gfx1200; fp16/bf16 reduced-precision reductions are not
  supported.
  <https://rocm.docs.amd.com/en/docs-7.2.3/compatibility/ml-compatibility/pytorch-compatibility.html>
- **MIOpen environment variables**, 3.6.3. `MIOPEN_FIND_MODE` values, `FAST` uses
  FindDb/immediate fallback, `MIOPEN_FIND_ENFORCE`, `MIOPEN_SEARCH_CUTOFF`.
  <https://rocm.docs.amd.com/projects/MIOpen/en/develop/reference/env_variables.html>
- **PyTorch TunableOp on ROCm**, AMD blog. `PYTORCH_TUNABLEOP_ENABLED=1`; offline
  tuning and cache.
  <https://rocm.blogs.amd.com/artificial-intelligence/pytorch-tunableop/README.html>

### CPU / other runtimes

- **ONNX Runtime thread management.** Intra-op defaults to physical cores,
  sequential execution by default, spinning and affinity knobs.
  <https://onnxruntime.ai/docs/performance/tune-performance/threading.html>
- **OpenVINO optimize-inference guide.** Latency/throughput performance hints,
  async API, device-specific scheduling.
  <https://docs.openvino.ai/2026/openvino-workflow/running-inference/optimize-inference.html>
- **OpenVINO SAM 3 notebook**, 2026. Converts SAM 3 image segmentation to
  OpenVINO IR.
  <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/sam3/sam3-segmentation.ipynb>
- **Intel Extension for PyTorch**, CPU BF16/oneDNN optimizations.
  <https://github.com/intel/intel-extension-for-pytorch>
- **Pillow PNG documentation.** `compress_level` 0–9; 1 is best speed, 6 is the
  default; `optimize` costs time.
  <https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html>
- **orjson.** Fast JSON encoder/decoder for Python, exact decoded output.
  <https://github.com/ijl/orjson>

### Code references in this repository

- `backend/models/sam3_backend.py` — encoding, compile, `cudnn_disabled`, restore,
  result formatting.
- `backend/models/ultralytics_backend.py` — SAM 2 encode/decode, compile,
  `predict_batch`, decoder warm-up.
- `backend/models/manager.py` — sessions, residency, eviction, predict/detect,
  mask PNG cache, response assembly.
- `backend/main.py` — upload decode, lifespan settings, routes, mask/export
  decoding, ZIP/PNG export.
- `backend/utils/compositing.py` — mask PNG encoding, feathering, compositing.
- `backend/utils/torch_threads.py` — thread resolution and `cudnn_disabled`.
- `backend/utils/model_integrity.py` — SHA-256 verification.
- `backend/utils/device.py` — device detection and `release_gpu_memory`.
- `backend/run.cmd`, `backend_amd_gpu/run.cmd` — thread, MIOpen, hipBLASLt, and
  cache environment.

---

## 19. Open questions to answer with measurements

1. On this hardware, what fraction of a first prompt is decode, encode, text,
   detector, mask head, upsample, PNG, JSON, and network?
2. Is `cudnn_disabled()` a workaround for a real driver bug, or an inherited
   precaution? If the former, which GPU/driver?
3. What does MIOpen FAST mode select for the 1008/1024 shapes, and how much does a
   tuned database improve them?
4. Does `torch.compile` on the ViT + pixel decoder actually produce a graph without
   breaks in the Ultralytics wrapper, on each platform?
5. Is `torch.compile` usable on Windows CUDA/ROCm with the shipped wheels, or only
   on CPU with MSVC?
6. Does the frontend correctly decode 1-bit PNG masks on every browser/canvas path?
7. Does TF32 change any mask on the golden set?
8. How much of a 4K, 20-subject response is mask PNG vs JSON vs base64?
9. Can the backend safely cache image features across sessions, or does the shared
   backend need per-session ownership first?
10. Is 1008 (the model's native resolution) faster *and* equal/better than 1024?
    This is a quality question, not a free speedup; test it separately.
11. Does the OpenVINO SAM 3 notebook support text prompts end-to-end, and can its
    export be adapted to the Ultralytics checkpoint?
12. Does TensorRT FP16 with the DART graph restructuring reproduce SAM 3 masks on
    this golden set, or only bounding boxes?

---

## 20. Bottom line

If the goal is a faster product with **identical** masks, the first pass should be
boring: keep models loaded, cache the hash, stop flushing the allocator, cache text
embeddings, encode 1-bit masks, send only what changed, and align the CPU thread
knobs. Those are T1 changes, and at 4K/multi-object they may be larger than any
model-level trick.

The second pass is measurement-driven: remove `cudnn_disabled`, compile the ViT and
pixel decoder the way the official implementations do, tune MIOpen once, and sweep
thread counts. Those are T2 and need the golden-mask gate.

TensorRT/OpenVINO/ONNX are real options, but they are projects, not switches. Do
not start there. Start with the harness in §17, get a phase breakdown, and let the
numbers choose the next change.
