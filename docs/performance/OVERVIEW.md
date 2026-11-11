# Performance overview

> **Research plan, not a current performance report.**
>
> This folder records experiments proposed on 2026-10-26. Some proposals may since
> have been implemented; verify the code before acting. Every "expected effect" is
> either a hypothesis about this codebase or an external number from a cited source;
> the two are labelled separately. The priority order is a starting point for
> experiments, not a result. Measured matte numbers live in
> [Matte mode](../modes/MATTE.md).

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

The measurement catalogue lives in [CATALOG.md](CATALOG.md); test design,
sources and open questions live in [TESTING.md](TESTING.md).

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
| `backend/utils/compositing.py:76-98` + `backend/main.py` export routes | ZIP export re-encodes PNGs with Pillow's default compression level (6) and puts them in a `ZIP_DEFLATED` archive, even though PNG is already compressed. The direct `export-image` route returns one encoded image and avoids the archive entirely. |
| `backend/run.cmd:35` | `OMP_NUM_THREADS` is capped at `min(6, physical cores)`, while `torch_threads.resolve_threads()` can set up to 32. The two knobs disagree. |
| `backend_amd_gpu/run.cmd:27-31` | MIOpen runs with `FIND_MODE=2` (fast), `FIND_ENFORCE=1` (none), `SEARCH_CUTOFF=1`; `backend_amd_gpu/main.py` sets `cudnn.benchmark=False`. `benchmark=True` was tested on torch 2.16 and was still autotuning after 60 s, so the guarded FAST path is intentional. |
| `backend_amd_gpu/run.cmd:33-34` | `TORCH_BLAS_PREFER_HIPBLASLT=1` and `DISABLE_ADDMM_CUDA_LT=1` are set, but `PYTORCH_TUNABLEOP_CACHE_DIR` is set without `PYTORCH_TUNABLEOP_ENABLED=1`. |
| `backend/models/yoloe_detector.py:49,120,174` | **Fixed.** The detector now passes `device=self._device` to `predict()` in both forks, so Ultralytics no longer chooses a different device. |
| `backend/main.py` CUDA setup | **Fixed.** Allocator configuration is set before `import torch`; capability-gated dtype/TF32 setup is in `utils/cuda_profile.py` and runs before model imports. The VRAM cap now runs only after the GPU passes the wheel/arch check. |

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
| D6 cuDNN SDPA | ✅ | ❌ | ❌ | Do not disable it preemptively |
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
