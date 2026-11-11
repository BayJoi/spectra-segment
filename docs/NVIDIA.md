# NVIDIA GPU support (NOT TESTED)

This document covers the CPU/NVIDIA fork (`backend/`). The AMD fork
(`backend_amd_gpu/`, ROCm) is separate and is covered by [`AMD.md`](AMD.md).

> **TEST STATUS: IT HAS NOT BEEN TESTED AT ALL ON PHYSICAL NVIDIA HARDWARE.**
> No NVIDIA card has been run by this project. The policy below is built only from
> published PyTorch and NVIDIA compatibility data (retrieved 2026-11-09), not from
> measurements or runtime validation here. Treat every setting as an untested
> default and verify it on the actual card. The startup log prints the detected
> profile so a mismatch is visible.

## 1. What the installer selects

`install.bat` reads `name, compute_cap, driver_version` from `nvidia-smi` and calls
[`launcher/select_nvidia_runtime.ps1`](../launcher/select_nvidia_runtime.ps1). The
helper is pure: it can also be run by hand to inspect the decision.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File launcher/select_nvidia_runtime.ps1 `
  -ComputeCapability 8.9 -DriverVersion 581.15
# nvidia|https://download.pytorch.org/whl/cu132|torch==2.15.0 torchvision==0.29.2|CUDA 13.2|...|https://download.pytorch.org/whl/cu126|torch==2.14.0 torchvision==0.29.0
```

| GPU generation | Compute capability | Cards (examples) | Selected wheel | Runtime precision |
|---|---|---|---|---|
| Maxwell | 5.0, 5.2, 5.3 | GTX 750 Ti, GTX 900 series, M40 | torch 2.14.0 `cu126` | fp32 |
| Pascal | 6.0, 6.1, 6.2 | GTX 10 series, P40/P100 | torch 2.14.0 `cu126` | fp32 |
| Volta | 7.0, 7.2 | V100, Titan V, Xavier | torch 2.14.0 `cu126` | fp16 |
| Turing | 7.5 | RTX 20 series, GTX 16 series, T4 | torch 2.15.0 `cu132`, or `cu126` with driver 525-579 | fp16 |
| Ampere | 8.0, 8.6, 8.7 | RTX 30 series, A100, A10, Orin | torch 2.15.0 `cu132`, or `cu126` with driver 525-579 | fp16 + TF32 |
| Ada Lovelace | 8.9 | RTX 40 series, L4, L40S | torch 2.15.0 `cu132`, or `cu126` with driver 525-579 | fp16 + TF32 |
| Hopper | 9.0 | H100, H200 | torch 2.15.0 `cu132`, or `cu126` with driver 525-579 | fp16 + TF32 |
| Blackwell | 10.0, 10.3, 12.0, 12.1 | B200, B300, RTX 50 series, DGX Spark | torch 2.15.0 `cu132` (driver 580+) | fp16 + TF32 |
| Rubin | 10.7 | Rubin generation | torch 2.15.0 `cu134`, driver 615+ | fp16 + TF32 |
| Kepler and older | below 5.0 | GTX 700 series and older | CPU | fp32 |

Driver rules:

- **525 or newer** is required for any CUDA 12.6 build.
- **580 or newer** is required for CUDA 13.x. With 525-579, Turing through Hopper use
  the CUDA 12.6 / PyTorch 2.14 build instead. Blackwell has no CUDA 12.6 wheel, so it
  runs on CPU below 580 rather than installing a wheel with no kernels for the card.
- **615 or newer** is required for the CUDA 13.4 Rubin build (`sm_107`).
- Below 525, or with a Kepler card, install.bat installs CPU PyTorch and prints a
  warning. It no longer installs a CUDA wheel that has no kernel for the card.

The current PyTorch release is 2.15.0 (2026-10-28), whose default wheel is `cu132`
(CUDA 13.2); `cu134` is also published. PyTorch 2.14 is the last release that
publishes `cu126`, which is why Maxwell/Pascal/Volta are pinned there. CUDA 13.0
removed offline support for those three generations. Turing and newer are covered by
CUDA 13.x, but CUDA 13.x requires driver 580+.

If the preferred wheel fails to install, `install.bat` retries the helper-provided
fallback once, then CPU. Turing through Hopper fall back to `cu126` (torch 2.14);
Maxwell/Pascal/Volta are already on `cu126`, and Blackwell/Rubin go straight to CPU
because no older wheel covers them. The CPU fallback installs from
`https://download.pytorch.org/whl/cpu`, not from PyPI, so it does not silently pull a
CUDA build. A successful install is followed by a `torch` verification line showing
the version, CUDA runtime, availability, device name and compiled arch list.

## 2. Runtime architecture check

`backend/utils/cuda_profile.py` builds one cached profile at startup:

- GPU name, compute capability, family and VRAM;
- installed torch and CUDA versions, plus `torch.cuda.get_arch_list()`;
- whether the wheel actually has a kernel for the card.

`arch_supported()` accepts:

- an exact `sm_XX` or `compute_XX` entry;
- an older PTX `compute_XX` entry, which the driver JIT-compiles forward; or
- an older SASS entry from the same major family (for example Ada `sm_89` can run an
  Ampere `sm_86` kernel, and a Blackwell 10.3 card can run an `sm_100` kernel).

Blackwell 10.x, Blackwell 12.x and Rubin 10.7 do not reuse each other's SASS, so a
`sm_100`-only wheel does not silently claim to support a `sm_120` card or Rubin
`sm_107`. Older PTX still takes the forward-compatible path when the driver can JIT it.

`detect_device()` returns `cuda:0` only when the GPU is available **and** supported.
An available but unsupported GPU returns CPU, logs the card and reason, and prints a
suggestion to install the matching wheel or set `SPECTRA_FORCE_CPU=1`. This replaces
the old behavior of using whatever GPU torch reported and failing later with a kernel
image error.

`start.bat` also runs this check before offering the GPU option. If the installed
wheel cannot run the detected card, it warns and launches in CPU mode instead of
writing a GPU memory profile that cannot work.

## 3. Runtime performance settings

| Setting | Maxwell/Pascal | Volta | Ampere and newer |
|---|---|---|---|
| SAM2/SAM3 working precision | fp32 | fp16 | fp16 |
| BiRefNet matte precision | fp32 | fp16 | fp16 |
| Detector precision | fp32 | fp16 | fp16 |
| TF32 for fp32 matmul/conv | off | off | on |
| cuDNN benchmark | on (CUDA fork) | on | on |

These are defaults, not hard requirements. The startup path sets them **before**
`backend.models.manager` is imported, because `matte_backend` computes its fp16 default
at import time and the SAM loaders read their precision variables during import.

TF32 is enabled through PyTorch's `fp32_precision` APIs on 2.9+ with an
`allow_tf32` fallback. Set `SPECTRA_TF32=0` to keep full fp32 on Ampere or newer.
FP16 is the largest win for the SAM/matte models; Pascal and Maxwell lack fast fp16,
so they stay fp32.

The CUDA allocator configuration is set before `import torch`, not inside the lifespan,
so the caching allocator actually sees it. `cudnn.benchmark` is enabled once at
startup by `configure_nvidia_runtime()`.

The VRAM hard cap (`SPECTRA_VRAM_HARD_CAP`, default `0.90`) is now applied after the
device check, so it only runs when the backend will actually use CUDA.

The CUDA setup also sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` unless the
user already set that variable. It is the allocator recommendation for reducing
fragmentation; it does not increase the physical VRAM budget.

Grounding DINO is not an SDPA model in the pinned transformers version: the loaded
class reports `_supports_sdpa = False`, so this fork keeps eager attention and does
not request an attention backend it cannot use. FP16 is already selected for Volta
and newer, and the detector processor cap remains the first memory control. For
NVIDIA specifically, keep `expandable_segments:True`; do not add `cudaMallocAsync`
or another allocator backend without hardware testing, because those change
allocation behavior rather than reducing live model memory. TensorRT / TAO
conversion could reduce inference memory, but it is a separate deployment path and
is not wired into this backend.

**These NVIDIA-only notes are static checks of the installed model/config and web
research; they were not run on NVIDIA hardware.** References: [PyTorch CUDA
allocator notes](https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf),
[NVIDIA TAO Grounding DINO](https://docs.nvidia.com/tao/tao-toolkit/text/cv_finetuning/pytorch/object_detection/grounding_dino.html).

### 4K and low-VRAM behavior

The same adaptive matte policy as the AMD fork is applied here, with NVIDIA defaults:

- `SPECTRA_MATTE_INPUT_SIZE` is capped at **1024** below 12 GB and **2048** at 12 GB or
  more. An explicit user value is preserved.
- When a guide supplies boxes or strokes, the native-2048² HR models use context crops at
  the cap; 1024-native models use a full pass and keep components overlapping the ROI. A
  prompt-free soft cutout always uses one capped full-image pass and upsamples the alpha.
- The full 2048² activation is not allocated on an 8 GB card unless the user explicitly
  raises the cap. This is a code path shared with the tested AMD implementation, but the
  NVIDIA numbers are **not measured here**.
- Automatic CPU fallback is enabled by default on placement failure or CUDA OOM. The
  failed model is moved to CPU and retried once; it is not a silent permanent switch.
  Set `SPECTRA_ALLOW_CPU_FALLBACK=0` to disable it.

**None of these NVIDIA 4K paths has run on physical NVIDIA hardware.** The code compiles
and follows the AMD implementation, but VRAM peaks, speed and numerical output remain
unverified on NVIDIA.

## 4. Environment overrides

| Variable | Default on NVIDIA | Effect |
|---|---|---|
| `SPECTRA_FORCE_CPU=1` | unset | Ignore the GPU, use CPU |
| `SPECTRA_TF32=0` | enabled on Ampere+ | Disable TF32 |
| `SAM2_QUANTIZE` / `SAM3_QUANTIZE` | `16` on Volta+, `32` on Maxwell/Pascal | SAM working precision |
| `SPECTRA_MATTE_FP16` | `1` on Volta+, `0` on Maxwell/Pascal | BiRefNet fp16 autocast |
| `SPECTRA_DETECTOR_FP16` | `1` on Volta+, `0` on Maxwell/Pascal | Grounding/Florence fp16 |
| `SPECTRA_VRAM_HARD_CAP` | `0.90` | Allocator fraction cap |
| `SPECTRA_MATTE_INPUT_SIZE` | `1024` below 12 GB, `2048` at 12 GB+ | BiRefNet input cap |
| `SPECTRA_ALLOW_CPU_FALLBACK` | enabled | `0` makes OOM a hard error |

The launchers normally choose CPU vs GPU. Any explicit value you set in the
environment wins over the automatic default; the code only fills in unset variables.
`SPECTRA_ALLOW_CPU_FALLBACK` is the exception: unset means **enabled**, and only
`0`, `false`, `no`, `off` or `disabled` turns it off.

## 5. Verification

After installing, run the explicit NVIDIA wrapper:

```bat
checks\check_gpu_nvidia.bat --skip-4k
```

It uses `backend\.venv`, validates the CUDA runtime and catalogue, and labels the
run as project-untested. Add `--all-models` only when the 33 checkpoints are
cached and you are ready for a long run; use `--skip-4k` for a quicker pass.
The wrapper does not turn this project into a validated NVIDIA implementation.

For a direct environment check:

```bash
backend\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_arch_list())"
```

Then start normally and check the backend log for lines beginning with `NVIDIA:`:

- `NVIDIA: <card> - <GB> - <family> sm_<cap> - fp16=... tf32=...`
- `NVIDIA: PyTorch <version> / CUDA <version> - <wheel support reason>`

If the card is reported as unsupported, the reason names the architecture and the
missing kernel. Re-run `install.bat`; if the correct wheel is already installed, check
that the GPU is visible to `nvidia-smi` and that the driver meets the minimum above.

## 6. Known limitations

- **No NVIDIA test or benchmark exists in this repository.** The card has not been
  run at all. The wheel/arch matrix and precision policy are from published
  compatibility data; device detection, kernel launch, throughput and VRAM behavior
  still need validation on physical hardware.
- **Old cards are supported, not fast.** Maxwell/Pascal can run the models in fp32 but
  may be too slow or run out of VRAM for the largest presets. Use the CPU profile or a
  smaller model if that happens.
- **MIG and multi-GPU systems use device 0.** The backend always selects `cuda:0`;
  it does not distribute across GPUs. Set `CUDA_VISIBLE_DEVICES` if another device
  should be device 0.
- **Jetson / embedded GPUs** are not a Windows launcher target. The profile can still
  classify them, but `install.bat` is written for desktop `nvidia-smi`.
- **cuDNN benchmark stays on.** It is a known first-call warmup cost. It has not been
  A/B tested on NVIDIA here; `cudnn_disabled()` remains in `utils/torch_threads.py`
  because it may be a driver workaround, not only a performance switch.
- **FP16 is not numerically identical to fp32.** If a particular checkpoint shows
  artifacts, set `SAM2_QUANTIZE=32`, `SAM3_QUANTIZE=32`,
  `SPECTRA_MATTE_FP16=0` or `SPECTRA_DETECTOR_FP16=0` to isolate it.
- **The 4K matte and automatic CPU fallback paths are untested on NVIDIA.** They
  compile and mirror the tested AMD design, but no NVIDIA run has measured their VRAM
  or output quality. Treat the first run as a validation run.

## 7. Sources

- PyTorch 2.15 release: <https://github.com/pytorch/pytorch/releases/tag/v2.15.0>
- PyTorch 2.14 release notes (last `cu126` release):
  <https://raw.githubusercontent.com/pytorch/pytorch/release/2.14/RELEASE.md>
- PyTorch CUDA semantics and TF32 controls:
  <https://docs.pytorch.org/docs/2.15/notes/cuda.html>
- CUDA 13.0 release notes (Maxwell/Pascal/Volta offline support removed):
  <https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html>
- PyTorch support matrix (2.14 cu126 architectures, 2.15 cu132/cu134):
  <https://github.com/pytorch/pytorch/blob/main/RELEASE.md>
- CUDA 13.4 release notes (Rubin `sm_107` developer preview, R615 driver):
  <https://docs.nvidia.com/cuda/archive/13.4.1/cuda-toolkit-release-notes/index.html>
- NVIDIA compute capability list: <https://developer.nvidia.com/cuda/gpus>
