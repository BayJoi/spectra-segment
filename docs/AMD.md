# AMD GPU support (Windows)

> **Tested on exactly one card: AMD Radeon RX 6600, 8 GB, `gfx1032`, on Windows 10.**
> The current `whl-next` stack and its 2.15 predecessor have been run on this
> machine. No other AMD GPU, operating system, driver, or wheel combination has
> been run by this project. Everything below is either a measurement from that
> machine or a clearly labelled upstream support statement. It is not a
> compatibility promise.

The AMD backend is `backend_amd_gpu/`. It is a separate fork of `backend/` and
uses AMD's ROCm PyTorch wheels built by
[TheRock](https://github.com/ROCm/TheRock). The CPU/NVIDIA fork and its policy are
covered in [`NVIDIA.md`](NVIDIA.md).

## 1. Verified environment

| Item | Value |
|---|---|
| GPU | AMD Radeon RX 6600, 8176 MB, `gfx1032` |
| OS | Windows 10 `10.0.19045` |
| Python | 3.12.13 |
| PyTorch | `2.16.0a0+rocm10.3.0a20261110` |
| torchvision | `0.30.0a0+rocm10.3.0a20261110` |
| torchaudio | `2.11.1+rocm10.3.0a20261110` |
| HIP | `7.18.26446` |
| Adrenalin driver | `32.0.21045.16010` |
| Package index used | TheRock `whl-next` (`https://nightly.repo.amd.com/rocm/whl-next/`) |

The installed stack is the newest 2026-11-10 `whl-next` build. The previous
environment was `torch 2.15.0+rocm10.3.0a20261110` /
`torchvision 0.29.2+rocm10.3.0a20261110`. Both stacks were tested with the same
isolated 33-model sweep after their MIOpen kernel caches had been warmed:

| Stack | Model count | Load total | Inference total | Highest peak | Result |
|---|---:|---:|---:|---:|---:|
| torch 2.15 / ROCm 10.3 | 33 | 121.05 s | 57.26 s | 9,677 MB | 33/33 passed |
| torch 2.16 / ROCm 10.3 | 33 | 122.34 s | 56.38 s | 9,677 MB | 33/33 passed |

This table is the **historical 2026-11-10 comparison**, before the checker separated
encode from prompt and before the 8 GB matte cap was used in the sweep. The current
2026-11-12 run is in section 2; do not compare its aggregate timings directly with
this table.

The two builds are effectively tied. The 2.16 nightly has **1.1 % higher aggregate
load time**, **1.5 % lower aggregate inference time**, and the same peak memory.
Its median inference time is 1.736 s versus 1.78 s. Per model, 10 inferences were
faster, 7 were slower and 16 were within ±2 %. Loads were 1 faster, 7 slower and
25 within ±2 %. The installer keeps this build because it is the newest resolved
nightly and passes the full suite, not because every model is faster.

The first 2.16 sweep was a cold-cache run and is not used for the comparison; it
included kernel compilation costs (SAM 3 load was 17.29 s and the text detectors
were around 9 s). Re-run a new wheel once before comparing warm timings.

No system-wide ROCm install or HIP SDK is required. The wheels carry
`rocm-sdk-core`, `rocm-sdk-libraries`, and the `device-gfxNNNN` package needed at
runtime.

## 2. What was actually run

The current run is from **2026-11-12** on the verified stack above. The checker
used one fresh Python process per model, with downloads disabled. The source was
a generic local test image; it is not part of the shipped app. Detectors used a
two-object prompt at confidence 0.05 with up to 20 detections. SAM 3 used a
one-object prompt at confidence 0.25. SAM 2 used a box prompt with multimask
output. Matte ran the full image with the 8 GB launcher cap
(`SPECTRA_MATTE_INPUT_SIZE=1024`), not the native 2048² mode used by the older
historical table. `peak` is allocator peak with the counter reset before
inference; it is not physical VRAM or driver overhead. Grounding DINO runs in
fp16 on the GPU with the processor capped at 1280 px on this card; the pinned
transformers build reports no SDPA support for this model, so it uses eager
attention.

### Current 33-model sweep

Command:

```
checks\check_gpu_amd.bat --all-models --image PATH\TO\TEST_IMAGE
```

The run passed **33/33** catalogue entries: **41 checks passed, 0 failed, 2
skipped** (the single-model smoke phase and the separate 4K phase were skipped in
this particular invocation). Wall time was **315.0 s**, including one Python
startup per model. Summed per-model load, encode and prompt/inference times are
shown below; they exclude process startup.

| Family | Entries | Summed load | Summed encode | Summed inference | Highest peak |
|---|---:|---:|---:|---:|---:|
| SAM 2 / 2.1 | 8 | 7.41 s | 12.32 s | 0.59 s | 1,674 MB |
| SAM 3 | 1 | 17.50 s | 7.07 s | 1.21 s | 5,932 MB |
| Grounding / Florence | 7 | 66.90 s | - | 13.80 s | 5,797 MB |
| YOLOE | 7 | 12.04 s | - | 20.18 s | 788 MB |
| BiRefNet matte | 10 | 45.70 s | - | 17.43 s | 3,058 MB |
| **Total** | **33** | **149.55 s** | **19.39 s** | **53.21 s** | **5,932 MB** |

The separate 4K pass (SAM 2.1 Base, 3840 × 2160 synthetic frame) passed **10
checks, 0 failed, 1 skipped** in 4.8 s: load 0.83 s, encode 1.55 s, prompt 0.09 s,
peak 1,383 MB of 8,176 MB. The 4K phase deliberately falls back to a synthetic
frame when `--image` is not itself at least 3000 px on its long edge.

### SAM 2 / SAM 2.1 checkpoints

| Checkpoint | Load | Encode | Prompt | Peak allocated |
|---|---:|---:|---:|---:|
| `sam2.1_b.pt` (Base) | 0.85 s | 1.54 s | 0.08 s | 1,383 MB |
| `sam2.1_l.pt` (Large) | 1.64 s | 1.73 s | 0.07 s | 1,674 MB |
| `sam2.1_s.pt` (Small) | 0.66 s | 1.44 s | 0.07 s | 728 MB |
| `sam2.1_t.pt` (Tiny) | 0.60 s | 1.46 s | 0.08 s | 714 MB |
| `sam2_b.pt` (legacy Base) | 0.83 s | 1.56 s | 0.08 s | 1,383 MB |
| `sam2_l.pt` (legacy Large) | 1.60 s | 1.70 s | 0.07 s | 1,674 MB |
| `sam2_s.pt` (legacy Small) | 0.64 s | 1.40 s | 0.07 s | 728 MB |
| `sam2_t.pt` (legacy Tiny) | 0.59 s | 1.49 s | 0.07 s | 714 MB |

All eight returned a mask. The legacy and 2.1 checkpoints have nearly identical
load and decode costs at the same size. Encode is reported separately from the
prompt because the old one-number table mixed them.

### SAM 3

`sam3.pt` is 3,290 MB and was downloaded and cache-verified before the sweep. On
the tested card it loaded in 17.50 s, encoded the test image in 7.07 s, returned
one mask in 1.21 s after that encode, and peaked at 5,932 MB. This is the
largest single-model resident footprint in the catalogue, but it completes on the
8 GB card. The sweep used the default encode size.

### Detection models

| Model | Result | Load | Inference | Peak allocated | Detections |
|---|---|---:|---:|---:|---:|
| `grounding-dino-tiny` | passed | 9.06 s | 0.93 s | 1,409 MB | 20 |
| `grounding-dino-base` | passed | 9.67 s | 0.99 s | 1,530 MB | 20 |
| `mm-gdino-base-all` | passed | 9.71 s | 0.99 s | 1,531 MB | 20 |
| `mm-gdino-large-all` | passed | 12.39 s | 6.39 s | 5,797 MB | 13 |
| `florence-2-base` | passed | 7.77 s | 0.84 s | 740 MB | 1 |
| `florence-2-large` | passed | 11.38 s | 1.83 s | 2,016 MB | 4 |
| `cogflorence-2.2-large` | passed | 6.92 s | 1.83 s | 2,004 MB | 4 |
| `yoloe-26l-seg` | passed | 1.79 s | 2.53 s | 220 MB | 8 |
| `yoloe-26m-seg` | passed | 1.72 s | 2.55 s | 203 MB | 5 |
| `yoloe-26s-seg` | passed | 1.64 s | 2.43 s | 119 MB | 6 |
| `yoloe-26n-seg` | passed | 1.68 s | 2.39 s | 79 MB | 7 |
| `yoloe-11l-seg` | passed | 1.76 s | 3.93 s | 788 MB | 7 |
| `yoloe-11m-seg` | passed | 1.73 s | 3.25 s | 767 MB | 7 |
| `yoloe-11s-seg` | passed | 1.72 s | 3.10 s | 696 MB | 7 |

All 14 detectors completed. The text-conditioned detectors are slower to load
than YOLOE, but their inference is generally faster on the test image. The detection
count is not a quality score: 20 is the request cap, and Florence reports one box
per prompt on this image. YOLOE masks come from the detector itself and are not
SAM-quality masks.

### BiRefNet matte models

All ten catalogued checkpoints ran on the full test image with the default fp16
path and the 8 GB cap. The two native-2048² HR models were therefore evaluated at
1024², which is the shipped 8 GB configuration. The historical native-2048² run
peaked at 9,677 MB and is not the 8 GB recommendation.

| Model | Effective input | Load | Inference | Peak allocated |
|---|---|---:|---:|---:|
| `birefnet-general` | 1024² | 4.67 s | 1.87 s | 3,058 MB |
| `birefnet-lite` | 1024² | 4.21 s | 1.48 s | 1,593 MB |
| `birefnet-matting` | 1024² | 4.68 s | 1.96 s | 3,058 MB |
| `birefnet-dynamic` | 1024² | 4.60 s | 1.85 s | 3,058 MB |
| `birefnet-dynamic-matting` | 1024² | 4.66 s | 1.86 s | 3,058 MB |
| `birefnet-lite-matting` | 1024² | 4.25 s | 1.43 s | 1,593 MB |
| `birefnet-hr` | 1024² (native 2048²) | 4.72 s | 1.89 s | 3,058 MB |
| `birefnet-hr-matting` | 1024² (native 2048²) | 4.58 s | 1.95 s | 3,058 MB |
| `birefnet-portrait` | 1024² | 4.71 s | 1.89 s | 3,058 MB |
| `birefnet-512` | 512² | 4.62 s | 1.25 s | 1,404 MB |

The allocator peak excludes the driver, display and other processes. A 2048²
pass on this card is possible only with an explicit override and carries memory
pressure; the measured native-2048² historical run reached 9,677 MB. The modern
8 GB path keeps the HR pair at 1024².

### Coverage and limits

All 33 catalogue entries passed: 8 SAM 2 variants, SAM 3, 14 detectors, and 10
BiRefNet checkpoints. That is coverage of the shipped catalogue on this one
machine, not proof that every prompt, ROI, mask, export, or UI path behaves the
same way. The sweep calls the backend classes directly and does not replace the
browser-based manual check described in the README. The 2026-11-10 comparison in
section 1 used an older harness and did not separate encode from prompt; do not
compare its aggregate timings directly with this table.

## 3. Windows GPU support

TheRock's [`SUPPORTED_GPUS.md`](https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md)
is the source of truth used by `amd_gfx.py`. The Windows 2026-11 table has two
levels:

| Level | Targets | Meaning |
|---|---|---|
| Release-ready | `gfx1010`-`gfx1012`, `gfx1030`-`gfx1036`, `gfx1100`-`gfx1103`, `gfx1150`-`gfx1153`, `gfx1200`, `gfx1201` | Built, sanity-tested and published for Windows |
| Build-only | `gfx1170`-`gfx1172`, `gfx900`, `gfx906`, `gfx90c`, `gfx908`, `gfx90a` | A Windows build exists, but upstream does not call it sanity-tested or release-ready |

`gfx942` and `gfx950` are Linux-only in TheRock's Windows table. The installer
refuses to treat them as Windows targets. `gfx117x` is the RDNA 4m APU family;
exact `gfx1170`/`gfx1171`/`gfx1172` targets can be forced, but are build-only.

This is upstream package support, not Spectra Segment hardware validation. The
only card runtime-tested for this app is the RX 6600 (`gfx1032`) on this machine.
Every other target is installed with conservative settings from the same matrix,
but cannot be tested here. Other AMD cards are therefore **not guaranteed** to
work, even when the wheel installs successfully.

### What the installer now does

1. Reads every AMD display adapter from `Win32_VideoController` by PCI device ID,
   then by product name. PCI matches win, and discrete cards win over iGPUs at
   equal VRAM. If several supported cards are present, the highest-scoring one is
   selected. `AMD_GFX_OVERRIDE` still forces an exact target.
2. Writes `backend_amd_gpu/spectra_amd_hardware.cfg` with the selected target,
   VRAM, family, support level, FP16 policy, detector resolution cap and
   recommended memory profile.
3. Seeds `spectra_launcher.cfg` only when it does not exist, so an existing user
   profile is never silently replaced.
4. Installs the exact `torch[device-gfxNNNN]` and
   `torchvision[device-gfxNNNN]` extras for that target from TheRock's nightly
   index. The mapping is in code, not duplicated in batch files.
5. Applies FP16 and quantization only where the target supports it, caps detector
   input for low-VRAM and integrated cards, and sets the matching memory profile.

Release-ready Radeon targets use native kernels. The installer does **not** set
`HSA_OVERRIDE_GFX_VERSION` automatically. Set `AMD_HSA_OVERRIDE` only for a known
workaround; it is written to the hardware config and applied explicitly.

## 4. 4K on an 8 GB card

The 4K path is staged instead of feeding a 3840x2160 image into every model:

| Stage | Effective input | Measured RX 6600 result |
|---|---|---|
| Upload + SAM 2 encode | 1024 px | 1.55 s, 1.38 GB peak |
| Grounding DINO large detection | 1280 px cap, retry 1024 | 6.4 s, 6.68 GB peak |
| BiRefNet HR on 8 GB | 1024 px, ROI crops for boxes | 1.4 s warm for two people, 3.06 GB peak |
| BiRefNet HR on 12 GB+ | 2048 px | full 2048 detail, 9.68 GB peak |

The 8 GB profile sets `SPECTRA_MATTE_INPUT_SIZE=1024`. That keeps the HR model below the card's memory limit instead of spilling into Windows shared GPU memory. When a text, box or stroke guide supplies ROIs to a native-2048² HR checkpoint, the capped backend runs context crops at 1024, which gives each person roughly native detail without allocating a 4K activation. The 1024-native checkpoints instead run one full 1024 pass and keep the components overlapping the ROI. A full soft cutout with no boxes runs one 1024 pass and upsamples the alpha to the original size.

A 2048 matte pass measured 9.68 GB peak on this 8 GB card, so it is reserved for cards with at least 12 GB. The 1024 ROI path produced a 0.9894 IoU and 0.0055 mean alpha error against the 2048 reference in the two-person region, while dropping the matte from 5.6 s to 1.4 s and from 9.68 GB to 3.06 GB. Set `SPECTRA_MATTE_INPUT_SIZE=2048` only when the GPU has the headroom.

Tiled SAHI detection was also measured. On this image it produced the same two person boxes as the whole-image pass while multiplying detector invocations, so the tiled path is not a default. Detection already runs on a bounded whole-image pass; the high-resolution work is spent on segmentation and matte ROIs where it changes the result.

## 5. Practical notes for this card

- Use `install_amd.bat` and `start_amd.bat`; the launcher sets the HIP, MIOpen,
  hipBLASLt, and SDMA environment.
- `install_amd.bat` always refreshes `whl-next`, upgrades torch/torchvision/
  torchaudio to the newest resolved build, replaces the old installed wheels, and
  runs `uv pip check` before verification. There is no silent fallback to an older
  ROCm index; a broken nightly fails the installer instead.
- `HIP_VISIBLE_DEVICES=1` selects the second GPU if an iGPU is being chosen first.
- The ROCm path is Windows-only here. Linux support in TheRock does not mean these
  `.bat` launchers work on Linux.
- Nightly wheels are development builds. Re-run `checks\check_gpu_amd.bat` after
  changing a wheel; add `--all-models` for the full catalogue. Health logs are
  generated locally on demand and are not checked into the project.
- The first run after a wheel change spends extra time compiling kernels. Treat it
  as a cold run and repeat before comparing model load or inference timings.
- The 8 GB card can run all ten matte checkpoints. HR models are automatically
  capped to 1024 px here, which keeps them around 3 GB instead of the 9.68 GB needed
  for a 2048² pass. The 2048² path is reserved for cards with at least 12 GB; on
  8 GB it is available only as an explicit `SPECTRA_MATTE_INPUT_SIZE=2048` override.
- `start_amd.bat -cpu` or `start_amd.bat -gpu` overrides the saved launcher choice for
  one run. Without a flag the launcher uses the saved profile.
- Automatic CPU fallback is enabled by default. On a placement failure or OOM, the
  affected model is moved to CPU and retried once; the rest of the launch keeps using
  the GPU. Set `SPECTRA_ALLOW_CPU_FALLBACK=0` to make the same failure a hard error.
- Every launch writes a new `backend_amd_gpu/logs/backend_<YYYYmmdd_HHMMSS>.log`.
  The old single `backend.log` is not appended to anymore.
- Keep the installed torch/torchvision pair together. The `gfx1032` device extras
  and the ROCm SDK packages are resolved by the same index and can change with it.
- Flash and memory-efficient SDP have no `gfx1032` kernels in this wheel; math SDP
  is the working attention backend. The launcher's RDNA2 guard is correct.
- `torch.cuda.memory.is_localization_supported()` is false on this card, so the new
  localized allocator API is not used. `torch.compile` default Inductor mode also
  needs Triton, which this Windows ROCm wheel does not ship; the optional
  `SAM2_COMPILE` / `SAM3_COMPILE` paths fall back to eager execution.
- MIOpen benchmark/autotune mode was tested and is deliberately off. A 1024² matte
  load was still searching after 60 s, while the guarded warm path was about 0.21 s
  per pass. The unbounded search is not a safe startup default.
- The wheel prints `CK grouped conv library not found` and a missing
  `gfx1032_14.HIP.fdb.txt` warning on this card. MIOpen falls back to its other
  kernels; all health checks and the 33-model sweep still pass. This is an upstream
  packaging warning, not a backend failure.

## 6. Sources

- TheRock supported GPUs and build/runtime warning:
  <https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md>
- TheRock release/index instructions:
  <https://github.com/ROCm/TheRock/blob/main/RELEASES.md>
- TheRock repository: <https://github.com/ROCm/TheRock>
