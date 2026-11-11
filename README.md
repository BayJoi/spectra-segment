# Spectra Segment

A local image background remover with four editing modes. Everything runs on
your own machine - nothing leaves your computer.

> **Note:** This project was built entirely from scratch using AI. There is no
> human-written code.

## What it does

Remove backgrounds with local models. Pick an image and a mode, then draw, type,
or run a soft cutout. The backend is a FastAPI app with two hardware forks: one
for CPU/NVIDIA CUDA and one for AMD ROCm on Windows. The React frontend drives
both through the same API.

## Modes

- **Brush** - paint positive strokes around what to keep. Multiple subjects get
  separate colours, layers and histories; right-drag subtracts a SAM-selected
  region. See [`docs/modes/BRUSH.md`](docs/modes/BRUSH.md).
- **Detect** - describe an object, let a detector place boxes, then segment the
  boxes with SAM. See [`docs/modes/DETECT.md`](docs/modes/DETECT.md).
- **SAM 3** - one text prompt for detection, segmentation and masking, with
  instance control and selectable encode quality. See
  [`docs/modes/SAM3.md`](docs/modes/SAM3.md).
- **Matte** - prompt-free soft alpha for hair, glass and blur. BiRefNet returns
  a continuous alpha plane; Auto and Stroke guides, subject layers, local
  Edge/Feather controls and per-layer undo are supported. See
  [`docs/modes/MATTE.md`](docs/modes/MATTE.md).

All modes share transparent preview, PNG/JPG export, and the console panel.
The mode guides are the source of truth for their controls and limits.

## Project layout

```text
Spectra Segment/
  backend/                 CPU/NVIDIA FastAPI backend
    main.py                app, startup, logging, request handlers
    models/                detector, SAM, matte and manager code
    utils/                 device, VRAM, CUDA profile, security helpers
  backend_amd_gpu/         AMD ROCm backend fork (same layout)
  frontend/                React + Vite UI
  launcher/                installer and runtime selection helpers
  checks/                  GPU health and memory-limit checks
  docs/                    guides, references and performance research
  install.bat              CPU/NVIDIA installer
  install_amd.bat          AMD installer
  start.bat                CPU/NVIDIA launcher
  start_amd.bat            AMD launcher
  stop.bat, stop_amd.bat   stop a running launch
  cleanup.bat, cleanup_amd.bat
```

Each backend writes a new `logs/backend_<YYYYmmdd_HHMMSS>.log` on every launch;
see [`docs/LOGGING.md`](docs/LOGGING.md).

## Install

Requirements: Windows 10 or later, 8 GB RAM minimum (16 GB recommended), and
10-20 GB free disk space for caches and the models you choose. The first model
download needs internet; later runs can be offline.

### CPU / NVIDIA

1. Run `install.bat`.
2. Run `start.bat`.
3. Open <http://localhost:3000>.

The installer chooses a CUDA or CPU PyTorch build from the detected GPU,
compute capability and driver. The backend then verifies that the installed
wheel contains a kernel for the card before using it. If a model later runs out
of VRAM, it retries that model on CPU automatically; set
`SPECTRA_ALLOW_CPU_FALLBACK=0` to make OOM a hard error instead.

> **NVIDIA TEST STATUS: IT HAS NOT BEEN TESTED AT ALL ON PHYSICAL NVIDIA
> HARDWARE.** The compatibility policy, wheel selection and precision defaults
> are based on published PyTorch/NVIDIA data. They have not been benchmarked on
> a real NVIDIA card in this repository. Use `start.bat` and inspect the
> startup `NVIDIA:` log lines on your hardware.

The full matrix, fallbacks, runtime checks and limitations are in
[`docs/NVIDIA.md`](docs/NVIDIA.md).

### AMD GPU (Windows only)

1. Run `install_amd.bat`.
2. Run `start_amd.bat`.
3. Open <http://localhost:3000>.

Add `-cpu` or `-gpu` to `start_amd.bat` to override the saved memory profile
for one launch. CPU fallback is automatic on OOM unless
`SPECTRA_ALLOW_CPU_FALLBACK=0` is set.

The AMD path uses ROCm PyTorch wheels built by
[TheRock](https://github.com/ROCm/TheRock). The installer detects the `gfx`
target and installs the newest matching `device-gfxNNNN` build from TheRock's
`whl-next` index. Re-running it refreshes and upgrades an existing stack instead
of keeping old torch wheels. The runtime wheels bundle the needed ROCm libraries;
the HIP SDK and a system-wide ROCm install are not required.

> **AMD TEST STATUS: ONLY THE AMD RADEON RX 6600 (8 GB, `gfx1032`) HAS BEEN
> TESTED.** All 33 catalog checkpoints passed again on 2026-11-12 under ROCm 10.3
> with torch 2.16: 8 SAM 2 variants, SAM 3, 14 detectors and 10 BiRefNet matte
> models. The project still has no runtime evidence for any other AMD GPU. The
> exact stack, timings and limits are in [`docs/AMD.md`](docs/AMD.md).

The supported upstream families currently include RDNA 1 (`gfx101x`), RDNA 2
(`gfx103x`), RDNA 3 (`gfx110x`), RDNA 3.5 (`gfx115x`) and RDNA 4
(`gfx120x`). That is package support, not proof that this app runs on those
cards. If an iGPU is selected instead of the discrete GPU, set
`HIP_VISIBLE_DEVICES=1`.

## First run

1. Run the installer for your hardware.
2. Run the matching start script. On first launch it asks for a memory profile;
   the defaults are fine.
3. Pick a mode, then select a model in the top-left selector if the mode needs
   one. Models download on first use.
4. Upload an image and edit. The frontend fits large images to 4096 x 4096
   (16.7 MP) and 8192 px on the longest side; direct API uploads are fitted the
   same way, and headers above 32 MP or 32768 px are rejected.

### Model downloads

- SAM 2, SAM 3 and YOLOE checkpoints come from pinned URLs and are
  SHA-256-verified before use.
- Grounding DINO, Florence and CogFlorence come from pinned Hugging Face
  revisions. They use the Hub's hashes rather than the local digest table.
- BiRefNet matte models are pulled from pinned Hugging Face revisions with
  `trust_remote_code=True`; the pin prevents the remote code from changing
  under an installed copy.
- Cached files are reused. Set `HF_HUB_OFFLINE=1` before starting if no Hub
  request should be attempted. See [`docs/modes/MATTE.md`](docs/modes/MATTE.md)
  for the matte catalogue and download sizes.

## Checks

- `checks\check_gpu.bat` auto-selects the installed AMD ROCm, NVIDIA CUDA, or CPU
  backend and runs the phased health check. It writes a timestamped log beside
  the script.
- `checks\check_gpu_amd.bat` and `checks\check_gpu_nvidia.bat` select a backend
  explicitly. The NVIDIA wrapper states clearly that NVIDIA hardware is untested
  by this project.
- Add `--all-models` to validate every one of the 33 catalogue entries in a fresh
  child process. The AMD RX 6600 passed 33/33 on 2026-11-12. Use `--image PATH`
  to test a real image, `--skip-smoke` to skip the single-model pass, and
  `--skip-4k` to skip the 3840x2160 memory pass.
- `checks\stress_limits.py` runs one model per family under a chosen VRAM/RAM
  cap; weights must already be downloaded.
- `frontend\use_web.bat <WebDir>` links the shared frontend into a backend's
  `web` folder; installers and launchers call it automatically.

There is no pytest suite. The health check is the repeatable test; browser
verification is still manual after it. The NVIDIA path, including the NVIDIA
health-check wrapper, remains unexercised on physical NVIDIA hardware.

## Cleanup and stopping

`cleanup.bat` / `cleanup_amd.bat` remove that backend's virtual environment,
caches, frontend workspace, toolchain and logs, optionally keeping downloaded
models. `stop.bat` / `stop_amd.bat` only stop the running backend and frontend, then
show a final `RESULT: STOPPED` or `RESULT: STOP INCOMPLETE` panel and wait for a key
press so the outcome stays visible.

## Documentation

- [`docs/README.md`](docs/README.md) - documentation index
- [`docs/AMD.md`](docs/AMD.md) - AMD support, TheRock facts and RX 6600 results
- [`docs/NVIDIA.md`](docs/NVIDIA.md) - NVIDIA wheel matrix and untested status
- [`docs/CODE_GUIDE.md`](docs/CODE_GUIDE.md) - architecture, startup, manager and
  verification
- [`docs/BACKEND_GUIDE.md`](docs/BACKEND_GUIDE.md) - routes, manager state,
  model backends and environment variables
- [`docs/FRONTEND.md`](docs/FRONTEND.md) - state, canvas, prompts and undo/redo
- [`docs/LOGGING.md`](docs/LOGGING.md) - log format and console panel
- [`docs/SECURITY.md`](docs/SECURITY.md) - auth fixes, dependency audit and
  residual risk
- [`docs/reference/BACKEND.md`](docs/reference/BACKEND.md) and
  [`docs/reference/FRONTEND.md`](docs/reference/FRONTEND.md) - per-file reference
- [`docs/performance/OVERVIEW.md`](docs/performance/OVERVIEW.md),
  [`CATALOG.md`](docs/performance/CATALOG.md) and
  [`TESTING.md`](docs/performance/TESTING.md) - performance research

## Security

The backend binds to localhost, validates the Host header, and requires a
per-launch token for every route except `/health` and `/api/token`. SSE and the
beacon release path accept the token in the query string because the browser
APIs cannot set headers. See [`docs/SECURITY.md`](docs/SECURITY.md) for the
2026 audit, the `transformers` exception, and the list of remaining risks.

## License

Spectra Segment's own source is GPL-3.0-only. It links `ultralytics`, which is
AGPL-3.0, so the combined distributed work must be AGPL-3.0; commercial or
closed-source distribution needs an Ultralytics Enterprise license. The SAM 3
weights have their own Meta SAM License. See
[`THIRD-PARTY-LICENSES.txt`](THIRD-PARTY-LICENSES.txt) for the full picture.

## Acknowledgments

- ROCm support was inspired by
  [patientx-cfz/comfyui-rocm](https://github.com/patientx-cfz/comfyui-rocm).
- AMD PyTorch wheels are built by [TheRock](https://github.com/ROCm/TheRock).
- Matte mode uses [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) by
  ZhengPeng7 and contributors (MIT).
