# Spectra Segment

A local image background remover with three editing modes. Everything runs on
your own machine — nothing leaves your computer.

> **Note:** This project was built entirely from scratch using AI. There is no
> human-written code.

---

## What It Does

Remove backgrounds from images using AI. Pick an image, choose a mode, and
start removing.

### Three Ways to Remove Backgrounds

**Brush Mode** — Draw strokes around the objects you want to keep. Select
multiple objects separately. Each object gets its own mask.

**Detection Mode** — Type what you want to remove (like "person" or "red car").
The AI finds it, shows a box around it, and removes the background when you
click. Add as many detections as you want.

**SAM 3 Mode** — Type a text prompt and SAM 3 handles detection, segmentation,
and masking in one step. The most powerful mode — single model, no tier
selection needed.

### Every Mode Has

- **Transparent button** — View your cutout with a transparent background
- **Feather slider** — Soften mask edges for smoother results
- **Export** — Save as PNG or JPG with transparent or colored backgrounds,
  as a single image or a ZIP per layer/subject
- **Undo/Redo** — Fix mistakes with Ctrl+Z / Ctrl+Y
- **Console panel** — Live backend log stream with download progress bars

### Brush Mode Extras

- **Brush size slider** — Adjust stroke width
- **Negative strokes** — Right-drag subtracts from the mask
- **Multiple subjects** — Keep separate objects in separate layers
- **Subjects panel** — Switch, add, or delete subjects

### Detection Mode Extras

- **Multiple detections** — Find and remove several objects at once
- **Layers panel** — View and manage all detected objects
- **Batch segmentation ("All")** — Segment every detection with one click
- **YOLOE masks toggle** — Use the detector's own masks (faster) or SAM masks
  (higher quality)

### SAM 3 Mode Extras

- **Text-to-mask** — Type what to segment, SAM 3 finds and masks automatically
- **Encode quality selector** — Fast (512), Balanced (1024), High (1500)
- **Keep loaded toggle** — Prevent the model from unloading between prompts
- **Per-instance control** — Keep/remove individual instances, undo/redo prompts

---

## Getting Started

Everything is portable. The installers fetch their own toolchain (uv, Python,
Bun), the launcher provisions a virtual environment inside the repo folder, and
model weights and runtime caches stay under `model_weights\`. The package cache
lives in the backend folder (`backend\.uv\` or `backend_amd_gpu\.uv_cache_amd\`),
and the installers and launchers redirect temp and cache paths into the project
folder so no system-wide install is needed.

### CPU / NVIDIA GPU

1. Run `install.bat`
2. Run `start.bat`
3. Open <http://localhost:3000>

On NVIDIA systems the installer detects the GPU automatically and installs the
CUDA build of PyTorch; otherwise it falls back to CPU.

> **Note:** The NVIDIA path has not been broadly tested yet.

### AMD GPU (Windows Only)

1. Run `install_amd.bat`
2. Run `start_amd.bat`
3. Open <http://localhost:3000>

The AMD path uses AMD's official ROCm PyTorch wheels built by
[TheRock](https://github.com/ROCm/TheRock). The installer detects your GPU's
architecture (gfx target) automatically and selects the matching `device-gfxNNNN`
extra from TheRock's multi-arch wheel index. Only the ROCm runtime wheels are
installed — the `rocm-sdk-devel` development package is deliberately skipped,
as it is only needed to compile HIP code and would add ~1.4 GB.

**Only the RX 6600 (8 GB, gfx1032) has been actually tested by this project.**
Other cards may work — the wheels exist for them — but nothing below is
verified. Treat it as experimental on any other GPU.

#### AMD Compatibility Notes

Wheel availability comes from TheRock's Windows builds; "app-tested" means
someone ran Spectra Segment on it.

| Family | Example cards | Wheels (TheRock) | App-tested |
|---|---|---|---|
| RDNA 1 | RX 5000 series | published | **no** |
| RDNA 2 | RX 6600/6700/6800/6900 | published | **RX 6600 only** |
| RDNA 3 | RX 7000 series | published | **no** |
| RDNA 3.5 | Strix APUs (880M/890M...) | published | **no** |
| RDNA 4 | RX 9000 series | published, runtime still being validated upstream | **no** |

If you try an untested card and hit crashes or black outputs, it is most
likely the wheel/runtime combination rather than the app — check TheRock's
issues for your gfx target first.

If your system has both an iGPU and a discrete GPU and torch picks the wrong
one, set `HIP_VISIBLE_DEVICES=1` (or disable the iGPU in BIOS).

---

## First Time Setup

1. **Install** — Run the install script for your hardware.
2. **Start** — Run the start script. On first launch it asks for a memory
   profile (how much VRAM/RAM to allow); the defaults are fine.
3. **Choose mode** — Pick Brush/Detect or SAM 3 from the home screen.
4. **Download model** — Click the model selector in the top left and choose
   one. Models download automatically with checksum verification on first use.
5. **Upload image** — Drag an image onto the app or click Upload.
6. **Start removing** — Draw strokes, type prompts, or let SAM 3 do the work.

Large photos are fitted to the working resolution automatically before upload
(the overlay says "Resizing & re-encoding..."), so phone-camera shots just
work. Images beyond 32 MP are rejected as a safety measure.

### Model Downloads

- The single-file checkpoints (SAM 2 / SAM 3 / YOLOE) come from pinned URLs and
  every one is sha256-verified against a published digest before it is used.
- The HuggingFace detector repos (GroundingDINO, Florence, CogFlorence) are pulled
  with `snapshot_download` and rely on the Hub's own integrity checks; they are not
  covered by the local digest table.
- HuggingFace-hosted models use chunked Xet downloads when available.
- Progress appears live in the console panel: one bar for the main weights
  file plus a per-file counter for the smaller support files.
- **Offline after the first download.** A model that is already on disk is
  loaded with `local_files_only` inside a forced `HF_HUB_OFFLINE` /
  `TRANSFORMERS_OFFLINE` guard, so the Hub is never contacted again. The network
  is only touched when a model is actually missing. This is what keeps the app
  usable with no internet once its models are cached.

---

## How It Works

### Brush Mode

1. Upload an image — it gets encoded by the segmentation model.
2. Draw strokes around objects you want to keep.
3. Each stroke produces a mask instantly; right-drag subtracts.
4. Add more subjects with the + button, switch via the subjects panel.
5. Toggle transparent view to inspect, then export.

### Detection Mode

1. Upload an image and load a detector (GroundingDINO, Florence, or YOLOE).
2. Type what to find — boxes appear over matching objects.
3. Click a box to segment it, or press "All" for everything at once.
4. Manage results in the layers panel, then export.

Florence-2 / CogFlorence run on current `transformers` (>= 4.50) again: their
remote code predates the release that removed `GenerationMixin` from
`PreTrainedModel`, so the loader re-attaches it (and a valid generation config)
at load time. Florence has no per-box confidence score, so the confidence
slider does not filter its results — GroundingDINO and YOLOE do report scores.

### SAM 3 Mode

1. Switch to SAM 3 mode and pick the SAM 3 model (~3.3 GB download).
2. Upload an image — encode quality follows your chosen setting.
3. Type a prompt like "red bottle" — every matching instance gets masked.
4. Stack prompts, remove instances, adjust quality, then export.

---

## Performance Notes (AMD / ROCm)

The launcher sets these automatically; they are listed here for reference:

- `TORCH_BLAS_PREFER_HIPBLASLT=1` + `DISABLE_ADDMM_CUDA_LT=1` — hipBLASLt is
  preferred for speed, but its addmm path is disabled because it crashes on
  some RDNA2 wheels.
- `MIOPEN_USER_DB_PATH` + `MIOPEN_CUSTOM_CACHE_DIR` — keep the MIOpen kernel
  database inside `model_weights\miopen` so it persists between runs.
- `HSA_ENABLE_SDMA=0` — avoids DMA-engine copy stalls reported on Windows.
- Encoders run under `torch.inference_mode()`, which stops autograd from building a
  live graph over the prompt encoder and decoder. That cut SAM2's per-image encoder
  memory dramatically and made encodes much faster.
- After any PyTorch/ROCm wheel update, double-click
  `checks\check_gpu.bat` (or run it from a terminal) to verify nothing
  regressed. Each run writes a timestamped log next to the script.

## Environment Knobs

Most knobs have sensible defaults set by the launcher:

| Variable | Effect |
|---|---|
| `SPECTRA_VRAM_FRACTION` | Fraction of VRAM this app may plan around (profile-based). |
| `SPECTRA_FORCE_CPU` | Ignore all GPUs. |
| `SPECTRA_KEEP_SEG_ON_DETECT` | Keep the SAM model resident across detector runs (faster alternating, more VRAM). |
| `SPECTRA_DETECTOR_FP16` | Load detectors in fp16 on GPU (halves their VRAM; CPU stays fp32). |
| `SAM2_QUANTIZE`, `SAM3_QUANTIZE` | Working precision override (16 = fp16). |
| `SAM2_OFFLOAD_ENCODER` | Keep the SAM2 image encoder on CPU between images. |

---

## System Requirements

- Windows 10 or later (all launchers are Windows-only)
- 8 GB RAM minimum (16 GB recommended)
- 10–20 GB free disk space for models and caches
- The first download of each model needs internet; after that the app runs
  fully offline

---

## Cleanup

To remove the app and start fresh:

1. Run `cleanup.bat` (CPU/NVIDIA) or `cleanup_amd.bat` (AMD)
2. Confirm with Y
3. Choose whether to keep downloaded models

This removes the virtual environment, package caches, frontend workspace,
toolchain binaries, logs, bytecode caches, and temp files. Each cleanup script
removes its own backend's files (plus shared installer temp and the frontend
lockfile); protected files (LICENSE, README, .gitignore, THIRD-PARTY-LICENSES.txt)
are never touched.

---

## Stopping the App

Run `stop.bat` or `stop_amd.bat`. This stops the backend and frontend without
removing files — start again anytime.

---

## Project Structure

```
spectra-segment/
├── install.bat           # CPU/NVIDIA installer (fetches uv, python, bun)
├── install_amd.bat       # AMD ROCm installer (TheRock wheels)
├── start.bat             # CPU/NVIDIA launcher
├── start_amd.bat         # AMD launcher (memory profile + GPU setup)
├── stop.bat / stop_amd.bat
├── cleanup.bat / cleanup_amd.bat
├── launcher/             # setup_memory.ps1 - memory profile (called by both start scripts)
├── checks/               # health-check script + bat wrapper + run logs
├── backend/              # FastAPI backend - CUDA/CPU fork
│   └── model_weights/    # downloaded models + runtime caches (CPU/NVIDIA)
├── backend_amd_gpu/      # FastAPI backend - ROCm fork
│   ├── model_weights/    # downloaded models + runtime caches (AMD)
│   └── detect_amd_gpu.py # AMD-only gfx target detection
├── frontend/             # React 19 + Vite UI (served on :3000)
└── tools/                # portable uv/python/bun (created by installers)
```

Backend API highlights: session lifecycle with token auth, stroke/bbox/SAM3
prompting, detector catalog with live status streaming, server-side compositing
for exports, SSE log and model-status streams. Both backend forks share the
same API surface and differ only in hardware-specific loading and tuning.

---

## License

Spectra Segment's own source is under the GNU General Public License v3.0
(GPL-3.0-only).

It also depends on `ultralytics`, which is **AGPL-3.0**. GPL-3.0 and AGPL-3.0 are
only one-way compatible, so the combined, distributed work must be released as
AGPL-3.0 — it is not accurate to call the whole app GPL-3.0 while ultralytics is
linked. Commercial or closed-source distribution requires an Ultralytics
Enterprise license.

Read [THIRD-PARTY-LICENSES.txt](THIRD-PARTY-LICENSES.txt) for the full picture,
including the Meta SAM License that governs the SAM 3 weights and the Apache-2.0
huggingface_hub / hf-xet stack.

---

## Acknowledgments

- ROCm support heavily inspired by
  [patientx-cfz/comfyui-rocm](https://github.com/patientx-cfz/comfyui-rocm).
- AMD PyTorch wheels by [TheRock](https://github.com/ROCm/TheRock).
