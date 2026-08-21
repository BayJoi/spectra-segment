# Spectra Segment

A local image background remover with three editing modes. Everything runs on your own machine — nothing leaves your computer.

> **Note:** This project was built entirely from scratch using AI. There is no human-written code.

---

## What It Does

Remove backgrounds from images using AI. Pick an image, choose a mode, and start removing.

### Three Ways to Remove Backgrounds

**Brush Mode** — Draw strokes around the objects you want to keep. Select multiple objects separately. Each object gets its own mask.

**Detection Mode** — Type what you want to remove (like "person" or "red car"). The AI finds it, shows a box around it, and removes the background when you click. Add as many detections as you want.

**SAM 3 Mode** — Type a text prompt and SAM 3 handles detection, segmentation, and masking in one step. The most powerful mode — single model, no tier selection needed.

### Every Mode Has

- **Transparent button** — View your cutout with a transparent background
- **Feather slider** — Soften mask edges for smoother results
- **Export** — Save as PNG or JPG with transparent or colored backgrounds
- **Undo/Redo** — Fix mistakes with Ctrl+Z / Ctrl+Y

### Brush Mode Extras

- **Brush size slider** — Adjust stroke width
- **Multiple subjects** — Keep separate objects in separate layers
- **Subjects panel** — Switch, add, or delete subjects

### Detection Mode Extras

- **Multiple detections** — Find and remove several objects at once
- **Layers panel** — View and manage all detected objects
- **Batch segmentation** — Segment all detected objects with one click

### SAM 3 Mode Extras

- **Text-to-mask** — Type what to segment, SAM 3 finds and masks it automatically
- **Encode quality selector** — Fast (512), Balanced (1024), or High (1500) — higher means better masks but slower encoding
- **Keep loaded toggle** — Prevent the model from unloading between prompts
- **Layers panel** — View and manage all SAM 3 instances
- **Undo/Redo** — Undo/redo individual prompts

---

## Getting Started

### CPU (No GPU Required)

1. Run `install.bat`
2. Run `start.bat`
3. Open `http://localhost:3000` in your browser

### NVIDIA GPU

1. Run `install.bat` (same as CPU — it detects your GPU automatically)
2. Run `start.bat`
3. Open `http://localhost:3000`

> **Note:** NVIDIA GPU support has not been tested yet.

### AMD GPU (Windows Only)

1. Run `install_amd.bat`
2. Run `start_amd.bat`
3. Open `http://localhost:3000`

> **Note:** AMD version is Windows only. Tested on RX 6600 (8GB VRAM). Support for other AMD GPUs is untested — it might work, it might not.

---

## First Time Setup

1. **Install** — Run the install script for your hardware
2. **Start** — Run the start script
3. **Choose mode** — Pick Brush/Detect or SAM 3 from the home screen
4. **Download model** — Click the model selector in the top left and choose one
5. **Upload image** — Click Upload or drag an image onto the app
6. **Start removing** — Draw strokes, type prompts, or let SAM 3 do the work

Models download automatically the first time you select them. After that, they're stored locally.

### SAM 3 First Time Setup

1. Switch to SAM 3 mode from the home screen
2. Select the SAM 3 model from the dropdown in the top bar
3. The model downloads and loads automatically (~3.3 GB)
4. Upload an image — the Upload button is disabled until the model is ready
5. Type a prompt and press Enter to segment

---

## How It Works

### Brush Mode

1. Upload an image
2. The image gets encoded (you'll see a loading screen)
3. Choose Brush mode
4. Draw strokes around objects you want to keep
5. Each stroke creates a mask
6. Add more subjects with the + button
7. Toggle transparent view to see results
8. Export when ready

### Detection Mode

1. Upload an image
2. The image gets encoded
3. Choose Detect mode
4. Type what you want to find (e.g., "person", "car", "eyes")
5. Click Detect or press Enter
6. Click on detected objects to create masks
7. Use "All" to segment everything at once
8. Export your results

### SAM 3 Mode

1. Switch to SAM 3 mode from the home screen
2. Select the SAM 3 model from the dropdown — downloads automatically on first use
3. Upload an image (the image encodes with the selected quality)
4. Type a text prompt (e.g., "person", "red bottle")
5. Press Enter — SAM 3 finds and segments matching objects
6. Each prompt creates a new layer — stack multiple prompts for complex scenes
7. Adjust encode quality if masks aren't precise enough
8. Export your results

---

## Features

- **Local processing** — Everything stays on your machine
- **Multiple formats** — Supports JPG, PNG, and WebP
- **Drag and drop** — Drop images directly onto the app
- **Replace image** — Swap images without restarting
- **End session** — Unload models and start fresh
- **Real-time logs** — Click Logs to see download progress and errors
- **Keyboard shortcuts** — Ctrl+Z undo, Ctrl+Y redo, Delete to remove
- **Zoom and pan** — Mouse wheel to zoom, Shift+drag to pan
- **Feather control** — Soften mask edges from 0-20 pixels
- **Export options** — PNG with transparency or JPG with background color

---

## System Requirements

- Windows 10 or later
- 8 GB RAM minimum (16 GB recommended)
- 10-20 GB free disk space

---

## Cleanup

To remove the app and start fresh:

1. Run `cleanup.bat` (CPU/NVIDIA) or `cleanup_amd.bat` (AMD)
2. Confirm with Y
3. Choose whether to keep downloaded models

This removes:
- Python and tools
- Virtual environment
- Package cache
- Frontend workspace
- Logs and temp files

> **Note:** Each cleanup script only removes its own backend. `cleanup.bat` removes CPU/NVIDIA files. `cleanup_amd.bat` removes AMD files. They don't touch each other. Protected files (LICENSE, README, .gitignore, THIRD-PARTY-LICENSES.txt) are never removed.

---

## Stopping the App

To stop the app without removing anything:

1. Run `stop.bat` (CPU/NVIDIA) or `stop_amd.bat` (AMD)
2. This stops the backend and frontend without removing any files

You can restart anytime by running `start.bat` or `start_amd.bat` again.

---

## Project Structure

```
spectra-segment/
├── install.bat          # CPU/NVIDIA installer
├── install_amd.bat      # AMD GPU installer
├── start.bat            # CPU/NVIDIA launcher
├── start_amd.bat        # AMD GPU launcher
├── stop.bat             # Stop CPU/NVIDIA app
├── stop_amd.bat         # Stop AMD app
├── cleanup.bat          # Remove CPU/NVIDIA setup
├── cleanup_amd.bat      # Remove AMD setup
├── backend/             # Python backend (NVIDIA/CPU)
├── backend_amd_gpu/     # Python backend (AMD)
├── frontend/            # Web interface
└── launcher/            # Memory profile setup
```

---

## License

This project is licensed under the GNU General Public License v3.0.

See [THIRD-PARTY-LICENSES.txt](THIRD-PARTY-LICENSES.txt) for all dependency licenses.

---

## Acknowledgments

ROCm support was heavily inspired by [patientx-cfz/comfyui-rocm](https://github.com/patientx-cfz/comfyui-rocm).
