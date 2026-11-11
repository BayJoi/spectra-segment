# Matte mode

Matte mode is the fourth main-page mode (Brush, Detect, Matte, and the separate
SAM 3 page). It produces a **soft alpha plane** instead of a thresholded mask, which
is what hair, fur, glass, smoke and motion blur need: a binary mask can only say
"inside" or "outside", while alpha can say "62 % covered".

Matte is chosen on the home screen next to B/D and SAM 3 and needs no segmentation
model: `POST /api/sessions` accepts `model_name: null`, so nothing is downloaded or
loaded until a matte model is. Pick a matte model, upload an image, then press
**Cut out** - nothing runs on upload. The toolbar exposes Auto and Stroke guides;
strokes are an optional correction and still run through the matte model.

Matte is a slightly heavier run than Brush or Detect. A run is a BiRefNet forward
at the model's input size (512², 1024² or 2048²), and the segmentation model is
evicted first; the Matte toolbar and home screen both say so. With no guide it is
one full-image pass. With a guide it normally takes the full pass and keeps the
components overlapping the ROI; it uses context crops only when the input cap is
below the model's native size and the image is large (the 8 GB HR case). On the
tested RX 6600 with ROCm 10.3 and the default fp16 path, the current 8 GB
configuration caps the native-2048² HR pair at 1024²; all ten checkpoints took
1.25-1.96 s per pass in the 2026-11-12 run. The older native-2048² pass took
5.77-5.82 s and is not the shipped 8 GB recommendation (section 1).

Every matte layer is an independent subject. It stores its own model, its own
alpha history and a `visible` flag. The active layer is the only one that receives
strokes. Visibility is the only preview control: the real alpha of every visible
subject is merged and composited over the image or checkerboard, so hiding one
subject leaves the others as actual cutouts, never coloured placeholders.

---

## 1. Models

All catalog entries are MIT-licensed BiRefNet repositories. The catalog is defined
once in each backend's `models/matte_backend.py`; the two copies must stay identical.
The console's Models section lists them separately from segmentation/detector models,
with downloaded counts and a Heavy/Medium/Light label. There is no VRAM compatibility
gate: every checkpoint is selectable. First use downloads the checkpoint from Hugging
Face and shows a progress bar in both console views.

| Key | Repo | Pinned revision | Weights | Input | Role |
|---|---|---|---|---|---|
| `birefnet-lite` | `ZhengPeng7/BiRefNet_lite` | `aa62cd87eafb9cc43056d08ef3615a14628b831d` | 169 MB | 1024² | Light - fastest |
| `birefnet-lite-matting` | `ZhengPeng7/BiRefNet_lite-matting` | `99c33412e3f58e1f33187abdc8c435c645243690` | 85 MB | 1024² | Fast soft edges |
| `birefnet-general` | `ZhengPeng7/BiRefNet` | `e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4` | 423 MB | 1024² | Default, best all-round |
| `birefnet-dynamic` | `ZhengPeng7/BiRefNet_dynamic` | `280306042f57b7a33854319da62fd86aaa89ec4c` | 424 MB | 1024² | Robust across source resolutions |
| `birefnet-dynamic-matting` | `ZhengPeng7/BiRefNet_dynamic-matting` | `074df545be87034e74a96bf71566ecbbc4c15f0a` | 424 MB | 1024² | Soft edges across resolutions |
| `birefnet-matting` | `ZhengPeng7/BiRefNet-matting` | `eccde0a8cbdce7ac5fecfeb06340fe7b949e85d9` | 843 MB | 1024² | Hair and fine edges |
| `birefnet-portrait` | `ZhengPeng7/BiRefNet-portrait` | `b6561965a70070d9143fd9e558f6ca3c481510db` | 844 MB | 1024² | People and portrait edges |
| `birefnet-hr` | `ZhengPeng7/BiRefNet_HR` | `a7a562f6fd16021180f2f4348f4de003a2d3d1e1` | 424 MB | 2048² | High-resolution detail |
| `birefnet-hr-matting` | `ZhengPeng7/BiRefNet_HR-matting` | `5d6b6f8adcb5b417c871b1d84ceaae9871355b7f` | 424 MB | 2048² | High-resolution hair and glass |
| `birefnet-512` | `ZhengPeng7/BiRefNet_512x512` | `678c37f9158280a2b3000b566ecd12bfbfd1a09c` | 424 MB | 512² | Draft resolution, fastest |

`birefnet-dynamic` is trained across 256-2304 px but is run at 1024² here. Both HR
variants have a native 2048² input; the 8 GB launcher caps them to 1024², while a
card with at least 12 GB uses the native size. Their weights are 424 MB each.The picker labels them Heavy. There is no VRAM compatibility gate; a pass that does not
fit retries at lower GPU input sizes and only then uses the CPU fallback instead of
being blocked up front.

Loading goes through
`AutoModelForImageSegmentation.from_pretrained(repo_id, revision=<sha>,
trust_remote_code=True)`. Pinning the revision is deliberate: `trust_remote_code`
executes Python from the model repo, so an unpinned `main` would let the code change
under an installed copy. A cached snapshot is loaded with `local_files_only=True` and
never contacts the Hub; only a missing or unreadable snapshot falls back to the pinned
download. The downloaded repository is still third-party code and should be treated
with the same care as any model checkpoint. Downloads land in
`model_weights/hf_cache` and use the Hub's file hashes; there is no local digest table
for these files. During a load the backend enables HuggingFace progress bars, routes
their output through `StderrInterceptor`, and logs
`Matte <name>: <file> N% complete` lines, which the console renders as download rows.
When the load finishes the first run reports `matte-loaded` on the model-status stream.

All ten checkpoints have been run on the project's hardware. The catalog entries
remain based on their published model cards and revision metadata; the measured
values below describe this card only.

### Measured on the project's hardware

Each row is a fresh process after load, with the default fp16 path, on a generic
test image. `Inference` is the full-image matte pass; peak is
`torch.cuda.max_memory_allocated()` with the counter reset before inference. The
2026-11-12 run used the shipped 8 GB cap, so the HR rows show their effective
1024² input. The full environment and method are in
[`AMD.md`](../AMD.md#2-what-was-actually-run). Other GPUs are unmeasured. The
frontend exports locally and does not call `export-matte` for combined, active or
separate downloads.

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

The historical native-2048² run measured 5.77-5.82 s and 9,677 MB for the HR pair.
It is not the 8 GB configuration.

---

## 2. Pipeline

1. `POST /api/sessions/{id}/matte` runs in `asyncio.to_thread`, so the event loop is
   never blocked by inference.
2. `ModelManager.run_matte` resolves the ROI (none, stroke box, text box, explicit
   box), ensures the matte model is resident, and calls `BiRefNetBackend.matte`.
3. With no ROI, the backend runs the full image at the selected input size. With a
   ROI it first tries the full-image alpha and keeps the connected components that
   overlap the ROI through `_roi_component_mask`. When the requested input is below
   the model's native size and the image is large, `_crop_matte` instead runs context
   crops around the boxes, with margins, so the subject keeps more detail. The tight-
   crop failure mode that motivated component selection is avoided by keeping those
   context margins; a crop that still produces no usable alpha falls back to the
   full pass.
4. Post-processing applies `edge_shift` (bias the alpha by +/-0.50), `edge_feather`
   (Gaussian sigma), and `invert` when the API request asks for them. The frontend
   sends zero for these and applies Edge, Feather and Invert locally instead.
5. The session stores the latest alpha as `uint8` plus a cached base64 PNG. The
   response carries the PNG; the frontend decodes it into a luminance-to-alpha canvas.
6. `POST /api/sessions/{id}/export-matte` still composites the last stored alpha with
   `feather_radius=0` and returns raw PNG/JPEG bytes. The current UI uses local
   compositing instead, because it has multiple layers and their raw alphas.

The alpha is 8-bit end to end. That matches the export path and keeps a 4096²
session at 16 MB; 16-bit alpha is not implemented.

### Client-side layers

The server's `MatteState` is still a single latest alpha per session. The editor's
multi-layer model lives in `frontend/src/lib/matte.ts` and `store/matte.ts`:

- every `MatteLayer` owns `rawAlpha`, `strokes`, `guide`, `info`, `history` and `redo`;
- each Auto or Stroke run appends a snapshot to that layer's history (capped at 12);
- undo/redo swaps the snapshot and re-applies the current local Edge/Feather/Invert;
- switching layers selects that layer's controls; the preview always shows the
  union of every visible subject as real cutouts;
- the eye controls visibility. A hidden subject is absent from the preview and from
  every export option. Hiding the active subject selects the next visible one, so
  strokes never silently target an invisible layer.

A model change re-runs the active layer when it already has an alpha. The model is
stored per layer, so two subjects can deliberately use different checkpoints (for
example `birefnet-portrait` for a person and `birefnet-general` for an object). The
dropdown labels heavy checkpoints `Heavy` but keeps every model selectable; there is no
automatic fallback. Add subject creates a fresh layer in Stroke
mode; the user paints the new subject, so the model gets a ROI instead of trying to
guess two subjects from a single Auto pass.

---

## 3. Guidance semantics

| Guide | What is sent | What happens |
|---|---|---|
| `auto` | nothing | Full-image saliency. Every salient object is kept, not just one |
| `stroke` | points + labels | Positive points become a box; components overlapping it are kept. Strokes are always positive here - Matte has no negative stroke or right-drag erase |
| `text` | text + confidence | The manager loads Grounding DINO Tiny if no detector is loaded, detects up to four boxes, and keeps components overlapping any of them |
| `box` | `[x1,y1,x2,y2]` | Same component selection, for callers that already have a box |

The toolbar exposes **Auto** and **Stroke** only. `text` and `box` remain available on
the API for callers that already have a detector box or a description.

Consequences worth knowing:

- Two objects that touch are one connected component, so a stroke on one keeps both.
  This is the trade-off that avoids crop artefacts.
- Text guidance is a detector plus the matte model. The detector never produces the
  alpha; it only supplies boxes.
- Text guidance needs a detector download on first use. It is not needed for auto or
  stroke.
- Every stroke is a positive guide. Matte has no negative stroke: subtracting by
  re-prompting a saliency model was unreliable, so the option was removed rather
  than left in the UI as a broken control.

---

## 4. Residency and VRAM

There is no per-model VRAM compatibility gate. Every matte checkpoint is selectable,
and heavy models are labelled `Heavy` instead of being hidden or disabled. Loading a
matte model evicts the segmentation model and any detector first
(`_evict_loaded_models`) to free memory, then attempts the load. If a pass does not
fit, the backend retries at 768 and then 512 px on the GPU before unloading to CPU
in fp32; the CPU fallback is logged. The 424 MB checkpoint downloads are the model
size; they are not a VRAM requirement.

Switching back to Brush or Detect reloads the SAM model on demand. The matte model is
not unloaded until another model needs the VRAM or the last session is destroyed.

GPU inference uses `torch.autocast("cuda", float16)` by default;
`SPECTRA_MATTE_FP16=0` switches to fp32. CPU inference is always fp32. On the RX 6600
with BiRefNet Lite and torch 2.16, five back-to-back passes over a 960x720 input
averaged 0.432 s in fp16 and 1.745 s in fp32. The alpha planes had a mean absolute
difference of 0.000117 and a maximum of 0.1013. The fp16 run peaked at 1.59 GB
allocated versus 3.44 GB for fp32, so it lowers the activation footprint as well as
running faster. The 33-model sweep above used the default fp16 path.

---

## 5. API

```
GET  /api/matte/models
  -> {models: [{name, display_name, input_size, tier, perf, soft_alpha,
                weight_mb, downloaded, loaded}]}

POST /api/sessions/{id}/matte
  {model, guide: "auto"|"stroke"|"text"|"box", points?, labels?, bbox?, text?,
   confidence?, max_detections: 1..4, crop_margin: 0..0.5, stroke_radius?,
   edge_shift: -50..50, edge_feather: 0..50, invert}
  -> {alpha: b64 png, width, height, model, guide, box, ms}

POST /api/sessions/{id}/matte/clear     -> {status: "cleared"}

POST /api/sessions/{id}/export-matte
  {format: "png"|"jpg"|"jpeg", background_color?: [r,g,b]}
  -> raw image bytes (last stored alpha only)
```

The frontend composes visible-subject exports on the client: one combined cutout,
the active visible subject, or one file per visible subject. Hidden subjects are
excluded. Downloads are image files; `export-matte` is kept for API callers that only
need the server's latest alpha. `export-matte` forces a white background when `jpg`
is requested without one, because JPEG has no alpha channel. `png` without
`background_color` keeps transparency.

Matte does not need a segmentation model. `POST /api/sessions` accepts
`model_name: null`, which skips the model load completely; the upload, matte,
`matte/clear`, `export-matte` and detection routes work normally, while `predict`,
`segment-batch` and the SAM 3 routes answer 400 "This session has no segmentation
model" until a model is loaded.

Errors: 404 for a missing session, 400 for a validation or guidance failure (for
example "No object matched '...'"), 507 for VRAM exhaustion, 500 for anything else.

---

## 6. Frontend

| File | Role |
|---|---|
| `lib/matte.ts` | Types, alpha PNG decoding, layered alpha, local Edge/Feather/Invert, shift cache, box-blur fallback |
| `store/matte.ts` | Models, subjects, active subject, guide, strokes, busy/history state, edge controls |
| `hooks/useMatte.ts` | Layer runs, per-layer snapshots, undo/redo, add/remove/select, reset |
| `components/MatteBar.tsx` | Two-row toolbar: model, guide, subjects, undo/redo, local Edge/Feather/Invert, export |
| `components/MatteLayersPanel.tsx` | Subject picker with per-subject cutout state, visibility and add/remove |
| `components/MatteExportDialog.tsx` | Combined/active/separate downloads of visible subjects |
| `components/Canvas.tsx` | Captures strokes for the active matte layer; merges visible alphas |
| `hooks/useCanvasRenderer.ts` | The visible-union cutout composited over the checkerboard |

The matte alpha is deliberately **not** a `PackedMask`. `PackedMask.data` is one byte
per pixel holding 0 or 1, and `decodeDetectionMask` thresholds at 127, so a soft
matte cannot round-trip through it. The alpha lives in its own atom as a canvas with
a real alpha channel.

Mode locking reuses `modeLockAtom` (`"brush" | "detect" | "matte"`). Matte now has a
subject/layer list and per-layer undo/redo; `Ctrl+Z` / `Ctrl+Y` route to the active
matte layer while Matte is active. Add subject starts a new Stroke layer. Every
visible subject's alpha is merged and composited once over the image or
checkerboard; hidden subjects are excluded from that union and from exports.

Edge and Feather are local canvas operations. `applyAlphaControls` uses a 256-entry
lookup table for Edge/Invert and caches the shifted canvas per layer, so changing
Feather does not repeat the full-resolution edge loop. If `ctx.filter` is unavailable
it uses a three-pass separable box blur instead of downscaling, keeping the same
alpha resolution.

---

## 7. Limitations

- Multiple layers are a frontend model. The server stores only the latest alpha per
  session; `export-matte` therefore exports that one alpha, not the whole layer set.
  Use the Matte export dialog for combined, active or separate visible-subject downloads.
- History snapshots are compressed alpha PNGs and are capped at 12 per layer. Undo
  re-decodes the snapshot, which is fast but not instant on very large images.
- The HR (2048²) checkpoints are 424 MB downloads and are labelled Heavy. They have
  no hard VRAM gate; a failed pass retries at 768 and 512 px on the GPU before the
  normal CPU fallback instead of being refused by a limit.
- Component selection can merge touching objects (see section 3).
- Alpha is 8-bit.
- Matte models are downloaded on first use and are not covered by
  `PUBLISHED_DIGESTS`; they rely on the Hub's own hashes plus the pinned revision.
- Matte layers, undo history and local Edge/Feather values are not persisted across
  a reload.

---

## 8. Video models: evaluated, not bundled

VideoMaMa (CVPR 2026, `cvlab-kaist/VideoMaMa`, arXiv 2601.14255) converts coarse
video masks into alpha mattes using a pretrained video diffusion prior. MatAnyone 2
(CVPR 2026, `pq-yang/MatAnyone2`) is a memory-based human video matting framework
with a learned quality evaluator. Both are strong video methods, and neither is a
drop-in replacement for a still-image matte model:

- VideoMaMa expects a video clip and a segmentation mask for the subject, then runs a
  diffusion model over frames. A still image would have to be treated as a one-frame
  clip, which does not use its temporal prior and adds a large diffusion dependency.
- MatAnyone 2 expects video frames plus an initial segmentation mask and propagates
  through time. It is designed for human video and has no still-image mode.
- Supporting either properly means adding video upload/decode, frame extraction,
  tracking/mask propagation, progress reporting over frames, and video export. That
  is a separate pipeline, not another entry in the matte model dropdown.
- VideoMaMa's repository is under the Stability AI Community License, and MatAnyone 2
  is research code with its own terms. Neither should be added to the default
  installer without a licence review.

Decision: not bundled. The image matte path remains BiRefNet-only. A future video
mode should be opt-in and licence-gated, with its own download and progress UI.

## 9. SAM2Matting: evaluated and rejected

SAM2Matting (ECCV 2026, `FudanCVL/SAM2Matting`) was investigated as an alternative
because it supports point, box, mask and text prompts with a MatAnyone-style matting
head.

Blocking facts, verified against the checked-out repository:

1. **License: CC BY-NC 4.0** (`LICENSE`: "Creative Commons
   Attribution-NonCommercial 4.0 International"). Non-commercial only. Spectra
   Segment is GPL-3.0-only and links AGPL-3.0 code, so this cannot be bundled or
   redistributed as part of the app.
2. **`requirements.txt` pins `torch-tensorrt==2.8.0`**, alongside
   `torch==2.8.0` and `torchvision==0.23.0`. TensorRT is NVIDIA-only; on the ROCm
   build used here (`torch 2.16.0a0+rocm10.3.0a20261110`) `import torch_tensorrt` raises
   `ModuleNotFoundError`, so the unmodified install cannot run.
3. Running it on ROCm would mean dropping TensorRT, unpinning torch, and re-testing
   the matting head - while still shipping non-commercial weights.

Decision: not bundled. If it is ever wanted, the clean shape is an optional,
user-supplied, licence-gated plugin that never enters the default catalogue.

---

## 10. Adding a matte model

1. Add a `MatteModelSpec` to `MATTE_MODELS` in `models/matte_backend.py`: registry
   key, HF repo id, pinned commit revision, display name, input size, tier, perf text,
   weight and VRAM estimates.
2. Confirm the repo actually loads through
   `AutoModelForImageSegmentation` with `trust_remote_code=True`.
3. Measure warm pass time on the target card and write a clear Heavy/Medium/Light
   label in `perf`. Do not add a VRAM compatibility gate; every model stays selectable.
4. Mirror `matte_backend.py` to the other fork (it is byte-identical).
5. Add the row to section 1 here. Update the README only with claims that have been
   measured or are factual model-card metadata.

No route or frontend change is needed: the selector is driven by
`GET /api/matte/models`.
