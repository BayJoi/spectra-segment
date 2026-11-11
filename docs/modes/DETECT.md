# Detect mode

Detect lets you name what you want instead of painting it. A text-conditioned
detector proposes boxes, and SAM turns the chosen box into a mask.

## Quick start

1. Pick **Detect** and a detector (Grounding DINO Tiny is the default).
2. Upload an image.
3. Type a prompt such as `person`, `red car` or `the dog on the left`.
4. Press **Detect**. The boxes appear on the canvas and in the detections list.
5. Press **Segment** on one box, or **All** to segment every box.
6. Export the active layer, all layers as separate files, or a combined image.

## The two-stage pipeline

**Stage 1 - detection.** The backend loads the selected detector and runs it on
the image with the prompt. Grounding DINO returns boxes and scores; Florence and
CogFlorence are supported alternatives. The threshold is exposed in the toolbar.
Detections are not masks: a box can contain background and can overlap another
box.

**Stage 2 - segmentation.** SAM is prompted with the chosen box and returns a
mask. This is why Detect produces cleaner edges than the raw detector boxes.
**YOLOE masks** is a shortcut: when the detector is YOLOE, use its own masks and
skip the SAM pass. That is faster and lower quality.

**All** runs stage 2 for every detection in one batch. Batch segmentation is
still one SAM encode plus one decode per box; it does not magically make a large
set free.

## Layers and subjects

Each segmented detection becomes its own layer with a colour and a visibility
toggle. The layer is independent: you can segment more detections, hide one to
inspect another, or delete it without disturbing the rest. The canvas draws the
active layer on top.

## Tips

- Be specific: `person` finds every person, `person on the left` narrows it.
- Lower the threshold when an object is missed; raise it when the detector
  returns clutter.
- If a box is too tight or too loose, draw a brush correction in Brush mode
  rather than re-running detection.
- Detector downloads happen on first use and are cached by pinned revision.
  The console shows the progress bar.

## Verification

All fourteen detector checkpoints were run on the tested RX 6600 with ROCm 10.3
and torch 2.16 on 2026-11-12: seven Grounding DINO / MM-GDINO / Florence-2 /
CogFlorence entries and seven YOLOE entries. The full 33-model sweep passed 33/33.
The detector path used a generic test image with a two-object prompt at
confidence 0.05; every detector completed, and the Grounding DINO/MM-GDINO
entries returned the 20-box cap. This is an integration smoke test, not a detection-quality
benchmark. Per-model load, inference and peak memory are in
[`AMD.md`](../AMD.md#2-what-was-actually-run).

## Limits

- Detection quality depends on the prompt and the detector; it is not a
  segmentation model on its own.
- Overlapping detections are kept as separate layers; there is no automatic
  non-maximum merge across prompts.
- YOLOE masks are the detector's own output and can be coarse.
- Text prompts are English-oriented, matching the underlying checkpoints.
- The detector and SAM both need VRAM; on 8 GB cards the manager evicts the
  other model when switching modes.
- The detector image processor honours the launcher cap (1280 px on the RX 6600)
  instead of its 1333 px default. If that still runs out of memory, the backend
  retries at 1024 and then 800 px on the GPU and reuses the working size for the
  rest of the process. CPU is only used after all GPU sizes fail.
- Grounding DINO is loaded in fp16 on GPU. The pinned transformers build reports
  `_supports_sdpa = False` for this model, so it uses eager attention; do not force
  SDPA or treat it like the SAM backbones. Its processor cap is the first memory
  control, not a generic resolution downscale applied to every detector.
- Before a VRAM retry, the failed attempt's traceback frames are cleared and the
  Python allocator runs GC before the CUDA/HIP cache flush. This drops the failed
  input/output tensors instead of leaving them resident during the smaller retry.
