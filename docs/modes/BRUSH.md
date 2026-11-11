# Brush mode

Brush is the direct way to cut out a subject: paint over it, and SAM turns the
stroke into a mask. Use it when the subject is easy to point at but hard to
describe in words.

## Quick start

1. Pick **Brush** on the home screen and choose a SAM 2 checkpoint (the small
   `sam2.1_s` is a good default on an 8 GB card).
2. Upload an image.
3. Paint over the subject. A short click or a small scribble is enough; you do
   not need a closed outline.
4. Use **Add subject** for another object. Each subject has its own mask, colour
   and undo history.
5. Export the active subject, every subject as separate files, or one combined
   image.

## How a stroke becomes a mask

Points are collected in image space with a 2 px minimum spacing and clamped to
the image bounds. `buildStrokePrompt` turns them into one of three prompts:

- **Point** - fewer than three points become one positive click.
- **Area** - an open or small stroke is rasterised into a compact set of
  positive points inside the painted area.
- **Ring** - a closed stroke becomes a bounding box plus up to 8 interior
  positive points. The box keeps the extent; the painted outline is not sent as
  negative points.

The point count is capped at 10 for open strokes so a long stroke does not become
a hundred decoder calls. Closed strokes were the weak spot: the old negative-ring
prompt covered 74.7% of the ball benchmark, while box + interior points covered
99.8% with 0.4% extra. The result is a SAM mask for the active subject.

## Negative strokes

Right-drag paints a negative stroke. It is not a SAM negative click and not a
fixed-width pixel erase. The stroke is used as an **automatic selection**:

1. SAM gets the stroke points as positive prompts plus a box around the stroke.
2. SAM selects the region under the stroke automatically.
3. The server subtracts that selected mask from the current subject.

This is the same mask selection SAM uses for a click, applied to the region you
painted. It removes what SAM recognises there, including soft edges, instead of
cutting a brush-shaped hole. A small dilation closes the selection's soft edge.

Benchmark on a 1920x1075 frame, five stroke points:

| Strategy | Stroke removed | Subject lost outside | IoU with ideal |
|---|---:|---:|---:|
| Negative-point prompt (`label=0`) | 100% | 99.4% | 0.6% |
| Geometric strip erase | 100% | 0% | 100% |
| SAM selection + subtract | 99.8% | 0.2% | 99.8% |

The negative-point prompt collapses the subject, which is why it is not used.
The selection path keeps model-driven edges and cannot erase the whole subject
from a stray negative click.

`stroke_radius` still travels with the request; it sizes the selection box and
the edge cleanup. The Brush toolbar no longer has a brush-size slider.

## Subjects and history

- **Add subject** creates a new layer and makes it active.
- **Toolbar chips** switch the active subject; the canvas draws the active mask
  on top.
- **Undo/redo** is per subject. A burst of clicks is coalesced, and one undo is
  one replay pass regardless of history length.
- **Clear** removes the active subject's mask; **Delete** removes the subject
  and its history.

## Tips

- Paint the boundary, not the whole object. A thin stroke along the edge gives
  SAM a cleaner prompt than filling the area.
- If a stroke grabs too much, right-drag over the extra part. Do not try to
  "subtract" with a long negative scribble; short erases are easier to control.
- If two subjects touch, paint them as separate subjects and use the layer
  visibility controls to check each one.
- Feather is applied at render/export time; it does not change the stored mask.

## Limits

- SAM 2 checkpoints are single-image models; there is no video tracking.
- Very thin structures (hair, wire) are better served by Matte mode.
- The right-drag gesture commits on pointer-leave because it has no pointer
  capture. The points collected before leaving are still exact.
- The model is unloaded when another mode needs the VRAM, then reloaded on the
  next stroke.
