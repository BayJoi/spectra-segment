# SAM 3 mode

SAM 3 is the separate text-to-mask page. Unlike Detect, there is no detector
stage: one model reads the prompt, finds instances and returns masks.

## Quick start

1. Open **SAM 3** from the home screen. The first visit downloads the checkpoint
   and the CLIP text encoder.
2. Upload an image.
3. Type a prompt such as `the person on the left` or `every red car`.
4. Choose an encode quality: **Fast (512)**, **Balanced (1024)** or
   **High (1500)**. Higher is slower and uses more VRAM.
5. Keep or remove individual instances from the results list.
6. Export the active instance, all instances, or a combined image.

## Encode quality

The encoder resolution is the main speed/quality dial. 512 is useful for
drafts, 1024 is the normal choice, and 1500 is for small details. The setting
is sent with the prompt; the backend re-encodes when it changes. It does not
persist across a reload, and neither does **Keep loaded**.

If a GPU encode runs out of memory, the backend retries at 768 and then 512 px
before moving the model to CPU. The successful lower size is reused until you
change the setting.

## Instances

SAM 3 can return several instances for one prompt. The results panel lists
them with their masks. **Keep** retains one; **Remove** drops it. The canvas
shows the kept instances, and the active one is drawn on top. This is different
from Detect's per-box layers: instances come from one model pass.

## Keep loaded

By default the model is unloaded when the session ends or another mode needs
the VRAM. **Keep loaded** pins it for the backend's lifetime, which avoids a
reload between prompts at the cost of holding the GPU memory.

## Tips

- Start at 1024. Drop to 512 if the first prompt is slow; raise to 1500 only
  when the mask misses small parts.
- Prompt with a noun phrase, not a sentence. `red car` works better than
  `can you find the red car for me`.
- Use the instance list to separate touching objects instead of writing a more
  complicated prompt.
- The first run needs the CLIP text encoder; the installer downloads it into
  `tools/clip_src`, and the model weights download on first use with a progress
  bar in the console.

## Verification

`sam3.pt` (3,290 MB) was downloaded and cache-verified before testing. On the RX 6600
with ROCm 10.3 and torch 2.16 on 2026-11-12 it loaded in 17.50 s, encoded a
generic test image in 7.07 s, ran a one-object prompt at confidence 0.25 in 1.21 s,
returned one mask, and peaked at 5,932 MB. The full 33-model sweep passed 33/33.
The test calls the backend directly, not the browser flow; see
[`AMD.md`](../AMD.md#2-what-was-actually-run).

## Limits

- The SAM 3 checkpoint is large (5,932 MB peak in the measured run). On an 8 GB
  card, keep the encode at 512 or 1024 and expect the model to be evicted when you
  switch modes.
- Settings are not persisted across reloads.
- The mode is separate from the Brush/Detect/Matte page and has its own session.
- Text prompts are English-oriented, matching the checkpoint.
- There is no per-instance model choice; one SAM 3 checkpoint serves the page.
