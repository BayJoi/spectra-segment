# Frontend code reference

Part 2 of the split code reference. Section numbers continue from the
[backend code reference](BACKEND.md).

## 15. Frontend: `lib/`

### `lib/api.ts`

`BASE` is built from `window.location.hostname` plus port 8000, with a
`localhost:8000` fallback outside the browser. `#ApiError` carries `status` and a
message built from the first 500 characters of the body with control characters
replaced.

`#getToken` memoises in `_token` and `_tokenPromise`; a rejected fetch clears the promise
so the next call retries. `#resetToken` clears both. `#getFreshToken` bypasses the cache,
fetches `/api/token`, stores the result and returns it; each SSE attempt calls it before
opening, so a backend restart is handled. `#apiFetch` sets `X-Local-Token`, retries
**once** after a 401 with a fresh token, and throws `ApiError` for any other non-OK
response.

| Method | Notes |
|---|---|
| `createSession`, `destroySession`, `sessionHealth` | `sessionHealth` returns a boolean and swallows the error |
| `releaseSession` | `navigator.sendBeacon(.../release?token=...)` (the one release path that accepts a query token) and then a `keepalive` `DELETE` with the cached token. It only sends the DELETE when a token is already cached; on 401 it refreshes and retries once. A page teardown therefore sends two destroy requests, both idempotent |
| `uploadImage` | `FormData`, own 401 retry, same body-sanitising error |
| `predict`, `undo`, `redo`, `clearObject` | All go through `toPredictResponse`, which decodes `object_masks` and passes `object_history` through |
| `sam3Prompt` | Decodes every mask |
| `sam3Undo`, `sam3Redo`, `sam3RemoveInstance` | Return `masks: []` - the server's masks are discarded and the local prompt stack stays the authority |
| `setSam3Settings`, `listDetectors`, `loadDetector` | `loadDetector` posts `{detector_name}` |
| `detect` | Sends `query`, `max_detections`, `use_yoloe_masks`. It never sends `confidence` |
| `exportZip` | Posts `root`, `files`, `include_whole`, `format`, `background_color`, `feather_radius` |
| `exportImage` | Posts `files`, `format`, `background_color`, `feather_radius`; returns a raw `Blob` |
| `getLogs`, `getModels` | `getLogs` asks for a limit and returns `{entries, total}` |
| `segmentBatch` | Keeps `null` for an empty mask and decodes the rest in parallel |

`#decodeObjectMasks` keeps `Map<objectId, {source, mask}>`. An entry is reused when the
base64 string is byte-identical, which is why the renderer can compare `PackedMask`
objects by identity. It also deletes cache entries for subjects that are no longer in the
response. `#resetObjectMaskCache` clears the map; `useSession.resetUndoRedoState` calls it
on session, model and image changes so a mask from the previous session cannot be reused.

`#decodeDetectionMask` base64 -> `Blob` -> `Image`, rejects anything over 8192 per side
or over 8192x8192 pixels in total, reads the red channel and stores `1` where it is
above 127, returning `{w, h, data}`. `#encodeMaskPng` writes white where the mask is set, calls `toDataURL`
and strips the data-URL prefix - synchronously, on the main thread.

### `lib/sse.ts`

`#connectSse(url, onMessage, onStateChange, onOpen?)` opens an `EventSource`, resets the
attempt counter on open, calls `onOpen` after reporting `true`, and on error closes,
reports `false`, and reconnects after `min(30000, 1000 * 2 ** attempt++)`. `url` may be a
string or an async provider, which is how each attempt fetches a fresh token. `close()`
marks the stream closed, clears the timer and reports `false`.

### `lib/utils.ts`

`#cn` wraps `clsx`. `SUPPORTED_IMAGE_EXTENSIONS` is `jpg, jpeg, png, webp`;
`#isSupportedImage` accepts by extension **or** MIME type. `#fitImageFile` decodes the
file once, reads `naturalWidth`/`naturalHeight` from that image, and only builds a canvas
when `w*h > 4096*4096` or the longest side exceeds 8192. The scale is
`min(1, sqrt(16.7 MP / (w*h)), 8192 / max(w, h))`, the output MIME follows the input
(PNG, WebP, otherwise JPEG at 0.92), and a file that will not decode is returned
unchanged.

### `lib/strokePrompt.ts`

Constants: `MAX_RING_POINTS=32`, `MAX_INTERIOR_POINTS=8`, `MAX_AREA_POINTS=10`,
`AREA_ERODE_FRACTION=0.3`, `RASTER_MAX_DIM=1024`.

`#buildStrokePrompt(stroke, brushSize, mode, imgW, imgH)`:

| Case | Result |
|---|---|
| Empty | no points |
| Fewer than 3 points | One point, label 1 or 0 |
| Negative | `buildNegativePrompt`; the server treats it as a positive selection |
| Positive and closed | `buildRingPrompt`; box + interior positive points |
| Positive and open | `buildAreaPrompt` with label 1 |

Closed means `gap <= max(brushSize * 1.5, 4)` **or** `gap <= 12%` of the stroke's
bounding-box diagonal, and at least 4 points.

`#buildRingPrompt` downsamples with a spacing of `max(6, minSide/96)`, caps at 32 ring
points, computes the centroid and mean radius, and pulls interior points in by
`clamp(brushSize / avgRadius, 0.15, 0.5)` of the radius: the centroid plus every
`ceil(len/8)`-th ring point, capped at 8 positives. It emits the ring's bounding box
plus those positive points; the ring outline is no longer negative. Fewer than 3
ring points degrades to a single positive point.

`#buildAreaPrompt` rasterises into an offscreen canvas whose longest side is 1024, strokes
with `lineWidth = brushSize*scale - 2*(0.3*brushSize)*scale` (60% of the brush narrower, so
sampled points sit inside the object), then samples on a grid of
`max(4, round(16*scale))` and caps at 10 points by stride. No sample falls back to the
first point.

All emitted coordinates are rounded and clamped to `0..w-1` / `0..h-1`.

### `lib/detectionEdit.ts`

`#removeDetectionMask` drops the removed index and shifts later keys down one.
`#removeDetectionIndexFromLayers` drops the matching layer and decrements every later
`detectionIndex`. `#shiftSelectedDetection` clears the selection when it is the removed
index, decrements it when a later one was removed, and leaves it otherwise. `Canvas.tsx`
and `useLayers.ts` both call these, so the two removal paths cannot drift.

### `lib/types.ts`, `lib/mask.ts`

`PackedMask` is `{w, h, data: Uint8Array}` with one byte per pixel. The response types
mirror the backend: `PredictResponse.objectMasks` is keyed by number, `objectHistory` by
string, `DetectorInfo.type` is `"grounding" | "yoloe"`, `Detection.mask` is an optional
base64 string that the UI clears once decoded.

---

## 16. Frontend: `store/`

Five files, all plain `atom()` on jotai's default store. There is no `Provider`, so every
component shares one store.

### `store/session.ts`

`sessionIdAtom`, `modelNameAtom`, `imageUrlAtom`, `imageWidthAtom`, `imageHeightAtom`,
`imageFileAtom`, `hasImageAtom` (derived from `imageUrl`), `masksAtom`,
`objectMasksAtom`, `perDetectionMasksAtom`, `brushObjectsAtom` (`[0]`),
`activeObjectIdAtom` (0), `objectUndoCountsAtom`, `objectRedoCountsAtom`,
`objectHistoryAtom`, `subjectMetaAtom`, `sam3ReadyAtom`, `modelsAtom`.

The three mask atoms mean different things:

| Atom | Keyed by | Meaning |
|---|---|---|
| `masksAtom` | position | Brush: `Object.values(objectMasks)` in id order. SAM 3: every instance, flattened. Also the "is there anything to export / show transparently" test |
| `objectMasksAtom` | subject id | The per-subject breakdown the colour overlay is built from |
| `perDetectionMasksAtom` | detection index | One mask per detection, which is why indices must stay stable |

`#SUBJECT_COLORS` is ten muted colours at hues 29.5, 54.2, 90.3, 125.8, 161.9, 198.1,
234.2, 269.7, 305.8 and 341.9 degrees: roughly 36-degree steps with a 24.7-degree minimum
between the first two and 47.6 between the last and the wrap.

`#subjectColor(id)` picks `SUBJECT_COLORS[id % 10]`, then shifts a repeat of the cycle up
or down in lightness by `0.13 * ceil(cycle / 2)` (alternating sign) via
`shiftLightness`, which converts to HSL, clamps lightness to 0.15..0.85 and converts back.
Results are memoised per id. Lightness clamping means two ids far apart in the cycle can
converge on the same swatch; the palette is not guaranteed unique beyond the first ten.

`#subjectName(id)` is `Subject <id+1>`.

### `store/ui.ts`

Dialog and overlay flags: `settingsOpenAtom`, `exportOpenAtom`, `showTransparentAtom`,
`hideBboxesAtom`, `endSessionOpenAtom`, `unsupportedFileAtom`, `modeLockAtom`
(`"brush" | "detect" | null`), `modeDialogOpenAtom`, `modeSwitchTargetAtom`,
`isExportingAtom`, `imageEncodingAtom`, `encodingMessageAtom`, `fitToViewAtom`,
`zoomInAtom`, `zoomOutAtom`, `uploadHoveredAtom`, `consoleOpenAtom`.

Four atoms persist through `atomWithStorage`: `consoleFilterAtom`,
`consoleSourceAtom`, `consoleCategoryAtom`, `consoleShowTimeAtom`.

`#pushToast` keeps at most the last three toasts (it slices to the last two before
appending), emits a `console` category entry and removes the toast after 4 seconds.

### `store/detection.ts`

`detectModeAtom`, `toolModeAtom` (`positive`/`negative`), `brushSizeAtom` (20),
`featherRadiusAtom` (0), `detectorsAtom`, `selectedDetectorAtom` (`""`),
`loadedDetectorAtom`, `detectorLoadingAtom`, `detectQueryAtom`, `detectionsAtom`,
`selectedDetectionAtom`, `isDetectingAtom`, `yoloeMasksEnabledAtom`.

### `store/layers.ts`

`Layer` is `{id, type: brush|detection|sam3, label, preview, objectId, detectionIndex?,
promptIndex?, instanceIndex?, createdAt}`. `#nextLayerId(existing)` scans the current ids
for the highest `layer-<n>`, advances a module-level counter past it and returns the next
id. Taking the id from a value captured once made every `addLayer` call in one synchronous
loop produce the same id, so `selectedLayers` addressed them all and `removeLayer(id)`
deleted them all.

### `store/sam3.ts`

`sam3ModeAtom`, `sam3PromptsAtom`, `sam3RedoStackAtom`, `sam3PromptInputAtom`,
`sam3PromptingAtom`, `sam3InstancesAtom`, `selectedSam3InstanceAtom`,
`sam3KeepLoadedAtom` (true), `sam3EncodeDimAtom` (1024; type `512 | 1024 | 1500`).

### `store/logs.ts`

`#emitUi(category, level, message, detail?)` builds `{id, source: "ui", level,
category, message, ts}` (`ts` is `HH:MM:SS` from `toTimeString`), applies `safeText`
(300 characters for the message, 200 for the detail) and notifies every subscriber
inside a `try/catch`. `#logErr(category, err)` is `emitUi(category, "ERROR", safeText(err))`.
`#subscribeLogs` returns an unsubscribe function. `uiCategoryLabels` is the twelve
categories: `session`, `upload`, `mode`, `detect`, `segment`, `brush`, `sam3`, `subject`,
`layer`, `export`, `settings`, `console`.

`safeText` coerces (`Error` -> `Name: message`, else `JSON.stringify` with a `String`
fallback), replaces `PATH_RE` (`X:\...` and `/home|Users|root|tmp|var|opt|srv|workspace|data/...`)
with `[path]`, `SECRET_RE` (`token|bearer|x-local-token` followed by `:` or `=` and a
value) with `[redacted]`, `UUID_RE` with `[id]`, and `LONGHEX_RE` (32+ hex characters)
with `[hash]`, then truncates with `...`. The backend has no equivalent filter.

---

## 17. Frontend: `hooks/`

### `useSession.ts`

Module-level state, because `useSession()` is instantiated in eight components (App,
Canvas, EmptyState, Header, ModeSwitchDialog, EndSessionDialog, SubjectsPanel, Toolbar)
and a ref inside the hook belongs to one instance:

| Name | Purpose |
|---|---|
| `uploadGuardRef` | One upload at a time |
| `endGuardRef` | One teardown at a time |
| `blobUrlRef` | The single live object URL, revoked exactly once (a per-instance ref leaked one per mode switch and end-session) |
| `sessionQueueMap` | One promise chain per session id |
| `deletedSubjects` | Ids the user deleted; `syncSubjects` refuses to recreate them |
| `pendingStrokesRef` | In-flight stroke count per subject |
| `historyRef` | The authoritative ledger copy the buttons read |
| `historyQueueRef`, `historyScheduledRef` | The coalescing undo/redo queue and its drain flag |

`#enqueue(sessionId, task)` appends to that session's chain and stores the caught promise,
so every mutation - predict, undo/redo burst, clear-object - is FIFO per session and
responses cannot land out of order.

`#applyHistory(history)` replaces the ledger: it rebuilds `objectHistoryAtom`,
`objectUndoCountsAtom` and `objectRedoCountsAtom` from the `object_history` the server
sent. Keys absent from the payload are dropped, which is the signal that the subject was
cleared. `#syncSubjects` creates metadata for ids it has not seen unless they are in
`deletedSubjects`, so a late response cannot resurrect a deleted subject.
`#patchHistory(oid, patch)` merges one subject and is what the 409 paths use - calling
`applyHistory({[oid]: ...})` there would replace the whole ledger and zero every other
subject's depth.

`#uploadImage` refuses a second concurrent upload, rejects an unsupported file into
`unsupportedFileAtom`, creates a session first when there is none (aborting if the model
changed mid-flight), revokes the previous blob URL, runs `fitImageFile`, sets the image
atom and resets masks/layers/detections/SAM 3 state, then uploads. On success it stores
the dimensions, logs the resize, and opens the mode dialog unless SAM 3 mode is active.
The guard is released when the upload settles.

`#switchModel` resets the undo/redo state and SAM 3 state, destroys the current session,
clears image-derived state and creates a session for the new model, re-uploading the
stored `File`. `#endSession` destroys, revokes the blob URL, resets every atom and
`resetUndoRedoState`. `#recoverSession` returns early when the health check passes;
otherwise it resets undo/redo, drops `brush` layers and recreates the session.
Detection masks and layers survive because they are client-side.

`#resetUndoRedoState` clears `deletedSubjects`, the object-mask cache, pending strokes, the
ledger, the queue and the counts, resets `brushObjects` to `[0]`, `activeObjectId` to 0 and
`subjectMeta` to a single subject 0.

`#predict` adjudicates the active subject id, counts the stroke as pending, enqueues the
call, applies the response and decrements the counter in `finally`. **404** triggers
`recoverSession`; **400** shows the backend's message as a toast (object cap, no image);
anything else shows "Prediction failed". It deliberately does not rebuild on 400 - doing
so used to destroy a live session whenever the object cap was hit.

`#requestHistory(oid, kind)` does nothing when the adjusted depth is zero, pushes the
intent and schedules a drain. `#runHistoryOps` takes the whole queue, clamps each op's
steps to the depth at that moment, calls undo or redo, applies the response and logs
"Undid N stroke(s) on Subject X". A **409** patches only that subject ("nothing left to
undo" also resets `strokes` and `has_mask`); a **404** recovers the session. Within one
drain every undo runs before every redo, so a mixed burst resolves undo-first rather than
in click order.

`#historyDepth` uses `availableSteps(depth, pending, kind)` from `lib/historyQueue.ts`:
undo depth plus pending strokes, redo depth minus pending. A pending stroke was enqueued
before the click, lands in the history first, and clears the redo stack, so a queued redo
would ask for an entry the server no longer has.

`#clearObjectHistory(oid)` runs `clear-object` through the same queue and falls back to
patching the ledger to zero when the call fails.

Returned: `sessionId`, `imageUrl`, `uploadImage`, `switchModel`, `endSession`,
`recoverSession`, `predict`, `undo`, `redo`, `undoSubject`, `redoSubject`, `canUndo`,
`canRedo` (both from the count atoms for the active subject), `clearObjectHistory`,
`applyHistory`, `forgetSubject`.

### `lib/historyQueue.ts`

`#availableSteps(depth, pendingStrokes, kind)` returns the usable depth, floored at 0.
`#HistoryQueue` accumulates steps per `(objectId, kind)` in two maps, so five clicks on one
subject become one op with `steps: 5`; `drain()` emits every undo then every redo and
clears both maps.

### `useSessionGuard.ts`

On `pagehide`, releases the session once per session id. On `pageshow` with `persisted`, it
re-arms and health-checks the session, recovering when it is gone. `recoverSession` is held
in a ref so the listeners never capture a stale closure.

### `useCanvasRenderer.ts`

Constants: `MASK_OVERLAY_ALPHA=0.42`, `ACTIVE_OVERLAY_ALPHA=0.56`,
`SUBJECT_LAYER_BUDGET_BYTES=192 MB`, `REBUILD_MIN_INTERVAL=80`, `REBUILD_TRAILING_DELAY=90`.

`currentDpr()` caches `devicePixelRatio`; `supportsCtxFilter()` probes a 16x16 canvas once
and caches the answer.

`#boxBlurFloat(src, w, h, r)` is a separable box blur over **interleaved RGBA** float data,
3 passes, clamping indices at the edges. `#blurPadded(source, radius, crop)` pads by the
radius so the border does not darken, uses `ctx.filter = blur(...)` when supported, else
premultiplies, runs the box blur three times and un-premultiplies. `#blurMaskCanvas` is
`blurPadded(..., crop=true)`.

`#computeState(zoom, x, y)` returns `baseScale = min(cw/iw, ch/ih, 1)`,
`scale = baseScale*zoom`, and offsets `floor((c - floor(i*scale))/2) + pan`, with the inner
`floor` matching the wheel handler's rounding so the pixel under the cursor stays put.

`#ensureSubjectLayer(id, mask, feather, img)` returns a cached full-resolution tinted
canvas when `mask` (identity), image (identity), feather radius and colour all match,
otherwise rebuilds: draw the mask white into the reused scratch canvas via a
`Uint32Array` view, feather it, `source-in` the subject colour **on that layer**, then
scale it onto an image-sized canvas. Layers above the 192 MB budget are not cached at all
and are rebuilt every time; the cache evicts the least recently inserted subject first
(the `Map` iteration order is refreshed on a hit) and never evicts the entry it is
currently building. The scratch canvas is reused, so the cached canvas is always a
separate allocation.

Tinting the accumulator instead would recolour every subject already composited and
multiply alpha by 0.42 per pass, leaving every mask one colour at `0.42^N` opacity.

`#rebuildSubjectMaskCache` drops layers whose subject disappeared or whose image changed,
orders subjects so the active one is drawn last, composites at 42%/56%, and caches under
`#subjectMaskKey(ids, activeId, featherRadius, revision)` - the same helper the caller uses
to compare. When nothing could be drawn it stores an **empty** key, which can never match,
so the next frame retries instead of painting a cached blank.

`#rebuildMaskCache` is the no-subject fallback: one union overlay tinted `0x783475d4`,
keyed on `f<feather>|r<revision>`, and it skips any mask whose size differs from the image.
`#rebuildTransparentComposite` draws the image then punches it with `destination-in`
through the feathered union mask, with the same key rule.

`#maybeRebuild(name, fn)` runs immediately when 80 ms have passed since the last rebuild;
otherwise it stores the job under its name (`subject`, `masks`, `transparent`), schedules
one 90 ms trailing timer and runs every pending job plus a redraw. The per-name map is
what stops a deferred subject rebuild from being overwritten by a mask rebuild.

`renderMasks` clears the mask canvas with the DPR transform, then either draws the
transparent composite (or the plain image when there are no masks), the per-subject overlay
when `objectMasks` is non-empty, or the union overlay.

`renderDetections` and `renderSam3Boxes` share the detection canvas and draw green
`#00ff88` boxes with a `#<n> <label> <score>%` / `#<n> <text>` chip, orange when
selected; both return early when `hideBboxes` is on in transparent mode.

`#noteMaskScale` emits one `subject` warning per `maskSize->imageSize` pair when a mask's
resolution differs; the mask is still stretched onto the image rect, because skipping it
left a silently empty composite cached forever.

`#redraw` sizes every backing store to `floor(css * dpr)` (skipping the assignment when it
is unchanged), resets each context transform, paints the checkerboard in transparent mode
(cached pattern per container size), and calls the three renderers. `#invalidateCache`
empties the three cache keys, bumps `maskRevisionRef` and drops the checker pattern; it
deliberately keeps the cached canvases so a deferred rebuild shows the previous overlay
rather than a blank one.

### `useBrushDrawing.ts`

Holds `isDrawingRef`, `strokePointsRef`, `activeStrokeModeRef` and the processing-ring
state. `#drawBrushPreview` clears the draw canvas and strokes the whole point list in
image space, using the active subject colour for positive strokes and `#ff4444` for
negative, at `brushSize * scale` width and 60% alpha. `#canvasToImage` maps a CSS point to
image coordinates and clamps to the image. `#isInsideImage` tests the unclamped mapping.
`#startProcessingRing` animates a pulsing ring at a point until `#stopProcessingRing`
clears it; the animation frame is cancelled on unmount.

### `useDetection.ts`

`#detect(sessionId)` requires a query, defaults the detector to `grounding-dino-tiny`,
loads it when it is not the loaded one, posts `max_detections: 20` with
`use_yoloe_masks`, logs the count and duration, appends the new detections and **keeps only
the existing mask entries**, so keys stay aligned with the array. It clears the selection,
returns `{detections, baseOffset}` and clears `loadedDetector` on failure.

### `useLayers.ts`

`#addLayer` mints an id from `nextLayerId(layers)` and appends. `#removeLayer` on a
detection layer reindexes masks, detections, layer indices and the selection through
`lib/detectionEdit.ts`; on a brush layer it drops the layer, and when no brush layer
remains it clears masks and calls `clear-object` for every brush object id. `#clearAllLayers`
does both lists and clears every server brush subject. `#selectLayer` replaces the set or
toggles with Ctrl/Meta. `#syncDetectionLayers(dets)` replaces all detection layers at
offset 0; `#appendDetectionLayers(dets, offset)` keeps existing detection layers and appends
with `offset` (or 0 when none exist).

### `useSam3.ts`

`#prompt(text)` posts the trimmed text, appends `{text, masks, scores, bboxes}`, clears the
redo stack and reconciles; it extracts a `"detail"` string from an error body for display.
`#syncFromPrompts` is the single reconciliation point: it flattens every prompt's masks
into `masksAtom`, builds `sam3InstancesAtom` as
`{promptIndex, instanceIndex, text, bbox, score}` and calls `#reconcileLayers`, which
reuses layers by the `${promptIndex}:${instanceIndex}` key (updating only the label) and
mints new ids from `nextLayerId`.

`#undo` and `#redo` are optimistic: they change the local stack, call the endpoint, and
restore the previous state if it fails. Because `api.sam3Undo`/`sam3Redo` discard the
server's masks, the local stack is the pixel authority; after a `recoverSession`, which
starts an empty server history, the two can diverge. `#removeInstance` filters the
instance out of its prompt's masks/scores/bboxes, drops the prompt when it becomes empty
(which shifts every later `promptIndex` on both sides) and clears the redo stack.

`canUndo`/`canRedo` are `prompts.length > 0` and `redoStack.length > 0`.

### `usePanZoom.ts`

`fitToView` resets zoom to 1 and pan to 0. `zoomIn` multiplies by 1.25 capped at 50,
`zoomOut` by 0.8 floored at 0.1. Each schedules a redraw.

### `usePanelDismiss.ts`

While open, listens on `document` for `mousedown` outside the panel and outside
`ignoredRef`, and for `keydown` Escape. `ignoredRef` is the panel's own toggle button:
without it the button's `mousedown` closes the panel and the following `click` reopens it,
so the button could never close anything.

---

## 18. Frontend: `components/`

### `App.tsx`

Renders `Header`, then either `EmptyState`, `Sam3Page`, or `Canvas` + `MatteBar`/`Toolbar`
based on `hasImageAtom`, `sam3ModeAtom` and `matteModeAtom`, then the dialogs,
`EncodingOverlay`, `ToastHost` and the upload vignette. `SettingsDialog` and `ExportDialog` are lazy-loaded.
While a session has an image, window-level `dragover`/`drop` handlers accept a file and
open `ReplaceImageDialog`; the same dialog opens from the header's Replace button. Escape
closes it before the other dialogs.

Global keys: Escape closes the mode-switch dialog, blurs a focused input, then closes
mode/end/settings/export or clears the detection selection in that order. Ctrl+Z /
Ctrl+Y are prevented and routed to SAM 3 or brush undo/redo (ignoring key repeat; Shift is
not considered). Delete/Backspace removes the selected SAM 3 instance or the last one, and
in non-detect mode removes the last layer. `beforeunload` prompts while a session or
unsaved masks/layers exist. On mount it fetches the detector list and fades out the
`#splash` element.

### `Canvas.tsx`

Four stacked canvases: `imageCanvas`, `maskCanvas`, `detectCanvas`, `drawCanvas`, the last
one owning the pointer handlers. `allMasks` is `perDetectionMasks + masks`, which is what
the union and transparent caches consume.

Pointer behaviour: left button in brush mode starts a stroke and takes pointer capture;
right button starts a **negative** stroke (also handled through `onMouseDown`, because a
right press does not always produce a pointerdown on the same target); middle button or
Shift+left pans; wheel zooms around the cursor with a 0.9/1.1 delta clamped to 0.1..50 and
recomputes pan from the image point under the cursor. Points are collected at 2 px
(`STROKE_COLLECT_SPACING`) minimum spacing and clamped to the image.

`#handlePointerUp` is bound to both `onPointerUp` and `onPointerLeave`. A left drag holds
pointer capture, so leaving the canvas keeps collecting; a right drag does not, so it
commits mid-gesture. On commit it builds the prompt, logs the sample count, clears the draw
canvas, starts the processing ring after 150 ms, awaits `predict`, and stops the ring only
if no newer stroke has started.

`#handleDetectClick` hit-tests detections from the topmost down, toggles the selection and
the matching layer, and segments on first click: the detector's own mask when YOLOE masks
are enabled and present (cleared from the detection afterwards), otherwise
`segmentBatch` for that single box. It adds a detection layer when none exists for that
index. A failure in the YOLOE-decode fallback path is not caught.

Delete/Backspace removes the selected detection, or the last one in detect mode, using the
same `detectionEdit` helpers. A `ResizeObserver` and the image-load effect both invalidate
the caches and schedule a redraw; mask, detection, feather and SAM 3 changes invalidate
too, while selection/transparent changes only redraw.

### `Toolbar.tsx`

Floating bar with two personalities. Detect: query input (Enter runs), detector select
grouped into Grounding DINO / Florence-2 / YOLOE with tier icons, hints and a "Loaded"
chip; the existing detector is disabled in the list; a YOLOE toggle appears only for YOLOE
detectors; the Detect button; when results exist, an "All" batch segment button, a result
count and the boxes toggle in transparent mode. Brush: undo/redo, subject switcher
(previous/next/open/add/delete), brush size 2..80, feather 0..20, and clear-all-masks when
there is anything to clear. Both modes: export when masks exist, and zoom in/fit/out.

`#handleDetect` remembers `offset = detections.length` and uses `syncDetectionLayers` for
the first batch or `appendDetectionLayers(dets, offset)` afterwards. `#handleSegmentAll`
sends every detection box in one `segmentBatch`, stores the non-null masks by index and
adds layers for detections that have none. `#addSubject` uses `max(ids) + 1`, so ids are
never reused - `clear_object` deletes server state for an id, and a reused id would inherit
the deleted subject's history. `#deleteSubject` refuses the last subject, moves the active
id to a neighbour, drops the metadata, marks the id deleted and clears it on the server.
`#clearAllBrushObjects` clears every subject on the server and resets to subject 0.

### `SubjectsPanel.tsx`

Lists `brushObjects` with colour swatch, name, stroke count, a mask dot, and per-subject
undo/redo/delete buttons that target that id. Selecting a row sets the active subject.
Missing metadata is filled on render. Delete is the same logic as the toolbar's, including
`forgetSubject` and the server clear.

### `LayersPanel.tsx`

Used by the detect toolbar and by the SAM 3 bar. Detection groups are keyed by the layer
label with the trailing score removed; each row shows `#<index+1> <label> - <score>%`, a
mask dot and a remove button. SAM 3 layers get their own section, select/toggle the SAM 3
selection and call `onRemoveSam3Instance`. Dismissal uses `usePanelDismiss`.

### `Header.tsx`

Owns the model selector (segmentation models, only matte models in Matte mode, or only SAM 3 models in SAM 3 mode), the
image size chip, the Transparent toggle, `ConsolePanel`, Settings, End Session and
Upload/Replace. Replace - and a file dropped on an active session - opens
`ReplaceImageDialog` instead of uploading immediately. It subscribes to
`/api/models/stream` and applies each payload to `modelsAtom` and the detector list,
deriving `loadedDetectorAtom` from the `loaded` flag. Stream events drive the overlay
label: `downloading` (10 min timeout), `loading` (30 s), `unloading` (3 s), `loaded`
(clears), plus `matte-downloading` (10 min), `matte-loading` (60 s) and `matte-loaded`
(clears and refreshes the matte model list). The model list is refetched when a local
load/download finishes. Switching a model shows "Downloading model..." when the target is
not on disk. Residency transitions are not logged by the client because the backend
already logs them.

### `Sam3Page.tsx`, `Sam3PromptBar.tsx`

`Sam3Page` renders `Canvas interactive={false}` plus the prompt bar. The bar has the prompt
input (Enter submits, clearing the input immediately), Segment (spinner while prompting),
undo/redo, a segment-count button opening `LayersPanel`, Export (disabled with no
instances), the boxes toggle in transparent mode, and an options popover with the
keep-loaded switch and the Fast/Balanced/High encode buttons (512/1024/1500) that call
`/api/settings/sam3`; changing the dimension shows a re-encoding spinner. Errors from
`useSam3` render inline. The popover closes on outside mousedown.

### `ExportDialog.tsx`

Three scopes: everything, one group, or picked sub-layers. Groups are SAM 3 prompts (by
text), one group per brush subject, or detections grouped by layer label. `includeWhole` is
true only for "everything". Paths are `<root>/<folder>/<slug>_<n>.png` for detections and
SAM 3, and `<root>_subj<n>.png` for brush subjects; the request `root` is `""` in brush
mode, so brush exports land at the archive top level.

`#slugify` lowercases and replaces non-alphanumerics with `_`; `#safeFolder` strips
separators, reserved characters and control characters and rejects `.`/`..`. Format is
PNG or JPG (`jpg` forces a solid background; a solid background disables transparent), the
colour comes from a colour input, and feather is the shared `featherRadiusAtom`. On export
it encodes every selected mask with `encodeMaskPng` (the brush lookup is
`objectMasks[oid] ?? masks[oid]`), posts them, converts the returned base64 into a Blob,
triggers a download and shows "Downloaded!" for two seconds. Selection is filtered to items
that actually have a mask.

### `ConsolePanel.tsx`

`MAX_LOG_ENTRIES=500`, `MIN_UI_ENTRIES=150`, `MAX_MESSAGE=400`, `RENDER_LIMIT=120`.
`LEVEL_RANK` maps `DEBUG`/`DOWNLOAD` to -1, `INFO` 0, `WARNING` 1, `ERROR` 2,
`CRITICAL` 3. `FILTER_OPTIONS` is the level ladder; `FILTER_COMBOS` is the six presets
(Everything, App only, Server only, Warnings & errors, Errors only, Debug & above) that set
a level **and** a source together.

`#trim` keeps the last 150 UI entries, fills the remaining slots with the newest other
entries and re-sorts by the numeric `seq`, so ordering stays chronological across the
merge. `#mergeEntries` deduplicates by a stable key before trimming: UI entries use their
local id, server entries use `rid|sid|ts|level|message`. That is what removes the same
record when it arrives both from live SSE and from the `/api/logs` backfill. It also
folds successful request entries and Brush stroke entries into stable operation groups
(`Brush updates`, `Segmentation`, `Detection`, `Matte updates`, `Brush strokes applied`)
with an `xN` count. `normalise` builds the search haystack from message, level, category,
source and `name`, and runs `safeText` on both UI and server messages. Only entries that
arrive before the initial `/api/logs` fetch settles are queued in `pendingRef`; the same
queue is drained if an SSE open triggers the history refresh first. `refreshHistory` is
called on every SSE open so a missed initial fetch is recovered.

Filtering is level rank, source, category (UI categories by name, all server entries under
`backend`) and a lowercase substring search. Download rows are always excluded from the
list and shown as progress bars instead: grouped by `name`, merging fields with the
previous row, removed on a `done` phase or on 100% without a file count, with the phase
chip (`Starting`/`Downloading`/`Verifying`), byte text, file name, ETA and a second bar for
files done/total. Category chips are listed **only for categories with a non-zero count**,
sorted by count. The model section lists downloaded/loaded state. Copy writes the filtered
rows plus a `[downloads]` block through the clipboard API when the context is secure,
falling back to a hidden textarea + `execCommand`. Unread count, error/warning badge and
the "Latest" jump button are all local state; stick-to-bottom is `pinnedRef` with a 30 px
threshold and a hover check.

### The smaller dialogs and panes

| Component | Behaviour |
|---|---|
| `EmptyState.tsx` | The home screen. Whole area is a drop target; the hidden file input accepts `image/jpeg,image/png,image/webp`; the Upload button is disabled until a model is chosen (Brush/Detect), `sam3Ready` is true (SAM 3), or Matte is selected (Matte loads no segmentation model). A drop is ignored unless the active mode is ready and the file has an `image/` MIME type. The four mode buttons set `modeSwitchTargetAtom`, which is what opens `ModeSwitchDialog`. The caption switches between "Choose a mode...", "Loading SAM 3 model...", the SAM 3 prompt hint, the matte hint, "Release to upload" while dragging, and "Drop an image here or click Upload". `uploadHoveredAtom` is set only while no mode/model is chosen, and paints a ring in `Sam3Page`/`App`. |
| `ModeDialog.tsx` | The Brush/Detect chooser shown after the first upload. Picking one writes `modeLockAtom`, closes itself, and sets `detectModeAtom` plus `toolModeAtom` (brush forces `positive`). There is no close button and the backdrop has no click handler, so one of the two must be picked. |
| `ModeSwitchDialog.tsx` | Confirms a Brush/Detect <-> SAM 3 switch. If a session exists it awaits `endSession()` first, then sets `sam3ModeAtom` from the target; the footer is the `accent` tone with a busy state, and Cancel is ignored while busy. |
| `EndSessionDialog.tsx` | Confirms teardown; awaits `endSession()` and closes in `finally`. `danger` footer tone, Cancel disabled while busy. Adds a `Session is still running` warning when `useSessionBusy` reports a task. |
| `ReplaceImageDialog.tsx` | Confirms replacing the image in an active session. Always warns that masks, subjects, mattes and undo history are discarded; when `useSessionBusy` reports a running upload, matte, detection, detector load or batch segmentation it names the task and disables Replace until it finishes. Cancel stays available. |
| `UnsupportedFileDialog.tsx` | Reports the rejected file name and extension and lists the formats built from `SUPPORTED_IMAGE_EXTENSIONS`. Dismissed by the backdrop, the X, or "Got it". |
| `SettingsDialog.tsx` | Shortcut list only. What it shows depends on `hasImageAtom` and `modeLockAtom`: no image -> "Open an image to see shortcuts", brush -> Esc/Del/Ctrl+Z/Ctrl+Y/Scroll/Shift+Drag/Right click, detect -> Esc/Enter/Del/Ctrl+Z/Ctrl+Y/Scroll/Shift+Drag. Lazy-loaded by `App`. |
| `EncodingOverlay.tsx` | Driven by `imageEncodingAtom` plus `encodingMessageAtom`. `SWEEP_MS = 850`, `BACKDROP_FADE_MS = 350`. While encoding it blocks input with a spinner and shimmer bar; when encoding ends it plays one screen-sweep, then holds `blocking` for another 100 ms before releasing pointer events. Title falls back to "Encoding image...". |
| `Toast.tsx` | `ToastHost` renders `toastAtom`; error toasts are red with a warning icon, info toasts orange. `role="status"`, bottom-centre, `pointer-events-none` container with interactive toasts. |
| `ui/Button.tsx` | `variant: default \| ghost \| accent`, `size: sm` only, disabled styling, passes the rest through to `<button>`. |
| `ui/Select.tsx` | Custom listbox (not a native `<select>`): `value`, `onChange`, `options` (`value`, `label`, `icon`, `hint`, `section`, `disabled`, `loaded`), `position: auto \| up \| down`, `disabled`. Auto mode flips up when the button plus 200 px would overflow the viewport. Arrow keys open and move focus, Enter/Space select, Escape closes. The current value is rendered as a disabled row, sections render a heading, and a `loaded` option gets a green "Loaded" chip. Closes on outside `mousedown`. |
| `ui/Tooltip.tsx` | Wraps a child and renders a fixed-position tip on hover or focus, `side: top \| bottom`, measures itself to clamp the x position into the viewport. Cancels its show timer on unmount. |
| `ui/TierIcon.tsx` | Maps `tiny` to a yellow bolt, `small` to a green feather, `large` to a purple server and anything else (including `medium`) to a blue chip. Used in the detector list. |
| `ui/BoxesToggle.tsx` | One chip labelled "Boxes" or "Boxes off"; orange when boxes are hidden. Used by the detect toolbar and the SAM 3 bar in transparent mode. |
| `ui/DialogFooter.tsx` | Shared Cancel + confirm pair with a busy spinner and a `danger`/`accent` gradient. Both buttons are disabled while `busy`; an optional `disabled` disables only the confirm button. |
| `ui/NumberInput.tsx` | Controlled numeric field used beside sliders. Keeps a local draft while focused, clamps to min/max on blur or Enter, reverts on Escape. |

---

## 19. Frontend: configuration

`vite.config.ts`: alias `@` -> `src`, `dedupe` React, dev server on port 3000 with
`strictPort`, `host: "localhost"` (both launchers override it with `--host 127.0.0.1`),
`fs.allow` covering the frontend directory, `frontend`, `backend/web` and
`backend_amd_gpu/web` (the junction targets), and headers: a CSP whose `connect-src`
allows `127.0.0.1:8000`, `localhost:8000` and the 3000 websocket, `X-Frame-Options: DENY`,
`nosniff`, `Referrer-Policy: no-referrer` and a `Permissions-Policy` disabling camera,
microphone, geolocation and USB. `preview` repeats them. Build: `dist`, no sourcemaps,
ES2020 targets, `react-vendor` and `jotai` chunks.

`uno.config.ts`: `presetUno`, the `transition-interactive` shortcut, Inter/JetBrains Mono
families and the orange scale `400 #d47534`, `500 #b85c2a`, `600 #96461d`, `950 #221005`.

`tsconfig.json`: ES2020, bundler resolution, `strict`, `noUnusedLocals`,
`noUnusedParameters`, `noFallthroughCasesInSwitch`, `noEmit`, and the `@/*` path.

`index.html` carries the splash element and the base colours; `main.tsx` imports
`virtual:uno.css`, `globals.css`, calls `initGrain()` and renders `App` in
`React.StrictMode`. `initGrain` paints a 256x256 random-noise PNG into `--grain-url`, which
`grain-bg` uses as a background layer.

`use_web.bat <WebDir>` creates junctions at `frontend/node_modules` and `frontend/dist`
pointing at the fork's `web` directory, hard-links `web/package.json` to the shared
manifest, and copies a root `frontend/bun.lock` into the web directory only when the web
copy is missing. `use_web.bat` never deletes the root lockfile. It replaces a real (non-reparse-point)
`node_modules` without asking, and resolves an existing junction that points elsewhere by
deleting and recreating it.

`bun_env.cmd` picks `BUN_DIR_OVERRIDE` or `tools/bun`, asks `bun_fetch.ps1` to populate it,
and fails with instructions when no executable exists. It sets `DO_NOT_TRACK=1`,
`BUN_INSTALL_CACHE_DIR`, `BUN_RUNTIME_TRANSPILER_CACHE_PATH` and `TMPDIR` inside that
directory and prepends it to `PATH`.

---

## 20. Reference entries

These entries record behaviour whose reason is not obvious from the call site. Each
was checked against the current code.

| Reference | What it explains |
|---|---|
| `main.py#run_prediction` | 404 only for a missing session, 400 for every other `ValueError`; the client depends on the split |
| `manager.py#ModelManager.create_session` | The `None` slot under the lock; the `type: ignore` marks it |
| `manager.py#ModelManager._object_masks_png` | Snapshot under the lock, encode outside it |
| `manager.py#ModelManager._restore_object_mask` | One decoder pass for the stroke that is now last; full replay when snapshots are misaligned |
| `grounding_detector.py#_normalize_auto_map` | A repo-prefixed `auto_map` value is trimmed to the module path |
| `grounding_detector.py#_florence_config_paths` | Snapshot paths, not `hf_hub_download`, whose resolution differs across hub versions |
| `grounding_detector.py#_florence_language_generation_config` | Inner config is the base because it carries the special-token ids |
| `grounding_detector.py#_enable_florence_generation` | Re-basing onto `GenerationMixin` for `transformers >= 4.50` |
| `compositing.py#_is_unit_scale` | Range test that ignores non-finite values; NaN/inf folding happens in `_to_mask_f32` |
| `compositing.py#_to_mask_u8` | A boolean mask skips the float plane |
| `device.py#_detect_nvidia` | `torch.version.cuda` is `None` on ROCm, so an AMD card is never reported as NVIDIA |
| `model_integrity.py#ensure_model_file` | Path when present, download when missing, directory named when offline |
| `offline.py#_apply` | The module globals are patched because the libraries read them, not the environment |
| `offline.py#offline_guard` | Cached loads run forced-offline, downloads run permissive, state restored in `finally` |
| `amd_gfx.py` | The three gfx families and the `None` fallback; importable before torch |
| `detect_amd_gpu.py` | `# noqa: E402` on the import after the `sys.path` bootstrap |
| `Header.tsx` model stream | The status stream drives the header only; residency is logged by the backend |
| `useDetection.ts#detect` | Masks are trimmed to the old length so keys stay aligned |
| `useSession.ts#enqueue` | One FIFO chain per session id |
| `useSession.ts#syncSubjects` | Deleted ids are skipped |
| `api.ts#resetObjectMaskCache` | Cleared on session/image change |
| `api.ts#decodeObjectMasks` | Byte-identical base64 is reused; changed masks decode in parallel |
| `api.ts#resetToken` | Both the token and the in-flight promise are cleared |
| `api.ts#releaseSession` | A stale token is refreshed and the DELETE retried once |
| `lib/detectionEdit.ts` | The three shared reindex helpers |
| `lib/utils.ts#decodeImage` | One decode returns the image and the object URL; failure is `null` |
| `store/session.ts#SUBJECT_COLORS` | Ten muted hues, 24.7 degrees minimum gap |
| `store/session.ts#subjectColor` | Stable colour per id, repeats shifted in lightness |
| `usePanelDismiss.ts` | `ignoredRef` stops the toggle button from reopening the panel |
| `ui/BoxesToggle.tsx`, `ui/DialogFooter.tsx` | The "Boxes" chip and the shared cancel/confirm pair |
| `useCanvasRenderer.ts#SUBJECT_LAYER_BUDGET_BYTES` | 192 MB, evicting the least recently used subject |
| `useCanvasRenderer.ts#SubjectLayer` | Full-resolution tinted layer; `mask` compared by identity |
| `useCanvasRenderer.ts#noteMaskScale` | A mismatched mask is stretched and reported once per size pair |
| `useCanvasRenderer.ts#subjectIds`, `#subjectMaskKey`, `#ensureSubjectLayer` | Ascending ids, one key format for both sides, cache-or-rebuild |

---

## 21. Limitations that are visible in the code

These are behaviours, not opinions; each one is reachable today.

1. `/detect` and `/segment-batch` return **400** for a missing session, while `/predict`
   returns 404. A client that "recovers on 404" will not recover from those two.
2. `POST /api/sessions` maps any `RuntimeError` to **429**, including a failed download or
   an integrity failure.
3. `load_detector` unloads the current detector before validating the new name; an unknown
   name leaves `_detector` registered but unloaded.
4. A failed encode during upload is logged and swallowed, so the client sees 200 and the
   next stroke fails with 400 "No image loaded".
5. The frontend never sends `confidence`; Grounding DINO therefore always runs at the
   transformers default (0.25 with `transformers 4.57.6`) and YOLOE at its own 0.05 default.
   There is no confidence control in the UI.
6. `export-zip` applies `format` to every layer and rewrites extensions to match. JPEG
   uses white when no background colour is supplied.
7. Export memory is unbounded: the ZIP is built in memory and base64-encoded into JSON.
8. Session memory is unbounded; `MAX_SESSIONS` limits the count, not the bytes.
9. `model_integrity` caches digests in `model_weights/.hash_cache.json`, keyed by size and
   `mtime_ns`; a file replaced with the same size and timestamp would skip rehashing.
10. `start.bat`/`stop.bat` match any `main:app`, so they also stop the AMD backend;
    `stop_amd.bat` matches only the AMD one.
11. `detect_amd_gpu.py` reports 0 MB for Strix Point, Strix Halo, Krackan, Phoenix,
    Van Gogh, Rembrandt, Mendocino and every Instinct, and `setup_memory.ps1` then uses a
    fallback budget without warning.
12. `get_or_create_token()` mints a new token per call; safe only because `lifespan` calls
    it once.
13. `SPECTRA_VRAM_MODE` and `SPECTRA_CPU_THREADS` are written and echoed only;
    `SPECTRA_CPU_RAM_MB` is enforced by the backend at startup.
14. `cudnn_disabled` serialises every cuDNN-disabled inference process-wide, and
    `cpu_threads` saves/restores a process-wide torch setting non-atomically.
15. The stderr and `torch.load` swaps use a per-module lock in the two SAM backends and no
    lock in the detectors, so concurrent loads can interleave them.
16. `_BufferHandler` drops a record emitted by a second thread during another thread's
    emit; the file handler still gets it.
17. SAM 3 undo/redo answer 404 for both "no session" and "empty history".
18. Removing a SAM 3 instance shifts every later `prompt_index`; the client's cached
    indices go stale silently.
19. The frontend has no `AbortController`; a hung request leaves a spinner latched, and
    `imageEncodingAtom` drives a full-screen overlay with no escape.
20. `SPECTRA_MAX_HISTORY` does not change the cost of an undo, only how deep it goes.
21. `torch` is not declared in `requirements.txt` even though the module catches
    `torch.OutOfMemoryError` at import time.
22. `sam3.pt` comes from an unvetted community mirror whose digest cannot be confirmed;
    `pin` changes there break every install.
23. Nothing except Python is pinned: AMD torch comes from a nightly index and the CLIP fork
    from a mutable `main` branch with no checksum.


## 22. Matte mode

`backend/models/matte_backend.py` (both forks, byte-identical)

- `#MATTE_MODELS` - registry keyed by the API model name. Each `MatteModelSpec` carries
  the HF repo id, a pinned commit `revision`, input size, tier/perf text and checkpoint
  download size. Matte has no VRAM compatibility gate: every model stays selectable,
  and heavy entries are labelled `Heavy`. The HR checkpoints are 444,473,596 bytes
  (424 MB) each, verified against the pinned files; an actual OOM is reported through
  the normal error path. The shipped entries are Lite, Lite-matting, General,
  Dynamic, Dynamic-matting, Matting, Portrait, HR, HR-matting and 512 (all MIT). Only
  Lite, General and Matting have measured timings.
- `#BiRefNetBackend#load_model` - `AutoModelForImageSegmentation.from_pretrained` with
  `revision` and `trust_remote_code=True`, caching under `model_weights/hf_cache/hub`.
  The pinned revision is what stops remote code from changing under an install.
- `#BiRefNetBackend#matte` - always a full-image forward. With a ROI it calls
  `_roi_component_mask`; `_crop_matte` is the fallback when no component overlaps.
  Cropping to a tight ROI was tried and returned empty alpha inside uniform objects.
- `#_roi_component_mask` - connected components of `alpha > 0.5`; keeps every label
  intersecting a ROI box, dilates 9 px, then Gaussian-blurs so the soft edge survives.
- `#subtract_negative_strokes` - rasterises negative labels as anti-aliased lines and
  multiplies `alpha * (1 - soft_mask)`.
- `#apply_alpha_controls` - bias (`edge_shift`), invert, Gaussian feather.
- `#alpha_to_png_b64` - uint8 grayscale PNG; used for the response and the cache.

`backend/main.py` (both forks)

- `#MatteRequest` - validators mirror `StrokeRequest` bounds; `model` must exist in
  `MATTE_MODELS`; `max_detections` is capped at 4 and `crop_margin` at 0.5.
- `#run_matte` - `asyncio.to_thread` around `manager.run_matte`; 404 only when the
  session is gone, 400 for guidance failures, 507 for VRAM.
- `#export_matte` - `composite_and_encode` with `feather_radius=0`; JPG without a colour
  gets a white background because JPEG has no alpha channel.

Frontend

- `lib/matte.ts#decodeAlphaPng` - draws the grayscale PNG, rewrites RGB to 255 and moves
  luminance into the alpha channel, which is what `destination-in` consumes.
  `applyAlphaControls` applies Edge/Invert with a 256-entry LUT, caches the shifted
  canvas per layer, and applies Feather with `ctx.filter` (three-pass box blur if the
  browser lacks it).
- `store/matte.ts` - separate from the mask atoms on purpose: `PackedMask` is binary, so
  a soft alpha cannot be represented there. Subjects, the active subject, guide/strokes,
  history/redo, local edge controls and the export flag live here.
- `hooks/useMatte.ts` - one request in flight; a newer run bumps `seqRef` so a stale
  decode cannot overwrite a newer alpha. Each Auto/Stroke run stores a compressed-alpha
  snapshot on its layer (12 max); undo/redo swaps the snapshot and re-applies controls.
- `lib/matte.ts#MatteLayer` - each subject owns `rawAlpha`, `strokes`, `guide`, `info`,
  `history` and `redo`.
- `components/MatteBar.tsx` - runs only when the user presses Cut out, and re-runs on
  model change when an alpha already exists. The guide control exposes Auto and Stroke;
  Edge and Feather use `ui/NumberInput` for exact values. Layers, add/remove and
  undo/redo are in the second toolbar row.
- `components/MatteLayersPanel.tsx` - subject picker with per-subject cutout state and
  an eye that controls both preview and export inclusion.
- `components/MatteExportDialog.tsx` - local combined/active/separate downloads of the
  visible subjects. The active option is disabled while that subject is hidden.
- `hooks/useCanvasRenderer.ts` - composites the merged alpha of all visible subjects
  over the checkerboard; hidden subjects contribute nothing.
- `components/Canvas.tsx` - merges every visible subject's `rawAlpha`, applies the
  local Edge/Feather/Invert controls once, and passes the result to the renderer.
