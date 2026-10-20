# Frontend

The browser side of Spectra Segment. This file covers how it is put together, where
state lives, and how a stroke on the screen becomes a mask on the server.

Source is in `frontend\`. The backend side is described in `docs\CODE_GUIDE.md` and the
console panel in `docs\LOGGING.md`.

---

## 1. Stack and how it runs

| Piece | Version | Job |
|---|---|---|
| React | 19.2.8 | The UI |
| Vite | 8.2.1 | Dev server and bundler |
| UnoCSS | 66.7.5 | Utility CSS, atomic output |
| jotai | 2.20.2 | State, using the default module store |
| TypeScript | 7.0.2 | Types, with `strict` and `noUnusedLocals` on |
| Bun | latest stable at install time | Package manager and script runner (not a runtime here) |

Scripts: `dev`, `build`, `preview`, `typecheck`. There is no test runner wired into
`package.json`; changes are checked with `bun x tsc --noEmit`, a build, and by hand
in the browser.

### There is no production serving path

The launchers always run the Vite **dev server** on `127.0.0.1:3000`:

- `start.bat` calls `use_web.bat`, then runs `vite.js --host 127.0.0.1 --port 3000`.
- `start_amd.bat` does the same through Bun.

`bun run build` still works and `install_amd.bat` calls it, but nothing reads `dist`.
The `preview` script is never invoked. So the app is effectively always in dev mode.

### How the workspace is wired

`frontend\use_web.bat` takes the target web directory (`backend\web` or
`backend_amd_gpu\web`) and wires it to the shared frontend. The junctions are
created in `frontend\` and point into that web directory, so Vite resolves them from
the source tree:

| Link | Location | Type | Why |
|---|---|---|---|
| `node_modules` | `frontend\node_modules` -> `<web>\node_modules` | Junction | Each backend's frontend dependencies stay separately cleanable |
| `dist` | `frontend\dist` -> `<web>\dist` | Junction | Build output lands in the right fork |
| `package.json` | `<web>\package.json` -> `frontend\package.json` | Hard link | One shared manifest, so editing either file changes both |

It also copies a stray `frontend\bun.lock` into the web directory and deletes the
frontend copy.

A junction is a Windows directory shortcut. Only one fork can own
`frontend\node_modules` at a time; running the AMD installer after the CPU one
re-points it.

`use_web.bat` also deletes a legacy real `node_modules` if it finds one, without
asking. It takes its target as an argument, so it is only safe because every caller
passes an in-repo path.

### Talking to the backend

`lib\api.ts` hardcodes `BASE = "http://localhost:8000"`. There is no environment
override, no proxy in `vite.config.ts`, and no relative URLs. The browser therefore
makes cross-origin requests to port 8000, which is why the backend's
`CORSMiddleware` allow-lists `localhost:3000` and `127.0.0.1:3000`.

`vite.config.ts` sets a Content-Security-Policy for the dev server. `connect-src`
allows both `127.0.0.1:8000` and `localhost:8000` so the API and the log stream
work. `server.fs.allow` lists `import.meta.dirname`, `../frontend`,
`../backend/web` and `../backend_amd_gpu\web` - that last part is what lets Vite
serve through the junctions. The repo root is deliberately excluded.

`server.host: "localhost"` in the config is dead. Both launchers override it with
`--host 127.0.0.1`, so the page origin is always `http://127.0.0.1:3000`.

---

## 2. State

Five stores, all plain `atom()` on jotai's default store. There is no custom
`Provider` anywhere, so every `useAtom` call in every component shares one store.

| Store | Holds |
|---|---|
| `session.ts` | Session id, model, image, masks, subject identity, history ledger |
| `ui.ts` | Dialog and overlay flags, console state, toasts, imperative callbacks |
| `detection.ts` | Detection mode, brush and feather values, detector state, the detections array |
| `layers.ts` | The layer list and the layer selection set |
| `sam3.ts` | SAM3 mode, prompt history, redo stack, instances, settings |

### What persists

Four atoms use `atomWithStorage`, so they survive a reload in `localStorage`:
`consoleFilterAtom`, `consoleSourceAtom`, `consoleCategoryAtom` and
`consoleShowTimeAtom`. Everything else is in-memory only, so a reload loses the
image, the masks and the session.

### The three mask atoms

These are easy to confuse:

| Atom | Keyed by | Meaning |
|---|---|---|
| `masksAtom` | Position in a list | In brush mode it is derived from `objectMasksAtom` - every subject's mask, in id order; in SAM3 mode it is every instance |
| `objectMasksAtom` | Subject id | Per-subject masks. This is what the per-subject colour overlay is built from |
| `perDetectionMasksAtom` | Detection index | One mask per detection in detection mode |

`masksAtom` carries three meanings at once: the brush subject list, the SAM3 instance
list, and "is there anything to export or show transparently". That overloading is why
the export dialog has to work out which mode it is in before it can find anything.

In brush mode, `applyPredict` in `useSession.ts` sets `masksAtom` to
`Object.values(objectMasks)` rather than reading a mask list off the response: a brush
response carries each subject's mask exactly once, in `object_masks` keyed by object
id, so there is no second positional copy to transfer or decode.

---

## 3. Subject identity

A subject is a maskable object the user carves out with the brush. In brush mode
several subjects can coexist, each keeping its own strokes and its own layer on
export.

| Atom | Holds |
|---|---|
| `brushObjectsAtom` | The ordered list of subject ids that exist |
| `activeObjectIdAtom` | Which one the brush is currently drawing on |
| `subjectMetaAtom` | `{ id, name, color }` per subject |
| `objectHistoryAtom` | The server's per-subject undo/redo depth |

### Ids only ever go up

`Toolbar.tsx`'s `addSubject` computes `Math.max(-1, ...brushObjects) + 1`. Ids are
never reused. That matters because `clear_object` on the server deletes the
`ObjectState` for an id - if ids were recycled, a re-added subject would inherit
the deleted one's stroke history.

### The server owns the history

`objectHistoryAtom` is filled from the `object_history` field that every mutating
endpoint now returns: `undo`, `redo`, `strokes`, `has_mask` per subject. The client
adopts it verbatim through `applyHistory` instead of keeping its own counter.

The client used to increment a local counter on each successful predict. The two
drifted whenever a subject was deleted mid-flight, a session was recovered with
fresh server state, or an id was reused. When the local counter said "you can undo"
but the server had no strokes, `/undo` returned 404. The client then retried the
same failing request in a loop.

`useSession.ts`'s `undo` and `redo` publish to `lib/historyQueue.ts` instead of
calling the endpoint straight away. The queue accumulates **per subject and kind**,
so five fast clicks on one subject become one `POST /undo?steps=5` - the backend
clamps `steps` to the depth that subject actually has. A drain runs as a task on the
session's `sessionQueueMap` chain, so undos can neither overtake an in-flight
prediction nor race each other, and bursts cannot produce out-of-order responses.

The ledger the guard reads, the in-flight stroke counter and the burst queue live at
module scope in `useSession.ts`, next to `sessionQueueMap`, not in a `useRef` inside
the hook. `useSession()` is called by several components: `Canvas` runs the
predictions, while `Toolbar`, `SubjectsPanel` and `App`'s Ctrl+Z handler own the
buttons. A per-hook copy of the ledger is only written by the instance that saw a
response, so the button owners' copies stay empty; their guard then reads depth 0,
drops every click, and no request is ever sent. The atoms are global for the same
reason - the refs have to match their scope.

The guard also counts strokes that are still in flight: `availableSteps` adds them to
the subject's undo depth and subtracts them from its redo depth. A stroke counted as
pending was already enqueued before the click, so it lands in the history first; it
also clears the redo stack, so a queued redo would ask for an entry the server no
longer has. When the drain runs, each subject's accumulated steps are clamped to the
depth it has at that moment, so a burst that outruns the history asks for exactly as
many steps as exist instead of earning a 409.

A drain applies every subject's undo clicks before any subject's redo clicks, so a
burst that mixes both resolves undo-first rather than in click order. That only differs
from click order when one subject has entries in both stacks at once.

Status handling on those calls: a 404 means the session is gone, so it gets rebuilt.
A 409 now only means the subject genuinely has nothing left; that subject's counter is
patched (`patchHistory`) and the other subjects are left untouched. It used to call
`applyHistory({ [oid]: ... })`, which replaces the whole ledger - zeroing every other
subject's undo/redo depth and disabling their buttons.

`predict` follows the same rule and recovers on **404 only**. A 400 is a rejection
with a reason, so the toast shows the backend's own message - "Maximum objects per
session (50) reached" - instead of silently rebuilding the session. It used to treat
both the same, which meant a stroke refused by the object cap destroyed the live
session and every mask in it.

Recovering rebuilds the session, and a replacement session starts empty, so
`recoverSession` drops the brush state that the old session owned (masks, history,
subject metadata, and any queued undo/redo, plus the brush layers) and logs it. The
overlay used to stay on screen after a recovery, showing masks the server could no
longer undo. Detection masks are computed on the client, so they survive.

### Colour is the identity

`subjectColor(id)` picks a stable palette entry from the subject id. That one colour
appears in:

- the subjects panel swatch,
- the toolbar subject chip,
- the brush preview for a positive stroke,
- the subject's mask overlay on the canvas.

So "which mask is which" has a single consistent answer everywhere. The active
subject is always drawn last in the overlay so it sits on top of the others, and it is
composited at a higher alpha (56% against 42%) so the one you are editing stands out.

The ten palette entries sit at roughly 36-degree hue steps, and `shiftLightness` moves a
repeat of the cycle up or down in lightness. The palette this replaced was not
hue-separated - its closest pair was only 7.2 degrees apart - so with eight or more subjects
two of them read as the same colour even once the overlay tinted each layer properly.

---

## 4. The canvas

Four `<canvas>` elements stacked on top of each other in `Canvas.tsx`:

| Canvas | Draws |
|---|---|
| `imageCanvas` | The photo, or a checkerboard in transparent mode |
| `maskCanvas` | Mask overlays |
| `detectCanvas` | Detection boxes and SAM3 instance boxes |
| `drawCanvas` | The brush preview and the processing ring |

### Coordinate space

All drawing happens in **CSS pixels**. `computeState` returns:

```
baseScale = min(containerWidth / imageWidth, containerHeight / imageHeight, 1)
scale     = baseScale * zoom
offsetX   = floor((containerWidth  - floor(imageWidth  * scale)) / 2) + panX
offsetY   = floor((containerHeight - floor(imageHeight * scale)) / 2) + panY
```

Screen to image is `(point - offset) / scale`, image to screen is
`point * scale + offset`. The wheel handler reproduces the same `floor` rounding so
the cursor stays anchored on the pixel under it. `zoomRef` is clamped to 0.1-50.

### HiDPI support

`redraw` reads `window.devicePixelRatio` and sizes each backing store as
`clientWidth * dpr` x `clientHeight * dpr`, then calls
`ctx.setTransform(dpr, 0, 0, dpr, 0, 0)` so every drawing call can keep using CSS
pixels. Without this the browser upscaled a 1:1 backing store, which made masks,
boxes and brush previews visibly soft on any high-density display.

The three render functions (`renderMasks`, `renderDetections`, `renderSam3Boxes`)
each set the same transform on their own context, because assigning `canvas.width`
resets it.

### The per-subject mask overlay

`rebuildSubjectMaskCache` builds one overlay for all subjects, tinting each subject's
own layer before compositing. For each subject it:

1. renders that subject's mask onto one **reused scratch canvas** as white,
2. feathers it if the feather radius is above zero,
3. applies `globalCompositeOperation = "source-in"` to *that layer* so only its own
   pixels take the subject's colour,
4. composites the tinted layer onto the accumulator at 42% alpha, or 56% when it is
   the active subject.

Steps 1-3 run only when that subject actually changed. `ensureSubjectLayer` caches
each subject's finished, full-resolution tinted layer and keys it on the `PackedMask`
**object identity** plus the image, feather radius and colour. Identity is exact here:
`decodeObjectMasks` in `lib/api.ts` hands back the same `PackedMask` object while a
subject's encoded mask is unchanged, so the renderer can tell "this subject did not
move" for free. A stroke or an undo on one subject therefore re-tints one subject and
re-composites the cached layers for the rest, and changing the active subject re-tints
nothing at all - only the alpha used while compositing changes.

The cache is bounded at 192 MB by dropping the least recently used subject first;
past the budget a subject is rebuilt every time, which is what it did before. Layers
are also dropped for subjects that disappeared or when the image is replaced.

The finished tinted layer is drawn into a canvas of its own, not onto the shared
scratch canvas. The scratch canvas is reused for the next subject, so it can never be
the object the cache holds.

Step 3 has to happen on the layer, and this is the whole point of the structure.
Tinting the accumulator instead keeps every pixel already composited but replaces its
colour with the fill, and `source-in` also multiplies alpha by 0.42 per pass. With
several subjects that left every mask in one colour at `0.42^N` alpha - faint enough
that subjects looked like they had vanished from the canvas while the panel still
listed them.

Subjects are drawn back to front with the active subject last. If there are no
per-subject masks it falls back to `rebuildMaskCache`, which builds a single
orange union overlay.

Transparent view is a separate route: `rebuildTransparentComposite` draws the image
and then punches holes in it with `globalCompositeOperation = "destination-in"`.

### Feathering and blur

`blurMaskCanvas` pads the mask by the radius before blurring, so the image border
does not darken. It then either uses `ctx.filter = "blur(Npx)"` or falls back to a
three-pass separable box blur implemented in `boxBlurFloat`.

The fallback works on **interleaved RGBA float** data. An older version allocated
a single-channel destination of `w * h` floats and was handed a `w * h * 4` RGBA
plane. So the horizontal pass only saw the first quarter of the image. The
un-premultiply loop then read out of range, which produced `NaN` and turned most of
the feathered edge fully transparent. It only shows up where `ctx.filter` is
unsupported, but it was badly wrong.

### Caching and throttling

The key is `subjectMaskKey(ids, activeObjectId, featherRadius, maskRevision)` - one
helper used by both the writer and the comparer. They used to disagree (the compare
side appended a revision the stored key did not have), so the key could never match
and the overlay was rebuilt on **every** redraw: a full-size canvas allocation per
subject, roughly eleven times a second, for as long as the editor was open. That is
what made panning and redrawing feel like they ate the machine.

`maybeRebuild` runs the rebuild immediately if 80 ms have passed since the last one.
Otherwise it defers by 90 ms and schedules a follow-up redraw. So during the defer
window the previous overlay is still on screen. Deferred jobs are held in a map keyed
by job name - `subject`, `masks`, `transparent` - which they used to share through a
single slot: a deferred subject rebuild was replaced by a mask-cache rebuild and then
simply never ran.

The flat caches (`rebuildMaskCache`, `rebuildTransparentComposite`) used to key on the
masks' widths alone, so a mask that changed content without changing size was served
stale forever. They key on the feather radius and the mask revision now.

`Canvas.tsx`'s `invalidateCache` empties every cache key whenever the mask arrays,
detections, feather radius or SAM3 instances change identity. It deliberately does
**not** drop the cached canvases: an emptied key still mismatches, so the rebuild
happens, but the frame in between redraws the previous overlay rather than a blank
one.

`scheduleRedraw` is `requestAnimationFrame`-coalesced, so pointer moves do not each
trigger a redraw.

If every mask in a rebuild turns out to be unrenderable, the composite is cached under
an **empty** key instead of a normal one. That key can never match, so the next frame
tries again - caching a real key there would leave a blank overlay on screen until
something else happened to invalidate it.

---

## 5. Brush to prompt

A stroke is captured in image space. `useBrushDrawing.ts` holds the shared refs and
provides `canvasToImage`, `isInsideImage` and `drawBrushPreview`. The pointer
handlers and the actual capture live in `Canvas.tsx`'s `handlePointerMove`, which
pushes into `strokePointsRef` with a minimum spacing of 2 px
(`STROKE_COLLECT_SPACING`) and clamps to the image bounds. Right-drag sets
`activeStrokeModeRef` to `"negative"` and the preview turns red.

`handlePointerUp` is wired to both `onPointerUp` and `onPointerLeave`. A left drag
has pointer capture, so leaving the canvas keeps collecting points. A right drag
does not, so it commits mid-gesture. That asymmetry is still present.

`buildStrokePrompt` then turns the raw points into the `points` and `labels` the API
expects:

| Case | What it produces |
|---|---|
| Fewer than 3 points | One point, label `+1` or `0` |
| Negative stroke | Always an area prompt |
| Positive and closed (gap within `max(brushSize * 1.5, 4)` or 12% of the diagonal, and at least 4 points) | A ring prompt |
| Positive and open | An area prompt |

`buildRingPrompt` builds a donut: it downsamples the ring to 32 points, works out a
centroid and mean radius, and emits up to 8 interior positive points plus the ring
itself as negative. The interior points are pulled in by
`clamp(brushSize / avgRadius, 0.15, 0.5)` of the radius.

`buildAreaPrompt` rasterises the stroke into an offscreen canvas at most 1024 px
wide. The line width is deliberately **narrowed by 60% of the brush on each side**,
so the sampled points sit inside the object instead of straddling the boundary. It
then samples on a grid of at least 4 image px, transforms back to image
coordinates, and caps the result at 12 points.

All coordinates are clamped to the image, so the server never receives anything out
of range.

---

## 6. Detection mode

### Detections accumulate

`useDetection.ts`'s `detect` **appends** to `detections` instead of replacing them.
Before, every query called `setDetections([])` and `setPerDetectionMasks({})`, so
searching for a second object made the first one silently vanish from the canvas and
from the results.

New detections land at the end of the array, which keeps the mask keys stable
because `perDetectionMasksAtom` is keyed by array position.

`handleDetect` picks between two layer helpers: `syncDetectionLayers` for the first
batch (offset zero) and `appendDetectionLayers(dets, offset)` afterwards, where the
offset is the number of detections that already existed.

### Why positions must line up

`perDetectionMasksAtom` is a `Record<number, PackedMask>` keyed by detection index.
Removing a detection in the middle requires reindexing four things:
`perDetectionMasks`, `detections`, `layers[].detectionIndex` and
`selectedDetection`.

That reindexing used to be implemented **twice** - once in `Canvas.tsx`'s
`removeDetectionAt` and once in `useLayers.ts`'s `removeLayer`. It now lives in one
shared module, `lib\detectionEdit.ts`, which exports `removeDetectionMask`,
`removeDetectionIndexFromLayers` and `shiftSelectedDetection`; both call sites use
it, so the two paths cannot drift.

The backend also has to cooperate: `segment-batch` returns one entry per requested
box with `null` for an empty mask. It used to drop the empty entries, which shifted
every later mask onto the wrong box.

### Layer ids

`addLayer` takes ids from the shared `nextLayerId()` in `store/layers.ts`. That
generator scans the ids already in the store, advances a module-level counter past
the highest one, and returns `layer-<n>`. Deriving the id from a value captured
once (rather than from the store) made every `addLayer` call inside one synchronous
loop produce the **same** id. With 20 detections that meant 20 layers with one name,
`selectedLayers` (a `Set<string>`) addressed them all at once, and
`removeLayer(id)` deleted them all at once.

---

## 7. SAM3 mode

SAM3 does detection, segmentation and masking in one step from a text prompt.

`useSam3.ts`'s `prompt` POSTs the text, appends `{text, masks, scores, bboxes}` to
`sam3PromptsAtom`, and clears the redo stack.

`syncFromPrompts` is the single reconciliation point. It flattens
`prompts[].masks` into `masksAtom` and builds `sam3InstancesAtom` as
`{promptIndex, instanceIndex, text, bbox, score}`, then calls `reconcileLayers`.

`reconcileLayers` reuses existing layers by the composite key
`${promptIndex}:${instanceIndex}`, and takes new ids from the shared
`nextLayerId()` in `store/layers.ts` - the same generator `addLayer` uses.

Undo and redo are optimistic: the local stack is changed first and restored if the
server call fails. Because `api.sam3Undo` and `api.sam3Redo` discard the masks the
server returns, the **local** prompt stack is the authority for the mask pixels.
After a `recoverSession` - which re-creates an empty server history - the two
diverge, and a later redo can restore a prompt the server no longer has.

Removing an instance drops any prompt whose mask list becomes empty. The server does
the same, which means every later `prompt_index` shifts down. The client's cached
indices go stale with no way to detect it.

---

## 8. Session lifecycle

`useSession` is instantiated in **eight** components: App, Canvas, EmptyState,
Header, ModeSwitchDialog, EndSessionDialog, SubjectsPanel and Toolbar. Anything
stored in a per-instance `useRef` is private to that one component, so the guards
are module-level singletons instead:

| Guard | Purpose |
|---|---|
| `uploadGuardRef` | One upload at a time |
| `endGuardRef` | One session teardown at a time |
| `blobUrlRef` | The single live object URL, revoked exactly once |
| `sessionQueueMap` | One promise chain per session id, serialising every session mutation (predict, undo/redo bursts, clear-object) |

The undo/redo ledger, the in-flight stroke counter, the burst queue and the drain flag
are module-level for the same reason (section 3): `Canvas` writes them through
`predict`, while the buttons and the keyboard handler read them.

The blob URL guard fixed a real leak. With a per-instance ref, the components that
ended a session never revoked the URL they did not create. So every mode switch and
every "End Session" leaked an image descriptor.

`recoverSession` health-checks the current session and, if it is gone, creates a new
one and re-uploads the file. The replacement starts with an empty server history, so
recovery runs `resetUndoRedoState` (subjects, masks, ledger, pending-stroke counts and
any queued undo/redo) and drops the brush layers. Detection masks are computed on the
client, so detection masks and layers survive.

`resetUndoRedoState` also runs on every upload, model switch and end session. Clearing
the queue there matters: an op still queued for the session that is going away must not
be replayed against the next one.

---

## 9. The API client

`lib\api.ts` wraps every endpoint. Notable behaviour:

- **Token.** `getToken()` memoises in `_token` and `_tokenPromise`. A 401 clears both
  and retries exactly once with a fresh token. The token never touches disk.
- **`ApiError`** carries the HTTP `status`, so callers can distinguish "the session
  is gone" (404, rebuild it) from "this subject has no strokes" (409, resync and
  stop). The old code threw a plain `Error` and every failure looked the same.
- **`sendBeacon` release.** `releaseSession` fires `navigator.sendBeacon` (which
  cannot set headers, so the backend exempts that path from auth) **and then** a
  token-bearing `DELETE`. A page teardown therefore sends two destroy requests. Both
  are idempotent.
- **Object mask decoding is cached by response string.** `decodeObjectMasks` keeps
  one entry per subject id holding the last base64 it decoded. A response that
  repeats a subject's mask byte-for-byte reuses the existing `PackedMask` instead of
  running the PNG decode again, so a stroke or an undo on one subject only decodes
  that subject. Changed masks decode in parallel (`Promise.all`), so a response that
  carries several changed subjects waits for one decode, not for their sum.
  `resetObjectMaskCache()` clears it on session/image changes.
- **Export encoding.** `encodeMaskPng` loops over every mask pixel on the main
  thread and calls `toDataURL`, then posts base64 to `/export-zip`. For 20 masks at
  4096x4096 that is a multi-second freeze with no progress feedback beyond a spinner.
  `/export-image` exists but is never called by the UI.

---

## 10. The console panel

See `docs\LOGGING.md` for the full logging story, including the backend side, the
phase vocabulary and the structured fields.

Frontend-only pieces:

- The UI log store (`store/logs.ts`). `emitUi(category, level, message, detail)`
  pushes entries into a subscriber set; `logErr` is the error-level shortcut.
  `subscribeLogs` lets `ConsolePanel.tsx` join its own stream to the SSE stream.
  Both then render in one list, with a `source: "ui" | "backend"` discriminator.
  Messages pass through `safeText`, which strips paths, tokens and long hex
  strings before they reach the panel.

---

## 11. Performance notes

| Area | Behaviour |
|---|---|
| Redraw | `requestAnimationFrame`-coalesced via `scheduleRedraw` |
| Mask overlay rebuild | Runs only when its content key changes, and is gated to 80 ms with a 90 ms deferred follow-up keyed per job, so panning triggers no rebuild at all |
| Mask overlay tinting | Each subject's tinted full-resolution layer is cached by mask identity, so a rebuild only re-tints the subjects that changed. Measured with eight 2048x2048 subjects and one changing per update: 74.6 ms full rebuild vs 9.7 ms cached |
| Undo/redo burst | Coalesced per subject into one request carrying `steps` |
| Object mask decode | Reused per subject while the response string repeats |
| Log append | Copies the array and trims to the cap; the two filter memos run over the retained list |
| Log render | Only the newest 120 rows render, with a "show older" button for the rest. Stable ids are used as React keys |
| Log entries | 500 cap, 150 reserved for UI entries |
| Toast retention | Last 3, 4 seconds each |
| SSE reconnect | Backoff 1s doubling to a 30s cap, reset on open. The backend also caps the log stream at 30 minutes (the model-status stream has no lifetime cap) |
| Upload fitting | `fitImageFile` decodes the file once, reads `naturalWidth`/`naturalHeight` from that decode and resizes from it; a file that will not decode is sent unchanged. Probed twice before, so an oversized photo paid for three full decodes |
| Export | Mask encoding is synchronous on the main thread |
| Brush preview | Redraws the whole point list per pointer move, so cost is O(n^2) per stroke with nothing capping the point count |

There is no `AbortController` anywhere. A hung request leaves `isDetectingAtom`,
`isExportingAtom` or `imageEncodingAtom` latched, and `imageEncodingAtom` drives a
full-screen overlay with no escape hatch.

---

## 12. Floating panels

`hooks\usePanelDismiss.ts` closes a floating panel when the user clicks outside it or
presses Escape. It takes an `ignoredRef`, which is the panel's own toggle button:
without it the button's `mousedown` closes the panel and the `click` that follows
reopens it, so the button could never close anything. While the panel is open the hook
listens on `document` for `mousedown` (outside the panel and outside `ignoredRef`) and
for `keydown` with Escape.

---

## 13. Known gaps

- **`Canvas.tsx`'s `handleDetectClick`** catches a failed YOLOE mask decode but not
the `segmentBatch` call it falls back to. If that request throws, the click handler
rejects with no toast, so the detection silently keeps no mask.
- **`sessionId` has no format validation.** The same client error yields 404, 400 or
  a 200 with `{"status": "destroyed"}` depending on the endpoint.
- **`SAM3` settings do not persist**, so the encode quality and keep-loaded toggle
  reset on reload even though the backend keeps them for its lifetime.
- **The export feather slider and the on-screen feather** are the same atom, so
  changing one changes the other.
- **The right-drag brush commits early** when the pointer leaves the canvas, because
  it has no pointer capture.
