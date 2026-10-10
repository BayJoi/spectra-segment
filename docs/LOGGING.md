# Logging

Everything the app says about itself goes to three places. This file explains what each
one gets, how to read a line, and what to look at when something is wrong.

All of it comes from `backend\main.py`, `backend\utils\security.py`,
`backend\utils\logctx.py`, `backend\models\manager.py`, the four model backends,
`frontend\src\store\logs.ts` and `frontend\src\components\ConsolePanel.tsx`.

---

## 1. The three destinations

| # | Destination | What it is | Level | Where it is set up |
|---|---|---|---|---|
| 1 | Rotating file | `backend\logs\backend.log` | `DEBUG` (everything) | `backend\main.py:94-105` |
| 2 | In-memory ring feeding SSE | 300-entry `deque`, sent to `/api/logs/stream` | `SPECTRA_UI_LOG_LEVEL`, default `INFO` | `backend\main.py:109`, `backend\main.py:201-205` |
| 3 | In-app console panel | the panel behind the "Logs" button in the header | same ring, plus browser-side entries | `frontend\src\components\ConsolePanel.tsx` |

The root logger is set to `DEBUG` (`backend\main.py:104`). Two handlers hang off it:

- `_file_handler` — a `RotatingFileHandler`. It keeps every record at `DEBUG` and above.
  Rotation: `maxBytes = SPECTRA_LOG_MAX_MB * 1024 * 1024`, `backupCount = SPECTRA_LOG_BACKUPS`.
- `_buf_handler` (a `_BufferHandler`) — copies records into `_log_buffer` and pushes them
  to every connected SSE subscriber.

Both handlers carry `_ctx_filter`, which stamps `rid` and `sid` onto every record
(`backend\main.py:112-120`). Without that filter the format string would raise on
`%(rid)s`.

### Environment variables

| Variable | Default | Meaning | Read at |
|---|---|---|---|
| `SPECTRA_LOG_MAX_MB` | `10` | Size of one `backend.log` before it rotates | import time |
| `SPECTRA_LOG_BACKUPS` | `3` | How many old files to keep (`.1`, `.2`, `.3`) | import time |
| `SPECTRA_UI_LOG_LEVEL` | `INFO` | Minimum level that reaches the ring, SSE and the console | import time |

All three are read when `backend\main.py` is imported, so you must set them **before**
starting the backend. Changing them needs a restart.

With the defaults you get about 40 MB of history on disk (10 MB × 4 files). The ring only
ever holds the last 300 entries.

Note that `SPECTRA_UI_LOG_LEVEL=DEBUG` does not make the file louder — the file already
takes everything. It only lets more into the SSE stream and the console panel.

---

## 2. How to read a line

The file format is (`backend\main.py:101-103`):

```
%(asctime)s [%(levelname)s] rid=%(rid)s sid=%(sid)s %(name)s: %(message)s
```

An example line:

```
2026-10-09 14:22:31,104 [INFO] rid=3f9a1c2b sid=0f8e7d6c backend.models.manager: Subject 0 mask ready in 412ms (1 mask(s))
```

Reading it left to right:

| Field | Meaning |
|---|---|
| `asctime` | Local time, milliseconds. No date. |
| `[INFO]` | Level. One of `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. |
| `rid=` | Request id, 8 hex characters. |
| `sid=` | Session id, a full UUID. `-` when there is no session. |
| `name` | The Python logger name, e.g. `backend.models.manager`. |
| message | The text. May carry structured extras (see section 4). |

### What a request id is, and why it exists

Every HTTP request gets a fresh `rid` from `RequestLoggingMiddleware.dispatch`
(`backend\utils\security.py:208`). It is stored in a `contextvars` variable
(`backend\utils\logctx.py`), so anything that runs inside that request inherits it
automatically — including work handed to `asyncio.to_thread`, which copies the context.

Why it exists: one brush stroke fires several log lines from several modules
(`backend.main`, `backend.models.manager`, `backend.models.ultralytics_backend`). Without
an id you cannot tell which lines belong together. With it, you can trace one click
end to end.

### Following one request across the three layers

1. **File.** `grep` the `rid` in `backend\logs\backend.log`:
   `findstr /c:"rid=3f9a1c2b" backend\logs\backend.log`
   This is the only place with the full picture, because it keeps `DEBUG`.
2. **Ring / SSE.** Same `rid` rides along in the JSON entry. The console panel shows it
   only if a line's message includes it — the panel does not render `rid` as a column.
3. **Panel.** Find the request line (`POST /api/sessions/.../predict 200 · 412ms`) and the
   matching UI line from the `brush` category (`Subject 1 mask ready in 415ms`). Match them
   by timestamp and duration.

To follow a whole *session* instead of one request, grep on `sid=`.

---

## 3. The phase vocabulary

`phase` is a plain string in the logging `extra` dict. It is the fastest way to filter a
log file by what the app was doing.

| `phase` | Meaning | Emitted from | Level |
|---|---|---|---|
| `request` | One HTTP request: method, path, status, duration | `RequestLoggingMiddleware` (`backend\utils\security.py:220,237`) | see ladder below |
| `session` | Session created, destroyed or released | `backend\main.py:940,962,972` | `INFO` |
| `upload` | Image upload accepted or rejected | `backend\main.py:1002,1011,1021` | `INFO` / `WARNING` |
| `encode` | The image is being encoded into model features | `backend\models\manager.py:422,431,442,451` | `INFO` / `DEBUG` |
| `decode` | A mask is being predicted for one subject (the SAM decoder step) | `backend\models\manager.py:534,539` | `INFO` / `DEBUG` |
| `segment` | Batch segmentation of one or more boxes | `backend\models\manager.py:614` | `INFO` |
| `sam3` | A SAM3 text prompt finished | `backend\models\manager.py:710` | `INFO` |
| `detect` | Which device the detector is running on | `backend\models\manager.py:1233` | `DEBUG` |
| `residency` | A model or detector was loaded, unloaded or kept | many places in `backend\models\manager.py` | `INFO` / `DEBUG` |
| `vram` | GPU memory bookkeeping | `backend\main.py:352,472`, `backend\models\manager.py:208` | `INFO` / `ERROR` |
| `download` | Download progress | the `dl` dict, see section 4 | level forced to `DOWNLOAD` |

### The `request` level ladder

The middleware picks the level for its own line like this
(`backend\utils\security.py:240-249`):

1. Path is in `_DEBUG_PATHS` (`/health`, `/api/token`, `/api/detectors`, `/api/logs`,
   `/api/logs/stream`, `/api/models/status`, `/api/models/stream`) → `DEBUG`
2. Status ≥ 500 → `ERROR`
3. Status ≥ 400 → `WARNING`
4. Path ends with one of `_ALWAYS_INFO`, or the request took > 1000 ms → `INFO`
5. Anything else → `DEBUG`

`_ALWAYS_INFO` (`backend\utils\security.py:188-202`) is `/predict`, `/segment-batch`,
`/detect`, `/sam3-prompt`, `/sam3-undo`, `/sam3-redo`, `/sam3-remove-instance`, `/undo`,
`/redo`, `/clear-object`, `/export-zip`, `/export-image`, `/image`. So a slow image upload
also shows up at `INFO` even though `/image` is listed — the rule is `or`, not `and`.

---

## 4. Structured fields that ride along

`_BufferHandler.emit` (`backend\main.py:159-175`) copies these off the record if they are
not `None`:

| Field | Type | Meaning |
|---|---|---|
| `rid` | str | Request id, or `-`. |
| `sid` | str | Session id, or `-`. |
| `model` | str | Segmentation model file name, e.g. `sam2.1_b.pt`, `sam3.pt`. |
| `detector` | str | Detector name, e.g. `grounding-dino-tiny`. |
| `phase` | str | See the table above. |
| `dur_ms` | int | Duration of that one step in milliseconds. |

These four (`model`, `detector`, `phase`, `dur_ms`) go into the JSON entry that SSE
sends. They reach the file only if you print them yourself, because the file format
string does not include them. They are in `record.__dict__`, so you have to log them
to see them.

`ts` in the SSE entry is `time.strftime("%H:%M:%S")` — time of day, no date, no milliseconds.

### The download dictionary

Downloads pass a dict under `extra={"dl": {...}}`. `_BufferHandler` flattens every key of
that dict into the entry (except any key literally named `level`, `message` or `ts`) and
then forces `entry["level"] = "DOWNLOAD"`.

| Key | Meaning | Set by |
|---|---|---|
| `name` | Model or detector name | `backend\utils\model_integrity.py`, `backend\models\grounding_detector.py` |
| `pct` | Percent complete, 0-100 | same |
| `phase` | `starting`, `download`, `verify` or `done` | same |
| `file` | File currently being fetched (HF downloads) | `backend\models\grounding_detector.py:120` |
| `files_done` / `files_total` | How many files of the repo are done | same |
| `done_bytes` / `total_bytes` | Bytes for the current file | `backend\utils\model_integrity.py:180-187` |
| `eta` | `"3m 07s"` or `"42s"` | `backend\utils\model_integrity.py:204` |

Two code paths produce `dl` entries:

- **Single-file downloads** (`download_model_file` in `backend\utils\model_integrity.py`)
  log a line every 10 percentage points, with `done_bytes`, `total_bytes` and `eta`, then a
  `verify` line, then a `done` line.
- **Hugging Face repo downloads** (`_emit` in `backend\models\grounding_detector.py:90`)
  log at INFO with `name`, `pct`, `files_done`, `files_total`, `file`, `done_bytes`,
  `total_bytes`. Progress is throttled to roughly every 0.4 s per file
  (`backend\models\grounding_detector.py:167`).

A second, cruder path also tags entries as `DOWNLOAD`. If a record has **no** `dl` dict,
`_BufferHandler` tries three regexes (`_DOWNLOAD_RES`, `backend\main.py:134-138`) against
the message text. Anything shaped like `name: 42% complete`, `name: 42% —` or
`name: 42%|` becomes a `DOWNLOAD` entry with `name` and `pct` pulled out of the text.
This is how the tqdm output captured by `backend\utils\stderr_progress.py` reaches the panel.

`_clean_message` (`backend\main.py:141-145`) runs on every message before it is buffered:
it strips ANSI escape sequences, strips `[K`, turns box-drawing characters (`━ ─ ╷`) into
`-`, and trims. Empty messages are dropped.

---

## 5. What the console panel can do

Open it with the **Logs** button in the header. The panel is
`frontend\src\components\ConsolePanel.tsx`.

| Control | What it does |
|---|---|
| Level filter | 5 options: `All`, `Debug+`, `Info+`, `Warn+`, `Errors` (`FILTER_OPTIONS`) |
| Source filter | 3 options: `All`, `App`, `Server` (`SOURCE_OPTIONS`) |
| Category chips | 12 chips from `uiCategoryLabels`, each with a live count of buffered entries |
| Search | Free text. Matches message + level + category + source + `name` |
| Clock button | Show or hide the `HH:MM:SS` timestamp on each row |
| Copy button | Copies the filtered rows as text, plus a `[downloads]` section |
| X button | Clears the buffer, the unread count and the badge |
| "Show N older" | Raises the render limit by 200 |
| "Latest" | Scrolls back to the bottom and re-pins |

Details worth knowing:

- **Levels rank** `DEBUG:-1, INFO:0, WARNING:1, ERROR:2, CRITICAL:3, DOWNLOAD:-1`
  (`LEVEL_RANK`). `DOWNLOAD` entries are always hidden from the main list — they are shown
  as progress bars instead.
- **Category counts** skip `DOWNLOAD` entries. UI entries count under their own category;
  server entries all count under the single key `backend`. That is why only 12 UI chips
  exist and there is no separate "server" chip.

The 12 categories, and what lands in each:

| Category | Emitted from |
|---|---|
| `session` | `useSession.ts` — session created, destroyed, recovered, released, health check |
| `upload` | `useSession.ts` — guard drops, unsupported file, resize, start, done, failure |
| `mode` | `ModeDialog.tsx`, `ModeSwitchDialog.tsx`, `EmptyState.tsx` |
| `detect` | `useDetection.ts` — detector choice, detector load, the run itself |
| `segment` | `Canvas.tsx` and `Toolbar.tsx` — per-detection and batch segmentation |
| `brush` | `Canvas.tsx` and `useSession.ts` — strokes, predictions, undo, redo, clear |
| `sam3` | `useSam3.ts` and `Sam3PromptBar.tsx` — prompts, undo, redo, remove instance, settings |
| `subject` | `Toolbar.tsx` and `SubjectsPanel.tsx` — add, switch, delete |
| `layer` | `useLayers.ts`, `Canvas.tsx`, `App.tsx` — add, remove, clear all |
| `export` | `ExportDialog.tsx` — no-op, skip, request, downloaded, failure |
| `settings` | `Header.tsx`, `Toolbar.tsx`, `Sam3PromptBar.tsx` — view and option toggles |
| `console` | `pushToast`, and the panel's own open, SSE, filter, search and copy |

The rule that keeps the two row kinds apart: `category` is only ever set on entries
whose `source` is `ui`. UI entries never use `level: "DOWNLOAD"` and never set `name`,
so they can never be consumed by `progressRows`, which needs both.
- **The entry cap is 500** (`MAX_LOG_ENTRIES`).
- **Per-source reserve:** when the buffer overflows, the last **150** UI entries
  (`MIN_UI_ENTRIES`) are always kept, even if that means dropping newer server lines. The
  splice in `trim()` re-sorts by `id` with `localeCompare`, which is string order, so
  `l123` sorts before `l45`. After a splice the order can drift a little.
- **Messages are cut to 400 characters** (`MAX_MESSAGE`).
- **Only 120 rows render at a time** (`RENDER_LIMIT`). The rest are counted as
  "N older".
- **Download progress bars** (`progressRows`): DOWNLOAD entries are grouped by `name`.
  Each row shows a percentage bar, a chip (`Starting` / `Downloading` / `Verifying`), the
  byte counts or the file name, and the ETA. A second thinner bar shows
  `files_done / files_total`. A `done` entry removes the row. The bar turns brighter above
  90%.
- **Backfill on open.** The panel calls `api.getLogs(500)` and merges in anything that
  arrived while waiting (`pendingRef`), then opens the SSE stream. The SSE client
  (`frontend\src\lib\sse.ts`) reconnects with exponential backoff, `1s → 2s → 4s …`,
  capped at 30 s.
- **Badges.** A red dot means an `ERROR` or `CRITICAL` arrived, yellow means a `WARNING`.
  If both, red wins. The number badge counts entries that arrived while the panel was
  closed.
- **Stick-to-bottom is a ref, not an atom.** `pinnedRef` tracks whether the list is
  scrolled to the bottom; `handleScroll` re-pins when the user is within 30 px of the
  end and the pointer is not hovering. New entries auto-scroll only while pinned.

---

## 6. How to add a new log line

### Backend

```python
LOGGER.info(
    "Subject %d mask ready in %s", object_id, format_duration(elapsed),
    extra={"phase": "decode", "sid": session_id, "dur_ms": int(elapsed * 1000),
           "model": session.model_name},
)
```

Rules:

- Use the module logger: `LOGGER = logging.getLogger(__name__)`.
- Put `extra` on the call, never on `LOGGER.info` without arguments.
- Every `extra` key that the file format string mentions must exist, so only add custom
  keys; `rid` and `sid` are filled in by `_ctx_filter`.
- Pick a `phase` from the table in section 3. Add a new one only if none fits.
- Use `format_duration` from `backend\utils\humantime.py` for human-readable times, and
  `dur_ms` for the machine-readable one. They are separate on purpose.
- Use `%s` placeholders, not f-strings, so the arguments are not formatted when the level
  is filtered out.
- Wrap anything that can throw in `try/except` and log with `LOGGER.exception(...)` if you
  want the traceback.

### Frontend

```ts
emitUi("brush", "INFO", `Subject ${oid + 1} mask ready in ${Date.now() - t}ms`);
```

Rules:

- One of four calls: `emitUi(category, level, message, detail?)`, `logInfo`, `logWarn`,
  `logErr` (`frontend\src\store\logs.ts`).
- The `category` must be one of the 12 keys of `UiCategory`: `session`, `upload`, `mode`,
  `detect`, `segment`, `brush`, `sam3`, `subject`, `layer`, `export`, `settings`, `console`.
  A typo compiles fine and silently lands nowhere.
- `message` is capped at 300 characters, `detail` at 200, both through `safeText`.
- `logErr(category, err)` accepts an `Error` and renders it as `Name: message`.
- Everything goes through `subscribeLogs`. There is no level filter on the producer side —
  the panel filters after the fact.

### What must never be logged

| Never log | Why |
|---|---|
| The local token, `X-Local-Token`, `Authorization`, any bearer token | It is the only thing standing between the app and the network. |
| Absolute filesystem paths | They leak the machine's layout. Use the model file name only. |
| Full request or response bodies | They contain user pixels and can be megabytes. |

`backend\utils\security.py` already redacts on the client side: `safeText` replaces absolute
paths with `[path]`, secret-looking `key: value` pairs with `[redacted]`, and long hex
strings with `[hash]` (`frontend\src\store\logs.ts:33-50`). Do not rely on that as an excuse
to log them in the first place — the backend has no equivalent filter, so anything you log
there lands raw in `backend.log`.

---

## 7. Redaction rules in `safeText`

`frontend\src\store\logs.ts:37-50`:

| Step | What it does |
|---|---|
| 1. Coerce | `Error` → `"Name: message"`; string → itself; anything else → `JSON.stringify`, falling back to `String()`. |
| 2. `PATH_RE` | `[A-Za-z]:\\...` and `/home|/Users|/root|/tmp|/var|/opt|/srv|/workspace|/data...` → `[path]` |
| 3. `SECRET_RE` | `token`, `bearer` or `x-local-token` followed by `:` or `=` and a value → `[redacted]` |
| 4. `LONGHEX_RE` | Any run of 32 or more hex characters → `[hash]` |
| 5. Truncate | `max` characters, default 300, with `…` appended |

Why these three:

- **Paths.** The whole app is portable and relocatable. A log line with
  `D:\Spectra Group\Spectra Segment\backend\logs` in it is useless to anyone else and
  identifies the machine.
- **Tokens.** The session token is a bearer credential. A pasted console log is a credential
  leak.
- **Long hex.** Model hashes and session ids are 32+ hex characters. They are stable
  fingerprints that add nothing to a log line.

`LONGHEX_RE` is greedy over digits and `a-f` only, so a normal timestamp or a percentage is
untouched.

---

## 8. Troubleshooting cheat sheet

### "The model seems stuck"

Look for, in order:

1. `phase=download` lines, or a `Residency: ...` line. A model that is not on disk has to
   download first, and that is the slowest thing the app does.
2. `Model <name> ready in <duration>` at `phase=residency` (`backend\models\manager.py:310`),
   with `dur_ms`. If you see `Residency: unloading previous model` right before it, the app
   switched models, which unloads and reloads.
3. `phase=vram` `reason=over-budget` (`backend\models\manager.py:204-209`) — the app decided
   it did not have the memory budget and started evicting models to make room.
4. `phase=residency` `reason=no-co-residency` — a detector and a segmentation model are not
   allowed to live together, so one gets thrown out to load the other.
5. The request line for the offending call. If it is `DEBUG` only, it was fast and
   successful; if you saw nothing at all, it never reached the handler.
6. `Eviction loop: removed N idle session(s)` at `DEBUG` — a background task, so it has
   `rid=-`. Sessions idle for more than `IDLE_TIMEOUT` (600 s) are destroyed and the model
   unloaded with them.

### "Detection returned nothing"

1. `Detected '<query>' — N found, M kept (<duration>)` at `INFO`
   (`backend\models\manager.py:1248`). This line carries **no** `phase`. Compare `found`
   and `kept`: if found is high and kept is low, non-maximum suppression dropped them.
2. `phase=detect` `Detector <name> running on <device>` at `DEBUG`
   (`backend\models\manager.py:1233`). If the device is `cpu`, detection will be slow.
3. `GroundingDINO raw: N boxes` and `GroundingDINO detect returning N detections` at
   `DEBUG` in `backend\models\grounding_detector.py:745,756`, or the `Florence raw:` /
   `Florence detect returning:` pair at `682,697`. `raw` is before filtering, `returning`
   is after.
4. For YOLOE, `YOLOE: setting classes to [...]` and `YOLOE detect: labels=..., conf=...`
   (`backend\models\yoloe_detector.py:167,172`). The default confidence when none is given
   is `0.05`.
5. `phase=segment` `Batch segment complete in ... — N mask(s)`
   (`backend\models\manager.py:614`). Zero masks means the backend returned an empty
   positional list.
6. `Segmentation returned no mask for detection #N` in the console, category `segment`.

### "The mask looks wrong"

1. `phase=encode` `Encoded WxH image in <duration>` (`backend\models\manager.py:442`). Check
   the dimensions against what you expected — an oversized upload is silently downscaled,
   and you will see `Resizing upload WxH -> WxH` at `INFO` from `backend\main.py:312,334`.
2. `subject N mask ready in ... (M mask(s))` at `phase=decode`
   (`backend\models\manager.py:539`). `M` above 1 means multimask output was on, which only
   happens on the first stroke of a subject.
3. `First-stroke refinement: pass1 mask (positives only) -> pass2 refine with N negatives`
   at `DEBUG` (`backend\models\manager.py:993`). A mixed positive/negative first stroke goes
   through the model twice; the second pass is the one that counts.
4. `Full replay: N stroke(s) (P positive, N negative in last)` at `DEBUG`
   (`backend\models\manager.py:940`). Undo and redo replay the whole stroke history, so a
   mask can change even though you only pressed undo.
5. `Stroke points capped to 2048` at `DEBUG` (`backend\models\manager.py:518`) — a very
   long brush stroke was downsampled, which changes the prompt.
6. In the console, category `brush`: `Stroke submitted (N sample points, positive|negative)`
   at `DEBUG`. `N` is what actually reached the backend after point sampling.
7. `set_image: encode OOM on <device> - running image encoder on CPU` or
   `set_image: falling back to full CPU model` at `WARNING`
   (`backend\models\ultralytics_backend.py:183,219`). A CPU fallback still produces a mask,
   just a slower one.

### "It ran out of memory"

1. `phase=vram` `reason=oom` at `ERROR` (`backend\main.py:464-473`). The handler catches
   `torch.OutOfMemoryError` anywhere in a request and returns HTTP 507. The line includes
   free and total MB when it can read them.
2. `phase=vram` `reason=over-budget` at `INFO` (`backend\models\manager.py:204`), which
   reports how far over budget it was and how much it needed.
3. `VRAM: hard cap set to N% of device memory` at `INFO` at startup
   (`backend\main.py:350`). Default cap is `0.90` (`SPECTRA_VRAM_HARD_CAP`), clipped to at
   most `0.95`.
4. `Residency: ... left resident (inference in flight)` at `DEBUG`
   (`backend\models\manager.py:162,168,1143`) — the eviction wanted to unload something but
   an inference was holding the lock, so it did not.
5. `Error unloading model for VRAM headroom` at `ERROR` with a traceback
   (`backend\models\manager.py:147`).
6. On CPU, `Detector on CPU: using fp32 (CPU fp16 is emulated and slow)` at `INFO`
   (`backend\models\grounding_detector.py:438`).

The 507 response body is the constant `VRAM_OOM_DETAIL` (`backend\main.py:440-444`).

### "The console shows nothing"

1. Check the green dot next to "Console" in the panel header. A grey dot means the SSE
   stream is not connected; the client retries for up to 30 s between attempts.
2. `GET /api/logs/stream` is exempt from token auth, but it is capped at **5 connections
   per IP** (`MAX_SSE_PER_IP`) and **100 subscribers total** (`MAX_SSE_SUBSCRIBERS`), and
   each connection lives at most **1800 seconds** (`MAX_SSE_LIFETIME_SECONDS`). Open tabs
   you forgot about eat into the per-IP budget. If you hit the cap you get 429; if you hit
   the global cap, 503.
3. `SPECTRA_UI_LOG_LEVEL` is `INFO` by default. Anything below that never enters the ring,
   so `DEBUG` lines will not appear even though they are in the file.
4. The ring only holds 300 entries and the panel asks for 500, so a fresh panel never shows
   more than the last 300 server entries.
5. `Ready.` / `Server ready.` at `INFO` in the file (`backend\main.py:404`) tells you the
   logger is wired up at all.

---

## 9. Known gaps

- **Third-party loggers are quieted, not hidden.** `_quiet_third_party()`
  (`backend\main.py`) sets ten loggers to `WARNING` or `ERROR`: `ultralytics`,
  `transformers`, `huggingface_hub`, `huggingface_hub.utils`, `urllib3`, `fsspec`,
  `matplotlib`, `sentence_transformers`, `PIL` and `asyncio`. Their detail still
  reaches the file at `DEBUG`, but their `INFO` chatter never reaches the ring.
  Real download progress still arrives, because it comes from the `dl` path rather
  than from those loggers.
- **The console panel does not show `rid` or `sid`.** To match a UI line with a
  server line, use the timestamp and the duration. Or read the file.
- **`download` appears in two places, with different meanings.** It is a value of
  the `dl.phase` field, which marks a download progress bar. It is also the value
  of the record-level `phase` that `backend\models\grounding_detector.py` builds in
  `_emit`, which is used for detector downloads. Both are invisible in the file,
  because the file format string prints neither `dl` nor `phase`.
- **Several lines carry useful numbers in the text but not in `extra`.**
  `dur_ms` and `model` are only attached on the `INFO` lines of `encode`, `decode`,
  `segment`, `sam3` and the `residency` model-load line. The `DEBUG` lines mostly
  have no extras. So a timing breakdown that only exists at `DEBUG` has no field to
  hang off.
- **`SPECTRA_UI_LOG_LEVEL` is a floor for the ring, not for the file.** Setting it
  to `CRITICAL` will not shrink `backend.log`. The file is always `DEBUG`.
- **`_BufferHandler` drops an entry when two threads emit at once.** The
  re-entrancy guard is thread-local, so a second thread emitting during another
  thread's emit is silently discarded. The file still gets both records, because
  that is a separate handler.
- **HTTP errors on `undo`, `redo` and `clear-object` only surface as the request
  line.** `main.py` does not log a reason for those three, so the console shows the
  method, path and status but not why.
- **`get_or_create_token()` makes a new token on every call.** It is safe only
  because `lifespan` calls it exactly once. A second call would invalidate every
  token already issued.
