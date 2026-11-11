# Security review

Last reviewed: **2026-11-12**. Scope: the two Python backends, the React
frontend, the packaged Vite build, and the direct runtime dependencies.

This is a focused hardening review, not a claim that the application is free of
vulnerabilities. The app is local-only, but a browser on the same machine can
still reach the API, so the token and host checks matter.

## Fixed in this pass

### 1. SSE streams and the release endpoint required no token

`TokenAuthMiddleware` used to exempt both SSE streams and
`POST /api/sessions/{id}/release`. Any local process able to reach port 8000
could therefore subscribe to logs or destroy sessions without the token.

The middleware now requires the token for:

- `GET /api/logs/stream`
- `GET /api/models/stream`
- `POST /api/sessions/{id}/release`

`/api/token` and `/health` remain exempt. `EventSource` and
`navigator.sendBeacon` cannot set request headers, so those three paths also
accept `?token=...`. No other route accepts a query token; in particular,
`GET /api/logs` requires `X-Local-Token`.

The frontend now:

- fetches a fresh token for every SSE connection and reconnect, which also
  handles a backend restart;
- puts the token on the `sendBeacon` release URL;
- keeps the token-bearing `DELETE` fallback, including its one retry after a
  401.

The token remains in memory only. It is not written to disk or logged, and the
Vite CSP restricts connections to the local backend.

### 2. Console redaction missed UUIDs

`safeText` already replaced paths, secret-looking assignments, and long hex
strings. Session IDs are UUID-shaped and contain hyphens, so the 32-hex rule did
not match them. UUIDs are now replaced with `[id]` before rendering.

### 3. Frontend build tool advisories

Vite `8.2.1` was affected by six npm advisories:

- GHSA-587h-7wmj-h5pm
- GHSA-698v-4r3x-3vcv
- GHSA-9jrq-w75r-8gcw
- GHSA-r825-w8mm-5pxw
- GHSA-rq7h-c2jc-7f22
- GHSA-vfpm-58rq-9qcg

All six are fixed in `8.2.4`. `frontend/package.json`,
`backend_amd_gpu/web/package.json`, and the packaged `bun.lock` are on
`8.2.4`; the audit now reports zero findings for that version.

### 4. Python dependency floors

The runtime floors are now:

| Package | Floor | Why |
|---|---|---|
| `transformers` | `>=4.57.6,<5.0` | Newest 4.x; 5.x is not compatible with the pinned Florence-2 remote config |
| `huggingface_hub[hf_xet]` | `>=0.36.2,<1.0` | Matches the 4.x transformers range and avoids older resolver output |
| `filelock` | `>=4.2.1,<5.0` | 4.0.8 had 14 OSV findings; fixed at 4.2.1 |
| `setuptools` | `>=84.1.0` | 82.0.1 had 3 OSV findings; fixed at 83.0.0 / 84.1.0 |

`filelock 4.2.10` and `setuptools 84.2.1` are installed in the AMD test venv
and return zero OSV findings.

### 5. Matte remote-code trust boundary

BiRefNet checkpoints use `trust_remote_code=True`. The catalog is fixed, but
`BiRefNetBackend.load_model()` now also refuses any repo that is not in
`TRUSTED_MATTE_REPOS`. Detector loads already had an equivalent allow-list plus
a pinned-revision check.

## Dependency audit

OSV API query on 2026-11-12, using the versions in the two `requirements.txt`
files and the packaged frontend lockfile.

- **Frontend:** 171 npm packages queried. Only Vite was affected; it is fixed.
- **Python:** `fastapi`, `uvicorn`, `starlette`, `h11`, `python-multipart`,
  `pillow`, `numpy`, `opencv-python-headless`, `ultralytics`,
  `huggingface-hub`, `accelerate`, `timm`, `einops`, `kornia`, and `supervision`
  returned zero OSV findings at the tested versions.
- **`transformers 4.57.6`:** 11 OSV findings. They include
  GHSA-29pf-2h5f-8g72 / CVE-2026-4372 (RCE before 5.3.0),
  GHSA-69w3-r845-3855 / CVE-2026-1839 (Trainer RCE),
  GHSA-fgcw-684q-jj6r / CVE-2026-5241 (LightGlue code execution),
  GHSA-x9r9-c232-4q39 / CVE-2026-80047 (custom generation code before consent),
  and PYSEC-2026-3929 (save_pretrained path traversal).

`bun pm scan` could not be used because this Bun installation has no security
scanner configured. The npm audit above was run directly against OSV.

## Why `transformers` was not upgraded to 5.x

The fixed releases were tested against the cached, pinned checkpoints:

- `5.22.2` loads BiRefNet successfully.
- `5.22.2` and `5.10.0` both fail the pinned Florence-2 remote config with
  `'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'`.
- `5.10.0` is additionally marked yanked on PyPI.

Pinning a 5.x release would therefore break the Detect mode's Florence
checkpoints. Blind upgrades are worse than a documented exception: they would
make the app look patched while one of its model families cannot load.

The remaining risk is reduced, not eliminated:

- Only fixed model IDs and fixed revisions are passed to
  `from_pretrained(..., trust_remote_code=True)`.
- User input cannot supply a repo id; session and matte model names are
  validated against closed catalogs.
- The app does not use `Trainer`, `save_pretrained`, LightGlue, or arbitrary
  user-supplied generation code.
- Cached models load with `local_files_only=True` first, and pinned revisions
  prevent a moving branch from changing the code.
- Matte and detector remote-code loads are allow-listed.

Revisit this exception when a 5.x release can load the pinned Florence-2
checkpoint. Test both a BiRefNet load and a Florence-2 config/model load before
lifting `<5.0`.

## Residual risks

- **`transformers 4.57.6` advisories remain.** The controls above reduce
  reachability but do not patch the library. The pinned upstream revisions are
  the main boundary; a compromised cache or upstream repository would still be
  trusted.
- **NVIDIA is untested on physical hardware.** The CPU/NVIDIA auth and
  dependency changes are covered by static checks and the shared middleware
  tests, not by a CUDA runtime test.
- **The SSE/release query token is visible to same-machine tooling** that can
  inspect browser or process state. It is scoped to those three paths and is not
  accepted elsewhere.
- **The app is local-only by policy.** Host validation, CORS allow-listing, the
  token, upload limits, and rate limits are not a substitute for exposing this
  service to a network.

## Verification performed

- `tsc --noEmit` and `vite build` pass on the updated frontend.
- `py_compile` and `pyflakes` pass for both security modules and the touched
  matte backend.
- Starlette `TestClient` matrix, run against both backend security modules:
  no-token `/api/logs` 401, header token 200, query-token `/api/logs` 401,
  SSE no-token 401, SSE query-token 200, wrong SSE token 401,
  release no-token 401, release query-token 200, `/health` 200.
- `requestSummary` and `safeText` smoke checks pass, including UUID redaction.
- OSV: Vite 8.2.4, filelock 4.2.10, and setuptools 84.2.1 return zero findings.
- The full AMD 33-model sweep ran on the installed stack on 2026-11-12
  (33/33 passed, 41 phase checks passed, 2 skipped), plus a 3840x2160 pass.
  It was not rerun after a dependency reinstall. NVIDIA hardware and browser
  automation remain untested.

## Related documents

- [Logging and console](LOGGING.md)
- [Backend guide](BACKEND_GUIDE.md)
- [Frontend guide](FRONTEND.md)
