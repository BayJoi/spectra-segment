# Documentation

Start with the mode you are using, then go deeper only when you need to change
the code.

## Modes

- [Brush](modes/BRUSH.md) - paint a subject, keep several subjects separate,
  and right-drag to erase
- [Detect](modes/DETECT.md) - describe an object, let a detector find it, then
  segment it with SAM
- [Matte](modes/MATTE.md) - soft alpha cutouts with BiRefNet, including
  per-subject models and Heavy/Medium/Light guidance
- [SAM 3](modes/SAM3.md) - text prompts, instance control and encode quality

## Architecture and workflow

- [Code guide](CODE_GUIDE.md) - project shape, startup chain, manager,
  invariants, verification and fork rules
- [Backend guide](BACKEND_GUIDE.md) - routes, manager state, model backends,
  environment variables and licensing
- [AMD support](AMD.md) - TheRock support matrix, tested RX 6600 results and
  Windows setup limits
- [NVIDIA support](NVIDIA.md) - untested compatibility matrix, wheel selection,
  precision/TF32 policy and limitations
- [Frontend guide](FRONTEND.md) - state, canvas, stroke prompts, undo/redo and
  known gaps
- [Logging](LOGGING.md) - log format, structured fields, console panel and
  troubleshooting
- [Security review](SECURITY.md) - fixed auth issues, 2026 dependency audit,
  residual risk and the transformers exception

## Checks and reference

- [GPU and model checks](../README.md#checks) - AMD, NVIDIA and CPU health
  wrappers, the 33-model catalogue sweep, and the 4K memory pass
- [Backend code reference](reference/BACKEND.md) - every Python module, route,
  symbol, installer and launcher
- [Frontend code reference](reference/FRONTEND.md) - every frontend module,
  component, hook and store

## Performance

- [Overview](performance/OVERVIEW.md) - the research plan, where time goes and
  the platform summary
- [Catalogue](performance/CATALOG.md) - proposed areas and external evidence:
  orchestration, SAM 3, SAM 2, CUDA, ROCm, CPU, export and startup
- [Testing](performance/TESTING.md) - experiment order, how to write a test,
  sources and open questions
