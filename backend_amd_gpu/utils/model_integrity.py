from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import urllib.request
import ssl
from pathlib import Path

LOGGER = logging.getLogger(__name__)

_download_locks: dict[str, threading.Lock] = {}
_download_locks_guard = threading.Lock()


def _download_lock(model_name: str) -> threading.Lock:
    with _download_locks_guard:
        lock = _download_locks.get(model_name)
        if lock is None:
            lock = threading.Lock()
            _download_locks[model_name] = lock
        return lock

MODELS_DIR = Path(__file__).resolve().parent.parent / "model_weights"
HASH_DB = MODELS_DIR / ".hashes.json"

PINNED_MODEL_URLS: dict[str, str] = {
    "mobileclip2_b.ts": "https://github.com/ultralytics/assets/releases/download/v8.4.0/mobileclip2_b.ts",
    "mobileclip_blt.ts": "https://github.com/ultralytics/assets/releases/download/v8.4.0/mobileclip_blt.ts",
    "sam2.1_b.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2.1_b.pt",
    "sam2.1_l.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2.1_l.pt",
    "sam2.1_s.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2.1_s.pt",
    "sam2.1_t.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2.1_t.pt",
    "sam2_b.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2_b.pt",
    "sam2_l.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2_l.pt",
    "sam2_s.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2_s.pt",
    "sam2_t.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2_t.pt",
    "sam3.pt": "https://huggingface.co/bodhicitta/sam3/resolve/main/sam3.pt",
    "yoloe-11l-seg.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-11l-seg.pt",
    "yoloe-11m-seg.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-11m-seg.pt",
    "yoloe-11s-seg.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-11s-seg.pt",
    "yoloe-26l-seg.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26l-seg.pt",
    "yoloe-26m-seg.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26m-seg.pt",
    "yoloe-26n-seg.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26n-seg.pt",
    "yoloe-26s-seg.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26s-seg.pt",
}

PUBLISHED_DIGESTS: dict[str, str] = {
    "mobileclip2_b.ts": "35d7f213e4d75f38514e4656ad3cb91158bd33e3805d8ac349f23b186f66982f",
    "mobileclip_blt.ts": "a67804d1b0f07b8b9a20c1761ec0847f34660f5fa338ec70e8f3fce68ed95e54",
    "sam2.1_b.pt": "f1a9cf2dd69d84bb463b5ad98246d03e2d47a130a9295db0ec967e6cd95e2e47",
    "sam2.1_l.pt": "ab7e1ac9cb9f6eb3bcf197ece044f06a707ec49129361a2b47e93e1db6989efd",
    "sam2.1_s.pt": "60f9e43f1307be192eef341437e02c40f32cd61cf36a97a203a0998a2952873a",
    "sam2.1_t.pt": "3c1e81ca9b037dd39d70a014ddb9a813d6c4c4e12555420db7eaff31689bd4e3",
    "sam2_b.pt": "39722bb0ce2a086058cf64e50dffd6f9e9931b5fcbee79e33b30441c6c40264d",
    "sam2_l.pt": "fd618bcfc7b84c8f2e0a6997548e197b907098196dd075387d01f64d9cf8a93b",
    "sam2_s.pt": "6ba91d739a6adfc1dcaf76336b2f380f3eae8d24a88f56c62798907f7fd62dad",
    "sam2_t.pt": "94375f988270836169320bd901960c67b5770e8bef3867d70102f01a8b5ca501",
    "sam3.pt": "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e",
    "yoloe-11l-seg.pt": "a993fb0fc7c8830939ae14e6434a925dd1179428158c2761482eb8a8d8a3699f",
    "yoloe-11m-seg.pt": "785199b7cc75a5f4041dec15782a4b8a4105e5a34d7ffb287a0a10781724c5bd",
    "yoloe-11s-seg.pt": "8e439445c87338b79d9ce21dec109f4621e26df67e94d26ea1a98c1e64dce3e3",
    "yoloe-26l-seg.pt": "a612d2d505f24e14d87ec82d688b823b6cb600646664f16125ce6c84ce360da9",
    "yoloe-26m-seg.pt": "585f5ec9028fd358035da8d860c27c56be285a795cba2076fba536a4391c2c83",
    "yoloe-26n-seg.pt": "1741c1f8da3cea47e2c01829c334a50dc0b9bbd05e685b90a3ce84fae32c8c1b",
    "yoloe-26s-seg.pt": "48f24206bc8680d60cbbfa296b0140da849669b9515058b72f5a945142df0654",
}


def _load_hash_db() -> dict[str, str]:
    if not HASH_DB.exists():
        return {}
    try:
        return json.loads(HASH_DB.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        LOGGER.debug("Corrupt hash database at %s — starting fresh", HASH_DB)
        return {}


def _save_hash_db(db: dict[str, str]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    HASH_DB.write_text(json.dumps(db, indent=2), encoding="utf-8")


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_model(model_path: Path, model_name: str) -> bool:
    if not model_path.exists():
        return False

    actual_hash = compute_sha256(model_path)

    published = PUBLISHED_DIGESTS.get(model_name)
    if published is not None:
        if actual_hash != published:
            LOGGER.error(
                "INTEGRITY FAILURE: Model %s hash mismatch against published digest!\n"
                "  Published: %s\n"
                "  Actual:    %s\n"
                "  File may be corrupted or tampered with. Delete and re-download.",
                model_name, published, actual_hash,
            )
            return False
        LOGGER.debug(
            "Model %s: verified against published GitHub digest (%s...)",
            model_name, actual_hash[:16],
        )
        return True

    db = _load_hash_db()
    stored_hash = db.get(model_name)

    if stored_hash is None:
        db[model_name] = actual_hash
        _save_hash_db(db)
        LOGGER.debug(
            "Model %s: no published digest available — TOFU hash recorded (%s...). "
            "This detects future cache tampering but does not independently "
            "verify the download source.",
            model_name, actual_hash[:16],
        )
        return True

    if actual_hash != stored_hash:
        LOGGER.error(
            "INTEGRITY FAILURE: Model %s TOFU hash mismatch!\n"
            "  Expected: %s\n"
            "  Actual:   %s\n"
            "  Local cache may be corrupted or tampered with. "
            "  Delete and re-download.",
            model_name, stored_hash, actual_hash,
        )
        return False

    LOGGER.debug("Model %s: TOFU hash verified (%s...)", model_name, actual_hash[:16])
    return True


def download_model_file(
    model_name: str,
    dest_path: Path,
    url: str | None = None,
) -> Path:
    if url is None:
        url = PINNED_MODEL_URLS.get(model_name)
    if not url:
        raise RuntimeError(f"No pinned URL for model {model_name}")

    with _download_lock(model_name):
        if dest_path.exists():
            if verify_model(dest_path, model_name):
                LOGGER.info("Model %s already downloaded — skipping download", model_name)
                return dest_path

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

        LOGGER.info("Downloading %s from %s", model_name, url)
        try:
            ctx = ssl.create_default_context()
            req = urllib.request.Request(url, headers={"User-Agent": "SpectraSegment/1.0"})

            with urllib.request.urlopen(req, context=ctx, timeout=120) as resp:
                total = resp.headers.get("Content-Length")
                total_bytes = int(total) if total else None
                total_mb = total_bytes / (1024 * 1024) if total_bytes else None
                downloaded = 0
                last_log_pct = -1
                start_time = time.monotonic()

                with open(tmp_path, "wb") as f:
                    while True:
                        chunk = resp.read(1 << 16)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_bytes:
                            pct = int(downloaded * 100 / total_bytes)
                            if pct >= last_log_pct + 10:
                                last_log_pct = pct
                                elapsed = time.monotonic() - start_time
                                speed = downloaded / elapsed if elapsed > 0 else 0
                                remaining = ((total_bytes - downloaded) / speed) if speed > 0 else 0
                                m, s = divmod(int(remaining), 60)
                                eta = f"{m}m {s:02d}s" if m else f"{s}s"
                                LOGGER.info(
                                    "  %s: %d%% — %.1f / %.1f MB (%.1f MB/s, ETA %s)",
                                    model_name, pct, downloaded / 1048576, total_mb,
                                    speed / 1048576, eta,
                                )
                        else:
                            LOGGER.info(
                                "  %s: %.1f MB downloaded", model_name, downloaded / 1048576,
                            )

            if not verify_model(tmp_path, model_name):
                tmp_path.unlink(missing_ok=True)
                raise RuntimeError(f"Integrity check failed for {model_name} after download")

            if dest_path.exists():
                dest_path.unlink()
            tmp_path.rename(dest_path)
            LOGGER.info("Downloaded and verified %s (%.1f MB)", model_name, dest_path.stat().st_size / 1048576)
            return dest_path

        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
