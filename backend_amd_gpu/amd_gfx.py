from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TargetInfo:
    gfx: str
    family: str
    support: str
    integrated: bool
    fp16: bool
    detector_max_side: int
    aotriton: bool


def _target(
    gfx: str,
    family: str,
    support: str = "release",
    integrated: bool = False,
    fp16: bool = True,
    detector_max_side: int = 1600,
    aotriton: bool = True,
) -> TargetInfo:
    if family in ("RDNA 1", "RDNA 2", "GCN 5.1", "GCN 5.0"):
        aotriton = False
    return TargetInfo(gfx, family, support, integrated, fp16, detector_max_side, aotriton)


WINDOWS_TARGETS: dict[str, TargetInfo] = {
    "gfx1201": _target("gfx1201", "RDNA 4"),
    "gfx1200": _target("gfx1200", "RDNA 4"),
    "gfx1172": _target("gfx1172", "RDNA 4m", support="build", integrated=True, detector_max_side=1024),
    "gfx1171": _target("gfx1171", "RDNA 4m", support="build", integrated=True, detector_max_side=1024),
    "gfx1170": _target("gfx1170", "RDNA 4m", support="build", integrated=True, detector_max_side=1024),
    "gfx1153": _target("gfx1153", "RDNA 3.5", integrated=True, detector_max_side=1024),
    "gfx1152": _target("gfx1152", "RDNA 3.5", integrated=True, detector_max_side=1024),
    "gfx1151": _target("gfx1151", "RDNA 3.5", integrated=True, detector_max_side=1280),
    "gfx1150": _target("gfx1150", "RDNA 3.5", integrated=True, detector_max_side=1024),
    "gfx1103": _target("gfx1103", "RDNA 3", integrated=True, detector_max_side=1024),
    "gfx1102": _target("gfx1102", "RDNA 3"),
    "gfx1101": _target("gfx1101", "RDNA 3"),
    "gfx1100": _target("gfx1100", "RDNA 3"),
    "gfx1036": _target("gfx1036", "RDNA 2", integrated=True, detector_max_side=1024),
    "gfx1035": _target("gfx1035", "RDNA 2", integrated=True, detector_max_side=1024),
    "gfx1034": _target("gfx1034", "RDNA 2"),
    "gfx1033": _target("gfx1033", "RDNA 2", integrated=True, detector_max_side=1024),
    "gfx1032": _target("gfx1032", "RDNA 2"),
    "gfx1031": _target("gfx1031", "RDNA 2"),
    "gfx1030": _target("gfx1030", "RDNA 2"),
    "gfx1012": _target("gfx1012", "RDNA 1"),
    "gfx1011": _target("gfx1011", "RDNA 1"),
    "gfx1010": _target("gfx1010", "RDNA 1"),
    "gfx906": _target("gfx906", "GCN 5.1", support="build", fp16=False, detector_max_side=1280, aotriton=False),
    "gfx90c": _target("gfx90c", "GCN 5.0", support="build", integrated=True, fp16=False, detector_max_side=1024, aotriton=False),
    "gfx900": _target("gfx900", "GCN 5.0", support="build", fp16=False, detector_max_side=1280, aotriton=False),
    "gfx90a": _target("gfx90a", "CDNA 2", support="build"),
    "gfx908": _target("gfx908", "CDNA 1", support="build"),
    "gfx942": _target("gfx942", "CDNA 3", support="linux-only"),
    "gfx950": _target("gfx950", "CDNA 4", support="linux-only"),
}

SUPPORT_LABELS = {
    "release": "release-ready",
    "build": "build-only, unverified",
    "linux-only": "Linux only",
}


def normalize_gfx(value: str | None) -> str | None:
    if not value:
        return None
    gfx = value.strip().lower().split(":", 1)[0]
    if not re.fullmatch(r"gfx[0-9a-z]{3,4}", gfx):
        return None
    return gfx


def target_info(gfx: str | None) -> TargetInfo | None:
    return WINDOWS_TARGETS.get(normalize_gfx(gfx) or "")


def support_label(support: str) -> str:
    return SUPPORT_LABELS.get(support, support)


def supported_windows_targets() -> list[str]:
    return [gfx for gfx, info in WINDOWS_TARGETS.items() if info.support in ("release", "build")]
