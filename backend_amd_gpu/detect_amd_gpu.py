from __future__ import annotations

import argparse
import csv
import ctypes
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from amd_gfx import (  # noqa: E402
    SUPPORT_LABELS,
    TargetInfo,
    normalize_gfx,
    support_label,
    supported_windows_targets,
    target_info,
)

PCI_DEV_TO_GFX: dict[str, tuple[str, str]] = {
    "7590": ("gfx1200", "Navi 44 (RX 9060 XT)"),
    "7550": ("gfx1201", "Navi 48 (RX 9070/9070 XT/9070 GRE)"),
    "7551": ("gfx1201", "Navi 48 (Radeon AI PRO R9700)"),
    "7580": ("gfx1201", "Navi 48 (RX 9070 XT)"),
    "7581": ("gfx1201", "Navi 48 (RX 9070)"),
    "7591": ("gfx1201", "Navi 44 (RX 9060 XT)"),
    "75a1": ("gfx1201", "Navi 48 (RX 9070 GRE)"),
    "75b0": ("gfx1201", "Navi 48 (RX 9070 XT)"),
    "150e": ("gfx1150", "Strix Point (880M/890M)"),
    "1586": ("gfx1151", "Strix Halo (8050S/8060S)"),
    "1114": ("gfx1152", "Krackan Point (840M/860M)"),
    "1590": ("gfx1150", "Strix Point (880M)"),
    "1591": ("gfx1150", "Strix Point (890M)"),
    "15d0": ("gfx1152", "Krackan Point (860M)"),
    "744c": ("gfx1100", "Navi 31 (RX 7900 XT/XTX/GRE/7900M)"),
    "7448": ("gfx1100", "Navi 31 (Pro W7900)"),
    "7449": ("gfx1100", "Navi 31 (Pro W7800 48GB)"),
    "744a": ("gfx1100", "Navi 31 (Pro W7900 Dual Slot)"),
    "744b": ("gfx1100", "Navi 31 (Pro W7900D)"),
    "745e": ("gfx1100", "Navi 31 (Pro W7800)"),
    "7460": ("gfx1101", "Navi 32 (Pro V710)"),
    "7461": ("gfx1101", "Navi 32 (Pro V710)"),
    "7470": ("gfx1101", "Navi 32 (Pro W7700)"),
    "747e": ("gfx1101", "Navi 32 (RX 7700 XT/7800 XT)"),
    "7480": ("gfx1102", "Navi 33 (RX 7600 series / Pro W7600)"),
    "7481": ("gfx1102", "Navi 33"),
    "7483": ("gfx1102", "Navi 33 (RX 7600M/7600M XT)"),
    "7487": ("gfx1102", "Navi 33"),
    "7489": ("gfx1102", "Navi 33 (Pro W7500)"),
    "748b": ("gfx1102", "Navi 33"),
    "7499": ("gfx1102", "Navi 33 (RX 7400/7300/Pro W7400)"),
    "749f": ("gfx1102", "Navi 33 (RX 7500)"),
    "15bf": ("gfx1103", "Phoenix1 (780M/760M/740M)"),
    "15c8": ("gfx1103", "Phoenix2 (780M/760M/740M)"),
    "164f": ("gfx1103", "Phoenix (780M/760M/740M)"),
    "73a1": ("gfx1030", "Navi 21 (Pro V620)"),
    "73a2": ("gfx1030", "Navi 21 (Pro W6900X)"),
    "73a3": ("gfx1030", "Navi 21 (Pro W6800)"),
    "73a5": ("gfx1030", "Navi 21 (RX 6950 XT)"),
    "73ab": ("gfx1030", "Navi 21 (Pro W6800X/W6800X Duo)"),
    "73ae": ("gfx1030", "Navi 21 (Pro V620 MxGPU)"),
    "73af": ("gfx1030", "Navi 21 (RX 6900 XT)"),
    "73bf": ("gfx1030", "Navi 21 (RX 6800/6800 XT/6900 XT)"),
    "73c3": ("gfx1031", "Navi 22"),
    "73ce": ("gfx1031", "Navi 22 (SRIOV MxGPU)"),
    "73df": ("gfx1031", "Navi 22 (RX 6700/6750 XT/6800M/6850M XT)"),
    "73e0": ("gfx1032", "Navi 23"),
    "73e1": ("gfx1032", "Navi 23 (Pro W6600M)"),
    "73e3": ("gfx1032", "Navi 23 (Pro W6600)"),
    "73ef": ("gfx1032", "Navi 23 (RX 6650 XT/6700S/6800S)"),
    "73ff": ("gfx1032", "Navi 23 (RX 6600/6600 XT/6600M)"),
    "73e2": ("gfx1032", "Navi 23 (RX 6600 OEM)"),
    "73f0": ("gfx1032", "Navi 23 (RX 6650 XT OEM)"),
    "163f": ("gfx1033", "Van Gogh (Steam Deck APU)"),
    "7421": ("gfx1034", "Navi 24 (Pro W6500M)"),
    "7422": ("gfx1034", "Navi 24 (Pro W6400)"),
    "7423": ("gfx1034", "Navi 24 (Pro W6300/W6300M)"),
    "7424": ("gfx1034", "Navi 24 (RX 6300)"),
    "743f": ("gfx1034", "Navi 24 (RX 6400/6500 XT/6500M)"),
    "1681": ("gfx1035", "Rembrandt (680M/660M)"),
    "1506": ("gfx1036", "Mendocino (610M)"),
    "7310": ("gfx1010", "Navi 10 (Pro W5700X)"),
    "7312": ("gfx1010", "Navi 10 (Pro W5700)"),
    "7319": ("gfx1010", "Navi 10 (Pro 5700 XT)"),
    "731b": ("gfx1010", "Navi 10 (Pro 5700)"),
    "731f": ("gfx1010", "Navi 10 (RX 5600/5700 series)"),
    "7360": ("gfx1011", "Navi 12 (Pro 5600M/V520/BC-160)"),
    "7362": ("gfx1011", "Navi 12 (Pro V520/V540)"),
    "7340": ("gfx1012", "Navi 14 (RX 5500/5500M/Pro 5300)"),
    "7341": ("gfx1012", "Navi 14 (Pro W5500)"),
    "7347": ("gfx1012", "Navi 14 (Pro W5500M)"),
    "734f": ("gfx1012", "Navi 14 (Pro W5300M)"),
    "74a0": ("gfx942", "Aqua Vanjaram (MI300A)"),
    "74a1": ("gfx942", "Aqua Vanjaram (MI300X)"),
    "74a2": ("gfx942", "Aqua Vanjaram (MI308X)"),
    "74a5": ("gfx942", "Aqua Vanjaram (MI325X)"),
    "74a9": ("gfx942", "Aqua Vanjaram (MI300X HF)"),
    "74b5": ("gfx942", "Aqua Vanjaram (MI300X VF)"),
    "74b9": ("gfx942", "Aqua Vanjaram (MI325X VF)"),
    "74bd": ("gfx942", "Aqua Vanjaram (MI300X HF)"),
    "75a0": ("gfx950", "Aqua Vanjaram (MI350X)"),
    "75a3": ("gfx950", "Aqua Vanjaram (MI355X)"),
    "738c": ("gfx908", "Arcturus (MI100)"),
    "738e": ("gfx908", "Arcturus (MI100)"),
    "7408": ("gfx90a", "Aldebaran (MI250X)"),
    "740c": ("gfx90a", "Aldebaran (MI250X/MI250)"),
    "740f": ("gfx90a", "Aldebaran (MI210)"),
    "6860": ("gfx900", "Vega 10 (Instinct MI25/V340/V320)"),
    "6861": ("gfx900", "Vega 10 (Pro WX 9100)"),
    "6862": ("gfx900", "Vega 10 (Pro SSG)"),
    "6863": ("gfx900", "Vega 10 (Vega Frontier Edition)"),
    "6864": ("gfx900", "Vega 10 (Pro V340/Instinct MI25x2)"),
    "6867": ("gfx900", "Vega 10 (Pro Vega 56)"),
    "6868": ("gfx900", "Vega 10 (Pro WX 8100/8200)"),
    "6869": ("gfx900", "Vega 10 (Pro Vega 48)"),
    "686b": ("gfx900", "Vega 10 (Pro Vega 64X)"),
    "686c": ("gfx900", "Vega 10 (Instinct MI25 MxGPU)"),
    "687f": ("gfx900", "Vega 10 (RX Vega 56/64)"),
    "66a0": ("gfx906", "Vega 20 (Pro / Instinct MI50)"),
    "66a1": ("gfx906", "Vega 20 (Pro VII / Instinct MI50)"),
    "66a3": ("gfx906", "Vega 20 (Pro Vega II / Vega II Duo)"),
    "66a7": ("gfx906", "Vega 20 (Pro Vega 20)"),
    "66af": ("gfx906", "Vega 20 (Radeon VII)"),
    "1636": ("gfx90c", "Renoir (Radeon Graphics)"),
    "1638": ("gfx90c", "Cezanne (Radeon Graphics)"),
    "164c": ("gfx90c", "Lucienne (Radeon Graphics)"),
    "15e7": ("gfx90c", "Barcelo (Radeon Graphics)"),
    "15d8": ("gfx902", "Picasso/Raven (Radeon Graphics)"),
}

GPU_NAME_TO_GFX: list[tuple[tuple[str, ...], str, str]] = [
    (("rx 9060",), "gfx1200", "RDNA 4"),
    (("rx 9070", "r9700", "r9600"), "gfx1201", "RDNA 4"),
    (("890m", "880m"), "gfx1150", "RDNA 3.5"),
    (("8065s", "8060s", "8050s", "8040s"), "gfx1151", "RDNA 3.5"),
    (("860m", "840m"), "gfx1152", "RDNA 3.5"),
    (("820m",), "gfx1153", "RDNA 3.5"),
    (("rx 7900", "w7900", "w7800"), "gfx1100", "RDNA 3"),
    (("rx 7800", "rx 7700", "w7700"), "gfx1101", "RDNA 3"),
    (("rx 7700s", "rx 7650", "rx 7600", "w7600", "w7500", "rx 7400", "w7400"), "gfx1102", "RDNA 3"),
    (("780m", "760m", "740m"), "gfx1103", "RDNA 3"),
    (("rx 6950", "rx 6900", "rx 6800", "w6800", "v620"), "gfx1030", "RDNA 2"),
    (("rx 6750", "rx 6700", "rx 6800m", "rx 6700m", "rx 6800s", "rx 6700s"), "gfx1031", "RDNA 2"),
    (("rx 6650", "rx 6600", "w6600", "rx 6650m", "rx 6600m", "rx 6600s"), "gfx1032", "RDNA 2"),
    (("van gogh", "amd custom apu 0405"), "gfx1033", "RDNA 2"),
    (("rx 6550", "rx 6500", "rx 6450", "rx 6400", "w6500", "w6400", "rx 6300", "w6300"), "gfx1034", "RDNA 2"),
    (("680m", "660m"), "gfx1035", "RDNA 2"),
    (("610m",), "gfx1036", "RDNA 2"),
    (("rx 5700", "rx 5600"), "gfx1010", "RDNA 1"),
    (("radeon pro v520",), "gfx1011", "RDNA 1"),
    (("rx 5500",), "gfx1012", "RDNA 1"),
    (("mi300", "mi325"), "gfx942", "CDNA 3"),
    (("mi350", "mi355"), "gfx950", "CDNA 4"),
    (("mi250", "mi210"), "gfx90a", "CDNA 2"),
    (("mi100",), "gfx908", "CDNA 1"),
    (("radeon pro vii", "radeon vii", "mi50", "mi60"), "gfx906", "GCN 5.1"),
    (("rx vega", "vega 64", "vega 56", "vega frontier", "mi25"), "gfx900", "GCN 5.0"),
]

PCI_DEV_TO_VRAM: dict[str, int] = {
    "7590": 16384,
    "7550": 16384,
    "7551": 32768,
    "7580": 16384,
    "7581": 16384,
    "7591": 16384,
    "75a1": 16384,
    "75b0": 16384,
    "744c": 24576,
    "7448": 49152,
    "7449": 49152,
    "744a": 49152,
    "744b": 49152,
    "745e": 49152,
    "7460": 16384,
    "7461": 16384,
    "7470": 16384,
    "747e": 16384,
    "7480": 8192,
    "7481": 8192,
    "7483": 8192,
    "7487": 8192,
    "7489": 8192,
    "748b": 8192,
    "7499": 8192,
    "749f": 8192,
    "73a1": 16384,
    "73a2": 16384,
    "73a3": 32768,
    "73a5": 16384,
    "73ab": 32768,
    "73ae": 16384,
    "73af": 16384,
    "73bf": 16384,
    "73c3": 12288,
    "73ce": 12288,
    "73df": 12288,
    "73e0": 8192,
    "73e1": 8192,
    "73e3": 8192,
    "73ef": 8192,
    "73ff": 8192,
    "73e2": 8192,
    "73f0": 8192,
    "7421": 4096,
    "7422": 4096,
    "7423": 4096,
    "7424": 4096,
    "743f": 4096,
    "7310": 8192,
    "7312": 8192,
    "7319": 8192,
    "731b": 8192,
    "731f": 8192,
    "7360": 8192,
    "7362": 8192,
    "7340": 8192,
    "7341": 8192,
    "7347": 8192,
    "734f": 8192,
    "74a0": 65536,
    "74a1": 196608,
    "74a2": 131072,
    "74a5": 196608,
    "74a9": 196608,
    "74b5": 196608,
    "74b9": 196608,
    "74bd": 196608,
    "75a0": 196608,
    "75a3": 196608,
    "738c": 32768,
    "738e": 32768,
    "7408": 131072,
    "740c": 131072,
    "740f": 65536,
    "6860": 8192,
    "6861": 8192,
    "6862": 8192,
    "6863": 8192,
    "6864": 8192,
    "6867": 8192,
    "6868": 8192,
    "6869": 8192,
    "686b": 8192,
    "686c": 8192,
    "687f": 8192,
    "66a0": 16384,
    "66a1": 16384,
    "66a3": 16384,
    "66a7": 16384,
    "66af": 16384,
    "1636": 0,
    "1638": 0,
    "164c": 0,
    "15e7": 0,
    "15d8": 0,
}

GPU_NAME_TO_VRAM: list[tuple[tuple[str, ...], int]] = [
    (("rx 7900 xtx",), 24576),
    (("rx 7900 xt", "rx 7900 gre", "rx 7900"), 20480),
    (("rx 9070", "r9700", "r9600"), 16384),
    (("rx 7800", "rx 7700", "w7700"), 16384),
    (("rx 9060", "rx 7600", "w7600", "w7500", "rx 7400", "w7400"), 8192),
    (("rx 6950", "rx 6900", "rx 6800", "w6800", "v620"), 16384),
    (("rx 6750", "rx 6700"), 12288),
    (("rx 6650", "rx 6600", "w6600"), 8192),
    (("rx 6550", "rx 6500", "rx 6450", "rx 6400", "w6500", "w6400", "rx 6300", "w6300"), 4096),
    (("rx 5700", "rx 5600"), 8192),
    (("rx 5500",), 8192),
    (("radeon vii", "vega 20", "vega 64", "vega 56", "vega frontier"), 8192),
    (("mi300", "mi325"), 196608),
    (("mi350", "mi355"), 196608),
    (("mi250", "mi210"), 131072),
    (("mi100",), 32768),
]


@dataclass
class GpuInfo:
    name: str
    gfx: str
    family: str
    support: str
    integrated: bool
    vram_mb: int
    pnp_id: str = ""
    driver: str = ""
    source: str = ""
    score: int = 0


def _log(*args, **kwargs) -> None:
    print(*args, file=sys.stderr, **kwargs)


def _parse_gpu_table(output: str) -> list[dict[str, str]]:
    lines = [line for line in output.splitlines() if line.strip()]
    if len(lines) < 2:
        return []
    gpus: list[dict[str, str]] = []
    try:
        for row in csv.DictReader(lines):
            name = (row.get("Name") or "").strip()
            pnp = (row.get("PNPDeviceID") or "").strip()
            if not name:
                continue
            is_amd = "VEN_1002" in pnp.upper() or "AMD" in name.upper() or "RADEON" in name.upper()
            if is_amd:
                gpus.append({
                    "name": name,
                    "pnp_id": pnp,
                    "adapter_ram": (row.get("AdapterRAM") or "").strip(),
                    "driver": (row.get("DriverVersion") or "").strip(),
                })
    except csv.Error:
        pass
    return gpus


def _detect_powershell() -> list[dict[str, str]]:
    try:
        command = (
            "Get-CimInstance Win32_VideoController | "
            "Select-Object Name,PNPDeviceID,AdapterRAM,DriverVersion | "
            "ConvertTo-Csv -NoTypeInformation"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True, text=True, check=False, timeout=15,
        )
        if result.returncode == 0:
            return _parse_gpu_table(result.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return []


def _detect_wmic() -> list[dict[str, str]]:
    try:
        result = subprocess.run(
            ["wmic", "path", "win32_videocontroller", "get", "name,pnpdeviceid,adapterram,driverversion", "/format:csv"],
            capture_output=True, text=True, check=False, timeout=10,
        )
        if result.returncode == 0:
            return _parse_gpu_table(result.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return []


def _device_id(gpu: dict[str, str]) -> str | None:
    match = re.search(r"DEV_([0-9A-Fa-f]{4})", gpu.get("pnp_id", ""))
    return match.group(1).lower() if match else None


def _match_gpu(gpu: dict[str, str]) -> tuple[str | None, str, str]:
    device_id = _device_id(gpu)
    if device_id and device_id in PCI_DEV_TO_GFX:
        gfx, description = PCI_DEV_TO_GFX[device_id]
        info = target_info(gfx)
        return gfx, info.family if info else description, "pci"
    name = gpu["name"].lower()
    for patterns, gfx, family in GPU_NAME_TO_GFX:
        if any(pattern in name for pattern in patterns):
            return gfx, family, "name"
    return None, "", ""


def _match_vram(gpu: dict[str, str], info: TargetInfo | None) -> int:
    device_id = _device_id(gpu)
    if device_id and device_id in PCI_DEV_TO_VRAM:
        return PCI_DEV_TO_VRAM[device_id]
    name = gpu["name"].lower()
    for patterns, vram in GPU_NAME_TO_VRAM:
        if any(pattern in name for pattern in patterns):
            return vram
    raw = gpu.get("adapter_ram", "")
    if raw:
        try:
            value = int(raw) & 0xFFFFFFFF
            if 512 <= value <= 0xFFFFFFFF:
                return int(value / (1024 * 1024))
        except ValueError:
            pass
    return 0 if info and info.integrated else 0


def _system_ram_mb() -> int:
    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(MemoryStatus)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return int(status.ullTotalPhys // (1024 * 1024))
    return 0


def _profile(info: TargetInfo, vram_mb: int, system_ram_mb: int) -> str:
    if info.support == "linux-only":
        return "cpu"
    if info.integrated:
        if info.gfx == "gfx1151" and system_ram_mb >= 32768:
            return "balanced"
        return "low"
    if vram_mb >= 16384:
        mode = "high"
    elif vram_mb >= 8192:
        mode = "balanced"
    elif vram_mb >= 6144:
        mode = "medium"
    else:
        mode = "low"
    if info.support == "build":
        order = ["low", "medium", "balanced", "high"]
        mode = order[max(0, order.index(mode) - 1)]
    return mode


def _detector_side(info: TargetInfo, vram_mb: int) -> int:
    if info.integrated:
        return min(info.detector_max_side, 1024)
    if vram_mb >= 16384:
        return info.detector_max_side
    if vram_mb >= 8192:
        return min(info.detector_max_side, 1280)
    if vram_mb >= 6144:
        return min(info.detector_max_side, 1024)
    return 1024


def _make_info(gpu: dict[str, str], gfx: str, family: str, source: str) -> GpuInfo | None:
    info = target_info(gfx)
    if info is None:
        return None
    vram = _match_vram(gpu, info)
    integrated = info.integrated
    score = (0 if integrated else 1000) + min(vram, 262144) // 1024
    if info.support == "release":
        score += 100
    elif info.support == "build":
        score += 25
    if source == "pci":
        score += 20
    if any(token in gpu["name"].lower() for token in ("rx ", "pro ", "instinct")):
        score += 10
    return GpuInfo(
        name=gpu["name"],
        gfx=gfx,
        family=family,
        support=info.support,
        integrated=integrated,
        vram_mb=vram,
        pnp_id=gpu.get("pnp_id", ""),
        driver=gpu.get("driver", ""),
        source=source,
        score=score,
    )


def detect_amd_gpus() -> list[GpuInfo]:
    gpus = _detect_powershell() or _detect_wmic()
    found: list[GpuInfo] = []
    for gpu in gpus:
        gfx, family, source = _match_gpu(gpu)
        if not gfx:
            continue
        info = _make_info(gpu, gfx, family, source)
        if info is not None:
            found.append(info)
    return found


def _torch_gpu() -> GpuInfo | None:
    try:
        import torch
        if not torch.cuda.is_available() or not getattr(torch.version, "hip", None):
            return None
        props = torch.cuda.get_device_properties(0)
        gfx = normalize_gfx(getattr(props, "gcnArchName", ""))
        if not gfx:
            return None
        info = target_info(gfx)
        if info is None:
            return None
        return GpuInfo(
            name=torch.cuda.get_device_name(0),
            gfx=gfx,
            family=info.family,
            support=info.support,
            integrated=info.integrated,
            vram_mb=int(props.total_memory // (1024 * 1024)),
            source="torch",
            score=10_000,
        )
    except Exception:
        return None


def _best(found: list[GpuInfo]) -> GpuInfo | None:
    usable = [gpu for gpu in found if gpu.support in ("release", "build")]
    return max(usable, key=lambda gpu: gpu.score) if usable else None


def _write_config(path: Path, info: GpuInfo, system_ram_mb: int) -> None:
    profile = _profile(target_info(info.gfx), info.vram_mb, system_ram_mb)
    target = target_info(info.gfx)
    values = {
        "AMD_GFX": info.gfx,
        "AMD_GPU_NAME": info.name,
        "AMD_VRAM_MB": str(info.vram_mb),
        "AMD_FAMILY": info.family,
        "AMD_GPU_SUPPORT": info.support,
        "AMD_GPU_INTEGRATED": "1" if info.integrated else "0",
        "AMD_PROFILE": profile,
        "AMD_FP16": "1" if target and target.fp16 else "0",
        "AMD_QUANTIZE": "16" if target and target.fp16 else "32",
        "AMD_DETECT_MAX_SIDE": str(_detector_side(target, info.vram_mb) if target else 1280),
        "SPECTRA_MATTE_INPUT_SIZE": "2048" if info.vram_mb >= 12288 and not info.integrated else "1024",
        "AMD_AOTRITON": "1" if target and target.aotriton else "0",
        "AMD_HSA_OVERRIDE": os.environ.get("AMD_HSA_OVERRIDE", "").strip(),
        "AMD_DRIVER": info.driver,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{key}={value}\n" for key, value in values.items()), encoding="ascii")


def _write_launcher_config(path: Path, profile: str, system_ram_mb: int, force: bool) -> None:
    if path.exists() and not force:
        return
    threads = min(32, max(1, os.cpu_count() or 4))
    ram = int(system_ram_mb * 0.90) if system_ram_mb else 0
    lines = ["vram_mode=" + profile, "cpu_threads=", f"cpu_ram_mb={ram}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    if profile == "cpu":
        _log("No Windows-supported AMD GPU: launcher seeded in CPU mode")


def _choose(gpus: list[GpuInfo], override: str | None) -> GpuInfo | None:
    if override:
        gfx = normalize_gfx(override)
        info = target_info(gfx)
        if info is None or info.support not in ("release", "build"):
            _log(f"Unknown or unsupported Windows override target: {override}")
            return None
        name = info.family + " (manual target)"
        return GpuInfo(
            name=name,
            gfx=info.gfx,
            family=info.family,
            support=info.support,
            integrated=info.integrated,
            vram_mb=0,
            source="override",
            score=10_000,
        )
    return _best(gpus)


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect a Windows AMD GPU for TheRock ROCm")
    parser.add_argument("--gfx", default=None, help="explicit gfx target (for example gfx1032)")
    parser.add_argument("--write-config", type=Path, default=None)
    parser.add_argument("--write-launcher-config", type=Path, default=None)
    parser.add_argument("--force-launcher-config", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--from-torch", action="store_true")
    parser.add_argument("--require-supported", action="store_true")
    args = parser.parse_args()

    override = (args.gfx or os.environ.get("AMD_GFX_OVERRIDE") or "").strip()
    gpus = detect_amd_gpus()
    torch_gpu = _torch_gpu() if args.from_torch else None
    if torch_gpu is not None:
        gpus = [torch_gpu] + [gpu for gpu in gpus if gpu.gfx != torch_gpu.gfx]
    chosen = _choose(gpus, override)
    if chosen is None:
        for gpu in gpus:
            _log(
                f"Found {gpu.name}: {gpu.gfx} ({gpu.family}) - "
                f"{SUPPORT_LABELS.get(gpu.support, gpu.support)}"
            )
        _log("No Windows-supported AMD GPU found.")
        _log("Supported targets: " + ", ".join(supported_windows_targets()))
        return 1
    if args.require_supported and chosen.support not in ("release", "build"):
        _log(f"{chosen.gfx} is {support_label(chosen.support)} on Windows")
        return 1

    system_ram_mb = _system_ram_mb()
    target = target_info(chosen.gfx)
    profile = _profile(target, chosen.vram_mb, system_ram_mb) if target else "low"
    side = _detector_side(target, chosen.vram_mb) if target else 1024

    if args.write_config:
        _write_config(args.write_config, chosen, system_ram_mb)
    if args.write_launcher_config:
        _write_launcher_config(args.write_launcher_config, profile, system_ram_mb, args.force_launcher_config)

    payload = asdict(chosen)
    payload.update({
        "profile": profile,
        "detector_max_side": side,
        "matte_input_size": 2048 if chosen.vram_mb >= 12288 and not chosen.integrated else 1024,
        "quantize": 16 if target and target.fp16 else 32,
        "support_label": support_label(chosen.support),
    })
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    print(chosen.gfx)
    print(chosen.vram_mb)
    print(os.environ.get("HSA_OVERRIDE_GFX_VERSION", "").strip())
    print(chosen.name)
    print(chosen.family)
    print(chosen.support)
    print(profile)
    print(side)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        _log(f"Fatal: {exc}")
        sys.exit(1)
