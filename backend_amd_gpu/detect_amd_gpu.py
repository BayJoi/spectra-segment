import csv
import re
import subprocess
import sys

PCI_DEV_TO_GFX: dict[str, tuple[str, str, bool]] = {
    "7590": ("gfx1200", "Navi 44 (RX 9060 XT)", True),
    "7550": ("gfx1201", "Navi 48 (RX 9070/9070 XT/9070 GRE)", True),
    "7551": ("gfx1201", "Navi 48 (Radeon AI PRO R9700)", True),
    "7580": ("gfx1201", "Navi 48 (RX 9070 XT)", True),
    "7581": ("gfx1201", "Navi 48 (RX 9070)", True),
    "7591": ("gfx1201", "Navi 44 (RX 9060 XT)", True),
    "75a1": ("gfx1201", "Navi 48 (RX 9070 GRE)", True),
    "75b0": ("gfx1201", "Navi 48 (RX 9070 XT)", True),
    "150e": ("gfx1150", "Strix Point (880M/890M)", True),
    "1586": ("gfx1151", "Strix Halo (8050S/8060S)", True),
    "1114": ("gfx1152", "Krackan Point (840M/860M)", True),
    "1590": ("gfx1150", "Strix Point (880M)", True),
    "1591": ("gfx1150", "Strix Point (890M)", True),
    "15d0": ("gfx1152", "Krackan Point (860M)", True),
    "744c": ("gfx1100", "Navi 31 (RX 7900 XT/XTX/GRE/7900M)", True),
    "7448": ("gfx1100", "Navi 31 (Pro W7900)", True),
    "7449": ("gfx1100", "Navi 31 (Pro W7800 48GB)", True),
    "744a": ("gfx1100", "Navi 31 (Pro W7900 Dual Slot)", True),
    "744b": ("gfx1100", "Navi 31 (Pro W7900D)", True),
    "745e": ("gfx1100", "Navi 31 (Pro W7800)", True),
    "7460": ("gfx1101", "Navi 32 (Pro V710)", True),
    "7461": ("gfx1101", "Navi 32 (Pro V710)", True),
    "7470": ("gfx1101", "Navi 32 (Pro W7700)", True),
    "747e": ("gfx1101", "Navi 32 (RX 7700 XT/7800 XT)", True),
    "7480": ("gfx1102", "Navi 33 (RX 7600 series / Pro W7600)", True),
    "7481": ("gfx1102", "Navi 33", True),
    "7483": ("gfx1102", "Navi 33 (RX 7600M/7600M XT)", True),
    "7487": ("gfx1102", "Navi 33", True),
    "7489": ("gfx1102", "Navi 33 (Pro W7500)", True),
    "748b": ("gfx1102", "Navi 33", True),
    "7499": ("gfx1102", "Navi 33 (RX 7400/7300/Pro W7400)", True),
    "749f": ("gfx1102", "Navi 33 (RX 7500)", True),
    "15bf": ("gfx1103", "Phoenix1 (780M/760M/740M)", True),
    "15c8": ("gfx1103", "Phoenix2 (780M/760M/740M)", True),
    "164f": ("gfx1103", "Phoenix (780M/760M/740M)", True),
    "73a1": ("gfx1030", "Navi 21 (Pro V620)", True),
    "73a2": ("gfx1030", "Navi 21 (Pro W6900X)", True),
    "73a3": ("gfx1030", "Navi 21 (Pro W6800)", True),
    "73a5": ("gfx1030", "Navi 21 (RX 6950 XT)", True),
    "73ab": ("gfx1030", "Navi 21 (Pro W6800X/W6800X Duo)", True),
    "73ae": ("gfx1030", "Navi 21 (Pro V620 MxGPU)", True),
    "73af": ("gfx1030", "Navi 21 (RX 6900 XT)", True),
    "73bf": ("gfx1030", "Navi 21 (RX 6800/6800 XT/6900 XT)", True),
    "73c3": ("gfx1031", "Navi 22", True),
    "73ce": ("gfx1031", "Navi 22 (SRIOV MxGPU)", True),
    "73df": ("gfx1031", "Navi 22 (RX 6700/6750 XT/6800M/6850M XT)", True),
    "73e0": ("gfx1032", "Navi 23", True),
    "73e1": ("gfx1032", "Navi 23 (Pro W6600M)", True),
    "73e3": ("gfx1032", "Navi 23 (Pro W6600)", True),
    "73ef": ("gfx1032", "Navi 23 (RX 6650 XT/6700S/6800S)", True),
    "73ff": ("gfx1032", "Navi 23 (RX 6600/6600 XT/6600M)", True),
    "73e2": ("gfx1032", "Navi 23 (RX 6600 OEM)", True),
    "73f0": ("gfx1032", "Navi 23 (RX 6650 XT OEM)", True),
    "163f": ("gfx1033", "Van Gogh (Steam Deck APU)", True),
    "7421": ("gfx1034", "Navi 24 (Pro W6500M)", True),
    "7422": ("gfx1034", "Navi 24 (Pro W6400)", True),
    "7423": ("gfx1034", "Navi 24 (Pro W6300/W6300M)", True),
    "7424": ("gfx1034", "Navi 24 (RX 6300)", True),
    "743f": ("gfx1034", "Navi 24 (RX 6400/6500 XT/6500M)", True),
    "1681": ("gfx1035", "Rembrandt (680M/660M)", True),
    "1506": ("gfx1036", "Mendocino (610M)", True),
    "7310": ("gfx1010", "Navi 10 (Pro W5700X)", True),
    "7312": ("gfx1010", "Navi 10 (Pro W5700)", True),
    "7319": ("gfx1010", "Navi 10 (Pro 5700 XT)", True),
    "731b": ("gfx1010", "Navi 10 (Pro 5700)", True),
    "731f": ("gfx1010", "Navi 10 (RX 5600/5700 series)", True),
    "7360": ("gfx1011", "Navi 12 (Pro 5600M/V520/BC-160)", True),
    "7362": ("gfx1011", "Navi 12 (Pro V520/V540)", True),
    "7340": ("gfx1012", "Navi 14 (RX 5500/5500M/Pro 5300)", True),
    "7341": ("gfx1012", "Navi 14 (Pro W5500)", True),
    "7347": ("gfx1012", "Navi 14 (Pro W5500M)", True),
    "734f": ("gfx1012", "Navi 14 (Pro W5300M)", True),
    "74a0": ("gfx942", "Aqua Vanjaram (MI300A)", True),
    "74a1": ("gfx942", "Aqua Vanjaram (MI300X)", True),
    "74a2": ("gfx942", "Aqua Vanjaram (MI308X)", True),
    "74a5": ("gfx942", "Aqua Vanjaram (MI325X)", True),
    "74a9": ("gfx942", "Aqua Vanjaram (MI300X HF)", True),
    "74b5": ("gfx942", "Aqua Vanjaram (MI300X VF)", True),
    "74b9": ("gfx942", "Aqua Vanjaram (MI325X VF)", True),
    "74bd": ("gfx942", "Aqua Vanjaram (MI300X HF)", True),
    "75a0": ("gfx950", "Aqua Vanjaram (MI350X)", True),
    "75a3": ("gfx950", "Aqua Vanjaram (MI355X)", True),
    "738c": ("gfx908", "Arcturus (MI100)", True),
    "738e": ("gfx908", "Arcturus (MI100)", True),
    "7408": ("gfx90a", "Aldebaran (MI250X)", True),
    "740c": ("gfx90a", "Aldebaran (MI250X/MI250)", True),
    "740f": ("gfx90a", "Aldebaran (MI210)", True),
    "6860": ("gfx900", "Vega 10 (Instinct MI25/V340/V320)", True),
    "6861": ("gfx900", "Vega 10 (Pro WX 9100)", True),
    "6862": ("gfx900", "Vega 10 (Pro SSG)", True),
    "6863": ("gfx900", "Vega 10 (Vega Frontier Edition)", True),
    "6864": ("gfx900", "Vega 10 (Pro V340/Instinct MI25x2)", True),
    "6867": ("gfx900", "Vega 10 (Pro Vega 56)", True),
    "6868": ("gfx900", "Vega 10 (Pro WX 8100/8200)", True),
    "6869": ("gfx900", "Vega 10 (Pro Vega 48)", True),
    "686b": ("gfx900", "Vega 10 (Pro Vega 64X)", True),
    "686c": ("gfx900", "Vega 10 (Instinct MI25 MxGPU)", True),
    "687f": ("gfx900", "Vega 10 (RX Vega 56/64)", True),
    "66a0": ("gfx906", "Vega 20 (Pro / Instinct MI50)", True),
    "66a1": ("gfx906", "Vega 20 (Pro VII / Instinct MI50)", True),
    "66a3": ("gfx906", "Vega 20 (Pro Vega II / Vega II Duo)", True),
    "66a7": ("gfx906", "Vega 20 (Pro Vega 20)", True),
    "66af": ("gfx906", "Vega 20 (Radeon VII)", True),
}

PCI_DEV_TO_VRAM: dict[str, int] = {
    "7590": 8192,
    "7550": 16384,
    "7551": 16384,
    "7580": 16384,
    "7581": 16384,
    "7591": 8192,
    "75a1": 16384,
    "75b0": 16384,
    "744c": 24576,
    "7448": 49152,
    "7449": 49152,
    "744a": 24576,
    "744b": 24576,
    "745e": 32768,
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
    "7499": 4096,
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
    "7360": 16384,
    "7362": 16384,
    "7340": 4096,
    "7341": 4096,
    "7347": 4096,
    "734f": 4096,
    "6860": 8192,
    "6861": 8192,
    "6862": 16384,
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
}

GPU_NAME_TO_VRAM: list[tuple[list[str], int]] = [
    (["rx 9060"], 8192),
    (["rx 9070", "r9700", "r9600"], 16384),
    (["rx 7900", "w7900", "w7800"], 24576),
    (["rx 7800", "rx 7700", "w7700"], 16384),
    (["rx 7600", "w7600", "w7500", "rx 7400", "w7400"], 8192),
    (["rx 6950", "rx 6900", "rx 6800", "w6800", "v620"], 16384),
    (["rx 6750", "rx 6700"], 12288),
    (["rx 6650", "rx 6600", "w6600"], 8192),
    (["rx 6550", "rx 6500", "rx 6450", "rx 6400", "w6500", "w6400", "rx 6300", "w6300"], 4096),
    (["rx 5700", "rx 5600"], 8192),
    (["rx 5500"], 4096),
    (["radeon vii", "vega 20", "vega 64", "vega 56", "vega frontier"], 8192),
]

GPU_NAME_TO_GFX: list[tuple[list[str], str, str, bool]] = [
    (["rx 9060"], "gfx1200", "RDNA 4", True),
    (["rx 9070", "r9700", "r9600"], "gfx1201", "RDNA 4", True),
    (["890m", "880m"], "gfx1150", "RDNA 3.5 (Strix)", True),
    (["8060s", "8050s", "8040s"], "gfx1151", "RDNA 3.5 (Strix Halo)", True),
    (["860m", "840m", "820m"], "gfx1152", "RDNA 3.5 (Krackan)", True),
    (["rx 7900", "w7900", "w7800"], "gfx1100", "RDNA 3", True),
    (["rx 7800", "rx 7700", "w7700"], "gfx1101", "RDNA 3", True),
    (["rx 7700s", "rx 7650", "rx 7600", "w7600", "w7500", "rx 7400", "w7400"], "gfx1102", "RDNA 3", True),
    (["780m", "760m", "740m"], "gfx1103", "RDNA 3 (Phoenix)", True),
    (["rx 6950", "rx 6900", "rx 6800", "w6800", "v620"], "gfx1030", "RDNA 2", True),
    (["rx 6750", "rx 6700", "rx 6800m", "rx 6700m", "rx 6800s", "rx 6700s"], "gfx1031", "RDNA 2", True),
    (["rx 6650", "rx 6600", "w6600", "rx 6650m", "rx 6600m", "rx 6600s"], "gfx1032", "RDNA 2", True),
    (["van gogh", "amd custom apu 0405"], "gfx1033", "RDNA 2 (Van Gogh)", True),
    (["rx 6550", "rx 6500", "rx 6450", "rx 6400", "w6500", "w6400", "rx 6300", "w6300"], "gfx1034", "RDNA 2", True),
    (["680m", "660m"], "gfx1035", "RDNA 2 (Rembrandt)", True),
    (["610m"], "gfx1036", "RDNA 2 (Mendocino)", True),
    (["rx 5700", "rx 5600"], "gfx1010", "RDNA 1", True),
    (["radeon pro v520"], "gfx1011", "RDNA 1 (Navi 12)", True),
    (["rx 5500"], "gfx1012", "RDNA 1", True),
    (["radeon pro vii"], "gfx906", "Radeon Pro VII / Vega 20", True),
    (["mi300", "mi325"], "gfx942", "CDNA 3 (MI300/MI325)", True),
    (["mi350", "mi355"], "gfx950", "CDNA 4 (MI350/MI355)", True),
    (["mi250", "mi210"], "gfx90a", "CDNA 2 (MI200)", True),
    (["mi100"], "gfx908", "CDNA 1 (MI100)", True),
    (["radeon vii", "vega 20"], "gfx906", "Vega 20 / GCN5", True),
    (["rx vega", "vega 64", "vega 56", "vega frontier"], "gfx900", "Vega 10 / GCN5", True),
]


def _log(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def _parse_gpu_table(output: str) -> list[dict[str, str]]:
    gpus = []
    lines = [l for l in output.splitlines() if l.strip()]
    if len(lines) < 2:
        return gpus
    try:
        for row in csv.DictReader(lines):
            name = (row.get("Name") or "").strip()
            pnp = (row.get("PNPDeviceID") or "").strip()
            if name and ("AMD" in name or "Radeon" in name or "Radeon" in name.title()):
                gpus.append({"name": name, "pnp_id": pnp})
    except csv.Error:
        pass
    return gpus


def _detect_wmic() -> list[dict[str, str]]:
    try:
        r = subprocess.run(
            ["wmic", "path", "win32_videocontroller", "get", "name,pnpdeviceid", "/format:csv"],
            capture_output=True, text=True, check=False, timeout=10,
        )
        if r.returncode == 0:
            return _parse_gpu_table(r.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass
    return []


def _detect_powershell() -> list[dict[str, str]]:
    try:
        ps_cmd = (
            "Get-CimInstance Win32_VideoController | "
            "Select-Object Name,PNPDeviceID | ConvertTo-Csv -NoTypeInformation"
        )
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            capture_output=True, text=True, check=False, timeout=10,
        )
        if r.returncode == 0:
            return _parse_gpu_table(r.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass
    return []


def _match_gpu(gpu: dict[str, str]) -> tuple[str | None, str | None, bool]:
    pnp = gpu.get("pnp_id", "")
    m = re.search(r"DEV_([0-9A-Fa-f]{4})", pnp)
    if m:
        hit = PCI_DEV_TO_GFX.get(m.group(1).lower())
        if hit:
            return hit
    name_lower = gpu["name"].lower()
    for patterns, gfx, arch, supported in GPU_NAME_TO_GFX:
        for pat in patterns:
            if pat in name_lower:
                return gfx, arch, supported
    return None, None, False


def _match_vram(gpu: dict[str, str]) -> int:
    pnp = gpu.get("pnp_id", "")
    m = re.search(r"DEV_([0-9A-Fa-f]{4})", pnp)
    if m:
        vram = PCI_DEV_TO_VRAM.get(m.group(1).lower())
        if vram:
            return vram
    name_lower = gpu["name"].lower()
    for patterns, vram in GPU_NAME_TO_VRAM:
        for pat in patterns:
            if pat in name_lower:
                return vram
    return 0


def gfx_override_value(gfx: str | None) -> str | None:
    if not gfx:
        return None
    if gfx.startswith("gfx101"):
        return "10.1.0"
    if gfx.startswith("gfx103"):
        return "10.3.0"
    if gfx.startswith("gfx110"):
        return "11.0.0"
    return None


def detect_amd_gpu() -> tuple[str | None, int]:
    _log("AMD GPU detection...")
    gpus = _detect_powershell()
    if not gpus:
        _log("PowerShell query failed, trying wmic fallback...")
        gpus = _detect_wmic()
    if not gpus:
        _log("No AMD GPU detected")
        return None, 0
    _log(f"Found {len(gpus)} AMD GPU(s)")
    for g in gpus:
        _log(f"  {g['name']}  [{g.get('pnp_id', '')}]")
    best: tuple[int, dict[str, str], str, str] | None = None
    for g in gpus:
        gfx, arch, supported = _match_gpu(g)
        if gfx and not supported:
            _log(f"Unsupported GPU: {g['name']} ({arch})")
            continue
        if not gfx:
            continue
        m = re.search(r"DEV_([0-9A-Fa-f]{4})", g.get("pnp_id", ""))
        score = (2 if m and m.group(1).lower() in PCI_DEV_TO_GFX else 0) + (
            1 if ("rx " in g["name"].lower() or "pro " in g["name"].lower()) else 0
        )
        if best is None or score > best[0]:
            best = (score, g, gfx, arch)
    if best is not None:
        _, g, gfx, arch = best
        _log(f"Matched: {g['name']} -> {arch} ({gfx})")
        vram = _match_vram(g)
        if vram:
            _log(f"VRAM (best-effort): {vram} MB")
        return gfx, vram
    _log("No supported AMD GPU identified")
    return None, 0


if __name__ == "__main__":
    try:
        gfx, vram = detect_amd_gpu()
        if gfx:
            print(gfx)
            print(vram)
            print(gfx_override_value(gfx) or "")
            sys.exit(0)
        sys.exit(1)
    except Exception as e:
        _log(f"Fatal: {e}")
        sys.exit(1)
