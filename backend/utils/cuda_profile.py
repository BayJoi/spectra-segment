from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from functools import lru_cache

import torch

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CudaProfile:
    available: bool = False
    name: str = ""
    capability: tuple[int, int] = (0, 0)
    family: str = "unknown"
    total_memory_mb: int = 0
    driver: str = ""
    torch_version: str = ""
    cuda_version: str = ""
    arch_list: tuple[str, ...] = ()
    supported: bool = False
    support_reason: str = "CUDA is not available"
    fp16: bool = False
    bf16: bool = False
    tf32: bool = False

    @property
    def capability_text(self) -> str:
        return f"{self.capability[0]}.{self.capability[1]}"


def classify_family(capability: tuple[int, int]) -> str:
    major, minor = capability
    if major == 5:
        return "Maxwell"
    if major == 6:
        return "Pascal"
    if major == 7 and minor in (0, 2):
        return "Volta"
    if major == 7 and minor == 5:
        return "Turing"
    if major == 8 and minor == 0:
        return "Ampere"
    if major == 8 and minor in (6, 7):
        return "Ampere"
    if major == 8 and minor == 9:
        return "Ada Lovelace"
    if major == 9 and minor == 0:
        return "Hopper"
    if major == 10 and minor == 0:
        return "Blackwell"
    if major == 10 and minor == 3:
        return "Blackwell Ultra"
    if major == 10 and minor == 7:
        return "Rubin"
    if major == 11 and minor == 0:
        return "Blackwell (Jetson Thor)"
    if major == 12:
        return "Blackwell"
    return "unsupported"


def _parse_arch(entry: str) -> tuple[str, tuple[int, int]] | None:
    value = entry.strip().lower()
    kind = ""
    if value.startswith("sm_"):
        kind = "sm"
        value = value[3:]
    elif value.startswith("compute_"):
        kind = "compute"
        value = value[8:]
    else:
        return None
    value = value.rstrip("af")
    if not value.isdigit():
        return None
    if "." in value:
        major, minor = value.split(".", 1)
    elif len(value) >= 2:
        major, minor = value[:-1], value[-1]
    else:
        return None
    try:
        return kind, (int(major), int(minor))
    except ValueError:
        return None


def arch_supported(capability: tuple[int, int], arch_list: tuple[str, ...]) -> tuple[bool, str]:
    if not arch_list:
        return True, "PyTorch did not report an architecture list"

    parsed = [item for item in (_parse_arch(entry) for entry in arch_list) if item]
    if not parsed:
        return True, "PyTorch architecture list was not recognizable"

    exact = f"sm_{capability[0]}{capability[1]}"
    for kind, cc in parsed:
        if kind == "sm" and cc == capability:
            return True, f"exact {exact} kernel"
        if kind == "compute" and cc == capability:
            return True, f"compute_{capability[0]}{capability[1]} PTX"

    for kind, cc in parsed:
        if kind == "compute" and cc <= capability:
            return True, f"compute_{cc[0]}{cc[1]} PTX forward-compatible"

    same_family = capability[0] < 10 or capability[0] in (10, 12)
    if capability == (10, 7):
        same_family = False
    if same_family:
        for kind, cc in parsed:
            if kind == "sm" and cc[0] == capability[0] and cc[1] < capability[1]:
                if capability[0] == 7 and not (capability == (7, 2) and cc == (7, 0)):
                    continue
                return True, f"sm_{cc[0]}{cc[1]} binary-compatible kernel"

    return False, f"the installed PyTorch build has no kernels for {exact}"


def _nvidia_driver_version() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        line = (result.stdout or "").splitlines()
        return line[0].strip() if line else ""
    except Exception:
        return ""


def _unavailable_profile(reason: str) -> CudaProfile:
    return CudaProfile(
        torch_version=getattr(torch, "__version__", ""),
        cuda_version=getattr(torch.version, "cuda", "") or "",
        support_reason=reason,
    )


@lru_cache(maxsize=1)
def get_cuda_profile() -> CudaProfile:
    try:
        if not torch.cuda.is_available():
            reason = "CUDA is not available"
            if getattr(torch.version, "cuda", None):
                reason = (
                    "PyTorch has a CUDA build but CUDA is not available; "
                    "check that the NVIDIA driver meets the wheel's minimum version"
                )
            return _unavailable_profile(reason)
        if not getattr(torch.version, "cuda", None):
            return _unavailable_profile("CUDA is not available")
        props = torch.cuda.get_device_properties(0)
        capability = (int(props.major), int(props.minor))
        arch_list = tuple(str(item) for item in torch.cuda.get_arch_list())
        supported, reason = arch_supported(capability, arch_list)
        return CudaProfile(
            available=True,
            name=torch.cuda.get_device_name(0),
            capability=capability,
            family=classify_family(capability),
            total_memory_mb=int(props.total_memory // (1024 * 1024)),
            driver=_nvidia_driver_version(),
            torch_version=getattr(torch, "__version__", ""),
            cuda_version=getattr(torch.version, "cuda", "") or "",
            arch_list=arch_list,
            supported=supported,
            support_reason=reason,
            fp16=capability >= (7, 0),
            bf16=capability >= (8, 0),
            tf32=capability >= (8, 0),
        )
    except Exception as exc:
        LOGGER.debug("NVIDIA profile detection failed: %s", exc, exc_info=True)
        return _unavailable_profile(f"CUDA detection failed: {exc}")


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw not in ("0", "false", "no", "off")


def _set_fp32_precision(owner: object, value: str) -> bool:
    try:
        if hasattr(owner, "fp32_precision"):
            setattr(owner, "fp32_precision", value)
            return True
    except Exception:
        pass
    return False


def _enable_tf32(profile: CudaProfile) -> None:
    requested = _env_flag("SPECTRA_TF32", profile.tf32)
    if not requested:
        LOGGER.info("NVIDIA: TF32 disabled (SPECTRA_TF32=0)")
        return
    if not profile.tf32:
        LOGGER.warning("NVIDIA: TF32 requested but %s does not support it", profile.family)
        return

    matmul = getattr(torch.backends, "cuda", None)
    matmul = getattr(matmul, "matmul", None)
    cudnn = getattr(torch.backends, "cudnn", None)
    conv = getattr(cudnn, "conv", None)

    matmul_ok = bool(matmul is not None and _set_fp32_precision(matmul, "tf32"))
    if not matmul_ok and matmul is not None:
        try:
            matmul.allow_tf32 = True
            matmul_ok = True
        except Exception:
            pass

    conv_ok = bool(conv is not None and _set_fp32_precision(conv, "tf32"))
    if not conv_ok and cudnn is not None:
        try:
            cudnn.allow_tf32 = True
            conv_ok = True
        except Exception:
            pass

    if matmul_ok or conv_ok:
        LOGGER.info(
            "NVIDIA: TF32 enabled for fp32 %s",
            " and ".join(part for part, ok in (("matmul", matmul_ok), ("conv", conv_ok)) if ok),
        )
    else:
        LOGGER.warning("NVIDIA: could not enable TF32 on PyTorch %s", profile.torch_version)


def prepare_nvidia_environment() -> CudaProfile:
    if os.environ.get("SPECTRA_FORCE_CPU") == "1":
        os.environ.setdefault("SPECTRA_MATTE_INPUT_SIZE", "1024")
        return _unavailable_profile("CPU was explicitly requested")
    profile = get_cuda_profile()
    if not profile.available or not profile.supported:
        os.environ.setdefault("SPECTRA_MATTE_INPUT_SIZE", "1024")
        return profile

    quantize = "16" if profile.fp16 else "32"
    detector_fp16 = "1" if profile.fp16 else "0"
    matte_size = "2048" if profile.total_memory_mb >= 12288 else "1024"
    os.environ.setdefault("SAM2_QUANTIZE", quantize)
    os.environ.setdefault("SAM3_QUANTIZE", quantize)
    os.environ.setdefault("SPECTRA_DETECTOR_FP16", detector_fp16)
    os.environ.setdefault("SPECTRA_MATTE_FP16", detector_fp16)
    os.environ.setdefault("SPECTRA_MATTE_INPUT_SIZE", matte_size)
    return profile


_configured = False


def configure_nvidia_runtime(profile: CudaProfile | None = None) -> None:
    global _configured
    if _configured:
        return
    _configured = True
    if os.environ.get("SPECTRA_FORCE_CPU") == "1":
        return
    profile = profile or get_cuda_profile()
    if not profile.available or not profile.supported:
        return

    _enable_tf32(profile)
    try:
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cudnn, "benchmark_limit"):
            torch.backends.cudnn.benchmark_limit = 4
    except Exception:
        LOGGER.debug("NVIDIA: could not enable cuDNN benchmark", exc_info=True)
