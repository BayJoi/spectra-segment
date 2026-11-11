param(
    [string]$ComputeCapability = "",
    [string]$DriverVersion = ""
)

$ErrorActionPreference = "Stop"

function Parse-Version([string]$text) {
    $clean = ($text -replace "[^0-9.]", "").Trim(".")
    if (-not $clean) { return $null }
    $parsed = $null
    if ([version]::TryParse($clean, [ref]$parsed)) { return $parsed }
    return $null
}

$cc = Parse-Version $ComputeCapability
$driver = Parse-Version $DriverVersion

$mode = "cpu"
$index = "-"
$spec = "-"
$label = "CPU"
$reason = ""
$fallbackIndex = "-"
$fallbackSpec = "-"

$legacyIndex = "https://download.pytorch.org/whl/cu126"
$legacySpec = "torch==2.14.0 torchvision==0.29.0"
$modernIndex = "https://download.pytorch.org/whl/cu132"
$modernSpec = "torch==2.15.0 torchvision==0.29.2"
$rubinIndex = "https://download.pytorch.org/whl/cu134"
$rubinSpec = "torch==2.15.0 torchvision==0.29.2"

if ($null -eq $cc) {
    if ($null -ne $driver -and $driver.Major -ge 580) {
        $mode = "nvidia"
        $index = $modernIndex
        $spec = $modernSpec
        $label = "CUDA 13.2"
        $reason = "compute capability unavailable; selected CUDA 13.2 from driver $DriverVersion"
    } elseif ($null -ne $driver -and $driver.Major -ge 525) {
        $mode = "nvidia"
        $index = $legacyIndex
        $spec = $legacySpec
        $label = "CUDA 12.6 legacy"
        $reason = "compute capability unavailable; selected CUDA 12.6 from driver $DriverVersion"
    } else {
        $reason = "compute capability and a usable NVIDIA driver were not detected"
    }
} elseif ($cc.Major -lt 5) {
    $reason = "compute capability $ComputeCapability predates Maxwell and has no PyTorch CUDA wheel"
} elseif ($cc.Major -eq 10 -and $cc.Minor -eq 7) {
    if ($null -ne $driver -and $driver.Major -ge 615) {
        $mode = "nvidia"
        $index = $rubinIndex
        $spec = $rubinSpec
        $label = "CUDA 13.4 (Rubin)"
        $reason = "Rubin sm_10.7 requires the CUDA 13.4 build and an R615 driver"
    } else {
        $reason = "Rubin needs NVIDIA driver 615 or newer"
    }
} elseif (($cc.Major -eq 5) -or ($cc.Major -eq 6) -or ($cc.Major -eq 7 -and $cc.Minor -lt 5)) {
    if ($null -ne $driver -and $driver.Major -ge 525) {
        $mode = "nvidia"
        $index = $legacyIndex
        $spec = $legacySpec
        $label = "CUDA 12.6 legacy (Maxwell/Pascal/Volta)"
        $reason = "sm_$ComputeCapability is only covered by the PyTorch 2.14 CUDA 12.6 build"
    } else {
        $reason = "Maxwell/Pascal/Volta needs NVIDIA driver 525 or newer"
    }
} elseif (
    ($cc.Major -eq 10 -and $cc.Minor -le 3) -or
    ($cc.Major -eq 11) -or
    $cc.Major -eq 12
) {
    if ($null -ne $driver -and $driver.Major -ge 580) {
        $mode = "nvidia"
        $index = $modernIndex
        $spec = $modernSpec
        $label = "CUDA 13.2"
        $reason = "sm_$ComputeCapability is covered by the current PyTorch CUDA 13.2 build"
    } else {
        $reason = "Blackwell needs NVIDIA driver 580 or newer; CUDA 12.6 has no Blackwell kernels"
    }
} elseif (
    ($cc.Major -eq 7 -and $cc.Minor -ge 5) -or
    ($cc.Major -eq 8) -or
    ($cc.Major -eq 9)
) {
    if ($null -ne $driver -and $driver.Major -ge 580) {
        $mode = "nvidia"
        $index = $modernIndex
        $spec = $modernSpec
        $label = "CUDA 13.2"
        $reason = "sm_$ComputeCapability is covered by the current PyTorch CUDA 13.2 build"
        $fallbackIndex = $legacyIndex
        $fallbackSpec = $legacySpec
    } elseif ($null -ne $driver -and $driver.Major -ge 525) {
        $mode = "nvidia"
        $index = $legacyIndex
        $spec = $legacySpec
        $label = "CUDA 12.6 legacy (driver compatibility)"
        $reason = "driver $DriverVersion predates CUDA 13; using the CUDA 12.6 build"
    } else {
        $reason = "NVIDIA driver 525 or newer is required"
    }
} else {
    $reason = "compute capability $ComputeCapability is not in the supported PyTorch range"
}

Write-Output "$mode|$index|$spec|$label|$reason|$fallbackIndex|$fallbackSpec"
