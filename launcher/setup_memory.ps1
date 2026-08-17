<#
    setup_memory.ps1 — Spectra Segment memory-profile setup.

    Reads/writes spectra_launcher.cfg (vram_mode / cpu_threads / cpu_ram_mb)
    and emits the derived environment variables to <ConfigPath>.env so the
    launcher .bat can `set` them for the backend process.

    Profiles control memory residency and the caching-allocator block
    budget (SPECTRA_VRAM_FRACTION, used for max_split_size_mb and eviction).
    The absolute safety ceiling (SPECTRA_VRAM_HARD_CAP, 90%) is applied via
    torch.cuda.set_per_process_memory_fraction(), so the process can NEVER
    cross the user's full VRAM while still letting workloads run on the GPU.

    Usage (from the launchers):
      powershell -NoProfile -ExecutionPolicy Bypass -File setup_memory.ps1 `
          -ConfigPath "D:\...\spectra_launcher.cfg" -Backend AMD -DetectedVramMb 8192
      # add -Reconfigure to force the interactive prompt

    All informational output goes to Write-Host (console only); the .env file
    is the only stdout-independent artifact the bat consumes.
#>
param(
    [string]$ConfigPath,
    [string]$Backend = "NVIDIA",
    [int]$DetectedVramMb = 0,
    [switch]$Reconfigure,
    [switch]$ForceCpu
)

$ErrorActionPreference = "Stop"

$ValidModes = @("high", "balanced", "medium", "low", "cpu")
$EnvFile = "$ConfigPath.env"

# VRAM budget per profile, as a fraction of total VRAM (medium = half, etc.).
# This controls allocator block sizing and residency/eviction (a soft target),
# NOT the hard ceiling - a workload may transiently exceed it to finish on GPU.
$ModeFraction = @{
    "high"     = 0.90
    "balanced" = 0.60
    "medium"   = 0.50
    "low"      = 0.30
}
# Absolute safety ceiling applied via set_per_process_memory_fraction() so the
# process can NEVER cross the user's full VRAM. Kept high enough that real
# workloads (e.g. SAM2 at 1024) still run on the GPU instead of CPU.
$HardCapFraction = 0.90
# Fallback split (MB) used when VRAM could not be detected.
$ModeFallbackMb = @{
    "high"     = 8192
    "balanced" = 4096
    "medium"   = 2048
    "low"      = 1024
}

function Read-Config {
    $cfg = @{}
    if (-not (Test-Path -LiteralPath $ConfigPath)) { return $cfg }
    foreach ($line in Get-Content -LiteralPath $ConfigPath) {
        $line = $line.Trim()
        if ($line -and -not $line.StartsWith("#")) {
            $idx = $line.IndexOf("=")
            if ($idx -gt 0) {
                $k = $line.Substring(0, $idx).Trim()
                $v = $line.Substring($idx + 1).Trim()
                $cfg[$k] = $v
            }
        }
    }
    return $cfg
}

function Write-Config {
    param([hashtable]$cfg)
    $dir = Split-Path -Parent $ConfigPath
    if ($dir -and -not (Test-Path -LiteralPath $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    $lines = @()
    foreach ($k in @("vram_mode", "cpu_threads", "cpu_ram_mb")) {
        if ($cfg.ContainsKey($k)) { $lines += "$k=$($cfg[$k])" }
    }
    Set-Content -LiteralPath $ConfigPath -Value $lines -Encoding ASCII
}

function Get-NvidiaVramMb {
    try {
        $out = & nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>$null
        if ($out) {
            $v = 0
            if ([int]::TryParse(($out | Select-Object -First 1).Trim(), [ref]$v)) { return $v }
        }
    } catch { }
    return 0
}

function Suggest-Mode {
    param([int]$vramMb)
    if ($vramMb -gt 0 -and $vramMb -lt 4096) { return "low" }
    if ($vramMb -ge 4096 -and $vramMb -lt 6144) { return "medium" }
    if ($vramMb -ge 6144 -and $vramMb -lt 16384) { return "balanced" }
    if ($vramMb -ge 16384) { return "high" }
    return "balanced"
}

function Get-TotalRamMb {
    try {
        $total = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory
        if ($total -and $total -gt 0) { return [int]($total / 1MB) }
    } catch { }
    return 0
}

function Get-CpuInfo {
    $name = "Unknown CPU"
    $cores = 0
    $logical = 0
    try {
        $cpus = @(Get-CimInstance Win32_Processor)
        if ($cpus.Count -gt 0) {
            $name = $cpus[0].Name
            if ([string]::IsNullOrWhiteSpace($name)) { $name = "Unknown CPU" }
            $cores = ($cpus | Measure-Object -Property NumberOfCores -Sum).Sum
            $logical = ($cpus | Measure-Object -Property NumberOfLogicalProcessors -Sum).Sum
        }
    } catch { }
    if ($logical -lt 1) { $logical = [Environment]::ProcessorCount }
    if ($cores -lt 1) { $cores = $logical }
    return @{ Name = $name; Cores = $cores; Logical = $logical }
}

function Resolve-Threads {
    param([string]$text, [int]$logical, [int]$cores)
    $t = $text.Trim().ToLower()
    if ([string]::IsNullOrWhiteSpace($t) -or $t -eq "all") { return "$logical" }
    if ($t -in @("phys", "physical", "p")) { return "$cores" }
    if ($t -in @("half")) { return "$([Math]::Max(1, [Math]::Ceiling($logical / 2)))" }
    $tv = 0
    if ([int]::TryParse($t, [ref]$tv) -and $tv -gt 0) { return "$tv" }
    return ""
}

function Get-SplitMb {
    param([string]$mode, [int]$vramMb)
    $frac = $ModeFraction[$mode]
    if (-not $frac) { return $ModeFallbackMb[$mode] }
    if ($vramMb -gt 0) {
        $split = [int][Math]::Floor($vramMb * $frac)
        return [Math]::Max(256, $split)
    }
    return $ModeFallbackMb[$mode]
}

function Emit-ProfileEnv {
    param(
        [string]$mode,
        [string]$threads,
        [string]$ram,
        [string]$backend,
        [int]$vramMb
    )
    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add("SPECTRA_VRAM_MODE=$mode")
    $lines.Add("SPECTRA_CPU_THREADS=$threads")
    $lines.Add("SPECTRA_CPU_RAM_MB=$ram")

    if ($mode -eq "cpu") {
        $lines.Add("SPECTRA_FORCE_CPU=1")
        $lines.Add("SPECTRA_VRAM_FRACTION=0")
        $lines.Add("SPECTRA_VRAM_HARD_CAP=0")
        $lines.Add("SAM2_OFFLOAD_ENCODER=0")
        $lines.Add("SAM3_IDLE_UNLOAD_TIMEOUT=300")
        $lines.Add("SPECTRA_NO_CO_RESIDENCY=0")
        if ($threads) {
            $lines.Add("SAM2_THREADS=$threads")
            $lines.Add("SAM3_THREADS=$threads")
            $lines.Add("SPECTRA_OMP_THREADS=$threads")
        }
    } else {
        $offload = 0
        $idle = 300
        $noCo = 0
        switch ($mode) {
            "high"   { $idle = 86400 }
            "medium" { $idle = 180 }
            "low"    { $offload = 1; $idle = 120; $noCo = 1 }
        }
        $splitMb = Get-SplitMb -mode $mode -vramMb $vramMb
        $conf = "garbage_collection_threshold:0.8,max_split_size_mb:$splitMb"
        $lines.Add("SPECTRA_FORCE_CPU=0")
        $lines.Add("SPECTRA_VRAM_FRACTION=$($ModeFraction[$mode])")
        $lines.Add("SPECTRA_VRAM_HARD_CAP=$HardCapFraction")
        $lines.Add("SAM2_OFFLOAD_ENCODER=$offload")
        $lines.Add("SAM3_IDLE_UNLOAD_TIMEOUT=$idle")
        $lines.Add("SPECTRA_NO_CO_RESIDENCY=$noCo")
        if ($backend -eq "AMD") {
            $lines.Add("PYTORCH_HIP_ALLOC_CONF=$conf")
            $lines.Add("PYTORCH_CUDA_ALLOC_CONF=$conf")
        } else {
            $lines.Add("PYTORCH_CUDA_ALLOC_CONF=$conf,expandable_segments:True")
        }
    }

    Set-Content -LiteralPath $EnvFile -Value $lines -Encoding ASCII
}

# --- main ---
$cfg = Read-Config

if (-not $Reconfigure -and $cfg.ContainsKey("vram_mode") -and ($cfg["vram_mode"] -in $ValidModes) -and (-not $ForceCpu -or $cfg["vram_mode"] -eq "cpu")) {
    $mode = $cfg["vram_mode"]
    $threads = ""
    if ($cfg.ContainsKey("cpu_threads") -and [string]$cfg["cpu_threads"] -match '^\d+$') { $threads = [string]$cfg["cpu_threads"] }
    $ram = ""
    if ($cfg.ContainsKey("cpu_ram_mb")) {
        $rv = 0
        if ([int]::TryParse([string]$cfg["cpu_ram_mb"], [ref]$rv) -and $rv -gt 0) { $ram = "$rv" }
    }
    if ($Backend -eq "NVIDIA") { $vramMb = Get-NvidiaVramMb } else { $vramMb = $DetectedVramMb }
    Write-Host ""
    Write-Host "  [INFO]  Memory profile: $mode (from $ConfigPath)"
    Write-Host "          To change it, run the launcher with: -reconfigure"
    Emit-ProfileEnv -mode $mode -threads $threads -ram $ram -backend $Backend -vramMb $vramMb
    exit 0
}

$vramMb = 0
$vramDesc = "unknown"
if ($Backend -eq "NVIDIA") {
    $vramMb = Get-NvidiaVramMb
} else {
    $vramMb = $DetectedVramMb
}
if ($vramMb -gt 0) { $vramDesc = "$vramMb MB" }
$suggested = Suggest-Mode -vramMb $vramMb

$cpuInfo = Get-CpuInfo
$ramMb = Get-TotalRamMb

Write-Host ""
Write-Host "  === Memory profile configuration ==="
Write-Host "  Detected GPU VRAM : $vramDesc"
Write-Host "  Suggested profile : $suggested"
Write-Host ""
Write-Host "  Profiles:"
Write-Host "    high     - keep models resident, biggest memory budget (budget ~90% VRAM)"
Write-Host "    balanced - default behavior (budget ~60% VRAM)"
Write-Host "    medium   - ~50% VRAM budget, tighter caches"
Write-Host "    low      - ~30% VRAM budget, encoder offload, no co-residency"
Write-Host "    cpu      - force CPU (uses only RAM + threads)"
Write-Host ""
Write-Host "  The profile is a residency target; the backend still uses the GPU and"
Write-Host "  may transiently exceed it to finish the job. An absolute 90% ceiling"
Write-Host "  guarantees VRAM is never crossed; only workloads that cannot fit in 90%"
Write-Host "  of VRAM fall back to CPU encoding (slower, but full quality)."
Write-Host ""

$map = @{
    "h" = "high"; "high" = "high"
    "b" = "balanced"; "balanced" = "balanced"
    "m" = "medium"; "medium" = "medium"
    "l" = "low"; "low" = "low"
    "c" = "cpu"; "cpu" = "cpu"
}
$mode = ""
if ($ForceCpu) {
    $mode = "cpu"
} else {
    while ($mode -notin $ValidModes) {
        $modeInput = Read-Host "  Memory profile [$suggested]"
        if ([string]::IsNullOrWhiteSpace($modeInput)) { $modeInput = $suggested }
        $modeInput = $modeInput.Trim().ToLower()
        if ($map.ContainsKey($modeInput)) { $mode = $map[$modeInput] }
    }
}

$threads = ""
if ($mode -eq "cpu") {
    Write-Host ""
    Write-Host "  Detected CPU: $($cpuInfo.Name)"
    Write-Host "  Physical cores: $($cpuInfo.Cores)   Logical processors/threads: $($cpuInfo.Logical)"
    $tinput = Read-Host "  CPU threads [default: $($cpuInfo.Logical), or 'phys' / 'half' / a number]"
    $threads = Resolve-Threads -text $tinput -logical $cpuInfo.Logical -cores $cpuInfo.Cores
    if (-not $threads) { $threads = "$($cpuInfo.Logical)" }
}

$ram = ""
if ($ramMb -gt 0) {
    $rinput = Read-Host "  CPU RAM budget (MB) [$ramMb]"
    if (-not [string]::IsNullOrWhiteSpace($rinput)) {
        $rv = 0
        if ([int]::TryParse($rinput.Trim(), [ref]$rv) -and $rv -gt 0) { $ram = "$rv" }
    }
    if (-not $ram) { $ram = "$ramMb" }
}

$cfg = @{ "vram_mode" = $mode; "cpu_threads" = $threads; "cpu_ram_mb" = $ram }
Write-Config -cfg $cfg
Write-Host ""
Write-Host "  [ OK ]  Saved memory profile: $mode (threads: $threads, RAM: $ram MB)"
Write-Host "          To change later, run the launcher with: -reconfigure"
Emit-ProfileEnv -mode $mode -threads $threads -ram $ram -backend $Backend -vramMb $vramMb
exit 0
