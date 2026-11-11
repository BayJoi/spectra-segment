param(
    [Parameter(Mandatory = $true)]
    [string]$ModelWeights
)

$ErrorActionPreference = "Stop"

$dirs = @(
    "hf_cache",
    "hf_cache\hub",
    "hf_cache\modules",
    "hf_cache\xet",
    "hf_datasets",
    "sentence_transformers",
    "torch_cache",
    "triton",
    "numba",
    "matplotlib",
    "cache",
    "config",
    "data",
    "tmp",
    "tunableop_cache",
    "miopen\cache"
)

foreach ($dir in $dirs) {
    New-Item -ItemType Directory -Path (Join-Path $ModelWeights $dir) -Force | Out-Null
}

Write-Host "  [ OK ]  Model cache directories are ready."
