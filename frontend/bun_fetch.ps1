param(
    [Parameter(Mandatory = $true)]
    [string]$Dir
)

$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

if (Test-Path -LiteralPath (Join-Path $Dir 'bun.exe')) {
    Write-Output '[BUN] Using existing project bun.exe (offline).'
    exit 0
}

$apiUrl = 'https://api.github.com/repos/oven-sh/bun/releases/tags/canary'
$zipUrl = 'https://github.com/oven-sh/bun/releases/download/canary/bun-windows-x64.zip'

$zipFile = Join-Path $Dir 'bun.zip'
$marker  = Join-Path $Dir '.bun_sha256'
$exe     = Join-Path $Dir 'bun.exe'

$web = New-Object System.Net.WebClient
$web.Headers.Add('User-Agent', 'spectra-segment')

$expected = $null
try {
    $json = $web.DownloadString($apiUrl)
    $rel = $json | ConvertFrom-Json
    $expected = ($rel.assets | Where-Object { $_.name -eq 'bun-windows-x64.zip' } | Select-Object -First 1).digest
    if ($expected) {
        $expected = ($expected -replace '^sha256:', '').Trim().ToLower()
    }
} catch {
    $expected = $null
}

if (-not $expected) {
    if (Test-Path -LiteralPath $exe) {
        Write-Output '[BUN] Could not reach GitHub API; using existing project bun.exe.'
        exit 0
    }
    Write-Output '[BUN] Could not reach GitHub API to fetch the canary checksum.'
    exit 1
}

$applied = $null
if (Test-Path -LiteralPath $marker) {
    $applied = (Get-Content -LiteralPath $marker | Select-Object -First 1).Trim()
}
if ((Test-Path -LiteralPath $exe) -and ($applied -eq $expected)) {
    Write-Output "[BUN] Project bun.exe is up to date (canary sha $expected)."
    exit 0
}

Write-Output '[BUN] A new canary build is available. Downloading bun-windows-x64.zip...'
$installed = $false
for ($attempt = 1; $attempt -le 4; $attempt++) {
    if ($attempt -gt 1) {
        Write-Output "[BUN] Retrying download (attempt $attempt of 4)..."
        Start-Sleep -Seconds 8
    }
    try {
        $web.DownloadFile($zipUrl, $zipFile)
        $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $zipFile).Hash.ToLower()
        if ($actual -ne $expected) {
            Write-Output "[BUN] SHA-256 mismatch (expected $expected, got $actual) - GitHub CDN may be updating."
            Remove-Item -LiteralPath $zipFile -Force -ErrorAction SilentlyContinue
            continue
        }
        if (Test-Path -LiteralPath $exe) {
            Remove-Item -LiteralPath $exe -Force
        }
        Expand-Archive -LiteralPath $zipFile -DestinationPath $Dir -Force
        $sub = Join-Path $Dir 'bun-windows-x64'
        if (Test-Path -LiteralPath (Join-Path $sub 'bun.exe')) {
            Move-Item -LiteralPath (Join-Path $sub 'bun.exe') -Destination $exe -Force
            Remove-Item -LiteralPath $sub -Recurse -Force -ErrorAction SilentlyContinue
        } elseif (-not (Test-Path -LiteralPath $exe)) {
            throw 'bun.exe was not found in the downloaded archive'
        }
        Set-Content -LiteralPath $marker -Value $expected -Encoding ascii
        Remove-Item -LiteralPath $zipFile -Force -ErrorAction SilentlyContinue
        Write-Output "[BUN] Installed verified canary build (sha $expected)."
        $installed = $true
        break
    } catch {
        Write-Output "[BUN] Failed to fetch latest canary: $($_.Exception.Message)"
        Remove-Item -LiteralPath $zipFile -Force -ErrorAction SilentlyContinue
        break
    }
}
if (-not $installed) {
    exit 1
}
exit 0
