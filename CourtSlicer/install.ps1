<#
    CourtSlicer installer (Windows / PowerShell)

    Usage:
        .\install.ps1            # interactive
        .\install.ps1 -Yes       # non-interactive, auto-installs uv if missing
        .\install.ps1 -NoUv      # force pip instead of uv
#>

param(
    [switch]$Yes,
    [switch]$NoUv
)

$ErrorActionPreference = "Stop"
$RepoDir = Split-Path -Parent $MyInvocation.MyCommand.Path

function Info($msg)    { Write-Host "[info]  $msg" -ForegroundColor Cyan }
function Success($msg) { Write-Host "[ok]    $msg" -ForegroundColor Green }
function Warn($msg)    { Write-Host "[warn]  $msg" -ForegroundColor Yellow }
function Die($msg)     { Write-Host "[error] $msg" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "  CourtSlicer installer (Windows)"
Write-Host "  ---------------------------------"
Write-Host ""

# ── Python check ──────────────────────────────────────────────────────────────
Info "Checking Python..."
$PythonCmd = $null
foreach ($candidate in @("python", "py")) {
    if (Get-Command $candidate -ErrorAction SilentlyContinue) {
        try {
            $verOutput = & $candidate -c "import sys; print('%d%03d' % sys.version_info[:2])" 2>$null
            if ($LASTEXITCODE -eq 0 -and [int]$verOutput -ge 3010) {
                $PythonCmd = $candidate
                break
            }
        } catch {}
    }
}
if (-not $PythonCmd) { Die "Python >=3.10 is required. Install it from https://python.org and re-run." }
$pyVersion = & $PythonCmd --version
Success "Found $pyVersion"

# ── tkinter check ─────────────────────────────────────────────────────────────
Info "Checking tkinter..."
& $PythonCmd -c "import tkinter" 2>$null
if ($LASTEXITCODE -ne 0) {
    Warn "tkinter is not available for $PythonCmd."
    Write-Host ""
    Write-Host "  tkinter ships with the standard python.org Windows installer."
    Write-Host "  Reinstall Python from https://python.org making sure 'tcl/tk and IDLE' is checked."
    Write-Host ""
    Die "tkinter is required (used for the GUI). Install it and re-run."
}
Success "tkinter OK"

# ── ffmpeg / ffprobe check ────────────────────────────────────────────────────
Info "Checking ffmpeg and ffprobe..."
$missingFF = @()
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue))  { $missingFF += "ffmpeg" }
if (-not (Get-Command ffprobe -ErrorAction SilentlyContinue)) { $missingFF += "ffprobe" }

if ($missingFF.Count -gt 0) {
    Warn "Missing: $($missingFF -join ', ')"
    Write-Host ""
    Write-Host "  Install ffmpeg (includes ffprobe):"
    Write-Host "    winget install ffmpeg"
    Write-Host "    (or)  choco install ffmpeg"
    Write-Host "    (or)  https://ffmpeg.org/download.html"
    Write-Host ""
    Die "ffmpeg and ffprobe are required. Install them and re-run."
}
Success "ffmpeg and ffprobe OK"

# ── VLC check ─────────────────────────────────────────────────────────────────
Info "Checking VLC (native library used by python-vlc)..."
$vlcFound = $false
if (Get-Command vlc -ErrorAction SilentlyContinue) { $vlcFound = $true }
foreach ($path in @("$env:ProgramFiles\VideoLAN\VLC\libvlc.dll", "${env:ProgramFiles(x86)}\VideoLAN\VLC\libvlc.dll")) {
    if (Test-Path $path) { $vlcFound = $true }
}
if ($vlcFound) {
    Success "VLC found"
} else {
    Warn "Could not confirm a VLC install (python-vlc needs the native VLC libraries, not just the pip package)."
    Write-Host ""
    Write-Host "  Install VLC:"
    Write-Host "    winget install VideoLAN.VLC"
    Write-Host "    (or)  https://www.videolan.org/vlc/"
    Write-Host ""
}

# ── uv detection / install ────────────────────────────────────────────────────
$UseUv = $false
if (-not $NoUv) {
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        $UseUv = $true
        $uvVersion = & uv --version
        Success "Found uv ($uvVersion)"
    } else {
        Warn "uv is not installed."
        $installUv = "y"
        if (-not $Yes) {
            $answer = Read-Host "  Install uv now? [Y/n]"
            if ($answer) { $installUv = $answer }
        }
        if ($installUv -match "^[Yy]") {
            Info "Installing uv..."
            powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
            if (Get-Command uv -ErrorAction SilentlyContinue) {
                $UseUv = $true
                Success "uv installed"
            } else {
                Warn "uv install completed but 'uv' not found in PATH; falling back to pip. Open a new terminal and re-run if this persists."
            }
        } else {
            Info "Skipping uv; will use pip."
        }
    }
}

# ── install dependencies ──────────────────────────────────────────────────────
Push-Location $RepoDir
try {
    if ($UseUv) {
        Info "Installing with uv sync..."
        uv sync
        Success "uv sync complete"
    } else {
        $VenvDir = Join-Path $RepoDir ".venv"
        if (Test-Path $VenvDir) {
            Info "Reusing existing .venv..."
        } else {
            Info "Creating .venv with $PythonCmd..."
            & $PythonCmd -m venv $VenvDir
        }
        Info "Installing with pip (editable)..."
        $venvPip = Join-Path $VenvDir "Scripts\pip.exe"
        & $venvPip install --quiet --upgrade pip
        & $venvPip install --quiet -e .
        Success "pip install complete"
    }

    # ── verify ────────────────────────────────────────────────────────────────
    Info "Verifying installation..."
    $courtSlicerBin = Join-Path $RepoDir ".venv\Scripts\court-slicer.exe"
    if (-not (Test-Path $courtSlicerBin)) { Die "court-slicer binary not found at $courtSlicerBin" }

    $venvPython = Join-Path $RepoDir ".venv\Scripts\python.exe"
    & $venvPython -c "import court_slicer"
    if ($LASTEXITCODE -ne 0) { Die "Failed to import court_slicer" }
    Success "court-slicer entry point verified"

    New-Item -ItemType Directory -Force -Path (Join-Path $RepoDir "videos_entrada") | Out-Null
    New-Item -ItemType Directory -Force -Path (Join-Path $RepoDir "videos_saida") | Out-Null
} finally {
    Pop-Location
}

# ── success summary ───────────────────────────────────────────────────────────
Write-Host ""
Write-Host "  Installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "  Activate the virtualenv:"
Write-Host "    .venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "  Or run directly:"
Write-Host "    .venv\Scripts\court-slicer.exe <video_file>"
Write-Host ""
Write-Host "  Put raw match videos in videos_entrada\, clips are written to"
Write-Host "  videos_saida\<match_name>\trecho_NN.mp4"
Write-Host ""
Write-Host "  Controls:"
Write-Host "    F         Flag current timestamp"
Write-Host "    Space     Pause / play"
Write-Host "    A         Rewind 5 seconds"
Write-Host "    D         Fast-forward 5 seconds"
Write-Host "    + / -     Increase / decrease playback speed"
Write-Host "    Q / Esc   Quit and cut clips"
Write-Host ""
