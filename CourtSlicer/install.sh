#!/usr/bin/env bash
set -euo pipefail

# ── colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; RESET='\033[0m'
info()    { echo -e "${CYAN}[info]${RESET}  $*"; }
success() { echo -e "${GREEN}[ok]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[warn]${RESET}  $*"; }
die()     { echo -e "${RED}[error]${RESET} $*" >&2; exit 1; }

# ── repo root (works when called from any directory) ──────────────────────────
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── CLI flags ─────────────────────────────────────────────────────────────────
AUTO_YES=false
FORCE_PIP=false
for arg in "$@"; do
  case "$arg" in
    --yes)    AUTO_YES=true ;;
    --no-uv)  FORCE_PIP=true ;;
    --help|-h)
      echo "Usage: bash install.sh [--yes] [--no-uv]"
      echo "  --yes     Non-interactive: auto-install uv if missing"
      echo "  --no-uv   Force pip install instead of uv"
      exit 0 ;;
    *) die "Unknown argument: $arg" ;;
  esac
done

echo ""
echo "  CourtSlicer installer (Linux/macOS)"
echo "  ────────────────────────────────────"
echo ""

# ── Python check ──────────────────────────────────────────────────────────────
info "Checking Python..."
PYTHON=""
for candidate in python3 python3.12 python3.11 python3.10; do
  if command -v "$candidate" &>/dev/null; then
    ver=$("$candidate" -c 'import sys; print("%d%03d" % sys.version_info[:2])')
    if [[ "$ver" -ge 3010 ]]; then
      PYTHON="$candidate"
      break
    fi
  fi
done
[[ -n "$PYTHON" ]] || die "Python >=3.10 is required. Install it and re-run."
success "Found $($PYTHON --version)"

# ── tkinter check ─────────────────────────────────────────────────────────────
info "Checking tkinter..."
if ! "$PYTHON" -c "import tkinter" &>/dev/null; then
  warn "tkinter is not available for $PYTHON."
  echo ""
  echo "  Install it for your distro/OS:"
  echo "    Debian/Ubuntu:  sudo apt install python3-tk"
  echo "    Fedora/RHEL:    sudo dnf install python3-tkinter"
  echo "    Arch:           sudo pacman -S tk"
  echo "    macOS (brew):   brew install python-tk"
  echo ""
  die "tkinter is required (used for the GUI). Install it and re-run."
fi
success "tkinter OK"

# ── ffmpeg / ffprobe check ────────────────────────────────────────────────────
info "Checking ffmpeg and ffprobe..."
MISSING_FF=()
command -v ffmpeg  &>/dev/null || MISSING_FF+=(ffmpeg)
command -v ffprobe &>/dev/null || MISSING_FF+=(ffprobe)

if [[ ${#MISSING_FF[@]} -gt 0 ]]; then
  warn "Missing: ${MISSING_FF[*]}"
  echo ""
  echo "  Install ffmpeg (includes ffprobe):"
  if command -v apt-get &>/dev/null; then
    echo "    sudo apt install ffmpeg"
  elif command -v dnf &>/dev/null; then
    echo "    sudo dnf install ffmpeg"
  elif command -v pacman &>/dev/null; then
    echo "    sudo pacman -S ffmpeg"
  elif command -v brew &>/dev/null; then
    echo "    brew install ffmpeg"
  else
    echo "    https://ffmpeg.org/download.html"
  fi
  echo ""
  die "ffmpeg and ffprobe are required. Install them and re-run."
fi
success "ffmpeg and ffprobe OK"

# ── VLC check ─────────────────────────────────────────────────────────────────
info "Checking VLC (native library used by python-vlc)..."
if command -v vlc &>/dev/null || [[ -e /Applications/VLC.app ]] || ldconfig -p 2>/dev/null | grep -q libvlc; then
  success "VLC found"
else
  warn "Could not confirm a VLC install (python-vlc needs the native VLC libraries, not just the pip package)."
  echo ""
  echo "  Install VLC:"
  if command -v apt-get &>/dev/null; then
    echo "    sudo apt install vlc"
  elif command -v dnf &>/dev/null; then
    echo "    sudo dnf install vlc"
  elif command -v pacman &>/dev/null; then
    echo "    sudo pacman -S vlc"
  elif command -v brew &>/dev/null; then
    echo "    brew install --cask vlc"
  else
    echo "    https://www.videolan.org/vlc/"
  fi
  echo ""
fi

# ── uv detection / install ────────────────────────────────────────────────────
export PATH="$HOME/.local/bin:$PATH"

USE_UV=false
if [[ "$FORCE_PIP" == false ]]; then
  if command -v uv &>/dev/null; then
    USE_UV=true
    success "Found uv ($(uv --version))"
  else
    warn "uv is not installed."
    if [[ "$AUTO_YES" == true ]]; then
      INSTALL_UV=y
    else
      echo -n "  Install uv now? [Y/n] "
      read -r INSTALL_UV
      INSTALL_UV="${INSTALL_UV:-y}"
    fi

    if [[ "$INSTALL_UV" =~ ^[Yy] ]]; then
      info "Installing uv..."
      curl -LsSf https://astral.sh/uv/install.sh | sh
      export PATH="$HOME/.local/bin:$PATH"
      if command -v uv &>/dev/null; then
        USE_UV=true
        success "uv installed ($(uv --version))"
      else
        warn "uv install completed but 'uv' not found in PATH; falling back to pip."
      fi
    else
      info "Skipping uv; will use pip."
    fi
  fi
fi

# ── install dependencies ──────────────────────────────────────────────────────
cd "$REPO_DIR"

if [[ "$USE_UV" == true ]]; then
  info "Installing with uv sync..."
  uv sync
  success "uv sync complete"
else
  # pip path
  VENV_DIR="$REPO_DIR/.venv"
  if [[ -d "$VENV_DIR" ]]; then
    info "Reusing existing .venv..."
  else
    info "Creating .venv with $PYTHON..."
    "$PYTHON" -m venv "$VENV_DIR"
  fi
  info "Installing with pip (editable)..."
  "$VENV_DIR/bin/pip" install --quiet --upgrade pip
  "$VENV_DIR/bin/pip" install --quiet -e .
  success "pip install complete"
fi

# ── verify ────────────────────────────────────────────────────────────────────
info "Verifying installation..."
COURT_SLICER_BIN="$REPO_DIR/.venv/bin/court-slicer"
[[ -f "$COURT_SLICER_BIN" ]] || die "court-slicer binary not found at $COURT_SLICER_BIN"

VENV_PYTHON="$REPO_DIR/.venv/bin/python"
"$VENV_PYTHON" -c "import court_slicer" || die "Failed to import court_slicer"
success "court-slicer entry point verified"

mkdir -p "$REPO_DIR/videos_entrada" "$REPO_DIR/videos_saida"

# ── success summary ───────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}  Installation complete!${RESET}"
echo ""
echo "  Activate the virtualenv:"
echo "    source .venv/bin/activate"
echo ""
echo "  Or run directly:"
echo "    .venv/bin/court-slicer <video_file>"
echo ""
echo "  Put raw match videos in videos_entrada/, clips are written to"
echo "  videos_saida/<match_name>/trecho_NN.mp4"
echo ""
echo "  Controls:"
echo "    F         Flag current timestamp"
echo "    Space     Pause / play"
echo "    A         Rewind 5 seconds"
echo "    D         Fast-forward 5 seconds"
echo "    + / -     Increase / decrease playback speed"
echo "    Q / Esc   Quit and cut clips"
echo ""
