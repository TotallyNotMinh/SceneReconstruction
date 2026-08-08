#!/usr/bin/env bash
# =============================================================================
# install_depth_anything_v3.sh
# Installs Depth Anything V3 from the official ByteDance-Seed GitHub repository.
# Run from the SceneReconstruction project root with your venv activated.
# Usage:
#   bash scripts/install_depth_anything_v3.sh
#   bash scripts/install_depth_anything_v3.sh --dir /path/to/install/dir
# =============================================================================

set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────────────
REPO_URL="https://github.com/ByteDance-Seed/Depth-Anything-3.git"
DEFAULT_INSTALL_DIR="$(dirname "$(pwd)")/Depth-Anything-3"
INSTALL_DIR="${DEFAULT_INSTALL_DIR}"

# ── Parse optional --dir argument ────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dir)
            INSTALL_DIR="$2"
            shift 2
            ;;
        *)
            echo "[ERROR] Unknown argument: $1"
            echo "Usage: bash scripts/install_depth_anything_v3.sh [--dir /path/to/dir]"
            exit 1
            ;;
    esac
done

# ── Check prerequisites ───────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Depth Anything V3 — Installation Script"
echo "============================================================"
echo ""

if ! command -v git &>/dev/null; then
    echo "[ERROR] git is not installed or not on PATH. Install git and retry."
    exit 1
fi

if ! command -v python &>/dev/null && ! command -v python3 &>/dev/null; then
    echo "[ERROR] python/python3 not found. Activate your virtual environment first."
    exit 1
fi

PYTHON=$(command -v python || command -v python3)

if ! "$PYTHON" -c "import pip" &>/dev/null; then
    echo "[ERROR] pip is not available in the current Python environment."
    echo "        Make sure your virtual environment is activated."
    exit 1
fi

echo "[✓] git     : $(git --version)"
echo "[✓] python  : $("$PYTHON" --version)"
echo "[✓] pip     : $("$PYTHON" -m pip --version | cut -d' ' -f1-2)"
echo ""
echo "[→] Install directory: ${INSTALL_DIR}"
echo ""

# ── Clone repository ──────────────────────────────────────────────────────────
if [[ -d "${INSTALL_DIR}" ]]; then
    echo "[!] Directory '${INSTALL_DIR}' already exists."
    echo "    Pulling latest changes instead of cloning..."
    git -C "${INSTALL_DIR}" pull
else
    echo "[→] Cloning ${REPO_URL}..."
    git clone "${REPO_URL}" "${INSTALL_DIR}"
fi

# ── Install in editable mode ──────────────────────────────────────────────────
echo ""
echo "[→] Installing package in editable mode (pip install -e .) ..."
"$PYTHON" -m pip install -e "${INSTALL_DIR}"

# ── Download model weights ────────────────────────────────────────────────────
WEIGHTS_SCRIPT="${INSTALL_DIR}/scripts/download_weights.sh"
echo ""
if [[ -f "${WEIGHTS_SCRIPT}" ]]; then
    echo "[→] Downloading pretrained model weights..."
    bash "${WEIGHTS_SCRIPT}"
else
    echo "[!] Weight download script not found at '${WEIGHTS_SCRIPT}'."
    echo "    Download weights manually from the repository README."
fi

# ── Verify installation ───────────────────────────────────────────────────────
echo ""
echo "[→] Verifying installation..."
if "$PYTHON" -c "from depth_anything_3.api import DepthAnything3; print('[✓] depth_anything_3 imported successfully')" 2>/dev/null; then
    echo ""
    echo "============================================================"
    echo "  Installation complete!"
    echo "  Depth Anything V3 is ready to use."
    echo "============================================================"
    echo ""
else
    echo ""
    echo "[ERROR] Import failed. Installation may be incomplete."
    echo "        Check the output above for errors and retry."
    exit 1
fi
