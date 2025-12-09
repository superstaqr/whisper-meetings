#!/usr/bin/env bash
#
# setup.sh - Quick setup script for whisper-meetings
#
# Usage:
#   ./setup.sh           # Full setup (venv + dependencies + .env)
#   ./setup.sh --dev     # Include development dependencies
#   ./setup.sh --help    # Show help
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"

print_header() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║              whisper-meetings setup                      ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_step() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

show_help() {
    echo "Usage: ./setup.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --dev       Include development dependencies (pytest, ruff, mypy)"
    echo "  --no-venv   Skip virtual environment creation (use existing Python)"
    echo "  --clean     Remove existing venv before setup"
    echo "  --help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./setup.sh              # Standard setup"
    echo "  ./setup.sh --dev        # Setup with dev tools"
    echo "  ./setup.sh --clean      # Fresh install"
}

check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        print_error "Python not found. Please install Python 3.10 or later."
        exit 1
    fi

    # Check Python version
    PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

    if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 10 ]); then
        print_error "Python 3.10+ required. Found: Python $PYTHON_VERSION"
        exit 1
    fi

    print_step "Found Python $PYTHON_VERSION"
}

check_ffmpeg() {
    if command -v ffmpeg &> /dev/null; then
        FFMPEG_VERSION=$(ffmpeg -version 2>&1 | head -n1 | awk '{print $3}')
        print_step "Found ffmpeg $FFMPEG_VERSION"
    else
        print_warning "ffmpeg not found. Install with: brew install ffmpeg"
        echo "    Whisper requires ffmpeg for audio processing."
    fi
}

create_venv() {
    if [ "$SKIP_VENV" = true ]; then
        print_warning "Skipping virtual environment creation"
        return
    fi

    if [ -d "$VENV_DIR" ]; then
        if [ "$CLEAN_INSTALL" = true ]; then
            print_step "Removing existing virtual environment..."
            rm -rf "$VENV_DIR"
        else
            print_warning "Virtual environment already exists at $VENV_DIR"
            echo "    Use --clean to remove and recreate it"
            return
        fi
    fi

    print_step "Creating virtual environment..."
    $PYTHON_CMD -m venv "$VENV_DIR"
}

activate_venv() {
    if [ "$SKIP_VENV" = true ]; then
        return
    fi

    if [ -f "${VENV_DIR}/bin/activate" ]; then
        source "${VENV_DIR}/bin/activate"
        print_step "Activated virtual environment"
    else
        print_error "Virtual environment not found at $VENV_DIR"
        exit 1
    fi
}

install_dependencies() {
    print_step "Upgrading pip..."
    pip install --upgrade pip --quiet

    if [ "$DEV_INSTALL" = true ]; then
        print_step "Installing package with development dependencies..."
        pip install -e ".[dev]" --quiet
    else
        print_step "Installing package..."
        pip install -e . --quiet
    fi

    print_step "Dependencies installed successfully"
}

setup_env_file() {
    if [ ! -f "${SCRIPT_DIR}/.env" ]; then
        if [ -f "${SCRIPT_DIR}/.env.example" ]; then
            cp "${SCRIPT_DIR}/.env.example" "${SCRIPT_DIR}/.env"
            print_step "Created .env file from template"
            print_warning "Edit .env and add your HUGGINGFACE_HUB_TOKEN"
        fi
    else
        print_step ".env file already exists"
    fi
}

create_model_dirs() {
    WHISPER_CACHE="${HOME}/models/whisper"
    HF_CACHE="${HOME}/models/huggingface"

    mkdir -p "$WHISPER_CACHE"
    mkdir -p "$HF_CACHE"

    print_step "Model cache directories ready"
    echo "    Whisper: $WHISPER_CACHE"
    echo "    HuggingFace: $HF_CACHE"
}

print_next_steps() {
    echo ""
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}Setup complete!${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "Next steps:"
    echo ""
    echo "  1. Activate the virtual environment:"
    echo -e "     ${YELLOW}source venv/bin/activate${NC}"
    echo ""
    echo "  2. Add your Hugging Face token to .env:"
    echo -e "     ${YELLOW}nano .env${NC}"
    echo "     (Get a token at https://huggingface.co/settings/tokens)"
    echo ""
    echo "  3. Accept Pyannote model terms:"
    echo "     https://huggingface.co/pyannote/speaker-diarization-3.1"
    echo ""
    echo "  4. Run a transcription:"
    echo -e "     ${YELLOW}whisper-meetings your-recording.m4a${NC}"
    echo ""
}

# Parse arguments
DEV_INSTALL=false
SKIP_VENV=false
CLEAN_INSTALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)
            DEV_INSTALL=true
            shift
            ;;
        --no-venv)
            SKIP_VENV=true
            shift
            ;;
        --clean)
            CLEAN_INSTALL=true
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main setup flow
cd "$SCRIPT_DIR"

print_header
check_python
check_ffmpeg
create_venv
activate_venv
install_dependencies
setup_env_file
create_model_dirs
print_next_steps
