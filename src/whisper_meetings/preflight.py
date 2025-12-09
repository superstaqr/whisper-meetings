"""Preflight checks and dependency management."""

import subprocess
import sys

from whisper_meetings.config import get_config
from whisper_meetings.utils import (
    get_ffmpeg_version,
    get_installed_version,
    get_outdated_packages,
)

# Packages to check for updates
TRACKED_PACKAGES = [
    "openai-whisper",
    "pyannote.audio",
    "torch",
    "python-dotenv",
    "huggingface-hub",
]


def run_preflight_checks(interactive: bool = True) -> None:
    """
    Run preflight dependency and environment checks.

    Args:
        interactive: Whether to prompt for package updates.
    """
    config = get_config()

    print("Preflight: checking dependencies and tools...")

    # Check ffmpeg
    ffmpeg_version = get_ffmpeg_version()
    if ffmpeg_version is None:
        print("  [!] ffmpeg not found. Install with: brew install ffmpeg")
    else:
        print(f"  [ok] ffmpeg: {ffmpeg_version}")

    # Check HF token
    if not config.hf_token:
        print("  [info] HUGGINGFACE_HUB_TOKEN not found.")
        print("         Pyannote may require it for gated models.")
        print("         See: https://huggingface.co/pyannote/speaker-diarization-3.1")

    # Check model cache directories
    config.ensure_cache_dirs()
    print(f"  [ok] Whisper cache: {config.whisper_cache}")
    print(f"  [ok] HuggingFace cache: {config.hf_cache}")

    # Check Python package versions
    _check_package_updates(interactive)


def _check_package_updates(interactive: bool) -> None:
    """Check for and optionally apply package updates."""
    config = get_config()
    current_versions = {pkg: get_installed_version(pkg) for pkg in TRACKED_PACKAGES}
    outdated = get_outdated_packages()

    packages_to_update = []

    for name in TRACKED_PACKAGES:
        current = current_versions.get(name)
        if name in outdated:
            latest = outdated[name]["latest"]
            print(f"  [update] {name}: {current} -> {latest}")
            packages_to_update.append(name)
        else:
            label = current if current else "not installed"
            print(f"  [ok] {name}: {label}")

    if not packages_to_update:
        print("  [ok] All tracked packages are up to date.")
        return

    # Determine whether to update
    do_update = False

    if config.auto_update_deps:
        do_update = True
    elif interactive and sys.stdin.isatty():
        try:
            response = input(
                f"\nUpdate {len(packages_to_update)} package(s)? [y/N]: "
            ).strip().lower()
            do_update = response in ("y", "yes")
        except EOFError:
            do_update = False

    if not do_update:
        print("\nSkipping updates. Update manually with:")
        print(f"  {sys.executable} -m pip install --upgrade " + " ".join(packages_to_update))
        return

    # Perform update
    print("\nUpdating packages...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", *packages_to_update],
            check=True
        )
        print("Updates installed. Please re-run the script.")
        sys.exit(0)
    except subprocess.CalledProcessError:
        print("Failed to update some packages. Try manually:")
        print(f"  {sys.executable} -m pip install --upgrade " + " ".join(packages_to_update))
