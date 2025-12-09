"""Utility functions for whisper-meetings."""

import json
import shutil
import subprocess
import sys
from importlib import metadata


def format_timestamp(start_seconds: float, end_seconds: float) -> str:
    """
    Convert seconds to timestamp range format.

    Args:
        start_seconds: Start time in seconds.
        end_seconds: End time in seconds.

    Returns:
        Formatted string like "[00:01:23.456 -> 00:01:30.789]"
    """
    def seconds_to_time(secs: float) -> str:
        hours = int(secs // 3600)
        minutes = int((secs % 3600) // 60)
        seconds = int(secs % 60)
        millis = int((secs - int(secs)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"

    return f"[{seconds_to_time(start_seconds)} -> {seconds_to_time(end_seconds)}]"


def calculate_overlap(
    a_start: float, a_end: float, b_start: float, b_end: float
) -> float:
    """
    Calculate overlap duration between two time intervals.

    Args:
        a_start: Start of interval A.
        a_end: End of interval A.
        b_start: Start of interval B.
        b_end: End of interval B.

    Returns:
        Duration of overlap in seconds (0 if no overlap).
    """
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def get_ffmpeg_version() -> str | None:
    """Get installed ffmpeg version, or None if not found."""
    if not shutil.which("ffmpeg"):
        return None

    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            check=True
        )
        first_line = result.stdout.splitlines()[0].strip() if result.stdout else ""
        parts = first_line.split()
        if len(parts) >= 3 and parts[0] == "ffmpeg" and parts[1] == "version":
            return parts[2]
    except (subprocess.CalledProcessError, IndexError):
        pass
    return None


def get_installed_version(package_name: str) -> str | None:
    """Get installed version of a Python package."""
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def get_outdated_packages() -> dict[str, dict[str, str]]:
    """
    Get mapping of outdated packages.

    Returns:
        Dict mapping package name to {"current": version, "latest": version}
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--outdated", "--format=json"],
            capture_output=True,
            text=True,
            check=True
        )
        data = json.loads(proc.stdout or "[]")
        return {
            item["name"]: {
                "current": item["version"],
                "latest": item["latest_version"]
            }
            for item in data
        }
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return {}
