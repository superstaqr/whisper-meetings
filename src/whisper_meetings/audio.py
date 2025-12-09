"""Audio processing utilities."""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable


def has_command(cmd: str) -> bool:
    """Check if a command is available in PATH."""
    return shutil.which(cmd) is not None


def probe_audio(input_path: Path) -> dict | None:
    """
    Get audio stream metadata using ffprobe.

    Args:
        input_path: Path to the audio file.

    Returns:
        Dictionary containing audio stream info, or None on failure.
    """
    if not has_command("ffprobe"):
        return None

    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=codec_name,channels,sample_rate,bit_rate",
                "-show_entries", "format=format_name",
                "-of", "json",
                str(input_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(proc.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return None


def needs_conversion(meta: dict | None, src_path: Path) -> bool:
    """
    Determine if audio needs conversion to optimal format.

    Optimal format is 16 kHz mono PCM WAV.

    Args:
        meta: Audio metadata from ffprobe.
        src_path: Path to source audio file.

    Returns:
        True if conversion is needed, False otherwise.
    """
    if meta is None:
        return True

    try:
        streams = meta.get("streams", [])
        if not streams:
            return True

        stream = streams[0]
        codec = stream.get("codec_name")
        channels = int(stream.get("channels", 0))
        sample_rate = int(stream.get("sample_rate", 0))
        container_ok = src_path.suffix.lower() == ".wav"

        return not (
            codec == "pcm_s16le"
            and channels == 1
            and sample_rate == 16000
            and container_ok
        )
    except (KeyError, ValueError, TypeError):
        return True


def prepare_audio(input_path: Path) -> tuple[Path, Callable[[], None] | None]:
    """
    Prepare audio file for transcription.

    Converts to 16 kHz mono PCM WAV if needed.

    Args:
        input_path: Path to input audio file.

    Returns:
        Tuple of (prepared_path, cleanup_function).
        cleanup_function should be called to remove temp file when done.
    """
    meta = probe_audio(input_path)

    if not needs_conversion(meta, input_path):
        return input_path, None

    if not has_command("ffmpeg"):
        print("Warning: ffmpeg not found; using original audio file.")
        return input_path, None

    # Create temporary file
    tmp = tempfile.NamedTemporaryFile(
        prefix="whisper_meetings_",
        suffix="_prepared.wav",
        delete=False
    )
    tmp_path = Path(tmp.name)
    tmp.close()

    print("Preparing audio (16 kHz mono PCM WAV)...")

    cmd = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-threads", "0",
        "-i", str(input_path),
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(tmp_path),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("Warning: Audio preparation failed; using original file.")
        tmp_path.unlink(missing_ok=True)
        return input_path, None

    def cleanup() -> None:
        tmp_path.unlink(missing_ok=True)

    return tmp_path, cleanup
