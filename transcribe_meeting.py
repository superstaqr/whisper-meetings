import os
import sys
import json
import shutil
import tempfile
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from typing import Callable, Optional

# Confirm models exist before running
WHISPER_CACHE = os.path.expanduser("~/models/whisper")
HF_CACHE = os.path.expanduser("~/models/huggingface")

print(f"✓ Whisper models location: {WHISPER_CACHE}")
print(f"✓ Hugging Face models location: {HF_CACHE}")

if not os.path.exists(WHISPER_CACHE):
    print("⚠️  WARNING: Whisper cache folder not found! Creating it...")
    os.makedirs(WHISPER_CACHE, exist_ok=True)

if not os.path.exists(HF_CACHE):
    print("⚠️  WARNING: Hugging Face cache folder not found! Creating it...")
    os.makedirs(HF_CACHE, exist_ok=True)

# Load environment variables from a .env file if present
load_dotenv()

# Prefer standard env var name; allow HF_TOKEN as fallback
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")

# --------------- New: Update & Health Check (runs before heavy imports) ---------------

def _have_cmd(cmd: str) -> bool:
    return shutil.which(cmd) is not None

def _ffmpeg_version() -> str | None:
    if not _have_cmd("ffmpeg"):
        return None
    try:
        out = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True).stdout
        first = out.splitlines()[0].strip() if out else ""
        # "ffmpeg version 8.0 ..." -> return "8.0"
        parts = first.split()
        if len(parts) >= 3 and parts[0] == "ffmpeg" and parts[1] == "version":
            return parts[2]
    except Exception:
        pass
    return None

def _pip_outdated() -> dict[str, dict]:
    """
    Return a mapping of dist_name -> {"current": str, "latest": str} for outdated packages.
    Uses: python -m pip list --outdated --format=json
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--outdated", "--format=json"],
            capture_output=True, text=True, check=True
        )
        data = json.loads(proc.stdout or "[]")
        # Normalize into dict for quick lookup
        return {item["name"]: {"current": item["version"], "latest": item["latest_version"]} for item in data}
    except Exception:
        return {}

def _installed_version(dist_name: str) -> str | None:
    try:
        from importlib import metadata
        return metadata.version(dist_name)
    except Exception:
        return None

def _check_and_offer_updates():
    print("🔍 Preflight: dependencies and tools check...")

    # ffmpeg check
    ffv = _ffmpeg_version()
    if ffv is None:
        print("❗ ffmpeg not found. Install it on macOS with:")
        print("   brew install ffmpeg")
    else:
        print(f"✓ ffmpeg: {ffv}")

    if not HF_TOKEN:
        print("ℹ️  HUGGINGFACE_HUB_TOKEN not found. Pyannote may require it if the model is gated.")
        print("   Create a token and accept terms at:")
        print("   - https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("   - https://huggingface.co/settings/tokens")

    # Packages to check (dist names on PyPI)
    targets = [
        "openai-whisper",
        "pyannote.audio",
        "torch",
        "python-dotenv",
        "huggingface-hub",
    ]

    # Current versions
    current = {name: _installed_version(name) for name in targets}
    # Outdated info from pip
    outdated = _pip_outdated()

    to_update = []
    for name in targets:
        cur = current.get(name)
        if name in outdated:
            latest = outdated[name]["latest"]
            print(f"⤴️  {name}: {cur} -> {latest} (update available)")
            to_update.append(name)
        else:
            label = cur if cur else "not installed"
            print(f"✓ {name}: {label}")

    if not to_update:
        print("✓ All tracked Python packages are up to date.")
        return

    # Offer to update
    auto = os.getenv("AUTO_UPDATE_DEPS")
    do_update = False
    if auto == "1":
        do_update = True
    elif sys.stdin.isatty():
        try:
            resp = input(f"Update {len(to_update)} package(s) now? [y/N]: ").strip().lower()
            do_update = resp in ("y", "yes")
        except EOFError:
            do_update = False

    if not do_update:
        print("ℹ️  Skipping updates. You can update later with:")
        print(f"   {sys.executable} -m pip install --upgrade " + " ".join(to_update))
        return

    print("⬇️  Updating packages via pip...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", *to_update],
            check=True
        )
        print("✅ Updates installed. Please re-run the script to use the updated packages.")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print("❌ Failed to update some packages. You may try manually:")
        print(f"   {sys.executable} -m pip install --upgrade " + " ".join(to_update))
        # continue without exiting

# --------------- End Update & Health Check ---------------

def _probe_audio(input_path: Path) -> dict | None:
    """Return ffprobe JSON for the first audio stream, or None on failure."""
    if not _have_cmd("ffprobe"):
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
    except Exception:
        return None

def _needs_conversion(meta: dict | None, src_path: Path) -> bool:
    """Decide if we should re-encode to 16 kHz mono PCM WAV."""
    try:
        if meta is None:
            # If we can't probe, be conservative and convert.
            return True
        streams = meta.get("streams", [])
        if not streams:
            return True
        s = streams[0]
        codec = s.get("codec_name")
        channels = int(s.get("channels", 0))
        sample_rate = int(s.get("sample_rate", 0))
        # We want: PCM s16le, mono, 16000 Hz, WAV container.
        container_ok = src_path.suffix.lower() == ".wav"
        return not (
            codec == "pcm_s16le"
            and channels == 1
            and sample_rate == 16000
            and container_ok
        )
    except Exception:
        return True

def ensure_optimal_audio(input_path: Path) -> tuple[Path, Optional[Callable[[], None]]]:
    """
    Ensure audio is 16 kHz mono PCM WAV.
    Returns (prepared_path, cleanup_fn). cleanup_fn() should be called to remove temp file.
    """
    meta = _probe_audio(input_path)
    if not _needs_conversion(meta, input_path):
        # Already optimal
        return input_path, None

    if not _have_cmd("ffmpeg"):
        print("⚠️  ffmpeg not found; proceeding with original audio (Whisper will attempt decode).")
        return input_path, None

    tmp = tempfile.NamedTemporaryFile(prefix="whisper_meetings_", suffix="_prepared.wav", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()

    print("🔧 Preparing audio (16 kHz mono PCM WAV)...")
    cmd = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-threads", "0",
        "-i", str(input_path),
        # Optional mild loudness normalization:
        # "-af", "loudnorm=I=-16:LRA=11:TP=-1.5",
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(tmp_path),
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print("⚠️  Audio preparation failed; falling back to original file.")
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        return input_path, None

    def _cleanup():
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    return tmp_path, _cleanup

def transcribe_multiperson(audio_file: str, num_speakers: int | None = None):
    """
    Transcribe multi-person conversation with speaker diarization
    Uses PREDOWNLOADED models from ~/models/
    """
    # Defer heavy imports until after update check
    import torch
    import whisper
    from pyannote.audio import Pipeline

    audio_path = Path(audio_file).expanduser()
    if not audio_path.exists():
        print(f"❌ Input not found: {audio_path}")
        print("   Tip: check the filename or use an absolute path.")
        return

    prepared_path, cleanup = ensure_optimal_audio(audio_path)
    try:
        # Allow overriding device via env var; else prefer MPS on Apple Silicon
        device_str = os.getenv("WHISPER_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"🎙️  Using device: {device_str}")
        if device_str == "mps":
            print("⚠️  MPS does not support Whisper word timestamps; disabling them for this run.")
            use_word_timestamps = False
        else:
            use_word_timestamps = True

        print("📝 Loading Whisper model from ~/models/whisper...")
        model = whisper.load_model("large", device=device_str, download_root=WHISPER_CACHE)

        print("   Transcribing audio...")
        try:
            result = model.transcribe(
                str(prepared_path),
                language="en",
                word_timestamps=use_word_timestamps,
                fp16=False if device_str == "mps" else True,
            )
        except Exception:
            if device_str == "mps":
                print("⚠️  Whisper on MPS failed. Falling back to CPU...")
                cpu_model = whisper.load_model("large", device="cpu", download_root=WHISPER_CACHE)
                result = cpu_model.transcribe(str(prepared_path), language="en", word_timestamps=True, fp16=False)
            else:
                raise

        # 👥 Run diarization using the same prepared audio
        print("👥 Loading Pyannote from ~/models/huggingface...")
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            cache_dir=HF_CACHE,
        )
        if HF_TOKEN and os.getenv("HUGGINGFACE_HUB_TOKEN") is None:
            # Ensure token in env for downstream libraries that read it at runtime
            os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

        print("   Identifying speakers...")
        diarization_out = (
            diarization_pipeline(str(prepared_path), num_speakers=num_speakers)
            if num_speakers else
            diarization_pipeline(str(prepared_path))
        )

        # Step 3: Merge results with best-overlap speaker assignment
        print("🔗 Combining transcription with speaker labels...")
        transcript_with_speakers = []

        # Normalize diarization output across pyannote versions
        diarization_tracks = []  # list of tuples (turn, speaker_label)
        if hasattr(diarization_out, "speaker_diarization"):
            # pyannote.audio >= 4.0 style: iterate over output.speaker_diarization
            for turn, speaker in diarization_out.speaker_diarization:
                diarization_tracks.append((turn, speaker))
        else:
            # Older style: either DiarizeOutput.annotation or raw Annotation
            annotation = getattr(diarization_out, "annotation", diarization_out)
            if hasattr(annotation, "itertracks"):
                for turn, _, speaker in annotation.itertracks(yield_label=True):
                    diarization_tracks.append((turn, speaker))
            else:
                raise AttributeError(
                    "Unsupported diarization output format. Update pyannote or open an issue with the traceback."
                )

        def overlap(a_start, a_end, b_start, b_end):
            return max(0.0, min(a_end, b_end) - max(a_start, b_start))

        for segment in result['segments']:
            start_time = segment['start']
            end_time = segment['end']
            text = segment['text'].strip()

            if not text:
                continue

            best_label = "Unknown"
            best_overlap = 0.0
            for turn, speaker_label in diarization_tracks:
                ov = overlap(start_time, end_time, turn.start, turn.end)
                if ov > best_overlap:
                    best_overlap = ov
                    best_label = speaker_label

            timestamp = format_timestamp(start_time, end_time)
            transcript_with_speakers.append({
                'timestamp': timestamp,
                'speaker': best_label,
                'text': text
            })
        
        # Step 4: Output results
        print("\n" + "="*60)
        print("MULTI-PERSON TRANSCRIPT WITH SPEAKER IDENTIFICATION")
        print("="*60 + "\n")
        
        for entry in transcript_with_speakers:
            print(f"{entry['timestamp']} {entry['speaker']}: {entry['text']}\n")
        
        # Step 5: Save to file
        in_path = Path(audio_file).resolve()
        out_file = in_path.with_suffix('.txt')
        with open(out_file, 'w') as f:
            f.write("MULTI-PERSON TRANSCRIPT WITH SPEAKER IDENTIFICATION\n")
            f.write("="*60 + "\n\n")
            for entry in transcript_with_speakers:
                f.write(f"{entry['timestamp']} {entry['speaker']}: {entry['text']}\n\n")
        
        print(f"✅ Transcript saved to: {out_file}\n")
        
        # Step 6: Speaker summary
        print("SPEAKER SUMMARY:")
        print("-"*60)
        speakers = {}
        for entry in transcript_with_speakers:
            speaker = entry['speaker']
            speakers.setdefault(speaker, 0)
            speakers[speaker] += 1
        
        for speaker, count in sorted(speakers.items()):
            print(f"{speaker}: {count} segments")
        
        return transcript_with_speakers
    finally:
        if cleanup:
            cleanup()

def format_timestamp(start_seconds, end_seconds):
    """Convert seconds to HH:MM:SS.sss format"""
    def seconds_to_time(secs):
        hours = int(secs // 3600)
        minutes = int((secs % 3600) // 60)
        seconds = int(secs % 60)
        millis = int((secs - int(secs)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"
    
    start = seconds_to_time(start_seconds)
    end = seconds_to_time(end_seconds)
    return f"[{start} → {end}]"

if __name__ == "__main__":
    # Run preflight updater before heavy imports/processing
    _check_and_offer_updates()

    if len(sys.argv) < 2:
        print("Usage: python transcribe_meeting.py <audio_file> [num_speakers]")
        print("Example: python transcribe_meeting.py meeting.wav 4")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    num_speakers = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    transcribe_multiperson(audio_file, num_speakers)
