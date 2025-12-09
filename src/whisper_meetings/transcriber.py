"""Core transcription functionality."""

import os
from dataclasses import dataclass
from pathlib import Path

from whisper_meetings.audio import prepare_audio
from whisper_meetings.config import Config, get_config
from whisper_meetings.utils import calculate_overlap, format_timestamp


@dataclass
class TranscriptEntry:
    """A single entry in the transcript."""

    timestamp: str
    speaker: str
    text: str


@dataclass
class TranscriptionResult:
    """Result of a transcription."""

    entries: list[TranscriptEntry]
    output_file: Path
    speaker_summary: dict[str, int]


def _get_device(config: Config) -> tuple[str, bool]:
    """
    Determine the compute device to use.

    Returns:
        Tuple of (device_string, use_word_timestamps)
    """
    import torch

    if config.device:
        device = config.device
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # MPS doesn't support word timestamps
    use_word_timestamps = device != "mps"

    return device, use_word_timestamps


def _load_whisper_model(config: Config, device: str):
    """Load the Whisper model."""
    import whisper

    print(f"Loading Whisper model '{config.whisper_model}'...")
    return whisper.load_model(
        config.whisper_model,
        device=device,
        download_root=str(config.whisper_cache)
    )


def _transcribe_audio(model, audio_path: Path, config: Config, device: str, use_word_timestamps: bool) -> dict:
    """Run Whisper transcription."""
    print("Transcribing audio...")

    try:
        return model.transcribe(
            str(audio_path),
            language=config.language,
            word_timestamps=use_word_timestamps,
            fp16=device != "mps",
        )
    except Exception as e:
        if device == "mps":
            print(f"MPS transcription failed ({e}). Falling back to CPU...")
            cpu_model = _load_whisper_model(config, "cpu")
            return cpu_model.transcribe(
                str(audio_path),
                language=config.language,
                word_timestamps=True,
                fp16=False,
            )
        raise


def _load_diarization_pipeline(config: Config):
    """Load the Pyannote diarization pipeline."""
    from pyannote.audio import Pipeline

    print("Loading speaker diarization model...")

    pipeline = Pipeline.from_pretrained(
        config.diarization_model,
        cache_dir=str(config.hf_cache),
    )

    # Ensure token is in environment for downstream libraries
    if config.hf_token and not os.getenv("HUGGINGFACE_HUB_TOKEN"):
        os.environ["HUGGINGFACE_HUB_TOKEN"] = config.hf_token

    return pipeline


def _run_diarization(pipeline, audio_path: Path, num_speakers: int | None) -> list[tuple]:
    """
    Run speaker diarization.

    Returns:
        List of (turn, speaker_label) tuples.
    """
    print("Identifying speakers...")

    if num_speakers:
        result = pipeline(str(audio_path), num_speakers=num_speakers)
    else:
        result = pipeline(str(audio_path))

    # Normalize output across pyannote versions
    tracks = []

    if hasattr(result, "speaker_diarization"):
        # pyannote.audio >= 4.0
        for turn, speaker in result.speaker_diarization:
            tracks.append((turn, speaker))
    else:
        # Older versions
        annotation = getattr(result, "annotation", result)
        if hasattr(annotation, "itertracks"):
            for turn, _, speaker in annotation.itertracks(yield_label=True):
                tracks.append((turn, speaker))
        else:
            raise ValueError(
                "Unsupported diarization output format. "
                "Please update pyannote.audio or open an issue."
            )

    return tracks


def _merge_transcription_with_speakers(
    whisper_result: dict,
    diarization_tracks: list[tuple]
) -> list[TranscriptEntry]:
    """Merge Whisper segments with speaker labels."""
    print("Combining transcription with speaker labels...")

    entries = []

    for segment in whisper_result["segments"]:
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"].strip()

        if not text:
            continue

        # Find best matching speaker
        best_speaker = "Unknown"
        best_overlap = 0.0

        for turn, speaker_label in diarization_tracks:
            overlap = calculate_overlap(
                start_time, end_time,
                turn.start, turn.end
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker_label

        entries.append(TranscriptEntry(
            timestamp=format_timestamp(start_time, end_time),
            speaker=best_speaker,
            text=text
        ))

    return entries


def _write_transcript(entries: list[TranscriptEntry], output_path: Path) -> None:
    """Write transcript to file."""
    with open(output_path, "w") as f:
        f.write("MULTI-PERSON TRANSCRIPT WITH SPEAKER IDENTIFICATION\n")
        f.write("=" * 60 + "\n\n")
        for entry in entries:
            f.write(f"{entry.timestamp} {entry.speaker}: {entry.text}\n\n")


def _get_speaker_summary(entries: list[TranscriptEntry]) -> dict[str, int]:
    """Count segments per speaker."""
    summary: dict[str, int] = {}
    for entry in entries:
        summary[entry.speaker] = summary.get(entry.speaker, 0) + 1
    return summary


def transcribe_meeting(
    audio_file: str | Path,
    num_speakers: int | None = None,
    config: Config | None = None,
) -> TranscriptionResult | None:
    """
    Transcribe a meeting recording with speaker diarization.

    Args:
        audio_file: Path to the audio file.
        num_speakers: Optional hint for number of speakers.
        config: Configuration object. Uses global config if not provided.

    Returns:
        TranscriptionResult containing the transcript entries and metadata,
        or None if the input file doesn't exist.
    """
    if config is None:
        config = get_config()

    config.ensure_cache_dirs()

    audio_path = Path(audio_file).expanduser().resolve()

    if not audio_path.exists():
        print(f"Error: Input file not found: {audio_path}")
        return None

    # Prepare audio
    prepared_path, cleanup = prepare_audio(audio_path)

    try:
        # Setup device
        device, use_word_timestamps = _get_device(config)
        print(f"Using device: {device}")

        if device == "mps":
            print("Note: Word timestamps disabled on MPS.")

        # Transcribe
        model = _load_whisper_model(config, device)
        whisper_result = _transcribe_audio(
            model, prepared_path, config, device, use_word_timestamps
        )

        # Diarize
        pipeline = _load_diarization_pipeline(config)
        diarization_tracks = _run_diarization(pipeline, prepared_path, num_speakers)

        # Merge results
        entries = _merge_transcription_with_speakers(whisper_result, diarization_tracks)

        # Output
        output_path = audio_path.with_suffix(".txt")
        _write_transcript(entries, output_path)

        # Print to console
        print("\n" + "=" * 60)
        print("MULTI-PERSON TRANSCRIPT WITH SPEAKER IDENTIFICATION")
        print("=" * 60 + "\n")

        for entry in entries:
            print(f"{entry.timestamp} {entry.speaker}: {entry.text}\n")

        print(f"Transcript saved to: {output_path}\n")

        # Speaker summary
        summary = _get_speaker_summary(entries)
        print("SPEAKER SUMMARY:")
        print("-" * 60)
        for speaker, count in sorted(summary.items()):
            print(f"{speaker}: {count} segments")

        return TranscriptionResult(
            entries=entries,
            output_file=output_path,
            speaker_summary=summary
        )

    finally:
        if cleanup:
            cleanup()
