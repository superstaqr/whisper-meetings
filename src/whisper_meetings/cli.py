"""Command-line interface for whisper-meetings."""

import argparse
import sys

from whisper_meetings import __version__
from whisper_meetings.preflight import run_preflight_checks
from whisper_meetings.transcriber import transcribe_meeting


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="whisper-meetings",
        description="Transcribe meetings with speaker diarization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  whisper-meetings meeting.wav
  whisper-meetings meeting.m4a --speakers 4
  whisper-meetings recording.mp3 --model medium --device cpu
        """,
    )

    parser.add_argument(
        "audio_file",
        help="Path to the audio file to transcribe",
    )

    parser.add_argument(
        "-s", "--speakers",
        type=int,
        default=None,
        help="Number of speakers (optional, auto-detected if not specified)",
    )

    parser.add_argument(
        "-m", "--model",
        default=None,
        help="Whisper model to use (tiny, base, small, medium, large)",
    )

    parser.add_argument(
        "-d", "--device",
        default=None,
        help="Device to use (cpu, mps, cuda)",
    )

    parser.add_argument(
        "-l", "--language",
        default=None,
        help="Language code (default: en)",
    )

    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip preflight checks",
    )

    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    # Run preflight checks
    if not args.skip_preflight:
        run_preflight_checks(interactive=True)
        print()  # Blank line after preflight

    # Build config overrides
    from whisper_meetings.config import get_config

    config = get_config()

    if args.model:
        config.whisper_model = args.model
    if args.device:
        config.device = args.device
    if args.language:
        config.language = args.language

    # Run transcription
    result = transcribe_meeting(
        audio_file=args.audio_file,
        num_speakers=args.speakers,
        config=config,
    )

    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
