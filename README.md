# whisper-meetings

Transcribe meeting recordings with automatic speaker identification using OpenAI's Whisper for speech-to-text and Pyannote for speaker diarization.

## Features

- **Automatic transcription** using OpenAI Whisper (supports multiple model sizes)
- **Speaker diarization** using Pyannote to identify who said what
- **Apple Silicon optimized** with MPS acceleration when available
- **Multiple audio formats** supported (wav, mp3, m4a, mp4, etc.)
- **Automatic audio preparation** converts to optimal format using ffmpeg
- **Configurable** via environment variables or command-line flags
- **Local model caching** for faster subsequent runs
- **Preflight checks** with optional automatic dependency updates

## Quick Start

### One-Line Setup

```sh
# Clone and setup (creates venv, installs dependencies)
git clone https://github.com/superstaqr/whisper-meetings.git
cd whisper-meetings
./setup.sh
```

### Manual Setup

```sh
# 1. Clone the repository
git clone https://github.com/superstaqr/whisper-meetings.git
cd whisper-meetings

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -e .  # Install as editable package

# 4. Configure your Hugging Face token
cp .env.example .env
# Edit .env and add your HUGGINGFACE_HUB_TOKEN
```

## Prerequisites

### Required

| Requirement | Description | Installation |
|-------------|-------------|--------------|
| Python 3.10+ | Python interpreter | [python.org](https://python.org) |
| ffmpeg | Audio processing | `brew install ffmpeg` (macOS) |
| Hugging Face Token | For Pyannote models | [Get token](https://huggingface.co/settings/tokens) |

### Hugging Face Setup

1. Create an account at [huggingface.co](https://huggingface.co)
2. Accept the model terms at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
3. Create an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Add the token to your `.env` file or export it:
   ```sh
   export HUGGINGFACE_HUB_TOKEN=hf_your_token_here
   ```

## Usage

### Command Line Interface

```sh
# Basic usage
whisper-meetings recording.m4a

# Specify number of speakers (improves accuracy)
whisper-meetings meeting.wav --speakers 4

# Use a smaller/faster model
whisper-meetings meeting.wav --model medium

# Force CPU (enables word timestamps)
whisper-meetings meeting.wav --device cpu

# Specify language
whisper-meetings meeting.wav --language en

# Skip preflight dependency checks
whisper-meetings meeting.wav --skip-preflight

# Show help
whisper-meetings --help
```

### Legacy Script

The original script is still available for backwards compatibility:

```sh
python transcribe_meeting.py recording.m4a
python transcribe_meeting.py meeting.wav 4  # with speaker count
```

### As a Python Library

```python
from whisper_meetings import transcribe_meeting
from whisper_meetings.config import Config

# Basic usage
result = transcribe_meeting("meeting.wav")

# With custom configuration
config = Config()
config.whisper_model = "medium"
config.device = "cpu"

result = transcribe_meeting(
    "meeting.wav",
    num_speakers=4,
    config=config
)

# Access results
for entry in result.entries:
    print(f"{entry.timestamp} {entry.speaker}: {entry.text}")

print(f"Saved to: {result.output_file}")
print(f"Speaker summary: {result.speaker_summary}")
```

## Output

The tool generates:

1. **Console output** - Real-time transcript with timestamps and speaker labels
2. **Text file** - Saved alongside your audio file (e.g., `meeting.txt`)
3. **Speaker summary** - Count of segments per speaker

### Example Output

```
[00:00:00.000 -> 00:00:05.230] SPEAKER_00: Welcome everyone to today's meeting.

[00:00:05.450 -> 00:00:12.100] SPEAKER_01: Thanks for having us. I wanted to discuss the quarterly results.

[00:00:12.340 -> 00:00:18.900] SPEAKER_00: Great, let's start with the sales figures.
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HUGGINGFACE_HUB_TOKEN` | Hugging Face API token | Required |
| `HF_TOKEN` | Alternative token variable | - |
| `WHISPER_MODEL` | Whisper model size | `large` |
| `WHISPER_DEVICE` | Compute device | Auto-detect |
| `WHISPER_LANGUAGE` | Transcription language | `en` |
| `WHISPER_CACHE` | Whisper model cache path | `~/models/whisper` |
| `HF_CACHE` | Hugging Face cache path | `~/models/huggingface` |
| `AUTO_UPDATE_DEPS` | Auto-update packages (1/0) | `0` |

### Whisper Models

| Model | Parameters | VRAM | Relative Speed |
|-------|------------|------|----------------|
| `tiny` | 39M | ~1GB | ~32x |
| `base` | 74M | ~1GB | ~16x |
| `small` | 244M | ~2GB | ~6x |
| `medium` | 769M | ~5GB | ~2x |
| `large` | 1550M | ~10GB | 1x |

Use smaller models for faster processing or limited hardware:
```sh
whisper-meetings meeting.wav --model small
```

## Project Structure

```
whisper-meetings/
├── src/whisper_meetings/
│   ├── __init__.py       # Package exports
│   ├── audio.py          # Audio processing (ffmpeg)
│   ├── cli.py            # Command-line interface
│   ├── config.py         # Configuration management
│   ├── preflight.py      # Dependency checking
│   ├── transcriber.py    # Core transcription logic
│   └── utils.py          # Helper functions
├── tests/                # Unit tests
├── transcribe_meeting.py # Legacy script
├── setup.sh              # Quick setup script
├── pyproject.toml        # Package configuration
├── requirements.txt      # Dependencies
└── README.md
```

## Troubleshooting

### MPS Tensor Error on Apple Silicon

```
TypeError: Cannot convert a MPS Tensor to float64
```

This is a known limitation with word timestamps on MPS. The tool automatically disables word timestamps on MPS. To enable them, force CPU:

```sh
whisper-meetings meeting.wav --device cpu
```

### Pyannote Authentication Error

```
OSError: You need to accept the user conditions...
```

1. Verify you've accepted terms at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
2. Check your token is set: `echo $HUGGINGFACE_HUB_TOKEN`
3. Ensure the token has read permissions

### ffmpeg Not Found

```sh
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows (with Chocolatey)
choco install ffmpeg
```

### Out of Memory

Try a smaller Whisper model:
```sh
whisper-meetings meeting.wav --model small
```

Or force CPU processing:
```sh
whisper-meetings meeting.wav --device cpu
```

### Poor Speaker Separation

- Ensure good audio quality (minimal background noise)
- Specify the number of speakers if known: `--speakers 4`
- Speakers should have distinct voices and minimal overlap

## Development

### Setup Development Environment

```sh
git clone https://github.com/superstaqr/whisper-meetings.git
cd whisper-meetings
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

### Run Tests

```sh
pytest
pytest --cov=whisper_meetings  # with coverage
```

### Code Quality

```sh
ruff check src/
mypy src/
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition
- [Pyannote](https://github.com/pyannote/pyannote-audio) - Speaker diarization
- [PyTorch](https://pytorch.org/) - Machine learning framework
