# whisper-meetings

Transcribe meetings with Whisper and add speaker labels using Pyannote diarization.

## Features
- Loads Whisper and Pyannote models from local caches (`~/models/whisper`, `~/models/huggingface`)
- Uses Apple Silicon acceleration (`mps`) when available
- Speaker attribution picks the label with the maximum time overlap per segment
- Saves a timestamped, speaker-tagged transcript next to your audio file
- Reads your Hugging Face token from environment or a `.env` file

## Prerequisites
- Python 3.10+
- ffmpeg (required by Whisper)
  - macOS (Homebrew): `brew install ffmpeg`
- A Hugging Face account and access token
  - Accept the terms for `pyannote/speaker-diarization-3.1`: https://huggingface.co/pyannote/speaker-diarization-3.1
  - Create a token: https://huggingface.co/settings/tokens
- For best performance on Apple Silicon, ensure macOS 12.3+; PyTorch will use MPS automatically when available.

## Setup
1. Create and activate a virtual environment
   - macOS/Linux (zsh):
     ```sh
     python3 -m venv venv
     source venv/bin/activate
     ```
2. Install dependencies
   ```sh
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   If you have trouble installing PyTorch, follow platform-specific instructions at https://pytorch.org/get-started/locally/

3. Provide your Hugging Face token
   - Option A: Use a `.env` file (recommended for local dev)
     ```sh
     cp .env.example .env
     # edit .env and paste your token for HUGGINGFACE_HUB_TOKEN
     ```
   - Option B: Export as an environment variable
     ```sh
     export HUGGINGFACE_HUB_TOKEN=hf_your_token_here
     ```

4. Ensure model cache folders exist (the script will create them if missing):
   - Whisper cache: `~/models/whisper`
   - Hugging Face cache: `~/models/huggingface`

Models will automatically download into those folders the first time you run the script.

## Usage
Basic usage:
```sh
activate your venv first
python transcribe_meeting.py "New Recording.m4a"
```
With an explicit number of speakers:
```sh
python transcribe_meeting.py path/to/meeting.wav 4
```

Device override:
- By default on Apple Silicon, the script uses MPS. To force CPU (e.g., to enable word timestamps), set an env var:
  ```sh
  export WHISPER_DEVICE=cpu
  python transcribe_meeting.py "New Recording.m4a"
  ```

Outputs:
- Prints a timestamped transcript with speaker labels to the console
- Writes a `.txt` file next to your audio (e.g., `New Recording.txt`)
- Shows a simple speaker summary at the end

## Notes
- Whisper model used: `large` by default; for slower machines, consider changing to a smaller model (e.g., `medium`, `small`) in the script.
- Speaker mapping relies on overlap between Whisper segments and Pyannote diarization turns; ties are broken by longest overlap.
- If you see an authentication error for Pyannote, verify you accepted the model terms and your `HUGGINGFACE_HUB_TOKEN` is set correctly.

## Troubleshooting
- "TypeError: Cannot convert a MPS Tensor to float64": Whisper's word timestamps path uses a float64 operation unsupported on MPS. The script automatically disables `word_timestamps` on MPS to avoid this. If you need word-level timestamps, force CPU:
  ```sh
  export WHISPER_DEVICE=cpu
  python transcribe_meeting.py path/to/audio.wav
  ```
