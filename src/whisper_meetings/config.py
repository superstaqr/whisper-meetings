"""Configuration management for whisper-meetings."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class Config:
    """Configuration settings for whisper-meetings."""

    # Model settings
    whisper_model: str = "large"
    whisper_cache: Path = field(default_factory=lambda: Path.home() / "models" / "whisper")
    hf_cache: Path = field(default_factory=lambda: Path.home() / "models" / "huggingface")
    diarization_model: str = "pyannote/speaker-diarization-3.1"

    # Runtime settings
    device: str | None = None  # None = auto-detect
    language: str = "en"

    # Feature flags
    auto_update_deps: bool = False

    def __post_init__(self) -> None:
        """Initialize from environment variables."""
        self.whisper_cache = Path(
            os.getenv("WHISPER_CACHE", str(self.whisper_cache))
        ).expanduser()
        self.hf_cache = Path(os.getenv("HF_CACHE", str(self.hf_cache))).expanduser()
        self.whisper_model = os.getenv("WHISPER_MODEL", self.whisper_model)
        self.device = os.getenv("WHISPER_DEVICE") or self.device
        self.language = os.getenv("WHISPER_LANGUAGE", self.language)
        self.auto_update_deps = os.getenv("AUTO_UPDATE_DEPS") == "1"

    @property
    def hf_token(self) -> str | None:
        """Get Hugging Face token from environment."""
        return os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")

    def ensure_cache_dirs(self) -> None:
        """Ensure model cache directories exist."""
        self.whisper_cache.mkdir(parents=True, exist_ok=True)
        self.hf_cache.mkdir(parents=True, exist_ok=True)


# Global default configuration
_config: Config | None = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config
