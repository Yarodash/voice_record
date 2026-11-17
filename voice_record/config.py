"""Typed view of config.json."""
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TerminalConfig:
    cols: int = 120
    rows: int = 30


@dataclass(frozen=True)
class AudioConfig:
    virtual_speaker: str = "Speakers (Steam Streaming Microphone)"
    virtual_mic: str = "Microphone (Steam Streaming Microphone)"
    cable_gain_db: float = -30  # the Steam driver adds exactly +30 dB
    mic_gain_db: float = 0


@dataclass(frozen=True)
class ClaudeConfig:
    model: str = "haiku"  # "" = the user's default model
    effort: str = ""  # "" = the user's default effort
    extra_args: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Config:
    terminal: TerminalConfig
    audio: AudioConfig
    claude: ClaudeConfig


def load_config(path: Path) -> Config:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return Config(
        terminal=TerminalConfig(**raw.get("terminal", {})),
        audio=AudioConfig(**raw.get("audio", {})),
        claude=ClaudeConfig(**raw.get("claude", {})),
    )
