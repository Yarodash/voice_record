"""Speech to text through the built-in dictation of a hidden Claude Code."""
from voice_record.audio import SAMPLE_RATE, load_wav, to_pcm48k
from voice_record.core import VoiceToText

__all__ = ["VoiceToText", "load_wav", "to_pcm48k", "SAMPLE_RATE"]
