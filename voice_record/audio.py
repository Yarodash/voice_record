"""Audio helpers: everything becomes PCM s16le mono 48 kHz, the format VoiceToText takes."""
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 48000


def to_pcm48k(samples: np.ndarray, rate: int) -> bytes:
    """float samples in [-1, 1], shape (n,) or (n, channels), any rate -> PCM s16le mono 48 kHz."""
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if rate != SAMPLE_RATE and len(samples):
        n_out = int(round(len(samples) * SAMPLE_RATE / rate))
        samples = np.interp(np.arange(n_out) * (rate / SAMPLE_RATE), np.arange(len(samples)), samples)
    return (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def load_wav(path: str | Path) -> bytes:
    """Any PCM WAV (8/16/24/32-bit, any rate and channel count) -> PCM s16le mono 48 kHz."""
    with wave.open(str(path), "rb") as w:
        rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128
    elif width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
    elif width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        ints = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        ints = np.where(ints & 0x800000, ints - 0x1000000, ints)
        data = ints.astype(np.float32) / 8388608
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648
    else:
        raise ValueError(f"unsupported sample width: {width} bytes")
    return to_pcm48k(data.reshape(-1, channels), rate)


class StreamResampler:
    """Linear resampling of a live stream, block by block, without clicks at block edges."""

    def __init__(self, rate: int):
        self._step = rate / SAMPLE_RATE  # input samples per output sample
        self._pos = 0.0  # position of the next output sample, relative to self._prev
        self._prev = np.zeros(0, dtype=np.float32)

    def process(self, mono: np.ndarray) -> bytes:
        if self._step == 1.0:
            return to_pcm48k(mono, SAMPLE_RATE)
        data = np.concatenate([self._prev, np.asarray(mono, dtype=np.float32)])
        if len(data) < 2:
            self._prev = data
            return b""
        positions = np.arange(self._pos, len(data) - 1, self._step)
        out = np.interp(positions, np.arange(len(data)), data)
        next_pos = (positions[-1] + self._step) if len(positions) else self._pos
        keep = int(next_pos)  # keep the sample just before the next position
        self._prev = data[keep:]
        self._pos = next_pos - keep
        return (np.clip(out, -1.0, 1.0) * 32767).astype("<i2").tobytes()
