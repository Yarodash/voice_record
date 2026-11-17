"""VoiceToText: audio in, text out, through the dictation of a hidden Claude Code."""
import asyncio
import json
import logging
import os
import re
from collections.abc import AsyncIterable
from pathlib import Path

from voice_record.audio import SAMPLE_RATE
from voice_record.audio_devices import CaptureDefaults, PycawBackend
from voice_record.claude_pty import CLAUDE_SETTINGS, ListeningDetector, PtyHost, build_claude_argv, child_env
from voice_record.config import Config, load_config
from voice_record.dictation import Dictation
from voice_record.screen import TerminalScreen, prompt_box
from voice_record.virtual_mic import VirtualMicPlayer

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CHUNK = SAMPLE_RATE * 2 // 50  # 20 ms
# Ctrl+U deletes one line of the prompt at a time, and a long dictation wraps into several.
CLEAR_PROMPT = "\x15" * 20
# Claude Code asks once per new folder whether it may trust it (the folder is our private one).
TRANSCRIBING = "Transcribing"  # Claude Code's footer while the last words are still being recognised
TRUST_DIALOG = re.compile(r"Yes, I trust this folder|Do you trust the files in this folder", re.I)
TRUST_YES_SELECTED = re.compile(r"[❯>]\s*(\d\.\s*)?Yes", re.I)


class VoiceToText:
    """
    async with VoiceToText() as vt:
        text = await vt.transcribe(pcm)          # PCM s16le mono 48 kHz, bytes or async chunks
    """

    def __init__(self, config: Config | None = None, state_dir: Path | None = None,
                 settle_s: float = 0.5, empty_wait_s: float = 3.0, result_timeout_s: float = 15.0,
                 lead_in_s: float = 0.3, prime: bool = True):
        local = ROOT / "config.json"  # your own copy of config.example.json (not in git)
        self.cfg = config or load_config(local if local.exists() else ROOT / "config.example.json")
        self.lead_in_s, self.prime = lead_in_s, prime
        self.state_dir = Path(state_dir or self.default_state_dir())
        self.settle_s, self.empty_wait_s, self.result_timeout_s = settle_s, empty_wait_s, result_timeout_s
        self.defaults = CaptureDefaults(PycawBackend(), self.cfg.audio.virtual_mic,
                                        self.state_dir / "default-mic.json")
        self.player = VirtualMicPlayer(self.cfg.audio.virtual_speaker,
                                        self.cfg.audio.cable_gain_db + self.cfg.audio.mic_gain_db)
        self._host: PtyHost | None = None
        self._screen: TerminalScreen | None = None
        self._dictation: Dictation | None = None
        self._lock = asyncio.Lock()

    @staticmethod
    def default_state_dir() -> Path:
        return ROOT / ".state"

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *exc):
        await self.stop()

    # --- lifecycle -------------------------------------------------------------------------

    async def start(self, timeout_s: float = 60.0) -> None:
        loop = asyncio.get_running_loop()
        workdir = self.state_dir / "claude-cwd"  # empty folder: no project CLAUDE.md is loaded
        workdir.mkdir(parents=True, exist_ok=True)
        settings_file = self.state_dir / "claude-settings.json"
        settings_file.write_text(json.dumps(CLAUDE_SETTINGS), encoding="utf-8")
        await asyncio.to_thread(self.defaults.prepare)  # also restores a mic left switched by a crash
        # PortAudio's first start-up takes seconds; doing it now keeps the first phrase fast.
        await asyncio.to_thread(self.player.warm_up)

        screen = TerminalScreen(self.cfg.terminal.cols, self.cfg.terminal.rows)
        self._screen = screen
        # listen_timeout_s: never start playing long before Claude Code records (the VR bridge uses 1.5)
        self._dictation = Dictation(self._write, ListeningDetector(), self.defaults, self.player,
                                    listen_timeout_s=4.0)

        def deliver() -> None:
            dictation = self._dictation
            if dictation is not None and dictation.state == "warmup":
                dictation.on_pty_output(screen.text())  # looks for `listening…` on the screen

        def on_output(text: str) -> None:  # PTY reader thread
            screen.feed(text)
            loop.call_soon_threadsafe(deliver)

        argv = build_claude_argv(self.cfg.claude, settings_file)
        log.info("starting hidden Claude Code: %s", " ".join(argv))
        self._host = PtyHost(argv, cwd=str(workdir), env=child_env(os.environ),
                             cols=self.cfg.terminal.cols, rows=self.cfg.terminal.rows, on_output=on_output)
        self._host.start()
        await self._wait_ready(timeout_s)
        if self.prime:
            await self._prime()

    async def _prime(self) -> None:
        """One silent dictation right after start-up. The first dictation of a fresh Claude Code
        sometimes lost its first ~2 s even after `listening…` was shown; later ones did not."""
        dictation = self._dictation
        await dictation.press()
        dictation.feed(b"\x00\x00" * SAMPLE_RATE)  # 1 s of silence
        await dictation.release()
        await dictation.wait_idle()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 5.0
        while TRANSCRIBING in self.screen_text and loop.time() < deadline:
            await asyncio.sleep(0.05)
        if self.draft():
            await self._clear_prompt()
        log.info("dictation primed")

    async def _wait_ready(self, timeout_s: float) -> None:
        """Ready = the prompt box has been on screen for a second without interruption."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        shown_since, keys_at = None, 0.0
        while loop.time() < deadline:
            if not self._host.alive:
                raise RuntimeError("Claude Code exited while starting:\n" + self._screen.text())
            text = self._screen.text()
            has_prompt = prompt_box(self._screen.snapshot()[0]) is not None
            if not has_prompt and TRUST_DIALOG.search(text) and loop.time() - keys_at > 0.5:
                # The highlighted option is the default "No, exit": move to "Yes" first and press
                # Enter only once "Yes" is highlighted. (Our own private folder: safe to trust.)
                keys_at = loop.time()
                if TRUST_YES_SELECTED.search(text):
                    log.info("trusting the private working folder %s", self.state_dir / "claude-cwd")
                    self._write("\r")
                else:
                    self._write("\x1b[B")  # arrow down
            if not has_prompt:
                shown_since = None
            elif shown_since is None:
                shown_since = loop.time()
            elif loop.time() - shown_since >= 1.0:
                log.info("Claude Code is ready")
                return
            await asyncio.sleep(0.1)
        raise TimeoutError("Claude Code did not show its prompt in time. Screen:\n" + self._screen.text())

    async def stop(self) -> None:
        if self._dictation is not None:
            try:
                await self._dictation.release()
                await asyncio.wait_for(self._dictation.wait_idle(), 10)
            except Exception:
                log.exception("dictation did not finish cleanly")
        try:
            await asyncio.to_thread(self.defaults.restore)
        except Exception:
            log.exception("could not restore the default microphone")
        if self._host is not None:
            self._host.stop()
            self._host = None

    def _write(self, data: str) -> None:
        if self._host is not None:
            self._host.write(data)

    @property
    def screen_text(self) -> str:
        return self._screen.text() if self._screen else ""

    def draft(self) -> str:
        return self._screen.draft() if self._screen else ""

    # --- recognition -----------------------------------------------------------------------

    async def transcribe(self, audio: bytes | AsyncIterable[bytes]) -> str:
        """PCM s16le mono 48 kHz in, recognised text out. Bytes are played at once (in real
        time by the audio device); an async iterable is streamed as its chunks arrive."""
        async with self._lock:
            if self._host is None or not self._host.alive:
                log.warning("Claude Code is not running; starting it")
                await self.start()
            if self.draft():
                await self._clear_prompt()
            dictation = self._dictation
            await dictation.press()
            dictation.feed(b"\x00\x00" * int(SAMPLE_RATE * self.lead_in_s))  # silence before the first word
            if isinstance(audio, (bytes, bytearray, memoryview)):
                data = bytes(audio)
                for start in range(0, len(data), CHUNK):
                    dictation.feed(data[start:start + CHUNK])
            else:
                async for chunk in audio:
                    dictation.feed(chunk)
            dictation.feed(b"\x00\x00" * (SAMPLE_RATE * 3 // 10))  # 0.3 s of silence after the last word
            await dictation.release()
            await dictation.wait_idle()
            if dictation.error:
                raise RuntimeError(f"dictation failed: {dictation.error}")
            text = await self._wait_settled()
            await self._clear_prompt()
            return text

    async def _wait_settled(self) -> str:
        """The text counts as final once Claude Code no longer shows `Transcribing…` and the
        text has not changed for `settle_s` (like the VR page)."""
        loop = asyncio.get_running_loop()
        start = loop.time()
        last, changed_at = self.draft(), start
        while loop.time() - start < self.result_timeout_s:
            await asyncio.sleep(0.05)
            now, text = loop.time(), self.draft()
            if text != last or TRANSCRIBING in self.screen_text:
                last, changed_at = text, now
            if now - changed_at >= self.settle_s and (last or now - start >= self.empty_wait_s):
                break
        else:
            log.warning("the text was still changing after %.0f s", self.result_timeout_s)
        return last

    async def _clear_prompt(self) -> None:
        loop = asyncio.get_running_loop()
        for _ in range(3):
            self._write(CLEAR_PROMPT)
            deadline = loop.time() + 2.0
            while loop.time() < deadline:
                await asyncio.sleep(0.05)
                if not self.draft():
                    return
        log.warning("could not empty the prompt; it still shows: %r", self.draft())
