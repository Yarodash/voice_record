"""The real interactive Claude Code, run inside a pseudo-terminal (only its dictation is used)."""
import logging
import subprocess
import threading
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path

from voice_record.config import ClaudeConfig

log = logging.getLogger(__name__)

# Applies to the hidden session only (passed with --settings), not to the user's own Claude Code.
# Nothing is submitted on release: the words stay in the prompt until we read and clear them.
CLAUDE_SETTINGS = {"voice": {"enabled": True, "mode": "hold", "autoSubmit": False}}

def build_claude_argv(cfg: ClaudeConfig, settings_file: Path) -> list[str]:
    # Always a fresh session: it is only a speech recogniser, there is nothing to remember.
    argv = ["claude", "--session-id", str(uuid.uuid4()), "--settings", str(settings_file)]
    if cfg.model:
        argv += ["--model", cfg.model]
    if cfg.effort:
        argv += ["--effort", cfg.effort]
    return [*argv, *cfg.extra_args]


def child_env(environ: Mapping[str, str]) -> dict[str, str]:
    """Drops the variables that mark "running inside another Claude Code session"."""

    def is_session_marker(name: str) -> bool:
        upper = name.upper()
        return upper in ("CLAUDECODE", "CLAUDE_PID") or upper.startswith("CLAUDE_CODE_")

    return {k: v for k, v in environ.items() if not is_session_marker(k)}


class ListeningDetector:
    """Spots the `listening…` footer that Claude Code shows once dictation is recording.

    Fed with the *rendered* screen, not the raw output: Claude Code repaints only the cells
    that changed, so after "keep holding…" the raw stream carries e.g. "listning…", never the
    whole word.
    """

    WORD = "listening"

    def __init__(self):
        self._armed = False
        self._seen = False

    def reset(self) -> None:
        self._armed, self._seen = True, False

    def feed(self, screen_text: str) -> bool:
        if self._armed and not self._seen:
            self._seen = self.WORD in screen_text
        return self._seen


class PtyHost:
    def __init__(
        self,
        argv: list[str],
        cwd: str,
        env: Mapping[str, str] | None,
        cols: int,
        rows: int,
        on_output: Callable[[str], None],
        on_exit: Callable[[], None] | None = None,
    ):
        self._argv, self._cwd, self._env = argv, cwd, env
        self._cols, self._rows = cols, rows
        self._on_output, self._on_exit = on_output, on_exit
        self._lock = threading.Lock()
        self._proc = None

    def start(self) -> None:
        from winpty import PtyProcess

        self._proc = PtyProcess.spawn(self._argv, cwd=self._cwd, env=self._env, dimensions=(self._rows, self._cols))
        threading.Thread(target=self._read_loop, name="pty-reader", daemon=True).start()

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.isalive()

    def write(self, data: str) -> None:
        with self._lock:
            if self.alive:
                self._proc.write(data)

    def stop(self) -> None:
        proc = self._proc
        if proc is None or not proc.isalive():
            return
        try:
            proc.terminate(force=True)
        except OSError:
            pass  # e.g. "Access is denied" from os.kill; fall back to taskkill below
        if proc.isalive():
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)

    def _read_loop(self) -> None:
        proc = self._proc
        try:
            while True:
                data = proc.read(4096)
                if data:
                    self._on_output(data)
        except EOFError:
            pass
        except Exception:
            log.exception("terminal reader stopped")
        if self._on_exit is not None:
            self._on_exit()
