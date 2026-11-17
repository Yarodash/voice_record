"""The hidden terminal's screen (pyte) and what is read from it: the dictated text.

Claude Code keeps the prompt in a box between two horizontal rules near the bottom of the
screen. The dictated words accumulate there, so that box *is* the recognised text.
Port of `extractDraft` from the VR project's web/src/draft.js.
"""
import re
import threading

import pyte

RULE_CHAR = "─"
LEVEL_BARS = re.compile(r"[▁-█]+$")  # the microphone-level cursor shown while recording
HINT = re.compile(r'^Try ".*"$')  # what Claude Code shows in an empty prompt of a fresh session
PROMPT_PREFIX = re.compile(r"^\s*[>❯]\s?")

# A row is a list of cells (char, skip); `skip` marks the second column of a wide character.
Row = list[tuple[str, bool]]

_ALT_SCREEN = 1049


class _Screen(pyte.Screen):
    """pyte has no alternate screen; switching to/from it at least starts from a clean page."""

    def set_mode(self, *modes, **kwargs):
        if kwargs.get("private") and _ALT_SCREEN in modes:
            self.erase_in_display(2)
        super().set_mode(*modes, **kwargs)

    def reset_mode(self, *modes, **kwargs):
        if kwargs.get("private") and _ALT_SCREEN in modes:
            self.erase_in_display(2)
        super().reset_mode(*modes, **kwargs)


class TerminalScreen:
    """Thread-safe: fed from the PTY reader thread, read from the asyncio loop."""

    def __init__(self, cols: int, rows: int):
        self._lock = threading.Lock()
        self._screen = _Screen(cols, rows)
        self._stream = pyte.Stream(self._screen)
        self.version = 0

    def feed(self, text: str) -> None:
        with self._lock:
            self._stream.feed(text)
            self.version += 1

    def snapshot(self) -> tuple[list[Row], tuple[int, int]]:
        """(rows, (cursor_x, cursor_y))."""
        with self._lock:
            s = self._screen
            rows = [
                [(s.buffer[y][x].data, s.buffer[y][x].data == "") for x in range(s.columns)]
                for y in range(s.lines)
            ]
            return rows, (s.cursor.x, s.cursor.y)

    def text(self) -> str:
        with self._lock:
            return "\n".join(line.rstrip() for line in self._screen.display)

    def draft(self) -> str:
        rows, cursor = self.snapshot()
        return extract_draft(rows, cursor)


def text_of(cells: Row) -> str:
    return "".join(ch or " " for ch, skip in cells if not skip)


def _is_rule(cells: Row) -> bool:
    text = text_of(cells).strip()
    if len(text) < 20:
        return False
    return text.count(RULE_CHAR) / len(text) > 0.5


def prompt_box(rows: list[Row]) -> tuple[int, int] | None:
    """[top, bottom] rule indexes of the prompt box, or None when it is not on screen."""
    rules = [i for i, cells in enumerate(rows) if _is_rule(cells)]
    if not rules:
        return None
    last = rules[-1]
    # The last rule has the prompt line right under it: it is the top of the box. The bottom
    # rule was not repainted, so the prompt runs to the end. (Older rules higher up are leftovers
    # of earlier screens that must not be taken for the top of the box.)
    if last + 1 < len(rows) and re.match(r"^\s*[>❯]", text_of(rows[last + 1])):
        return last, len(rows)
    if len(rules) >= 2:
        return rules[-2], last
    return None


def extract_draft(rows: list[Row], cursor: tuple[int, int] | None = None) -> str:
    """The prompt text *before the cursor*: whatever Claude Code draws after it (a hint, a
    suggestion) looks the same but was not said."""
    box = prompt_box(rows)
    if box is None:
        return ""
    top, bottom = box
    cursor_in_box = cursor is not None and top < cursor[1] < bottom
    last_row = cursor[1] if cursor_in_box else bottom - 1

    lines = []
    for i in range(top + 1, last_row + 1):
        cells = rows[i][: cursor[0]] if cursor_in_box and i == cursor[1] else rows[i]
        text = text_of(cells)
        lines.append((PROMPT_PREFIX.sub("", text, count=1) if i == top + 1 else text).strip())
    draft = re.sub(r"\s+", " ", LEVEL_BARS.sub("", " ".join(lines))).strip()
    return "" if not cursor_in_box and HINT.match(draft) else draft
