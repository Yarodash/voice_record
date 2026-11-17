"""python -m voice_record — speech to text through a hidden Claude Code's dictation.

    python -m voice_record phrase.wav [more.wav ...]   # print the text of each file
    python -m voice_record --mic                       # Enter: start, Enter: stop, q: quit
    python -m voice_record --stdin < audio.pcm         # raw PCM s16le mono 48 kHz
"""
import argparse
import asyncio
import logging
import sys
import time

from voice_record.audio import StreamResampler, load_wav
from voice_record.audio_devices import CaptureDefaults, PycawBackend
from voice_record.core import VoiceToText

log = logging.getLogger("voice_record")


async def run_files(vt: VoiceToText, paths: list[str]) -> None:
    for path in paths:
        pcm = load_wav(path)
        started = time.perf_counter()
        text = await vt.transcribe(pcm)
        log.info("%s: %.1f s of audio recognised in %.1f s", path, len(pcm) / 96000, time.perf_counter() - started)
        print(text, flush=True)


async def run_stdin(vt: VoiceToText) -> None:
    pcm = await asyncio.to_thread(sys.stdin.buffer.read)
    print(await vt.transcribe(pcm), flush=True)


def _wasapi_inputs() -> list[tuple[int, dict]]:
    import sounddevice as sd

    return [(index, dev) for index, dev in enumerate(sd.query_devices())
            if "WASAPI" in sd.query_hostapis(dev["hostapi"])["name"] and dev["max_input_channels"] > 0]


def _open_mic(name: str, on_pcm):
    """Opens the user's own microphone by name (WASAPI, shared mode): runs on the audio thread."""
    import sounddevice as sd

    inputs = _wasapi_inputs()
    found = [(i, d) for i, d in inputs if d["name"] == name] or [(i, d) for i, d in inputs if name in d["name"]]
    if not found:
        names = "\n  ".join(d["name"] for _, d in inputs)
        raise LookupError(f"микрофон {name!r} не найден. Есть:\n  {names}")
    index, dev = found[0]
    rate = int(dev["default_samplerate"])
    resampler = StreamResampler(rate)

    def callback(indata, frames, time_info, status):
        on_pcm(resampler.process(indata.mean(axis=1)))

    stream = sd.InputStream(device=index, samplerate=rate, channels=min(2, dev["max_input_channels"]),
                            dtype="float32", blocksize=rate // 50, callback=callback)
    stream.start()
    return stream


async def run_mic(vt: VoiceToText, device: str | None) -> None:
    loop = asyncio.get_running_loop()
    # The user's own microphone, captured by name now: while recording, the Windows default
    # becomes the virtual mic, but an already open stream keeps reading the real device.
    name = device or await asyncio.to_thread(vt.defaults.user_default_name)
    if name == vt.cfg.audio.virtual_mic:
        inputs = await asyncio.to_thread(vt.player.call, _wasapi_inputs)
        names = "\n  ".join(d["name"] for _, d in inputs if d["name"] != name)
        raise SystemExit(f"Микрофон по умолчанию в Windows — сам виртуальный ({name}).\n"
                         f"Укажите настоящий: --device \"<имя>\". Есть:\n  {names}")
    target: list[asyncio.Queue | None] = [None]

    def on_pcm(pcm: bytes) -> None:  # audio thread
        queue = target[0]
        if queue is not None and pcm:
            loop.call_soon_threadsafe(queue.put_nowait, pcm)

    stream = await asyncio.to_thread(vt.player.call, _open_mic, name, on_pcm)
    print(f"Микрофон: {name}", file=sys.stderr, flush=True)
    try:
        while True:
            line = await asyncio.to_thread(input, "Enter — говорить, q — выход: ")
            if line.strip().lower() in ("q", "й", "quit", "exit"):
                return
            queue: asyncio.Queue = asyncio.Queue()
            target[0] = queue

            async def chunks():
                while (pcm := await queue.get()) is not None:
                    yield pcm

            async def wait_stop():
                await asyncio.to_thread(input, "Запись... Enter — стоп: ")
                target[0] = None
                queue.put_nowait(None)

            stopper = asyncio.create_task(wait_stop())
            started = time.perf_counter()
            text = await vt.transcribe(chunks())
            await stopper
            log.info("recognised %.1f s after the start", time.perf_counter() - started)
            print(text or "(ничего не распознано)", flush=True)
    finally:
        await asyncio.to_thread(vt.player.call, lambda: (stream.stop(), stream.close()))


async def main_async(args) -> None:
    vt = VoiceToText()
    started = time.perf_counter()
    await vt.start()
    log.info("hidden Claude Code ready in %.1f s", time.perf_counter() - started)
    try:
        if args.mic:
            await run_mic(vt, args.device)
        elif args.stdin:
            await run_stdin(vt)
        else:
            await run_files(vt, args.wav)
    finally:
        await vt.stop()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="python -m voice_record",
                                     description="Речь в текст через встроенную диктовку Claude Code.")
    parser.add_argument("wav", nargs="*", help="WAV-файлы (любая частота и число каналов)")
    parser.add_argument("--mic", action="store_true", help="говорить в свой микрофон: Enter — старт, Enter — стоп")
    parser.add_argument("--device", help="для --mic: имя микрофона (по умолчанию — микрофон Windows по умолчанию)")
    parser.add_argument("--stdin", action="store_true", help="сырой PCM s16le моно 48 кГц со стандартного входа")
    parser.add_argument("-v", "--verbose", action="store_true", help="подробный журнал в stderr")
    args = parser.parse_args()
    if not (args.wav or args.mic or args.stdin):
        parser.error("нужен WAV-файл, --mic или --stdin")
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, stream=sys.stderr,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        pass
    finally:
        # Safety net: whatever happened above, the user's own microphone is the default again.
        vt_defaults = CaptureDefaults(PycawBackend(), "", VoiceToText.default_state_dir() / "default-mic.json")
        vt_defaults.restore()


if __name__ == "__main__":
    main()
