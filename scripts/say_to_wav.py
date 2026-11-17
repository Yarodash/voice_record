"""Synthesises a phrase with Windows TTS (System.Speech) into a WAV file, for testing.

    .venv\\Scripts\\python scripts\\say_to_wav.py "Привет, это проверка" phrase.wav
"""
import os
import subprocess
import sys
from pathlib import Path

TTS = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$ru = $s.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -eq 'ru-RU' } | Select-Object -First 1
if ($ru) { $s.SelectVoice($ru.VoiceInfo.Name) }
[Console]::Error.WriteLine("voice: " + $s.Voice.Name)
$s.SetOutputToWaveFile($env:TTS_WAV)
$s.Speak($env:TTS_TEXT)
$s.Dispose()
"""


def main() -> None:
    text, out = sys.argv[1], Path(sys.argv[2]).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "TTS_WAV": str(out), "TTS_TEXT": text}
    subprocess.run(["powershell.exe", "-NoProfile", "-Command", TTS], check=True, env=env)
    print(out)


if __name__ == "__main__":
    main()
