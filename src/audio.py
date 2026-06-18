"""Audio sources for Stage 2.

Three concrete sources:
  - MicSource:      live capture from the default input device via sounddevice
  - FileSource:     read a 16-bit mono WAV file in fixed-size chunks (tests + replay)
  - PlaybackSource: play a WAV through the speakers AND yield its bytes at
                    real-time speed — used by the simulator so you hear the
                    narration AND the STT pipeline sees the same audio.

All yield raw PCM bytes at the WAV's native rate (16 kHz mono is what Vosk wants).
"""

from __future__ import annotations

import threading
import time
import wave
from queue import Empty, Queue
from typing import Iterator

from .pcm import resample  # offline PCM resample (audioop-backed, 3.13-safe)


# ---------- file source ----------

class FileSource:
    """Read a 16-bit mono WAV file in chunks. Used for tests and offline replay."""

    def __init__(self, path: str, chunk_ms: int = 200) -> None:
        self.path = path
        self.chunk_ms = chunk_ms
        with wave.open(path, "rb") as w:
            self.sample_rate = w.getframerate()
            self.channels = w.getnchannels()
            self.sampwidth = w.getsampwidth()
        if self.channels != 1 or self.sampwidth != 2:
            raise ValueError(
                f"{path}: need mono 16-bit PCM, got {self.channels} ch / {self.sampwidth*8} bit"
            )

    def chunks(self) -> Iterator[bytes]:
        frames_per_chunk = int(self.sample_rate * self.chunk_ms / 1000)
        with wave.open(self.path, "rb") as w:
            while True:
                buf = w.readframes(frames_per_chunk)
                if not buf:
                    return
                yield buf

    def stop(self) -> None:
        pass  # file iteration ends naturally


# ---------- microphone source ----------

class MicSource:
    """Live mic capture via sounddevice. Background-thread, yields PCM chunks.

    sounddevice opens a callback-driven InputStream. The callback runs on an
    audio thread and pushes raw bytes into a Queue; `chunks()` drains the
    queue from the consumer thread. `stop()` is safe to call from anywhere.
    """

    def __init__(self, sample_rate: int = 16000, chunk_ms: int = 200, device: int | str | None = None) -> None:
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self.device = device
        self.blocksize = int(sample_rate * chunk_ms / 1000)
        self._queue: Queue[bytes | None] = Queue()
        self._stop = threading.Event()
        self._stream = None  # type: ignore[assignment]
        self._int16 = True   # ask PortAudio for native int16; fall back to float32

    def _callback(self, indata, frames, time_info, status) -> None:
        if self._stop.is_set():
            return
        if self._int16:
            # PortAudio delivered native int16 — no float round-trip/truncation
            self._queue.put(indata[:, 0].tobytes())
            return
        import numpy as np
        pcm16 = (indata[:, 0] * 32767.0).clip(-32768, 32767).astype(np.int16)
        self._queue.put(pcm16.tobytes())

    def _open(self, sd, dtype: str):
        return sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=dtype,
            blocksize=self.blocksize,
            device=self.device,
            callback=self._callback,
        )

    def start(self) -> None:
        import sounddevice as sd
        try:
            self._int16 = True
            self._stream = self._open(sd, "int16")
            self._stream.start()
        except Exception:
            # device refused int16 — fall back to float32 + manual conversion
            self._int16 = False
            self._stream = self._open(sd, "float32")
            self._stream.start()

    def chunks(self) -> Iterator[bytes]:
        if self._stream is None:
            self.start()
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.5)
            except Empty:
                continue
            if item is None:
                return
            yield item

    def stop(self) -> None:
        self._stop.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self._queue.put(None)  # unblock any waiting iterator


# ---------- playback source (speakers + STT in lockstep) ----------

class PlaybackSource:
    """Plays a 16-bit mono WAV through the speakers AND yields the same bytes.

    The yields are real-time-paced because `sounddevice.RawOutputStream.write`
    blocks until the audio buffer can accept more — so the consumer
    (STT engine) sees audio at exactly the speed the listener hears it. That
    keeps the highlight tracking aligned with the spoken word.
    """

    def __init__(self, wav_path: str, chunk_ms: int = 100, mute: bool = False) -> None:
        self.wav_path = wav_path
        self.chunk_ms = chunk_ms
        self.mute = mute
        self._stop = threading.Event()
        with wave.open(wav_path, "rb") as w:
            self.sample_rate = w.getframerate()
            self.channels = w.getnchannels()
            self.sampwidth = w.getsampwidth()
        if self.channels != 1 or self.sampwidth != 2:
            raise ValueError(
                f"{wav_path}: need mono 16-bit PCM, got {self.channels} ch / {self.sampwidth*8} bit"
            )
        self.chunk_frames = max(1, int(self.sample_rate * chunk_ms / 1000))

    def _start_playback(self) -> bool:
        """Begin decoupled speaker playback of the whole WAV.

        Playback is driven by PortAudio's own callback thread from a full
        in-memory buffer, so STT/embedding work on the consumer side can't starve
        the device buffer (the old design wrote to the device *inside* the STT
        loop, which underran → ALSA xrun → native crash). The buffer is also
        resampled to the output device's native rate so ALSA never has to plug-
        resample 16 kHz on a 44.1/48 kHz device. Returns True if playback began.
        """
        import sounddevice as sd
        try:
            import numpy as np
            with wave.open(self.wav_path, "rb") as w:
                pcm_all = w.readframes(w.getnframes())
            play_pcm, play_sr = pcm_all, self.sample_rate
            try:
                dev_sr = int(sd.query_devices(kind="output")["default_samplerate"])
            except Exception:
                dev_sr = 0
            if dev_sr and dev_sr != self.sample_rate:
                play_pcm = resample(pcm_all, self.sample_rate, dev_sr)
                play_sr = dev_sr
            data = np.frombuffer(play_pcm, dtype=np.int16)
            sd.play(data, samplerate=play_sr, blocking=False)
            return True
        except Exception as e:
            print(f"[PlaybackSource] no audio device ({e}); pacing without playback")
            return False

    def chunks(self) -> Iterator[bytes]:
        import sounddevice as sd  # local import — keeps headless callers happy
        playing = self._start_playback() if not self.mute else False
        try:
            # Yield 16 kHz chunks paced by wall-clock for STT — independent of the
            # speaker device, so playback and recognition can't deadlock or starve
            # each other. Both advance at real time, staying aligned for the highlight.
            with wave.open(self.wav_path, "rb") as w:
                last_t = time.monotonic()
                while not self._stop.is_set():
                    pcm = w.readframes(self.chunk_frames)
                    if not pcm:
                        return
                    target_t = last_t + (self.chunk_ms / 1000.0)
                    delay = target_t - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                    last_t = time.monotonic()
                    yield pcm
        finally:
            if playing:
                try:
                    sd.stop()
                except Exception:
                    pass

    def stop(self) -> None:
        self._stop.set()
