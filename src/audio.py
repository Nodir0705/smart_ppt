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

    def _callback(self, indata, frames, time_info, status) -> None:
        if self._stop.is_set():
            return
        # indata is float32 by default; we want 16-bit PCM bytes for Vosk
        import numpy as np
        pcm16 = (indata[:, 0] * 32767.0).clip(-32768, 32767).astype(np.int16)
        self._queue.put(pcm16.tobytes())

    def start(self) -> None:
        import sounddevice as sd
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.blocksize,
            device=self.device,
            callback=self._callback,
        )
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

    def chunks(self) -> Iterator[bytes]:
        import sounddevice as sd  # local import — keeps headless callers happy
        stream = None
        if not self.mute:
            try:
                stream = sd.RawOutputStream(
                    samplerate=self.sample_rate, channels=1, dtype="int16",
                    blocksize=self.chunk_frames,
                )
                stream.start()
            except Exception as e:
                # no audio device — fall back to wall-clock pacing
                print(f"[PlaybackSource] no audio device ({e}); pacing without playback")
                stream = None
        try:
            with wave.open(self.wav_path, "rb") as w:
                last_t = time.monotonic()
                while not self._stop.is_set():
                    pcm = w.readframes(self.chunk_frames)
                    if not pcm:
                        return
                    if stream is not None:
                        stream.write(pcm)  # blocks → real-time pacing
                    else:
                        # no speakers: pace by wall clock so STT runs at real-time
                        target_t = last_t + (self.chunk_ms / 1000.0)
                        delay = target_t - time.monotonic()
                        if delay > 0:
                            time.sleep(delay)
                        last_t = time.monotonic()
                    yield pcm
        finally:
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass

    def stop(self) -> None:
        self._stop.set()
