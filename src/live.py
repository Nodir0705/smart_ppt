"""Glue between an AudioSource → STTEngine → RollingBuffer → callback.

Lives in its own module so the viewer's Tk code stays thin and the
audio→STT→buffer pipeline can be exercised end-to-end without a window.

    transcriber = LiveTranscriber(source, engine, buffer, hotwords)
    transcriber.run(on_update=lambda text: ...)   # blocks until source ends

For Tk integration, run this in a daemon thread and have the callback push
the buffer text into a queue that the Tk main loop drains.
"""

from __future__ import annotations

import threading
from typing import Callable, Optional

from .stt import Partial, RollingBuffer, STTEngine


class LiveTranscriber:
    def __init__(
        self,
        source,
        engine: STTEngine,
        buffer: Optional[RollingBuffer] = None,
        hotwords: Optional[list[str]] = None,
    ) -> None:
        self.source = source
        self.engine = engine
        self.buffer = buffer or RollingBuffer()
        self.hotwords = hotwords
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def run(self, on_update: Callable[[str, Partial], None]) -> None:
        """Pump audio→STT→buffer until the source runs out (or stop() is called).

        `on_update(buffer_text, partial)` is called for every Partial emitted,
        with the buffer's text *after* the partial has been folded in.
        """
        for partial in self.engine.transcribe_stream(self._chunks(), hotwords=self.hotwords):
            if self._stop.is_set():
                break
            self.buffer.push(partial)
            on_update(self.buffer.text, partial)

    def _chunks(self):
        """Source chunks, each shown to the buffer (pause detection) before STT."""
        sr = getattr(self.source, "sample_rate", 16000)
        for chunk in self.source.chunks():
            if chunk:
                self.buffer.audio(chunk, sr)
            yield chunk

    def start_in_thread(
        self,
        on_update: Callable[[str, Partial], None],
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> None:
        """Run `run()` on a daemon thread. Returns immediately.

        If the pump dies (e.g. the mic stream drops), `on_error(exc)` is invoked
        from the worker thread so the host can surface it / restart; without a
        handler the traceback is printed (the old behaviour) so it isn't silent.
        """
        if self._thread is not None:
            return
        def _target():
            try:
                self.run(on_update)
            except Exception as e:  # surface but don't crash the host
                if on_error is not None:
                    on_error(e)
                else:
                    import traceback
                    traceback.print_exception(type(e), e, e.__traceback__)
        self._thread = threading.Thread(target=_target, daemon=True, name="LiveTranscriber")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self.source.stop()
        except Exception:
            pass
