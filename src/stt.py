"""Stage 2 — speech to text.

This module defines the STT abstractions and a couple of concrete engines.
The viewer (or any other consumer) only ever depends on the `STTEngine`
Protocol and the `RollingBuffer`; engines are swappable.

    AudioSource.chunks()  ──► STTEngine.transcribe_stream()  ──► Partial
                                                                     │
                                                                     ▼
                                                               RollingBuffer
                                                                     │
                                                                     ▼
                                                   matcher.match(buffer.text, slide_no)

Hotwords:
    `hotwords_from_index(index)` returns the deck's distinctive vocabulary
    so an engine can bias toward those terms. VoskEngine supports two modes:
      - free-form (default): standard model, hotwords ignored
      - grammar=True:        constrained to deck vocabulary (perfect on-deck
                             recognition, cannot transcribe off-deck speech)
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Protocol


# ---------- partial transcript record ----------

@dataclass
class Partial:
    """A single STT emission. `is_final` means the engine has committed this
    text to the transcript (it will not change). Non-final partials are
    living guesses that the engine may revise as more audio arrives.
    """

    text: str
    is_final: bool
    timestamp: float = field(default_factory=time.monotonic)


# ---------- engine protocol ----------

class STTEngine(Protocol):
    """Streaming speech-to-text. Yields Partial events for each chunk."""

    sample_rate: int

    def transcribe_stream(
        self,
        audio_chunks: Iterator[bytes],
        hotwords: list[str] | None = None,
    ) -> Iterator[Partial]:
        ...


# ---------- rolling buffer ----------

class RollingBuffer:
    """Assembles a rolling window of recently spoken text for the matcher.

    The matcher cares about *what the speaker is currently saying*, not the
    full conversation history. We keep:
      - the last `window_seconds` of finalised segments
      - whatever the current non-final partial is

    `text` returns the concatenation of those two — that's what the matcher
    operates on each tick.
    """

    def __init__(self, window_seconds: float = 4.0) -> None:
        self.window_seconds = window_seconds
        self._finals: deque[tuple[float, str]] = deque()
        self._current_partial: str = ""

    def push(self, partial: Partial) -> None:
        if partial.is_final:
            txt = partial.text.strip()
            if txt:
                self._finals.append((partial.timestamp, txt))
            self._current_partial = ""
            self._prune(partial.timestamp)
        else:
            self._current_partial = partial.text.strip()
            self._prune(partial.timestamp)

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._finals and self._finals[0][0] < cutoff:
            self._finals.popleft()

    @property
    def text(self) -> str:
        parts = [t for _, t in self._finals]
        if self._current_partial:
            parts.append(self._current_partial)
        return " ".join(parts).strip()

    def tail(self, n_words: int) -> str:
        """Return only the last `n_words` of the buffer text.

        Vosk often does not finalise mid-sentence — its partial grows for the
        entire utterance. Feeding the whole partial to the matcher means
        early-mentioned elements stay in play even after the speaker has
        moved on. Slicing to the last N words mimics "recent speech only"
        and lets the highlight track the *current* phrase.
        """
        words = self.text.split()
        if n_words <= 0 or n_words >= len(words):
            return " ".join(words)
        return " ".join(words[-n_words:])

    def clear(self) -> None:
        self._finals.clear()
        self._current_partial = ""


# ---------- hotword extraction ----------

_WORD = re.compile(r"[A-Za-z][A-Za-z']+")

_GENERIC_STOPWORDS = {
    "the", "a", "an", "is", "in", "on", "of", "and", "or", "to", "for", "with",
    "this", "that", "it", "its", "as", "by", "at", "be", "are", "was", "were",
    "but", "not", "no", "yes", "if", "then", "than", "so", "do", "does", "did",
    "from", "into", "out", "we", "you", "they", "i", "he", "she",
    "your", "their", "our", "his", "her", "my", "me",
    "have", "has", "had", "will", "would", "can", "could", "should",
}


def hotwords_from_index(index: dict, min_len: int = 2) -> list[str]:
    """Extract distinctive lowercased single words from an index for STT biasing.

    We pull every word that appears in any element's `trigger_phrases`, drop
    generic stopwords, and de-duplicate. Single-word output because most
    streaming engines (Vosk, faster-whisper) accept a *vocabulary list*, not
    multi-word phrases.
    """
    seen: set[str] = set()
    out: list[str] = []
    for slide in index.get("slides", []):
        for el in slide.get("elements", []):
            for phrase in el.get("trigger_phrases") or []:
                for tok in _WORD.findall(phrase.lower()):
                    if len(tok) < min_len:
                        continue
                    if tok in _GENERIC_STOPWORDS:
                        continue
                    if tok in seen:
                        continue
                    seen.add(tok)
                    out.append(tok)
    return out


# ---------- mock engine (deterministic; used in tests + offline demo) ----------

@dataclass
class MockScript:
    """One emission from a mocked transcription run."""
    text: str
    is_final: bool
    delay_s: float = 0.0  # sleep before yielding (lets a consumer breathe)


class MockEngine:
    """Replay a scripted sequence of Partial events.

    Ignores the audio entirely — useful for deterministic pipeline tests and
    for demos where you don't want to plug a microphone in.
    """

    def __init__(self, script: list[MockScript], sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self._script = list(script)

    def transcribe_stream(
        self,
        audio_chunks: Iterator[bytes],
        hotwords: list[str] | None = None,
    ) -> Iterator[Partial]:
        # we don't consume audio_chunks, but we don't want to leave a generator
        # half-consumed for the caller either — just leave it untouched.
        del audio_chunks, hotwords
        for step in self._script:
            if step.delay_s:
                time.sleep(step.delay_s)
            yield Partial(text=step.text, is_final=step.is_final)


# ---------- Vosk engine (offline, streaming, local) ----------

class VoskEngine:
    """Offline streaming STT via Vosk (small-en-us model by default).

    Two modes:
      - free-form (default): full standard model, accepts arbitrary English.
        Hotwords are ignored at the engine level — we rely on the matcher's
        semantic embedding to be robust to STT errors on deck-specific terms.
      - grammar=True: the recognizer is constrained to the supplied hotwords
        (deck vocabulary). On-deck terms transcribe near-perfectly; off-deck
        speech is forced into the closest match (or rejected). Use this when
        the speaker stays strictly on-topic.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        model_lang: str = "en-us",
        model_path: str | None = None,
        grammar: bool = False,
    ) -> None:
        import vosk  # local import — heavy
        vosk.SetLogLevel(-1)
        if model_path:
            self._model = vosk.Model(model_path=model_path)
        else:
            self._model = vosk.Model(lang=model_lang)
        self._Recognizer = vosk.KaldiRecognizer
        self.sample_rate = sample_rate
        self.grammar_mode = grammar

    def _make_recognizer(self, hotwords: list[str] | None):
        if self.grammar_mode and hotwords:
            # Vosk accepts a JSON list of allowed phrases; "[unk]" lets it
            # produce <UNK> tokens for off-vocab speech instead of refusing.
            allowed = json.dumps(list(dict.fromkeys(hotwords + ["[unk]"])))
            return self._Recognizer(self._model, self.sample_rate, allowed)
        return self._Recognizer(self._model, self.sample_rate)

    def transcribe_stream(
        self,
        audio_chunks: Iterator[bytes],
        hotwords: list[str] | None = None,
    ) -> Iterator[Partial]:
        rec = self._make_recognizer(hotwords)
        for chunk in audio_chunks:
            if not chunk:
                continue
            if rec.AcceptWaveform(chunk):
                final = json.loads(rec.Result()).get("text", "")
                if final:
                    yield Partial(text=final, is_final=True)
            else:
                partial = json.loads(rec.PartialResult()).get("partial", "")
                if partial:
                    yield Partial(text=partial, is_final=False)
        # flush any tail
        tail = json.loads(rec.FinalResult()).get("text", "")
        if tail:
            yield Partial(text=tail, is_final=True)


# ---------- sherpa-onnx engine (offline, streaming, lower-latency) ----------

# Vendored model dir (mirrors Vosk's ~/.cache pattern). Override with
# SMARTPPT_SHERPA_MODEL. Fetch with scripts/fetch_sherpa_model.sh.
SHERPA_MODEL_DIR = Path(
    os.environ.get("SMARTPPT_SHERPA_MODEL")
    or Path.home() / ".cache" / "sherpa-onnx" / "sherpa-onnx-streaming-zipformer-en-2023-06-26"
)
_SHERPA_SUFFIX = "-epoch-99-avg-1-chunk-16-left-128.int8.onnx"


def _sherpa_model_present(model_dir: Path = SHERPA_MODEL_DIR) -> bool:
    return (model_dir / "tokens.txt").exists() and (
        model_dir / f"encoder{_SHERPA_SUFFIX}"
    ).exists()


class SherpaEngine:
    """Offline streaming STT via sherpa-onnx (streaming zipformer transducer).

    Lower per-word latency than the Vosk small model (~0.30s vs ~0.40s median word
    cadence on the deck) at equal grounding accuracy. Fully offline (local int8
    model, onnxruntime CPU) and deterministic: single-threaded greedy decoding,
    no sampling — same audio yields the same transcript. Endpoint detection lets
    a finished utterance finalise and reset, mirroring Vosk's partial/final flow.
    """

    def __init__(self, sample_rate: int = 16000, model_dir: Path | None = None) -> None:
        import sherpa_onnx  # local import — heavy, optional dependency

        self.sample_rate = sample_rate
        md = Path(model_dir) if model_dir else SHERPA_MODEL_DIR
        s = _SHERPA_SUFFIX
        self._rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(md / "tokens.txt"),
            encoder=str(md / f"encoder{s}"),
            decoder=str(md / f"decoder{s}"),
            joiner=str(md / f"joiner{s}"),
            num_threads=1,                      # deterministic CPU decode
            sample_rate=sample_rate,
            feature_dim=80,
            decoding_method="greedy_search",    # no sampling -> reproducible
            provider="cpu",
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.4,
            rule2_min_trailing_silence=1.2,
            rule3_min_utterance_length=300,
        )

    def transcribe_stream(
        self,
        audio_chunks: Iterator[bytes],
        hotwords: list[str] | None = None,
    ) -> Iterator[Partial]:
        import numpy as np

        del hotwords  # the matcher's embedding handles deck-term robustness
        rec = self._rec
        stream = rec.create_stream()
        last = ""
        for chunk in audio_chunks:
            if not chunk:
                continue
            samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            stream.accept_waveform(self.sample_rate, samples)
            while rec.is_ready(stream):
                rec.decode_stream(stream)
            text = rec.get_result(stream).lower()  # match Vosk's lowercase contract
            if rec.is_endpoint(stream):
                if text:
                    yield Partial(text=text, is_final=True)
                rec.reset(stream)
                last = ""
            elif text and text != last:
                yield Partial(text=text, is_final=False)
                last = text


def load_default_stt_engine(sample_rate: int = 16000, grammar: bool = False):
    """Return the best available offline STT engine.

    Prefers the lower-latency sherpa-onnx streaming model when it's installed and
    the model is present; otherwise falls back to Vosk. `grammar` forces Vosk
    (its deck-vocabulary constraint mode has no sherpa equivalent here).
    """
    if not grammar and _sherpa_model_present():
        try:
            return SherpaEngine(sample_rate=sample_rate)
        except Exception as e:  # missing dep / bad model -> fall back, don't crash
            print(f"[stt] sherpa-onnx unavailable ({e}); falling back to Vosk")
    return VoskEngine(sample_rate=sample_rate, grammar=grammar)
