"""Stage 2 tests.

Layers:
  - RollingBuffer state machine (finals + non-final partial, time pruning)
  - hotword extraction from the index
  - MockEngine + LiveTranscriber pipeline (deterministic; no audio required)
  - end-to-end mock: scripted STT → matcher → policy → expected highlight
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.live import LiveTranscriber
from src.stt import MockEngine, MockScript, Partial, RollingBuffer, hotwords_from_index

ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / "out" / "index.json"


# ---------- RollingBuffer ----------

class TestRollingBuffer:
    def test_partial_alone_appears_in_text(self):
        buf = RollingBuffer(window_seconds=4.0)
        buf.push(Partial("the matcher", is_final=False, timestamp=1.0))
        assert buf.text == "the matcher"

    def test_final_persists_and_partial_clears(self):
        buf = RollingBuffer(window_seconds=10.0)
        buf.push(Partial("the matcher", is_final=False, timestamp=1.0))
        buf.push(Partial("the matcher box", is_final=True, timestamp=1.5))
        # partial gets dropped on final; final is in history
        assert buf.text == "the matcher box"
        buf.push(Partial("and the", is_final=False, timestamp=1.8))
        assert buf.text == "the matcher box and the"

    def test_old_finals_get_pruned(self):
        buf = RollingBuffer(window_seconds=2.0)
        buf.push(Partial("alpha", is_final=True, timestamp=0.0))
        buf.push(Partial("beta", is_final=True, timestamp=1.0))
        buf.push(Partial("gamma", is_final=True, timestamp=4.0))  # past 2s window from "alpha"
        # at t=4.0 with window=2.0, cutoff=2.0; alpha (0.0) and beta (1.0) both dropped
        assert buf.text == "gamma"

    def test_clear_resets(self):
        buf = RollingBuffer()
        buf.push(Partial("hello", is_final=True, timestamp=1.0))
        buf.push(Partial("world", is_final=False, timestamp=1.1))
        assert buf.text
        buf.clear()
        assert buf.text == ""

    def test_finals_in_order(self):
        buf = RollingBuffer(window_seconds=100.0)
        for i, t in enumerate(["one", "two", "three", "four"], start=1):
            buf.push(Partial(t, is_final=True, timestamp=float(i)))
        assert buf.text == "one two three four"


# ---------- hotword extraction ----------

class TestHotwords:
    def test_dedup_and_lowercase(self):
        idx = {
            "slides": [{"elements": [
                {"trigger_phrases": ["The Matcher", "matcher box", "MATCHER", "the matcher"]},
                {"trigger_phrases": ["overlay", "Overlay UI"]},
            ]}],
        }
        hot = hotwords_from_index(idx)
        # "matcher" appears 4 times across phrases, must be once in output
        assert hot.count("matcher") == 1
        assert "overlay" in hot
        assert "ui" in hot

    def test_drops_stopwords(self):
        idx = {"slides": [{"elements": [{"trigger_phrases": ["the and is on"]}]}]}
        hot = hotwords_from_index(idx)
        assert hot == []

    def test_handles_punctuation_and_numbers(self):
        idx = {"slides": [{"elements": [{"trigger_phrases": ["< 1 ms latency!", "0.5 threshold"]}]}]}
        hot = hotwords_from_index(idx)
        assert "ms" in hot
        assert "latency" in hot
        assert "threshold" in hot
        # digits are not [A-Za-z]+ words so they're dropped — expected
        assert "1" not in hot and "0" not in hot

    @pytest.mark.skipif(not INDEX_PATH.exists(), reason="run src.index_deck first")
    def test_real_index_yields_distinctive_vocabulary(self):
        idx = json.loads(INDEX_PATH.read_text())
        hot = hotwords_from_index(idx)
        # deck-specific terms must show up
        for word in ["matcher", "microphone", "overlay", "render", "sync", "highlight"]:
            assert word in hot, f"expected hotword: {word}"
        # generic English shouldn't
        for stop in ["the", "and", "is"]:
            assert stop not in hot


# ---------- MockEngine + LiveTranscriber pipeline ----------

class _FakeAudioSource:
    sample_rate = 16000
    def chunks(self):
        # one empty chunk; engine ignores it
        yield b""
    def stop(self):
        pass


class TestMockEnginePipeline:
    def test_mock_engine_drives_buffer(self):
        script = [
            MockScript("the matcher", is_final=False),
            MockScript("the matcher box", is_final=True),
            MockScript("and overlay", is_final=False),
        ]
        engine = MockEngine(script)
        buf = RollingBuffer(window_seconds=100.0)
        transcriber = LiveTranscriber(_FakeAudioSource(), engine, buf)
        seen: list[str] = []
        transcriber.run(on_update=lambda text, _p: seen.append(text))
        assert seen == [
            "the matcher",                  # partial set
            "the matcher box",              # finalised; partial cleared
            "the matcher box and overlay",  # partial appended after final
        ]


# ---------- end-to-end mock: STT → matcher → policy ----------

@pytest.mark.slow
@pytest.mark.skipif(not INDEX_PATH.exists(), reason="run src.index_deck first")
class TestMockedFullPipeline:
    """Scripted Vosk-like emissions drive the live matcher; the highlight
    should land on the expected element after each phrase finalises.
    """

    def test_three_phrases_each_switch_highlight(self, real_encoder):
        from src.matcher import Index, Matcher, MatchPolicy, PolicyConfig

        enc = real_encoder
        idx = Index(INDEX_PATH, enc)
        matcher = Matcher(idx, enc)
        policy = MatchPolicy(matcher, PolicyConfig())

        # script the same trace I use in the REPL on slide 6. The finals are
        # spaced > the buffer window so each one prunes its predecessor — without
        # this, the mock stamps all finals at ~the same monotonic() time, the
        # window never prunes, and the buffer feeds polluted multi-topic text.
        script = [
            MockScript("let me explain the", is_final=False),
            MockScript("let me explain the matcher", is_final=True, delay_s=1.2),
            MockScript("audio comes in through the microphone", is_final=True, delay_s=1.2),
            MockScript("and finally we draw the overlay", is_final=True, delay_s=1.2),
        ]
        engine = MockEngine(script)
        buf = RollingBuffer(window_seconds=1.0)  # short window so each final dominates
        emitted_actives: list[str | None] = []

        def on_update(text: str, partial: Partial):
            if not partial.is_final:
                return  # only step the policy on finals (simulates the throttle)
            tail = buf.tail(8) or text  # match the live viewer's tail-of-buffer feed
            _, active, _ = policy.step(tail, slide_no=6)
            emitted_actives.append(active)

        transcriber = LiveTranscriber(_FakeAudioSource(), engine, buf)
        transcriber.run(on_update=on_update)

        # three finals → three highlight switches
        assert len(emitted_actives) == 3
        # the final actives must be Matcher → Microphone → Overlay
        id_to_text = {ref.id: (ref.text or "").lower() for ref in idx.elements_on(6)}
        labels = [id_to_text.get(a or "") for a in emitted_actives]
        assert labels == ["matcher", "microphone", "overlay"], f"got {labels}"


# ---------- WhistleEngine (segmentation with a fake decoder) + sherpa hotwords ----------

import numpy as np

from src import stt


def _pcm(*parts: tuple[str, float]) -> bytes:
    """('tone'|'sil', seconds) parts -> PCM16 bytes at 16 kHz."""
    out = []
    for kind, sec in parts:
        n = int(round(sec * 16000))
        out.append(3000 * np.sin(np.arange(n) * 0.3) if kind == "tone" else np.zeros(n))
    return np.concatenate(out).astype(np.int16).tobytes()


def _chunks(pcm: bytes, ms: int = 100):
    step = 16000 * ms // 1000 * 2
    return [pcm[i:i + step] for i in range(0, len(pcm), step)]


def _whistle(decode=None):
    calls: list[int] = []

    def fake(pcm: bytes) -> str:     # one word per started half second, written form
        calls.append(len(pcm))
        return " ".join(["Word,"] * (len(pcm) // 16000 + 1))
    return stt.WhistleEngine(audio_clock=True, decode=decode or fake), calls


class TestWhistleEngine:
    AUDIO = _pcm(("sil", 0.3), ("tone", 1.2), ("sil", 0.4),                 # short segment
                 ("tone", 4.4), ("sil", 0.06), ("tone", 2.0), ("sil", 0.5))  # long, with a 60 ms dip

    def test_segments_partials_and_finals(self):
        eng, calls = _whistle()
        out = list(eng.transcribe_stream(iter(_chunks(self.AUDIO))))
        finals = [p for p in out if p.is_final]
        assert len(finals) == 3                    # short, long cut at the dip, its rest
        assert finals[0].timestamp == pytest.approx(1.7)   # 0.16 s after speech stops
        assert max(calls) <= int(5.12 * 16000) * 2        # never past 64 encoder frames
        # the cut is at the dip (0.1 s pre-roll + 4.4 s tone + dip), not a hard 5.12 s cut
        assert any(abs(c / 32000 - 4.56) <= 0.04 for c in calls)
        # partials re-decode every 0.5 s of segment audio (0.1 s pre-roll + speech from 0.3 s),
        # in sherpa's text form
        assert all(p.text and set(p.text.split()) == {"word"} for p in out)
        partial_ts = [p.timestamp for p in out if not p.is_final and p.timestamp < 1.7]
        assert partial_ts == pytest.approx([0.7, 1.2])

    def test_buffer_words_accumulate_and_run_is_deterministic(self):
        runs = []
        for _ in range(2):
            eng, _calls = _whistle()
            buf = RollingBuffer(window_seconds=100)
            for p in eng.transcribe_stream(iter(_chunks(self.AUDIO))):
                buf.push(p)
            runs.append(buf.words())
        assert runs[0] == runs[1] and len(runs[0]) >= 3

    def test_empty_final_clears_a_shown_partial(self):
        texts = iter(["Hello there.", ""])         # partial, then the final decodes to nothing
        eng, _ = _whistle(decode=lambda pcm: next(texts))
        out = list(eng.transcribe_stream(iter(_chunks(_pcm(("tone", 0.6), ("sil", 0.3))))))
        assert [(p.text, p.is_final) for p in out] == [("hello there", False), ("", True)]

    def test_open_segment_is_flushed_at_end_of_audio(self):
        eng, _ = _whistle()
        out = list(eng.transcribe_stream(iter(_chunks(_pcm(("sil", 0.2), ("tone", 0.3))))))
        assert out[-1].is_final and out[-1].timestamp == pytest.approx(0.5)

    def test_spoken_text(self):
        assert stt.spoken_text('It\'s 100% "offline"; 0.3 s, 1,234 runs on 8 GB.') == \
            "it's one hundred percent offline zero point three s one thousand two hundred thirty four runs on eight gb"

    def test_explicit_whistle_never_falls_back(self, monkeypatch):
        monkeypatch.setenv("SMARTPPT_STT", "whistle")
        monkeypatch.setattr(stt, "WHISTLE_PYTHON", Path("/nonexistent/python"))
        with pytest.raises(RuntimeError, match="venv python"):
            stt.load_default_stt_engine()
        with pytest.raises(ValueError):
            stt.load_default_stt_engine(engine="bogus")

    @pytest.mark.skipif(not (stt.WHISTLE_PYTHON.exists() and stt.WHISTLE_WEIGHTS.exists()),
                        reason="needs ~/.venvs/smartppt-whistle and ~/models/whistle/whistle.cact")
    def test_server_round_trip(self):
        wav = sorted((ROOT / "out/bench/AI_Handout_Checker/presenter/audio").glob("slide_001_*.wav"))
        if not wav:
            pytest.skip("no bench audio")
        from src.audio import FileSource
        eng = stt.WhistleEngine(audio_clock=True)
        try:
            chunks = list(FileSource(str(wav[0]), chunk_ms=100).chunks())[:40]   # 4 s
            text = " ".join(p.text for p in eng.transcribe_stream(iter(chunks)) if p.is_final)
        finally:
            eng.close()
        assert "everyone" in text and eng.proc.poll() is not None


class _FakeStream:
    def accept_waveform(self, sr, samples): pass


class _FakeRec:
    def __init__(self): self.created = []
    def create_stream(self, *args): self.created.append(args); return _FakeStream()
    def is_ready(self, s): return False
    def get_result(self, s): return ""
    def is_endpoint(self, s): return False


class TestSherpaHotwords:
    def _engine(self, score):
        eng = object.__new__(stt.SherpaEngine)
        eng.sample_rate, eng.audio_clock, eng.hotwords_score, eng._rec = 16000, False, score, _FakeRec()
        return eng

    @pytest.mark.parametrize("score,expected", [(0.0, ()), (1.5, ("KERNEL/SMMU UNIT",))])
    def test_hotwords_reach_the_stream_only_when_on(self, score, expected):
        eng = self._engine(score)
        list(eng.transcribe_stream(iter([bytes(3200)]), hotwords=["kernel", "smmu unit"]))
        assert eng._rec.created == [expected]

    @pytest.mark.skipif(not stt._sherpa_model_present() or not (stt.SHERPA_MODEL_DIR / "bpe.vocab").exists(),
                        reason="needs the sherpa model with bpe.vocab")
    def test_real_recognizer_builds_with_hotwords(self):
        pytest.importorskip("sherpa_onnx")
        eng = stt.SherpaEngine(hotwords_score=1.5)
        assert list(eng.transcribe_stream(iter([bytes(3200)] * 5), hotwords=["kernel"])) == []
