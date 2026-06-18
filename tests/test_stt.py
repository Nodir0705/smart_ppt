"""Stage 2 tests.

Layers:
  - RollingBuffer state machine (finals + non-final partial, time pruning)
  - hotword extraction from the index
  - MockEngine + LiveTranscriber pipeline (deterministic; no audio required)
  - end-to-end mock: scripted STT → matcher → policy → expected highlight
"""

from __future__ import annotations

import json
import os
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

    def test_three_phrases_each_switch_highlight(self):
        if os.environ.get("SKIP_SEMANTIC"):
            pytest.skip("SKIP_SEMANTIC set")
        from src.encoder import load_default_encoder
        from src.matcher import Index, Matcher, MatchPolicy, PolicyConfig

        enc = load_default_encoder()
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
