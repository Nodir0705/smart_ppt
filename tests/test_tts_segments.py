import wave

from src.tts import TARGET_RATE, PiperTTS, synthesize_slides


class _FakeTTS(PiperTTS):
    """PiperTTS without a voice model: each segment is 0.1 s of a constant tone."""
    def __init__(self):
        self.calls = 0

    def _pcm16k(self, text):
        self.calls += 1
        return b"\x10\x00" * (TARGET_RATE // 10)


def _frames(path):
    with wave.open(str(path), "rb") as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (TARGET_RATE, 1, 2)
        return w.getnframes()


def test_synthesize_segments_inserts_gaps(tmp_path):
    tts = _FakeTTS()
    wav = tmp_path / "s.wav"
    tts.synthesize_segments(["a", "b", "c"], wav, gap_ms=500)
    # 3 x 0.1 s speech + 3 x 0.5 s gaps (between segments and one trailing)
    assert _frames(wav) == 3 * TARGET_RATE // 10 + 3 * TARGET_RATE // 2


def test_synthesize_slides_caches_by_content(tmp_path):
    tts = _FakeTTS()
    slides = [{"slide": 1, "segments": ["hello"]}, {"slide": 2, "segments": ["world"]}]
    m1 = synthesize_slides(slides, "voice.onnx", tmp_path, gap_ms=900, tts=tts)
    assert [e["slide"] for e in m1] == [1, 2] and tts.calls == 2
    m2 = synthesize_slides(slides, "voice.onnx", tmp_path, gap_ms=900, tts=tts)
    assert m2 == m1 and tts.calls == 2          # cache hit, no re-synthesis
    changed = [{"slide": 1, "segments": ["hello again"]}]
    m3 = synthesize_slides(changed, "voice.onnx", tmp_path, gap_ms=900, tts=tts)
    assert m3[0]["wav"] != m1[0]["wav"] and tts.calls == 3   # new content -> new file
    m4 = synthesize_slides(slides[:1], "voice.onnx", tmp_path, gap_ms=600, tts=tts)
    assert m4[0]["wav"] != m1[0]["wav"]          # gap is part of the key
