"""Sound-based ASR repair: homophones and spelled letters go back to the slide's
words; correctly heard and ordinary words stay."""
from __future__ import annotations

from src.asr_repair import SoundRepair, distance, g2p


def _rep(*texts: str, **unit) -> SoundRepair:
    units = [{"uid": f"u{i}", "text": t, "summary": "", "talk": [], "terms": []} for i, t in enumerate(texts)]
    if unit:
        units.append({"uid": "x", "text": "", "summary": "", "talk": [], "terms": [], **unit})
    return SoundRepair(units, 0.25)


def test_homophones_both_ways():
    assert _rep("Linux kernel")("going back to the colonel at the bottom") == "going back to the kernel at the bottom"
    assert _rep("Colonel Mustard")("the kernel did it") == "the colonel did it"
    assert _rep("Device tree node")("the display knowed in the tree") == "the display node in the tree"


def test_correct_and_ordinary_words_stay():
    r = _rep("Linux kernel", "Four CPU cores", "Battery life")
    for text in ("the kernel and the colonel's men", "for the right score we wait",
                 "a fore court", "it scores well on battery"):
        assert r(text) == text                                # ("colonel's": contractions never change)
    assert r("the core count") == "the core count"            # "core" is not "cores": no exact sound match


def test_spelled_letters_and_garbles():
    assert _rep("HAL & Vendor Layer")("one level down is the h l and vendor layer") == \
        "one level down is the hal and vendor layer"
    assert _rep("Status LEDs")("the elly dees blink") == "the leds blink"
    assert _rep("SMMU and IOMMU")("the esa memu is the protection side") == "the smmu is the protection side"
    assert _rep("CPU core")("kor the driver reads") == "core the driver reads"
    assert _rep("The codec driver")("the codak driver") == "the codec driver"
    assert _rep("CPU core", talk=["the cpu side"])("the sea p u runs it") == "the cpu runs it"


def test_letters_only_match_acronyms_and_ties_stay():
    assert _rep("The GPU uses its own view")("the g u sees its own view") == "the g u sees its own view"
    # nowed sounds as close to node as to knowed: a tie between two words leaves it alone
    assert _rep("node", "knowed")("the nowed") == "the nowed"


def test_phoneme_distance_and_fallback():
    assert distance("K ER N AH L".split(), "K ER N AH L".split()) == 0.0
    assert distance("L IH N AA K S".split(), "L IH N AH K S".split()) == 0.5 / 6   # vowel swap costs half
    assert g2p("kor") == "K AO R" and g2p("memu") == "M EH M UW"
