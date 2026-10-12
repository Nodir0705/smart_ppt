import numpy as np

from tools.stt_lag import emission, naming_word, stem, trim


def _p(t, text, final=False):
    return {"t": t, "final": final, "text": text}


def test_emission_two_utterances():
    script = "Hi everyone, the class has 30 pupils. Now grading starts.".split()
    partials = [_p(0.5, "hi"), _p(0.8, "hi everyon"), _p(1.1, "hi everyone the"),
                _p(1.5, "hi everyone the class has thirty"),
                _p(2.0, "hi everyone the class has thirty pupil"),
                _p(3.2, "hi everyone the class has thirty pupils", final=True),
                _p(3.6, "now grating"), _p(3.9, "now grating starts")]
    got = dict(zip(script, emission(script, partials)))
    assert got["Hi"] == 0.5
    assert got["everyone,"] == 1.1          # the fragment "everyon" does not count
    assert got["30"] == 1.5                 # "30" -> "thirty"
    assert got["pupils."] == 3.2            # "pupil" is not "pupils"
    assert got["Now"] == 3.6                # second utterance restarts the partial
    assert got["grading"] is None           # misrecognised: never emitted
    assert got["starts."] == 3.9


def test_emission_fragment_cannot_confirm_a_later_word():
    # one long utterance: the fragment "work" (of "worksheet") at 1.0 must not
    # confirm the "work" said at the end
    script = "the same worksheet thirty times and that is student work".split()
    got = emission(script, [_p(1.0, "the same work"),
                            _p(2.0, "the same worksheet thirty times and that is student work")])
    assert got == [1.0, 1.0] + [2.0] * 8


def test_trim_moves_end_to_last_voiced_frame():
    v = np.array([0, 1, 1, 1, 0, 0, 0, 1], bool)       # 10 ms frames
    assert trim(0.01, 0.07, v) == 0.04
    assert trim(0.04, 0.07, v) == 0.07                  # nothing voiced: unchanged


def test_naming_word_is_first_distinctive():
    units = {"u1": "THE PROBLEM", "u2": "A teacher marks the same worksheet 30+ times",
             "u3": "15–33 gradable items per page"}
    s = "Here's the problem: picture a teacher marking the same worksheet".split()
    assert naming_word(s, ["u2"], units) == 5           # "problem" belongs to u1
    s = "Each of those pages has things to grade".split()
    assert naming_word(s, ["u3"], units) == 3           # pages ~ page
    assert naming_word("Nothing to see here".split(), ["u3"], units) is None
    assert stem("grading") == stem("gradable") == stem("grade") == "grad"
