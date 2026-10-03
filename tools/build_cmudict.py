"""Build data/cmudict.txt.gz (the runtime pronunciation dictionary of
src/asr_repair.py) from a CMU Pronouncing Dictionary 0.7a text file:

    python3 -m tools.build_cmudict ~/nltk_data/corpora/cmudict/cmudict

One line per lowercase word: ``word PH ON ES|ALT PRON``, stress digits stripped,
every distinct pronunciation kept; only plain words (letters and apostrophes).
"""
from __future__ import annotations

import gzip
import re
import sys
from pathlib import Path

NOTICE = """# CMU Pronouncing Dictionary 0.7a, stress stripped (tools/build_cmudict.py).
# Copyright (C) 1993-2008 Carnegie Mellon University. All rights reserved.
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met: 1. Redistributions of source code must retain the
# above copyright notice, this list of conditions and the following disclaimer. The contents of this
# file are deemed to be source code. 2. Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
# THIS SOFTWARE IS PROVIDED BY CARNEGIE MELLON UNIVERSITY ``AS IS'' AND ANY EXPRESSED OR IMPLIED
# WARRANTIES ARE DISCLAIMED. IN NO EVENT SHALL CARNEGIE MELLON UNIVERSITY NOR ITS EMPLOYEES BE LIABLE
# FOR ANY DAMAGES ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE.
"""


def main(src: str, out: str = "data/cmudict.txt.gz") -> None:
    prons: dict[str, list[str]] = {}
    for line in Path(src).read_text(encoding="latin-1").splitlines():
        p = line.split()
        if len(p) < 3 or not re.fullmatch(r"[a-z][a-z']*", w := p[0].lower()):
            continue
        ph = " ".join(re.sub(r"\d", "", x) for x in p[2:])
        if ph not in prons.setdefault(w, []):
            prons[w].append(ph)
    body = "\n".join(f"{w} {'|'.join(v)}" for w, v in sorted(prons.items()))
    Path(out).write_bytes(gzip.compress((NOTICE + body + "\n").encode(), 9, mtime=0))
    print(f"{len(prons)} words -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
