#!/usr/bin/env python3
"""Normalise a `VREND_DEBUG=shader` log into the shader fixture.

The C prints, per shader created, "TGSI received:" followed by `tgsi_dump` of the tokens it
parsed and "GLSL:" followed by the text it emitted, each fragment prefixed with the context's
name by `vrend_print_context_name`. This strips the prefixes, drops everything that is not one of
those blocks (driver warnings, readback errors), and writes the blocks one after another:

    TGSI received:
    <the dump, one token per line>

    GLSL:
    <the GLSL>

The Rust tests read the fixture and hold the TGSI parser and the translator to it byte for byte.
Record it with the debug C prefix, which is the only one that compiles the dump in:

    VIRGL_PREFIX=.../harness/vm/prefix-debug VREND_DEBUG=shader \\
        ./vrend-replay.sh ../vm/captures/vrend.bin --score /dev/null 2> shader.log
    ./vrend-shader-log.py shader.log > fixtures/vrend-shaders.txt

The Rust prefix prints the same blocks under `VIRGLRS_DEBUG=shader`, with the mark on its own
line already; this reads either, so the two runs diff after the same pass.
"""

import re
import sys

PREFIX = re.compile(r"^limina-replay: ", re.M)
MARK = re.compile(r"^(TGSI received:|GLSL:)\n?", re.M)


def main() -> int:
    if len(sys.argv) != 2:
        sys.stderr.write(__doc__)
        return 2
    with open(sys.argv[1], encoding="utf-8", errors="replace") as f:
        text = f.read()
    # The context name is printed before every debug fragment, not every line: the dump's first
    # line follows "TGSI received:" on the same line, and the prefix lands wherever the next
    # fragment started.
    text = text.replace("limina-replay: ", "")
    text = MARK.sub(r"\1\n", text)
    out = []
    pos = 0
    while True:
        m = re.search(r"^TGSI received:\n", text[pos:], re.M)
        if not m:
            break
        start = pos + m.start()
        g = text.index("GLSL:\n", start)
        n = re.search(r"^TGSI received:\n", text[g:], re.M)
        end = g + n.start() if n else len(text)
        block = text[start:end]
        # What follows the last block on stderr is the rest of the run's chatter; a block ends at
        # its own trailing empty line.
        tail = block.index("\n\n", g - start) + 2
        out.append(block[:tail])
        pos = end
    sys.stdout.write("".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
