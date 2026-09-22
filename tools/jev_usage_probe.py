#!/usr/bin/env python3
"""Does Jev bill for questions, or only for state? Ask the API's own usage counter.

    TYPESAFE_API_KEY=... python tools/jev_usage_probe.py

Sends one state with 1, 8 and then 32 questions and prints the ``input_tokens`` the
API reports for each. ``jevctx/types.py`` and ``demo.py`` assume questions are free;
this is the one-command check of that assumption.
"""

from __future__ import annotations

import sys

from jevctx import HttpJevClient
from jevctx.types import Noul

STATE = {
    "task": "The build fails on a dependency conflict. Find which package is pinned wrong.",
    "items": [
        {"ref": f"i{i}", "text": f"npm WARN deprecated glob@7.2.{i}: no longer supported"}
        for i in range(32)
    ],
}


def question(i: int) -> Noul:
    return Noul(
        instructions=f"Considering item i{i} only: will this item still be needed later in "
        "the task described in `task`?",
        true="The item carries information a later step may need.",
        false="The item is noise.",
    )


def main() -> int:
    client = HttpJevClient()
    rows: list[tuple[int, int]] = []
    for n in (1, 8, 32):
        before = client.usage.input_tokens
        client.ask(STATE, {f"i{i}": question(i) for i in range(n)})
        rows.append((n, client.usage.input_tokens - before))

    print(f"{'questions':>10}{'input_tokens':>14}")
    for n, tokens in rows:
        print(f"{n:>10}{tokens:>14,}")
    one, many = rows[0][1], rows[-1][1]
    print()
    if many > one * 1.5:
        print("→ input_tokens grow with the question count: questions ARE billed as input.")
    else:
        print("→ input_tokens barely move with the question count: billed by state, "
              "questions are (near) free.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
