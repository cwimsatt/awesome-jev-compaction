#!/usr/bin/env python3
"""Gate every tests/fixtures/ file through admit() against REAL Jev, two labelled passes.

    TYPESAFE_API_KEY=proxy-attached python tools/real_jev_results/gate_fixtures.py

Pass A: default GateConfig() — every fixture is under the 400-token min_gate_tokens floor, so
        admit() returns gated=False and scores nothing (no Jev call).
Pass B: GateConfig(min_gate_tokens=0) — every fixture is scored for real.

One shared HttpJevClient() per pass, so the printed usage is that pass's alone. A failed request
fails open to 1.0 inside admit(), so each failed score is printed with a FAILED marker.
"""

from __future__ import annotations

import pathlib

from jevctx import (
    GateConfig,
    HttpJevClient,
    InMemoryStore,
    Origin,
    ShadowLog,
    admit,
    reconstruct,
    segment,
)

TASK = "The build fails on a dependency conflict. Find which package is pinned wrong."
FIXTURES = pathlib.Path("tests/fixtures")


def run_pass(label: str, config: GateConfig) -> None:
    print(f"\n{'=' * 90}\nPASS {label}: {config}\n{'=' * 90}")
    client = HttpJevClient()
    for path in sorted(FIXTURES.iterdir()):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        origin = Origin(source="tool:bash", ref=path.name, turn=1)
        store, log = InMemoryStore(), ShadowLog(path=None)
        res = admit(text, origin, task_digest=TASK, turn=1, client=client, store=store, log=log,
                    config=config)
        by_id = {r.item_id: r for r in res.scores}
        failed = [r for r in res.scores if r.failed]
        byte_exact = reconstruct(res.text, store) == text
        print(f"\n── {path.name}  gated={res.gated}  "
              f"{res.original_tokens}→{res.result_tokens} tok  pointers={len(res.pointers)}  "
              f"tripwire={res.tripwire or '-'}  byte_exact={byte_exact}  failed_scores={len(failed)}")
        for seg in segment(text, origin):
            r = by_id.get(seg.id)
            if r is None:
                mark = "(not scored)"
            elif r.failed:
                mark = f"FAILED: {r.error}"
            else:
                mark = f"{r.score:.2f}"
            first60 = " ".join(seg.text.split())[:60]
            print(f"     {mark:<10} {seg.kind:<10} {first60}")
    u = client.usage
    print(f"\n  [usage] PASS {label}: requests {u.requests}  input_tokens {u.input_tokens:,}  "
          f"output_tokens {u.output_tokens:,}")


def main() -> int:
    run_pass("A (default GateConfig)", GateConfig())
    run_pass("B (min_gate_tokens=0)", GateConfig(min_gate_tokens=0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
