#!/usr/bin/env python3
"""Transcript-shaped test runs of jevctx against a snapshot of a Claude Code session's raw .jsonl.

    python tools/transcript_experiments.py <session.jsonl> <out_dir>

A session transcript lives at ~/.claude/projects/<project-slug>/<session-id>.jsonl.

Offline by default against a clearly labelled scripted stand-in (a keyword heuristic, NOT
Jev's judgement). Set TYPESAFE_API_KEY and the identical code path talks to real Jev,
exactly as demo.py does.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

from jevctx import (
    FakeJevClient,
    HttpJevClient,
    InMemoryStore,
    JsonlStore,
    Origin,
    ShadowLog,
    admit,
    estimate_tokens,
    format_pointer,
    reconstruct,
    retrieve,
    segment,
)
from jevctx.store import summarise
from jevctx.types import PRICE_PER_INPUT_TOKEN, Record, content_id

SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
for stale in ("store.jsonl", "shadow.jsonl"):
    (OUT / stale).unlink(missing_ok=True)

SNAP = OUT / "transcript_snapshot.jsonl"
shutil.copyfile(SRC, SNAP)
raw_bytes = SNAP.read_bytes()
raw_text = raw_bytes.decode("utf-8")
lines = raw_text.splitlines(keepends=True)
N = len(lines)
CHUNK = f"transcript_chunk_lines_1-{N}.jsonl"   # what save-transcript would name this span

TASK = ("Compare the awesome-jev-compaction repo with the save-transcript and "
        "verbatim-memories skills: are they complementary or redundant, and could Jev "
        "enhance save-transcript?")


def usd(tokens: int) -> str:
    return f"${tokens * PRICE_PER_INPUT_TOKEN:.6f}"


def rule(title: str) -> None:
    print(f"\n{title}\n" + "─" * 78)


# ---- stand-in ---------------------------------------------------------------
EVIDENCE = re.compile(r"(md5|sha256|exit: ?\d|passed|failed|FAIL|Error|Traceback|✓|✗|assert|"
                      r"byte.exact|reconstruct|IDENTICAL)", re.I)
TOPIC = re.compile(r"(transcript|verbatim|jev|save|manifest|chunk|compaction|pointer|elided|"
                   r"memor)", re.I)


def stand_in(text: str) -> float:
    if EVIDENCE.search(text):
        return 0.90
    if TOPIC.search(text):
        return 0.55
    return 0.12


def make_client():
    if os.environ.get("TYPESAFE_API_KEY"):
        print("✓ TYPESAFE_API_KEY found — running against real Jev")
        return HttpJevClient()
    print("! No TYPESAFE_API_KEY — scripted stand-in (keyword heuristic, NOT Jev's judgement).")
    print("  Mechanics (segmentation, relocation, byte-exact expand, logging, replay) are real;")
    print("  the scores are not.")
    return FakeJevClient.by_text(stand_in)



def reconstruct_scoped(text: str, store, pointers) -> str:
    """Substitute only the exact pointer lines the gate emitted for this block.

    jevctx.reconstruct() substitutes ANY pointer-looking text whose id is in the store,
    including a pointer line quoted verbatim inside kept content (a transcript of a
    session that uses or discusses jevctx contains such quotes). This variant is the
    inverse of what admit() actually did.
    """
    out = text
    for pointer in pointers:
        record = store.get(pointer.id)
        if record is not None:
            out = out.replace(format_pointer(pointer) + "\n", record.text, 1)
    return out

client = make_client()

# ---- filing report, the way save-transcript would print it -------------------
rule("0. The corpus (filing-report style)")
print(f"  file   {SNAP.name}")
print(f"  lines  {N}    bytes {len(raw_bytes):,}    md5 {hashlib.md5(raw_bytes).hexdigest()}")
print(f"  save-transcript would file this span as {CHUNK}")
print(f"  estimate_tokens(whole file) = {estimate_tokens(raw_text):,}")

# ---- A. segment() on the raw jsonl as one text ------------------------------
rule("A. segment() on the raw .jsonl as one blob (no transcript awareness)")
segs = segment(raw_text, Origin(source="file:transcript", ref=SNAP.name, turn=0))
kinds = Counter(s.kind for s in segs)
lossless = "".join(s.text for s in segs) == raw_text
big = max(segs, key=lambda s: s.tokens)
print(f"  segments {len(segs)}   kinds {dict(kinds)}   lossless tiling {lossless}")
print(f"  oversized {sum(1 for s in segs if s.meta.get('oversized'))}   "
      f"split {sum(1 for s in segs if s.meta.get('split'))}")
print(f"  largest segment {big.tokens:,} tok at lines {big.line_span}")
print("  → the raw ledger is opaque to the splitter: every segment is kind 'text', cut only by")
print("    token size. A transcript adapter has to cut at record / content-block boundaries.")


# ---- transcript adapter: content blocks -------------------------------------
def blocks_of(record: dict, line_no: int):
    """Yield (kind, ref, text) for every content block in a user/assistant record."""
    t = record.get("type")
    if t not in ("user", "assistant"):
        return
    m = record.get("message") or {}
    content = m.get("content")
    if isinstance(content, str):
        yield (f"{t}_text", f"L{line_no}", content)
        return
    if not isinstance(content, list):
        return
    for b in content:
        if not isinstance(b, dict):
            continue
        bt = b.get("type")
        if bt == "text" and b.get("text"):
            yield (f"{t}_text", f"L{line_no}", b["text"])
        elif bt == "thinking" and b.get("thinking"):
            yield ("thinking", f"L{line_no}", b["thinking"])
        elif bt == "tool_use":
            name = b.get("name", "?")
            inp = b.get("input", {})
            text = inp.get("command", "") if name == "Bash" and isinstance(inp, dict) \
                else json.dumps(inp, ensure_ascii=False, indent=1)
            yield (f"tool_use:{name}", f"L{line_no}:{b.get('id', '')}", text)
        elif bt == "tool_result":
            c = b.get("content", "")
            if isinstance(c, list):
                c = "\n".join(x.get("text", "") for x in c
                              if isinstance(x, dict) and x.get("type") == "text")
            yield ("tool_result", f"L{line_no}:{b.get('tool_use_id', '')}", str(c))


records = []
for i, line in enumerate(lines, 1):
    try:
        d = json.loads(line)
    except ValueError:
        continue
    for kind, ref, text in blocks_of(d, i):
        if text:
            records.append((i, kind, ref, text))

# ---- B. the admit gate, one call per content block --------------------------
rule("B. admit() per content block (what a transcript-aware gate would do live)")
store = JsonlStore(OUT / "store.jsonl")
log = ShadowLog(OUT / "shadow.jsonl")
rows = []
for line_no, kind, ref, text in records:
    origin = Origin(source=kind, ref=ref, turn=line_no)
    res = admit(text, origin, task_digest=TASK, turn=line_no, client=client,
                store=store, log=log)
    ok = reconstruct(res.text, store) == text
    ok_scoped = reconstruct_scoped(res.text, store, res.pointers) == text
    rows.append((line_no, kind, ref, text, res, (ok, ok_scoped)))

gated = [r for r in rows if r[4].gated]
orig = sum(r[4].original_tokens for r in rows)
after = sum(r[4].result_tokens for r in rows)
print(f"  content blocks {len(rows)}   gated (≥400 tok) {len(gated)}   "
      f"passed through small {len(rows) - len(gated)}")
print(f"  tokens {orig:,} → {after:,}   saved {orig - after:,} ({(orig - after) / orig:.0%})")
print(f"  pointers {sum(len(r[4].pointers) for r in rows)}   "
      f"tripwire fired {sum(1 for r in rows if r[4].tripwire)}   "
      f"records in store {len(store)}")
lib_fail = [r for r in rows if not r[5][0]]
print(f"  byte-exact via jevctx.reconstruct(): {not lib_fail}   "
      f"via scoped reconstruct: {all(r[5][1] for r in rows)}")
if lib_fail:
    print(f"  → {len(lib_fail)} block(s) fail only because their KEPT content quotes a pointer line "
          f"whose id is in the store (lines {[r[0] for r in lib_fail]}); reconstruct() expands the quote.")
print()
print(f"  {'line':>4}  {'kind':<22}{'orig':>7}{'after':>7}{'ptrs':>5}  tripwire")
for line_no, kind, _ref, _text, res, _ok in sorted(gated, key=lambda r: -(r[4].saved_tokens))[:12]:
    print(f"  {line_no:>4}  {kind[:22]:<22}{res.original_tokens:>7,}{res.result_tokens:>7,}"
          f"{len(res.pointers):>5}  {res.tripwire or '-'}")
seg_kinds = Counter()
for line_no, kind, ref, text, _res, _ok in gated:
    for s in segment(text, Origin(source=kind, ref=ref, turn=line_no)):
        seg_kinds[s.kind] += 1
print(f"\n  segment kinds the splitter assigned inside gated blocks: {dict(seg_kinds)}")
ptr = next((p for r in gated for p in r[4].pointers), None)
if ptr is not None:
    print(f"  example pointer: {format_pointer(ptr)[:120]}…")

print("\n  replay of the logged scores at other thresholds (no re-run, no Jev calls):")
print(f"  {'threshold':<12}{'kept':>8}{'elided':>9}{'tokens saved':>15}")
for th in (0.1, 0.35, 0.6, 0.9):
    st = log.replay(th)
    print(f"  {th:<12.2f}{st.by_action['kept']:>8}{st.by_action['elided']:>9}"
          f"{st.elided_tokens:>15,}")

# ---- C. persistence round trip ----------------------------------------------
rule("C. JsonlStore survives a restart (append-only, like a chunk)")
reopened = JsonlStore(OUT / "store.jsonl")
ok_all = all(reconstruct_scoped(res.text, reopened, res.pointers) == text
             for _, _, _, text, res, _ in rows)
print(f"  reopened store records {len(reopened)}   scoped byte-exact reconstruct after reload {ok_all}")
print(f"  store file bytes {(OUT / 'store.jsonl').stat().st_size:,}   "
      f"shadow log bytes {(OUT / 'shadow.jsonl').stat().st_size:,}")

# ---- D. retrieve(): the transcript as memory, reranked by Jev ----------------
rule("D. retrieve() over the transcript as a memory store (the verbatim-memories shape)")
mem = InMemoryStore()
for line_no, kind, ref, text in records:
    if not text.strip():
        continue
    mem.put(Record(id=content_id(text, salt=ref, prefix="t"), text=text, kind=kind,
                   origin=Origin(source=kind, ref=ref, turn=line_no),
                   tokens=estimate_tokens(text), created_turn=line_no))
probe = client
before_calls = len(getattr(probe, "calls", []))
rlog = ShadowLog(path=None)
digest = mem.digest(budget_tokens=24_000)
hits = retrieve(TASK, turn=N + 1, client=probe, store=mem, log=rlog, k=5, threshold=0.5)
calls = getattr(probe, "calls", [])[before_calls:]
state_tokens = sum(estimate_tokens(c.state) for c in calls)
q_tokens = sum(estimate_tokens(q.to_payload()) for c in calls for q in c.questions.values())
full_tokens = sum(r.tokens for r in mem.all_records())
print(f"  memory records {len(mem)} ({full_tokens:,} tok of verbatim text)")
print(f"  digest entries that fit the 24k-token window {len(digest)} "
      f"(most recent first; {len(mem) - len(digest)} dropped)")
print(f"  Jev requests {len(calls)}   state tokens {state_tokens:,}   "
      f"question tokens {q_tokens:,} (repo docstrings say questions are not billed)")
print(f"  est. cost, state only {usd(state_tokens)}   if questions were billed too "
      f"{usd(state_tokens + q_tokens)}")
print(f"  vs. putting the verbatim text itself in state: {full_tokens:,} tok ≈ {usd(full_tokens)}")
print(f"  vs. re-reading the whole raw ledger with the host model: "
      f"{estimate_tokens(raw_text):,} tok per read")
print(f"\n  top-{len(hits)} verbatim memories for the task, cited by save-transcript span:")
for r in hits:
    print(f"    {CHUNK}:L{r.created_turn:<4} {r.kind:<20} {summarise(r.text, 66)!r}")
if calls:
    c = calls[0]
    first_q = next(iter(c.questions.values()))
    print(f"\n  what one request looks like — state keys {list(c.state.keys())}, "
          f"{len(c.state['items'])} items, {len(c.questions)} questions")
    print(f"    question: {first_q.instructions[:160]}…")

# ---- E. BM25 prefilter + Jev rerank for a long transcript ------------------------
rule("E. search() prefilter, then Jev rerank (what scales past the 24k digest window)")
cands = mem.search("md5 manifest chunk byte-exact", limit=20)
print(f"  BM25 candidates {len(cands)} → one Jev request of ≤32 questions")
for r in cands[:5]:
    print(f"    {CHUNK}:L{r.created_turn:<4} {r.kind:<20} {summarise(r.text, 66)!r}")
print("\nDone.")
