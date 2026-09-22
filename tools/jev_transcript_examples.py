#!/usr/bin/env python3
"""Four Jev examples over a save-transcript chunk (a Claude Code session's raw .jsonl).

    python tools/jev_transcript_examples.py <chunk.jsonl> [--query "..."] [--evidence-lines 505,1069]
        [--audit-lines 444,484] [--checkpoints 419,497] [--readings readings.json]

Examples
  1  verbatim-memory retrieval: BM25 prefilter -> one Jev request of <=32 relevance Nouls -> cite chunk:line
  2  evidence gate for a transcript flattener: block-level Noul, then per-segment Nouls -> pointers, not a char cap
  3  audit-event screen: Choice / Noul / Score questions in the verification-audit ontology's vocabulary
  4  phase and commit-point detection over a window of recent turns

Without TYPESAFE_API_KEY the judgments come from --readings (an analyst's numbers, NOT Jev's); every request
shape, batch, pointer, citation and byte-exact expand is real jevctx code. With a key the same requests go to
Jev and the readings, if given, print beside Jev's answers for comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

from jevctx import (
    FakeJevClient,
    GateConfig,
    InMemoryStore,
    Origin,
    ShadowLog,
    admit,
    estimate_tokens,
    format_pointer,
    segment,
)
from jevctx.scorer import build_state, score_items
from jevctx.store import summarise
from jevctx.types import Choice, Noul, Record, Score, ScoreItem, content_id

# ----------------------------------------------------------------------------- blocks


@dataclass(frozen=True)
class Block:
    line: int
    kind: str          # user_text | assistant_text | thinking | tool_use | tool_result
    tool: str          # tool name for tool_use / tool_result, else ""
    text: str
    summary: str

    @property
    def cite(self) -> str:
        return f"L{self.line}"


_LINENO = re.compile(r"^\s*\d+(?:\t| {1,2}|→)")


def summarise_block(kind: str, tool: str, text: str, limit: int = 110) -> str:
    """Transcript-aware one-liner: provenance first, then the first line that means something.

    jevctx.store.summarise() takes the first non-empty line; on a numbered file read that is
    "1 ---" and on a JSON tool input it is "{", which is useless as Jev state.
    """
    body = ""
    if kind == "tool_use":
        try:
            inp = json.loads(text)
        except ValueError:
            inp = {}
        if isinstance(inp, dict):
            body = str(inp.get("command") or inp.get("file_path") or inp.get("pattern")
                       or inp.get("query") or inp.get("description") or json.dumps(inp))
            body = body.strip().splitlines()[0] if body.strip() else ""
    else:
        for raw in text.splitlines():
            line = _LINENO.sub("", raw).strip()
            if not line or line in ("{", "}", "---", "[", "]") or line.startswith('{"result":"{'):
                continue
            body = line
            break
        if not body:
            body = summarise(text, limit)
    body = " ".join(body.split())
    head = f"{kind}" + (f"[{tool}]" if tool else "")
    return f"{head}: {body[:limit]}"


def load_blocks(path: str) -> tuple[list[Block], list[str]]:
    lines = open(path, encoding="utf-8").read().splitlines(keepends=True)
    blocks: list[Block] = []
    tool_names: dict[str, str] = {}
    for i, raw in enumerate(lines, 1):
        try:
            d = json.loads(raw)
        except ValueError:
            continue
        t = d.get("type")
        if t not in ("user", "assistant"):
            continue
        content = (d.get("message") or {}).get("content")
        if isinstance(content, str):
            if content.strip():
                blocks.append(Block(i, f"{t}_text", "", content, summarise_block(f"{t}_text", "", content)))
            continue
        for b in content or []:
            if not isinstance(b, dict):
                continue
            bt = b.get("type")
            if bt == "text" and b.get("text", "").strip():
                blocks.append(Block(i, f"{t}_text", "", b["text"], summarise_block(f"{t}_text", "", b["text"])))
            elif bt == "thinking" and b.get("thinking", "").strip():
                blocks.append(Block(i, "thinking", "", b["thinking"], summarise_block("thinking", "", b["thinking"])))
            elif bt == "tool_use":
                name = b.get("name", "?")
                tool_names[b.get("id", "")] = name
                text = json.dumps(b.get("input", {}), ensure_ascii=False)
                blocks.append(Block(i, "tool_use", name, text, summarise_block("tool_use", name, text)))
            elif bt == "tool_result":
                c = b.get("content", "")
                if isinstance(c, list):
                    c = "\n".join(x.get("text", "") for x in c if isinstance(x, dict) and x.get("type") == "text")
                c = str(c)
                if c.strip():
                    name = tool_names.get(b.get("tool_use_id", ""), "?")
                    blocks.append(Block(i, "tool_result", name, c, summarise_block("tool_result", name, c)))
    return blocks, lines


# ----------------------------------------------------------------------------- clients


class ReadingsClient(FakeJevClient):
    """Offline stand-in that answers from an analyst's readings keyed by the item's `L<line>` prefix.

    Every item text this script builds starts with "L<line> " (retrieval) or "L<line>:<a>-<b> "
    (segments), so the key is recoverable from the state alone, exactly as Jev would see it.
    """

    def __init__(self, table: Mapping[str, float], default: float = 0.15) -> None:
        self.table: Mapping[str, float] = table  # swapped per example, so line keys never collide

        def answer(state, questions, key):
            if isinstance(state, Mapping):
                for item in state.get("items") or []:
                    if isinstance(item, Mapping) and item.get("ref") == key:
                        m = re.match(r"(L\d+(?::\d+-\d+)?)\b", str(item.get("text", "")))
                        return float(self.table.get(m.group(1), default)) if m else default
            return default
        super().__init__(answer)


def make_client(readings: Mapping[str, float] | None):
    key = os.environ.get("TYPESAFE_API_KEY")
    if key:
        from jevctx import HttpJevClient
        if key == "proxy-attached":
            print("✓ auth via proxy-attached credential — judgments below are REAL Jev answers")
        else:
            print("✓ TYPESAFE_API_KEY found — judgments below are REAL Jev answers")
        return HttpJevClient(), "jev"
    print("! No TYPESAFE_API_KEY — judgments below are the analyst's READINGS (not Jev), driving real jevctx code")
    return ReadingsClient(readings or {}), "readings"


def _print_usage(client) -> None:
    """Print the real client's API usage. A failed request would have raised, not defaulted to 1.0."""
    usage = getattr(client, "usage", None)
    if usage is not None:
        print(f"  [usage] Jev requests {usage.requests}  input_tokens {usage.input_tokens:,}  "
              f"output_tokens {usage.output_tokens:,}")


def rule(title: str) -> None:
    print(f"\n{'=' * 96}\n{title}\n{'=' * 96}")


def show_json(obj, limit: int = 1400) -> None:
    s = json.dumps(obj, ensure_ascii=False, indent=1)
    print(s if len(s) <= limit else s[:limit] + f"\n … [{len(s) - limit:,} more chars]")


# ----------------------------------------------------------------------------- example 1

_WORD = re.compile(r"\w+", re.UNICODE)


def bm25_rank(records, query: str, k1: float = 1.2, b: float = 0.75) -> list:
    """Standard BM25 with document-length normalisation (jevctx.store.search has none).

    Diagnostic only: shows what a prefilter with length normalisation would hand Jev.
    """
    docs = {r.id: _WORD.findall(f"{r.summary}\n{r.text}".casefold()) for r in records}
    n = len(docs)
    avgdl = sum(len(d) for d in docs.values()) / max(n, 1)
    terms = set(_WORD.findall(query.casefold()))
    sets = {rid: set(d) for rid, d in docs.items()}
    df = {t: sum(1 for d in sets.values() if t in d) for t in terms}
    scored = []
    for r in records:
        d = docs[r.id]
        cnt = Counter(d)
        score = 0.0
        for t in terms:
            f = cnt.get(t, 0)
            if not f:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            score += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(d) / avgdl))
        if score > 0:
            scored.append((score, r))
    scored.sort(key=lambda x: -x[0])
    return [r for _, r in scored]


RETRIEVE_Q = Noul(
    instructions=("Is this saved transcript item relevant to the question in `task`? Answer true if a "
                  "reader answering `task` would want to open this item verbatim. Answer false if it "
                  "is unrelated, or superseded by a later item that covers the same fact."),
    true="A reader answering `task` would want this item.",
    false="Not useful for `task`.",
)


def example_retrieval(blocks: list[Block], chunk_name: str, query: str, client, source: str, readings,
                      prefilter: int = 32, expect_lines: list[int] | None = None) -> None:
    rule(f"EXAMPLE 1 — verbatim-memory retrieval over {chunk_name}\n  query: {query!r}")
    mem = InMemoryStore()
    for b in blocks:
        mem.put(Record(id=content_id(b.text, salt=b.cite, prefix="t"), text=b.text, kind=b.kind,
                       origin=Origin(source=b.kind, ref=b.cite, turn=b.line), tokens=estimate_tokens(b.text),
                       created_turn=b.line, summary=f"{b.cite} {b.summary}"))
    if expect_lines:
        lite = mem.search(query, limit=len(mem))
        full = bm25_rank(mem.all_records(), query)
        pos_lite = {r.created_turn: i for i, r in enumerate(lite, 1)}
        pos_full = {r.created_turn: i for i, r in enumerate(full, 1)}
        print("  candidate coverage — rank of the lines the analyst says hold the answer:")
        print(f"  {'line':<7}{'store BM25-lite':>16}{'BM25 + length norm':>20}")
        for ln in expect_lines:
            print(f"  L{ln:<6}{str(pos_lite.get(ln, '-')):>16}{str(pos_full.get(ln, '-')):>20}")
        hit_lite = sum(1 for ln in expect_lines if pos_lite.get(ln, 10**9) <= 32)
        hit_full = sum(1 for ln in expect_lines if pos_full.get(ln, 10**9) <= 32)
        print(f"  in a 32-candidate request: {hit_lite} of {len(expect_lines)} (lite) vs {hit_full} of {len(expect_lines)} (length-normalised)")
    if prefilter <= 0:
        cands = sorted(mem.all_records(), key=lambda r: r.created_turn)
        print("  no prefilter: every block's one-line summary goes to Jev (the whole digest)")
    else:
        cands = mem.search(query, limit=prefilter)
    items = [ScoreItem(id=r.id, text=r.summary, tokens=estimate_tokens(r.summary), meta={"line": r.created_turn})
             for r in cands]
    first = items[:32]
    state = build_state(query, first, [f"i{k}" for k in range(len(first))])
    n_req = -(-len(items) // 32)
    all_state = sum(estimate_tokens(build_state(query, items[k:k + 32], [f"i{j}" for j in range(len(items[k:k + 32]))]))
                    for k in range(0, len(items), 32))
    print(f"  memory records {len(mem)}  ->  BM25 prefilter {len(cands)} candidates  ->  {n_req} Jev request(s), "
          f"{all_state:,} state tokens in total (computed from the batch plan, not measured; see [usage] below)")
    print(f"  first request: state tokens {estimate_tokens(state):,}   questions {len(first)} × ~{estimate_tokens(RETRIEVE_Q.to_payload()):,} tok")
    print("\n  the state Jev sees (verbatim, one-line summaries with provenance):")
    show_json(state, 2600)
    print("\n  one of the 32 questions (they differ only in the item ref):")
    show_json({"i0": {**RETRIEVE_Q.to_payload(), "instructions": "Considering item i0 only: " + RETRIEVE_Q.instructions}})
    results = score_items(client, query, items, RETRIEVE_Q, on_error="raise")
    by_line = {r.created_turn: r for r in cands}
    ranked = sorted(zip(results, items, strict=True), key=lambda p: -p[0].score)
    print(f"\n  answers ({source}):")
    print(f"  {'p(relevant)':>11}  {'cite':<8} {'summary'}")
    for res, item in ranked[:12]:
        print(f"  {res.score:>11.2f}  L{item.meta['line']:<7} {item.text[len(str(item.meta['line']))+2:][:82]}")
    print("\n  top 3, expanded VERBATIM from the chunk (the verbatim-memories deliverable):")
    for res, item in ranked[:3]:
        rec = by_line[item.meta["line"]]
        excerpt = " ".join(rec.text[:420].split())
        print(f"    {chunk_name}:L{rec.created_turn}  [{rec.kind}]  p={res.score:.2f}\n      {excerpt}{' …' if len(rec.text) > 420 else ''}")


# ----------------------------------------------------------------------------- example 2

BLOCK_EVIDENCE_Q = Noul(
    instructions=("Does this tool result contain verification evidence produced by running something: a test "
                  "or self-test result, a version or liveness probe, a hash or checksum, an exit code, a process "
                  "or memory census, a solver or build log, an error or traceback? Answer false if it is reference "
                  "material the session merely read: a skill file, documentation, source code listing, or search hits."),
    true="Evidence from execution that an audit would anchor to verbatim.",
    false="Reference material that was read, not produced.",
)
SEGMENT_EVIDENCE_Q = Noul(
    instructions=("Considering this segment of the tool result only: would a verification audit need it verbatim? "
                  "Answer true for lines carrying a result, measurement, version, hash, pid, error, or warning. "
                  "Answer false for headers, repeated boilerplate, and routine info lines that add no fact."),
    true="An audit would quote this segment.",
    false="Filler an audit would not miss; keep it behind a pointer.",
)


class _SpanPrefixed:
    """Ask the same questions, with the segment's line span prefixed to each item text.

    The analyst's readings are keyed ``L<line>:<a>-<b>``; putting that key in the state lets
    the offline stand-in answer per segment, and real Jev simply sees a provenance prefix.
    """

    def __init__(self, inner, line: int, spans: Mapping[str, tuple[int, int]]) -> None:
        self._inner, self._line, self._spans = inner, line, spans

    def ask(self, state, questions):
        if isinstance(state, Mapping):
            items = []
            for it in state.get("items") or []:
                span = self._spans.get(it.get("text"))
                items.append({**it, "text": f"L{self._line}:{span[0]}-{span[1]} " + it["text"]} if span else it)
            state = {**state, "items": items}
        return self._inner.ask(state, questions)


def example_evidence_gate(blocks: list[Block], chunk_name: str, ev_lines: list[int], client, source: str) -> None:
    rule("EXAMPLE 2 — evidence gate for the audit flattener (replaces its fixed 1500-char cap)")
    by_line = {b.line: b for b in blocks if b.kind == "tool_result"}
    print("  block-level question:")
    show_json(BLOCK_EVIDENCE_Q.to_payload())
    print("  segment-level question:")
    show_json(SEGMENT_EVIDENCE_Q.to_payload())
    for ln in ev_lines:
        b = by_line.get(ln)
        if b is None:
            print(f"\n  L{ln}: no tool_result on that line")
            continue
        print(f"\n  ── L{ln} tool_result[{b.tool}]  {len(b.text):,} chars, {estimate_tokens(b.text):,} tok; the audit flattener keeps 1,500 chars")
        # block-level: one question over the head of the block
        head_item = ScoreItem(id=f"blk:{ln}", text=f"L{ln} " + b.text[:1500], tokens=estimate_tokens(b.text[:1500]))
        blk = score_items(client, "verification audit of this session", [head_item], BLOCK_EVIDENCE_Q,
                          on_error="raise")[0]
        print(f"  block-level p(evidence) = {blk.score:.2f} ({source})")
        if blk.score < 0.5:
            print("  → relocate the WHOLE block behind one pointer; the audit cites it by chunk:line only:")
            print(f"    [[elided id={content_id(b.text, salt=b.cite, prefix='r')} lines=1-{len(b.text.splitlines())} tokens={estimate_tokens(b.text):,} \"{b.summary[:60]}…\"]]  → {chunk_name}:L{ln}")
            continue
        origin = Origin(source="tool_result", ref=b.cite, turn=ln)
        segs = segment(b.text, origin)
        store, log = InMemoryStore(), ShadowLog(path=None)
        cfg = GateConfig(keep_threshold=0.5, min_gate_tokens=100, max_elide_fraction=1.0, protected_kinds=frozenset({"stacktrace", "diff"}))

        spans = {s.text: s.line_span for s in segs}
        res = admit(b.text, origin, task_digest="verification audit of this session", turn=ln,
                    client=_SpanPrefixed(client, ln, spans),
                    store=store, log=log, config=cfg)
        by_id = {r.item_id: r for r in res.scores}
        failed = [r for r in res.scores if r.failed]
        if failed:
            raise SystemExit(f"Jev scoring failed inside admit() at L{ln}: {failed[0].error}")
        print(f"  {'span':>9} {'kind':<10}{'tok':>6}  {'p(need)':>7}  action   first line")
        for s in segs:
            sc = by_id[s.id].score
            act = "KEEP " if sc >= cfg.keep_threshold or s.kind in cfg.protected_kinds else "elide"
            print(f"  {s.line_span[0]:>4}-{s.line_span[1]:<4} {s.kind:<10}{s.tokens:>6}  {sc:>7.2f}  {act}    {' '.join(s.text[:70].split())}")
        kept_chars = sum(len(s.text) for s in res.kept)
        print(f"  result: {res.original_tokens:,} → {res.result_tokens:,} tok, {len(res.pointers)} pointer(s); verbatim chars kept {kept_chars:,} vs the flattener's 1,500 cap")
        print("  what the flattened corpus would contain (first 14 lines):")
        for line in res.text.splitlines()[:14]:
            print("    " + (line[:100] + ("…" if len(line) > 100 else "")))
        # byte-exact, scoped to the emitted pointers (see tools/transcript_experiments.py for why)
        back = res.text
        for p in res.pointers:
            rec = store.get(p.id)
            back = back.replace(format_pointer(p) + "\n", rec.text, 1)
        print(f"  expand() restores the original byte-exact: {back == b.text}")


# ----------------------------------------------------------------------------- example 3

VERIFIER_CLASS = Choice(
    instructions=("Which class of verifier produced the check described in `block`, per the audit ontology? "
                  "Use `context` (the surrounding assistant turns) to understand what the block was for."),
    criteria={
        "HX": "A human expert's statement, quoted verbatim.",
        "DC": "Deterministic code or math whose tool or method is certified for this purpose.",
        "DU": "Deterministic code or math, uncertified: a script, a solver run, a probe, a hash, a process census.",
        "LS": "A single LLM checking by reading and reasoning, possibly over tool output it did not compute.",
        "LC": "Several LLMs agreeing, with votes or dissent recorded.",
        "UC": "The user confirming, correcting, or rating from their own observation or knowledge.",
        "PS": "Comparison against a primary source: a spec sheet, vendor documentation, published data.",
        "RW": "A real-world physical outcome: a part fitting, a system operating.",
    },
)
RESULT_Q = Choice(
    instructions=("What is the outcome of the check in `block`? Read `context`: an error that the session "
                  "deliberately provoked to confirm a rejection is a pass, not a fail."),
    criteria={"pass": "The check confirmed what it set out to confirm.",
              "fail": "The check found the claim false or the system broken.",
              "partial": "Some of it confirmed, some not, or confirmed under a caveat.",
              "corrected": "An earlier claim was overturned by this check.",
              "not_a_check": "This block is not a verification event at all."},
)
RENDERER_Q = Choice(
    instructions="Who rendered the pass/fail verdict in `block`: a script that computed it, an LLM reading a printout, or a human?",
    criteria={"code": "A program computed and emitted the verdict itself, e.g. a field like pass: true.",
              "llm_reading_instrument": "An LLM read numbers or logs and judged them.",
              "human": "A person judged it."},
)
LOAD_BEARING_Q = Noul(
    instructions="Did later work in this session depend on the claim this block checks? Use `context`.",
    true="Later decisions or work rested on it.",
    false="Nothing downstream depended on it.",
)
GRADE_Q = Score(
    instructions="How strong is the evidence in `block` for an auditor who has only this transcript?",
    criteria=["A summary or recollection with no artifact behind it.",
              "A delegated or relayed report: tool-accessed then summarised, raw output not in the transcript.",
              "Primary tool output or data present verbatim in the transcript."],
)


def example_audit_screen(blocks: list[Block], audit_lines: list[int], readings: Mapping | None, client, source: str) -> None:
    rule("EXAMPLE 3 — audit-event screen (Choice / Noul / Score in the verification-audit vocabulary)")
    print("  questions asked over every candidate block, in one request per block:")
    for name, q in (("verifier_class", VERIFIER_CLASS), ("result", RESULT_Q), ("verdict_renderer", RENDERER_Q),
                    ("load_bearing", LOAD_BEARING_Q), ("evidence_grade", GRADE_Q)):
        print(f"   · {name}: {q.instructions[:120]}…  ({type(q).__name__}{'' if isinstance(q, Noul) else ': ' + ', '.join(list(q.criteria)[:8] if isinstance(q.criteria, Mapping) else ['0', '1', '2'])})")
    idx = {}
    for b in blocks:
        idx.setdefault(b.line, []).append(b)
    for ln in audit_lines:
        targets = idx.get(ln) or []
        if not targets:
            print(f"\n  L{ln}: nothing on that line")
            continue
        b = targets[-1]
        # context = nearest assistant text before and after (within 6 lines), truncated
        before = [x for x in blocks if x.kind == "assistant_text" and ln - 6 <= x.line < ln]
        after = [x for x in blocks if x.kind == "assistant_text" and ln < x.line <= ln + 6]
        ctx = {"before": " ".join((before[-1].text if before else "")[:500].split()),
               "after": " ".join((after[0].text if after else "")[:500].split())}
        state = {"context": ctx, "block": {"cite": b.cite, "kind": b.kind, "tool": b.tool, "text": b.text[:2500]}}
        print(f"\n  ── L{ln} {b.kind}[{b.tool}]  state ≈ {estimate_tokens(state):,} tok")
        print(f"     block head: {' '.join(b.text[:160].split())}")
        if source == "jev":
            answers = client.ask(state, {"verifier_class": VERIFIER_CLASS, "result": RESULT_Q, "verdict_renderer": RENDERER_Q,
                                         "load_bearing": LOAD_BEARING_Q, "evidence_grade": GRADE_Q})
            for k, a in answers.items():
                print(f"     {k:<17} {getattr(a, 'value', a)}  {getattr(a, 'probabilities', '')}")
        else:
            r = (readings or {}).get("audit", {}).get(f"L{ln}")
            if r:
                for k, v in r.items():
                    print(f"     {k:<17} {v}")
            else:
                print("     (no readings supplied for this line)")


# ----------------------------------------------------------------------------- example 4

STAGE_Q = Choice(
    instructions="What is the session doing in these recent turns?",
    criteria={"building": "Writing or editing code, manifests, or documents.",
              "verifying": "Running gates, tests, probes, or reading results back.",
              "recovering": "Diagnosing a crash, a failure, or an unexpected state.",
              "correcting": "Retracting or amending an earlier claim.",
              "documenting": "Recording results, updating notes, memory, handoffs, or skills.",
              "closing": "Summarising, handing off, or saving the session."},
)
FINISHED_Q = Noul(
    instructions=("Has the work batch these turns describe reached a verified stopping point, so that a checkpoint "
                  "(a transcript save, a frozen-prefix commit) would not cut a task in half?"),
    true="A gate or check has just passed and nothing is mid-flight.",
    false="Something is still open: a failing check, an unanswered question, a pending restart.",
)


def example_phase(blocks: list[Block], checkpoints: list[int], readings: Mapping | None, client, source: str) -> None:
    rule("EXAMPLE 4 — phase and commit-point detection (dispatch-monitor / save-transcript cadence)")
    print("  questions over a window of the last 4 user/assistant text blocks before each checkpoint:")
    show_json({"stage": STAGE_Q.to_payload(), "finished": FINISHED_Q.to_payload()}, 1600)
    texts = [b for b in blocks if b.kind in ("assistant_text", "user_text")]
    for ln in checkpoints:
        window = [b for b in texts if b.line <= ln][-4:]
        state = {"turns": [{"cite": b.cite, "role": b.kind.replace("_text", ""), "text": " ".join(b.text[:600].split())} for b in window]}
        print(f"\n  ── checkpoint L{ln}: window {[b.cite for b in window]}  state ≈ {estimate_tokens(state):,} tok")
        print(f"     last turn: {' '.join(window[-1].text[:150].split())}…")
        if source == "jev":
            a = client.ask(state, {"stage": STAGE_Q, "finished": FINISHED_Q})
            print(f"     stage {a['stage'].value} {a['stage'].probabilities}   finished {a['finished'].value:.2f}")
        else:
            r = (readings or {}).get("phase", {}).get(f"L{ln}")
            print(f"     {r if r else '(no readings supplied)'}")


# ----------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("chunk")
    ap.add_argument("--query", default="Where is the ccx binary on the host, what version is it, and how was it proven to run?")
    ap.add_argument("--evidence-lines", default="")
    ap.add_argument("--audit-lines", default="")
    ap.add_argument("--checkpoints", default="")
    ap.add_argument("--readings", default=None)
    ap.add_argument("--prefilter", type=int, default=32, help="BM25 candidates handed to Jev (32 per request); 0 = whole digest")
    ap.add_argument("--expect-lines", default="", help="lines the analyst says answer the query, for the coverage check")
    args = ap.parse_args()
    readings = json.load(open(args.readings)) if args.readings else None
    blocks, lines = load_blocks(args.chunk)
    chunk_name = os.path.basename(args.chunk)
    print(f"chunk {chunk_name}: {len(lines)} lines, {len(blocks)} content blocks")
    client, source = make_client((readings or {}).get("retrieve", {}))
    ints = lambda s: [int(x) for x in s.split(",") if x.strip()]  # noqa: E731
    example_retrieval(blocks, chunk_name, args.query, client, source, readings, prefilter=args.prefilter,
                      expect_lines=ints(args.expect_lines) or None)
    _print_usage(client)
    if args.evidence_lines:
        if source == "readings":
            client.table = (readings or {}).get("evidence", {})
        example_evidence_gate(blocks, chunk_name, ints(args.evidence_lines), client, source)
        _print_usage(client)
    if args.audit_lines:
        example_audit_screen(blocks, ints(args.audit_lines), readings, client, source)
        _print_usage(client)
    if args.checkpoints:
        example_phase(blocks, ints(args.checkpoints), readings, client, source)
        _print_usage(client)
    return 0


if __name__ == "__main__":
    sys.exit(main())
