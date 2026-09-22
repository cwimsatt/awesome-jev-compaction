# TypeSafe docs vs. the constants hard-coded in jevctx

Fetched 2026-09-22 from docs.typesafe.ai (pages saved under `docs/`). The live model is
`jev-1.13.0` (aliases `jev-latest` and `jev-preview` both point to it).

## (a) The six claims in `jevctx/types.py` / `demo.py`

| # | Repo claim | Verdict | Evidence (verbatim) | Page |
|---|---|---|---|---|
| 1 | `MAX_QUESTIONS_PER_REQUEST = 32` | **NOT STATED** | No per-request question count appears in the docs. The only stated caps are token budgets (see 2, 3) and, per question type, "a maximum of 255 options per Choice" and a Score "up to 10" levels. The real limit is the token budget, not a fixed count of 32. | models.md, api.md |
| 2 | state + all questions ≤ 64,000 tok | **CONFIRMED** | "The 64k budget covers the `state` plus all questions combined" | models.md |
| 3 | state + longest question ≤ 32,000 tok | **CONFIRMED** | "the 32k budget applies to the `state` plus the single longest question" | models.md |
| 4 | 1,200 requests/min | **CONFIRMED** (repo omits a second cap) | "250,000 tokens per second / 1,200 requests per minute". The repo models the rpm cap but not the 250k tokens/sec cap. Also flagged: "Rate limits are adjusting dynamically … can change without notice". | models.md |
| 5 | $0.042 per million input tokens | **CONFIRMED** | "Price (per Btok / per Mtok) \$42 / \$0.042" and "Charged per input token. Output tokens are free." Cross-check: the rerank cookbook reports "1200 TypeSafe calls used 1,536,002 input … costing \$0.0645" → 1,536,002 × 0.042/1e6 = \$0.0645 exactly. | models.md |
| 6 | Questions are free; Jev "bills for state, not for questions" | **CONTRADICTED** | Docs: "Charged per input token" (every input token), and the 64k budget "covers the `state` plus all questions combined." The `usage_probe.txt` run confirms it directly: one shared 32-item state cost **1,414** input tokens with 1 question, **1,834** with 8, **3,296** with 32 — about **61 input tokens billed per added question**. Questions are billed. | models.md + usage_probe.txt |

**What the batching optimization actually is.** The repo's premise ("many questions over one
state, questions free") overstates it, but the underlying win is real and the docs describe it:
the *state* is what repeats. parallel_questions.md: "the 13 single-question calls re-send the
article 13 times; the batched call sends [it once] … the 13x token cost stays" for the document.
So batching N questions saves (N−1)×(state tokens), not the question tokens. With a small state
and many questions the saving shrinks; the questions are never free.

## (b) Guidance relevant to the jevctx design

**Thresholding a Noul (noul.md).** "A value near 0.5 means the model gives yes and no similar
probability." "Where to set the threshold depends on the cost of being wrong. Use 0.5 when yes
and no are equally easy to act on. Raise it when acting on a false yes is expensive … Lower it
when missing a true yes is expensive … Values in the middle can go to a person." This is exactly
jevctx's `keep_threshold` (default 0.35, biased toward keeping) and the shadow-log replay for
tuning it — the docs endorse tuning the threshold in code against the cost of a wrong drop.

**Confidence vs probability (confidence.md).** Confidence summarises how concentrated a Choice/
Score distribution is; it is separate from the probability of any one option. jevctx uses Noul
for admit/retrieve, which has no confidence field — correct, since a keep/drop gate wants the
yes-probability directly, not distribution concentration.

**How much to put in state (concepts_state.md + jev-1.13.md).** state.md: "Use an object for
most requests so each part of the state has a descriptive name." The jaggedness page is blunter
and directly relevant to Example 1's whole-digest scan: "Accuracy falls as the state grows with
content unrelated to the decision. Unrelated detail acts as a distractor," and "Jev suffers from
context rot, so unrelated material in the `state` costs you accuracy." This is a real argument
for a prefilter (or per-item batching) over dumping all 830 digest items into one scan — not
just cost, but accuracy.

**Rerank decomposition (rerank_typesafe.md).** "BM25 builds a fast search shortlist of 30
candidates for each of 40 queries, then TypeSafe re-ranks each shortlist" using "one TypeSafe
question per query-candidate pair." Top-1 rose 5%→18%, top-10 38%→62%. This is precisely
jevctx's `store.search()` BM25 prefilter → Jev rerank pattern (Example 1 and Example E), and the
docs confirm the shortlist-then-rerank shape rather than scoring the whole corpus.

**Citation check (citation_check.md).** "One `Choice` question decides whether the quote's
context supports the claim" — supports / contradicts / says-nothing. This is the shape Example 3
borrows for the audit-event screen (Choice over a fixed vocabulary), and the verification-audit
skill's own claim-vs-evidence check.

**Classifying RAG passages (classifying_rag_passages.md).** "Score each retrieved passage with
one TypeSafe request, then decide in code which ones reach the answering model … thresholds in
`route()`, first match wins." This is the admit-gate pattern exactly: score, then let code (not
the model) decide keep/relocate against a threshold.

**Jaggedness caveats that bite jevctx (jev-1.13.md).** (1) "does not count reliably … items in a
long list" — don't ask Jev to count; jevctx never does, it scores each item. (2) "A Choice over
options and one Noul per option answer different questions: the Choice is relative … while each
Noul is absolute and can be low for all of them." jevctx's per-item admit Nouls are absolute, so
all items in a block can score low at once — which is exactly why the `max_elide_fraction`
tripwire exists and why it fired on several fixtures and 16 transcript blocks in these runs.

## (c) Cookbook and pattern pages listed in llms.txt

```
cookbooks/autoformat
cookbooks/autoresearch_feature_discovery
cookbooks/citation_check
cookbooks/classification_using_confidence
cookbooks/classifying_rag_passages
cookbooks/consistency_choice_cookbook
cookbooks/consistency_noul_cookbook
cookbooks/date_extraction_cookbook
cookbooks/entity_alignment
cookbooks/function_calling
cookbooks/hierarchical_classification
cookbooks/llm_guardrails
cookbooks/parallel_questions
cookbooks/pre_parsed_value_extraction_cookbook
cookbooks/rerank_typesafe
cookbooks/sde_cascade
cookbooks/semantic_find
cookbooks/skill_suggestion
patterns/composite-scoring
patterns/confidence-routing
patterns/fan-out
patterns/intent-routing
```
