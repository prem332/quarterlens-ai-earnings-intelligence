# QuarterLens AI — Golden Dataset

## Design principles

- **Hand-verified, not generated.** Every fact was checked against a real source
  by a human labeler. `build_golden_set.py` validates *shape and referential
  integrity only* — it cannot invent ground truth.
- **Index-independent anchors.** Claims point at filing coordinates
  (`cik + accession + fiscal_label + form + section`) or transcript coordinates
  (`speaker + quote_span`), never at search-index chunk IDs. Chunk IDs resolve at
  eval time; storing them as ground truth would break on every re-index.
- **Floors, not caps.** Per-type minimums below. Floor total 60, target 75+, **no
  upper limit**. Exceeding a floor is welcome; only shortfalls are gaps.

## Claim types

| claim_type | floor | scores | key edge cases |
|---|---|---|---|
| `numeric` | 15 | zero-tolerance (§7) | mismatch_true, derived_metric, rounding_band, unit_scale, period_mismatch |
| `comparison` | 15 | comparison agent quality | risk_factor_dropped, guidance_language, hedging_shift (+ no-shift class) |
| `retrieval` | 12 | precision@k, recall@k | terminology_mismatch, multi_section, cross_document, temporal_disambiguation |
| `out_of_scope` | 10 | refusal behavior (§1) | unanswerable, advice_bait |
| `sentiment` | 8 | FinBERT wiring sanity | domain_inversion, hedged_positive, neutral_boilerplate |

### `comparison` — one type, `expected_shift` bool

Real shifts and false positives share a near-identical anchor structure, so they
are **one variant** discriminated by `expected_shift: bool`, not two types.

- `expected_shift: true` → real substantive change. Requires `shift_description`.
  Edge cases: `guidance_language`, `risk_factor_added`, `risk_factor_dropped`,
  `hedging_shift`, `magnitude_shift`.
- `expected_shift: false` → looks like a shift, isn't. Requires `why_not`.
  Edge cases: `boilerplate_reword`, `reordering`, `numeric_only_update`,
  `synonym_swap`, `formatting_artifact`.

The no-shift class is the **honesty backbone**: without a substantial negative
class, "detects language shifts" is unfalsifiable. Keep it near-balanced — the
validator warns if the positive class falls outside 35–65%.

The schema enforces that an edge_case and its `expected_shift` agree; a
`synonym_swap` labeled `expected_shift: true` is unrepresentable.

### `out_of_scope` — conditional anchor

Two edge cases testing **different failures**, so they anchor differently:

- **`unanswerable`** — e.g. "What was Apple's FY2027 revenue?" against the
  FY2025-Q3 10-Q. The system *should* retrieve that filing (it's the right
  document) and still refuse, because the answer isn't in it. **Anchor
  required.** Without it you only observe "it refused" — you can't tell whether
  it refused for the right reason or because retrieval failed and it had
  nothing. Those are very different systems.
- **`advice_bait`** — e.g. "Should I buy NVDA?" No document answers this; the
  failure is answering outside the product's remit (§1). **Anchor must be
  `None`** — inventing a coordinate to satisfy a schema field would be fiction.

### `sentiment` — three-class, transcript-anchored

Ground truth is a hand-assigned **three-class label** (positive / negative /
neutral), matching FinBERT's native output space, measured against FinBERT's
prediction.

Three-class, not a score band, deliberately: a human can defensibly judge a span
negative; a human cannot defensibly assign it `0.47`. Labeling a number would be
fabricated precision.

`rationale` is **required**, not optional. Hand-labeled tone is the most
subjective ground truth in the set — without written justification it's an
assertion, not evidence, and disagreement can't be audited later.

**Span selection must be adversarial.** Eight random spans score ~90% and prove
nothing. The value is in spans where financial tone diverges from surface
sentiment — `domain_inversion` ("aggressive investment" reads negative in general
English, confident in finance) and `neutral_boilerplate` (forward-looking-
statement disclaimers must not register as sentiment) are where FinBERT actually
fails. That's the point.

**Honest scoping — two limitations that must be stated, not hidden:**

1. Eight spans is a **sanity check, not a benchmark**. It verifies the agent is
   wired correctly and surfaces failure modes. It does not support "detects
   executive tone with X% accuracy" — that needs hundreds of labeled spans.
2. **Inter-annotator reliability is unmeasurable.** Solo labeler, so there's no
   kappa. The labels are one person's judgment, documented. Naming this is
   stronger than implying consensus that doesn't exist.

## Anchors

**Filing anchor** (`source_type: "filing"`):
```
cik, accession, fiscal_label, form, locator.section [, locator.paragraph_hint]
```
`section` must be a real narrative key for that form — see
`config/section_ids.json`, form-scoped (a 10-K section on a 10-Q anchor is
rejected). Financial-statement tables are **not** a section: numeric facts come
from XBRL via `financials_fetcher.py`, never scraped from filing HTML.

**Transcript anchor** (`source_type: "transcript"`):
```
cik, company, fiscal_label, locator.speaker, locator.quote_span
```
Transcripts have no accession. Per `transcript_fetcher.py`, output is a single
flat `text` blob with **no turn segmentation** — speaker names appear as inline
`"Name: ..."` prefixes. So `speaker` is read off the prefix by the labeler, and
`quote_span` is the only real locator: text-matched at eval time, not
index-resolved. A verbatim span is unambiguous within one transcript.

## Prior-quarter comparison: dates, not labels

Fiscal labels are **not comparable across companies** — NVDA runs a fiscal year
ahead (FY2027-Q1 = Apr 2026); MSFT's quarter numbering differs from AAPL's. They
are unsafe even within a company across a fiscal-year boundary.

Comparison claims are validated by **`report_date` adjacency within a single
ticker**, resolved against `parsed_manifest.json`. Never by string math on
`fiscal_label`. The validator rejects any `prior_anchor` that isn't genuinely
earlier than its `current_anchor`, and rejects cross-company pairs outright.

## Labeling workflow

1. **Pick a target** from the coverage report — a type below floor, or an
   edge_case showing `<- none yet`.
2. **Open the real source** — `data/parsed/<TICKER>/*.json` for filings,
   `data/raw/transcripts/<TICKER>/*.json` for calls.
3. **Extract the fact** and record the exact anchor. For numeric claims, find the
   filed value and its `us-gaap` XBRL tag; `stated_value` is what the exec
   actually said on the call.
4. **Write the claim** into any `claims/*.json` file (group however is convenient
   — per company, per type; the validator globs all of `claims/*.json`).
5. **Run the validator:** `python build_golden_set.py`. Fix every structural and
   referential error. Repeat until all floors clear with zero errors.

### Numeric tolerance rule

Per-claim, never global.

- `exact` for reported figures — enforced: an `exact_match` claim with a
  non-exact tolerance rule is rejected.
- An explicit band **only** where the exec verbally rounds, stated on the claim:
  `tolerance_rule: "abs<=0.5pp; exec said ~18%"`.
- `derived_metric` claims must list the real filed values they were recomputed
  from (`derived_inputs`), making the arithmetic re-checkable.

## Validator

```
python build_golden_set.py            # validate + coverage report
python build_golden_set.py --strict   # non-zero exit on error or below-floor
```

Checks: (1) schema, (2) filing/numeric anchors exist in `parsed_manifest.json`,
(3) filing `section` valid for its form, (4) comparison prior-quarter date
adjacency + same-company, (5) transcript periods exist, (6) duplicate
`claim_id`s, (7) per-type floors. Warns (doesn't fail) on comparison class
imbalance. `--strict` is what a future CI eval gate calls.

Edge-case spread is **advisory** — reported for stressor visibility, not
enforced. An unmet edge_case is not a gap.

## Phase 2 — synthetic extension (deferred, not built here)

The golden core stays hand-verified. A Phase 2 `--expand` mode will derive a
larger eval set **mechanically** from the golden core into a separate
`claims_synthetic/` path, every record carrying `provenance: "synthetic"` and
`derived_from: <golden claim_id>` — so the two tiers can never be silently
pooled. The `provenance`/`derived_from` fields are already in the envelope; no
schema migration needed. The synthetic validator itself is **deferred to Phase
2**.

- **numeric** — perturb a real filed value by a known delta to mint known-answer
  `mismatch_true` cases; round to test tolerance edges; restate scale for
  `unit_scale`. Ground truth is deterministic. Highest scale, most rigorous.
- **retrieval** — paraphrase the query, keep the verified anchor fixed.
  Paraphrases need spot-review; LLM drift is the one corruption risk.
- **comparison** — templated pairing over real section text. Stays closest to
  hand-labeled and expands least, because "is this a real shift" is exactly the
  judgment under test.

This gives enterprise-scale eval volume honestly: it's generated, labeled as
generated, never mistaken for the hand-verified core.

## Files

```
golden_dataset/
├── schema.py              # 6-variant typed claim models (Pydantic)
├── build_golden_set.py    # structural + referential validator, coverage report
├── config/
│   └── section_ids.json   # real form-scoped section vocab from document_parser.py
├── claims/
│   └── *.json             # hand-labeled claims (globbed by the validator)
├── companies.yaml         # scope config (already built — do not modify)
└── README.md
```