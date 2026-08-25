# Decisions

Architectural and methodological decisions with rationale. If you're about
to propose "let's add a second entry point" or "let's just tune the
threshold after seeing the held-out result," check here first — it's
probably already been tried and rejected, and the reason is recorded below.

Status values: `Active` (binding), `Superseded` (replaced, see the entry
that replaced it).

---

## D001 — Single canonical public API

**Decision**: One entry point, `FuzzyXAI.wrap(model, ...).explain(...)`, and
one visualization namespace, `fuzzyxai.visualization`. `fuzzyxai.visual` and
`fuzzyxai.viz` exist only as compatibility shims for one release cycle.

**Reason**: Early development produced parallel visual namespaces during the
Explanation Experience milestone (v1.1.0rc1); keeping both live long-term
would let call sites drift onto an unmaintained path silently.

**Alternatives considered**: keeping `visual`/`viz` as permanent aliases —
rejected, they'd become permanent parallel maintenance surfaces.

**How to apply**: New code always imports `fuzzyxai.visualization`. If you
find code importing `visual`/`viz`, that's a migration bug, not a valid
usage.

**Status**: Active. See [`AGENTS.md`](../AGENTS.md).

---

## D002 — Evidence-first, fail-closed disclosure

**Decision**: Every claim carries an explicit origin: native, surrogate, or
missing. Unavailable evidence produces `insufficient_evidence` or an
explicit limitation — never a fabricated or inferred metric.

**Reason**: This is the dissertation's central defended property, not just
a style preference — see `RESEARCH.md`'s Core Hypothesis. Multiple research
cycles (H10-C5c development gate, H10-C7R development gates) failed
*because* their abstention/coverage thresholds were honored rather than
loosened after seeing a disappointing result. That discipline is what makes
the negative results in `RESEARCH.md` trustworthy.

**How to apply**: Never add a fallback that silently substitutes a plausible
value for a missing evidence channel. Adapters declare capabilities and
return partial evidence for unsupported channels (`CONTRIBUTING.md` rule 2).

**Status**: Active.

---

## D003 — Operator manifest traceability

**Decision**: Every defended operator must be mapped in
`framework/fuzzyxai/operators_manifest.yaml` to a callable, input/output
schemas, tests, artifacts, and a visualization policy. `make
operator-manifest-check` enforces this.

**Reason**: With 41+ registered operators spanning core theory, diagnostics,
and multiple research protocols, an operator without this mapping becomes
untestable and unreproducible — exactly the failure mode the evidence-first
policy (D002) is designed to prevent internally.

**How to apply**: Adding or changing a defended operator without a manifest
row is incomplete work, not a follow-up task.

**Status**: Active.

---

## D004 — Strict core/operators/evidence/visualization layering

**Decision**: `fuzzyxai.core` owns all mathematical implementation.
`fuzzyxai.operators` is a typed public facade that only delegates to `core`
— no computation lives there. `fuzzyxai.evidence` observes data, training,
model knowledge, similar cases, and counterfactual tests.
`fuzzyxai.visualization` never computes scientific metrics, only renders
already-computed evidence.

**Reason**: Keeps presentation code from silently becoming a second,
untested implementation of the math, and keeps the proof scripts
(`proofs/*.py`) checking the same code path the runtime actually uses.

**How to apply**: If you're writing a formula inside a visualization
renderer or a CLI/report script, it belongs in `core` or `evidence`
instead, with a corresponding test.

**Status**: Active. See [`docs/architecture.md`](../docs/architecture.md).

---

## D005 — Repair is never implicit

**Decision**: `repair_mode="execute"` requires an explicit
`RepairExecutionContext`, a registered operation, satisfied preconditions, a
preserved source artifact, and a rollback path. Batch repair execution is
disabled by default.

**Reason**: A diagnostic framework that can also mutate the system it's
diagnosing is a much larger blast radius than a read-only auditor. Every
research cycle that touches repair (H10-C3b, H10-C4, ML Pipeline v2) treats
"full recertification with zero new critical violations" as a hard gate,
not a nice-to-have — repair without a guaranteed, verified rollback path
would make that gate meaningless.

**How to apply**: Never add a repair code path that executes without a
caller-supplied context object. Diagnosis and repair-planning stay
side-effect-free by default.

**Status**: Active.

---

## D006 — External adapters are observation-only

**Decision**: Adapters connecting FuzzyXAI to independently built external
pipelines (used for the current release's headline validation, "External ML
Pipeline Validation v1") must not modify the registered core. They observe
and report; they cannot become a second implementation of core contracts.

**Reason**: This is what let the 2026-08-01 external validation claim "zero
core files changed by adapters" — the whole point of testing against
independently implemented fixtures is that the fixtures weren't shaped to
fit FuzzyXAI's internals. An adapter that reaches into core to make a
fixture pass would invalidate exactly the claim it's supposed to support.

**How to apply**: Code in `framework/fuzzyxai/fuzzyxai/external_adapters/`
never edits, monkeypatches, or special-cases core/diagnostics modules for a
specific external pipeline.

**Status**: Active.

---

## D007 — Sealed/held-out confirmatory protocol: single opening, no post-hoc tuning

**Decision**: Confirmatory (as opposed to formative/development) data is
sealed. An immutable opening record is written *before* decryption/scoring.
After one opening — success or failure — that data can never be reused as
"the" confirmatory run. No threshold, feature, cost, or budget may be tuned
after opening. Confirmatory controller inputs must be observable before
scoring; held-out labels may be targets but never feature channels.

**Reason (concrete precedent, not abstract caution)**: This rule exists
*because* of failures, not despite the absence of them:
- The original H10-C3 deterministic 240-template sealed set was found to be
  reconstructible from public source before scoring — invalidated,
  replaced with a secret-derived AES-256-GCM payload (v23.3). Its opening
  count stayed at zero; nothing was lost because it was never opened.
- The v19 H10-L/H10-R Gold-generation process was found, after positive
  results were already reported, to share a taxonomy with the evaluated
  solver (target leakage). Because the numbers were already public in the
  ledger, they could only be *relabeled* invalid, not silently removed —
  see `RESEARCH.md`'s Architectural Evolution and the ledger's own
  insistence on preserving negative/blocked results.
- `FXAI-FINAL-ONE-ZIP-PRACTICAL-CLOSURE`'s original scoring attempt failed
  *before* labels were opened (an envelope-unwrapping bug), and the
  recovery run is explicitly flagged in the ledger as
  `invalid_after_label_opening` / a declared protocol deviation — kept
  fully visible rather than presented as a clean run.

**How to apply**: If you're building or modifying any confirmatory
pipeline, the opening-record-before-decryption pattern and the
single-reuse-forbidden rule are not optional scaffolding — they're the
reason anyone can trust a positive result in this repo at all.

**Status**: Active.

---

## D008 — DubnaXAI generated site quarantined off `main`

**Decision**: The earlier generated DubnaXAI prototype site lives only on a
historical archive branch. It must never be restored to `main`.

**Reason**: `main` now represents the installable research framework as the
primary product (`AGENTS.md`); the generated site was superseded by that
framework and kept only for historical reference, not as a maintained
surface.

**How to apply**: If you find generated site artifacts reappearing in a
PR/branch destined for `main`, that's a regression, not new work to finish.

**Status**: Active.

---

## D009 — Version path B: no retroactive stable tags

**Decision**: When `v1.2.0`'s human/domain external gates stayed open,
the project chose not to retroactively stamp a `v1.2.0` stable tag once
work had moved on to `v1.3.0rc1`. The next allowed tag became `v1.3.0rc2`,
contingent on the same external gates plus a green main-branch CI run.

**Reason**: A stable tag is a claim about what was actually verified at
that point in time; back-dating one after building on top of unverified
work would misrepresent the verification boundary.

**How to apply**: Don't propose tagging a stable release retroactively to
"clean up" version history. If a release candidate's gates never closed,
the version number stays a candidate, permanently.

**Status**: Active.

---

## D010 — Gold/source-truth must be structurally independent of the evaluated solver

**Decision**: Ground-truth ("Gold") generation for any confirmatory study
must come from an independently implemented oracle with import-level
isolation from the solver being evaluated — it cannot derive labels from
the same taxonomy, catalog, or mutation generator the solver itself uses.

**Reason**: This is the direct lesson from the v19 target-leakage finding
(see D007) — the oracle had import-level independence and a separately
implemented cut solver, but its *label source* (the mutation catalog) still
exactly duplicated the evaluated H10 taxonomy, which was enough to
invalidate the confirmatory claim even though nothing else was wrong.
`gold_oracle/` (`source_truth.py`, `repair_truth.py`, `cut_oracle.py`) now
derives truth from low-level graph transactions instead.

**How to apply**: When reviewing a new confirmatory protocol, check where
Gold labels come from before checking anything about the solver. If Gold
and solver share a taxonomy or generator, the design is invalid regardless
of measured results.

**Status**: Active.

---

## D011 — AI review output is formative-only, never expert evidence

**Decision**: AI-generated pre-review output can inform formative
development but can never substitute for, or be presented as, independent
human/expert confirmation. It requires its own blinding and
leakage-remediation cycle before even formative use.

**Reason**: An earlier 360-case review bundle was frozen as a technical
prototype but explicitly barred from scoring because it disclosed outcome,
expected-action, and answer-key fields to reviewers — a leakage bug that
would have let any reviewer (human or AI) "solve" the case from the
input alone. The replacement (240 cases / 720 variants) strips those fields
and passed an automated leakage audit before any review could be scored.

**How to apply**: Don't treat AI agreement with a method's output as
validation evidence in dissertation claims. It's a development signal only.

**Status**: Active.

---

## D012 — Detection recall parity ≠ superiority claim

**Decision**: When a simple/local/pairwise baseline matches FuzzyXAI's raw
contract-recall on a benchmark, the project does not claim detection
superiority — it reports the actual measured advantage (root-cause
attribution, fewer redundant repair actions) and states explicitly where
statistical significance was or wasn't reached.

**Reason**: This happened repeatedly — Cross-Pipeline Practical v1
(pairwise rules also hit contract recall `1.0`) and ML Pipeline v2
Comparative (Holm-adjusted McNemar `p=0.5`, not significant, despite a
positive-looking bootstrap CI). Overselling detection recall when the
honest advantage is elsewhere would contradict D002.

**How to apply**: When writing up or summarizing a new benchmark result,
report where the framework's real advantage is (localization, repair
minimality, evidence completeness), not just whichever metric happens to
look best.

**Status**: Active.
