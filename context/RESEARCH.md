# FuzzyXAI Research History

Compressed synthesis of `PROJECT_MEMORY.md` (1510 lines, ~90 dated
milestones, append-only, never edited). This file answers "what do we know
and where do things stand" — it does not replace the ledger as the source of
exact commit hashes, SHA256 evidence hashes, or CI run IDs; go to
`PROJECT_MEMORY.md` (`grep` by protocol name, e.g. `H10-C4`) when you need
those.

## Research Goal

Build a framework that makes an ML/XAI pipeline's evidence chain
machine-verifiable — provable relationships between data, model, prediction,
and explanation — and evaluate, as a doctoral dissertation, whether that
verification (a) has sound mathematical foundations, (b) transfers to
practical action policy (accept/review/block), and (c) transfers to causal
localization of naturally occurring software/ML faults.

## Core Hypothesis

An explicit, typed, evidence-graph representation of an ML/XAI route — with
composition operators, semantic-disagreement metrics, and a fuzzy
representation hierarchy (F0/interval/hesitant/neutrosophic/multilevel) —
can detect contract violations, localize their root cause via a minimal
diagnostic cut, and support a safe, reversible repair/recertification cycle,
*without* fabricating evidence when a channel is unavailable.

## Key Findings (currently valid, confirmed or supported)

- **Chapters 2-3 core theory** (composition, semantic disagreement `d_E`,
  loss/index `L(E)`/`I(E)`, representation hierarchy and reductions,
  category/HoTT extension) — implemented and proof-checked, not a
  statistical hypothesis; see `proofs/`, `reports/thesis_validation.json`.
- **H10-C3 (route diagnostic effectiveness)** — `CONFIRMATORY_PASS`
  (2026-07-24, one-time sealed opening, independently reproduced postopen).
  Effect `0.527` (95% CI `[0.460, 0.597]`), positive in 6/6 pipeline
  families, zero false certification.
- **H10-C4 (operational repair cost)** — `H10_C4_SUPPORTED` (2026-07-25).
  Global minimal-cut repair strategy cuts mean normalized cost from `1.0`
  (apply-all) to `0.517`, action/component/recert-check counts from
  `3.1/3.1/14.3` to `1.0/1.0/8.0`.
- **H10-C6 / H10-C6-N (cut robustness under perturbation)** — `SUPPORTED`,
  but scope is narrow: registered truth-preserving structural/feature-noise
  perturbations only.
- **Engineering validation program** (ML Vertical v1 -> ML Pipeline v2 -> v2
  comparative -> cross-pipeline practical v1 -> external ML pipeline
  validation v1) — this is the basis for the *current release's* validation
  claims in `README.md`/`RELEASE_STATUS.md`. See its own section below.
- **H5-A (controlled route-contract validity)** — consistently near-`1.0`
  F1 and near-`0` false certification across every cycle that measured it.
  This is the most robust result in the whole program precisely because it's
  a structural-consistency claim, not a predictive one.

## Research Programs

### 1. Core theory (chapters 2-3, category/HoTT extension)

Not a hypothesis-tested program — a mathematical construction with
proof-checked invariants. The operator route is
`E -> T -> gamma -> Delta -> rDelta -> rho -> D -> chi -> action`.
`fuzzyxai.core` owns the math; `fuzzyxai.operators` only delegates.
Chapter-to-code mapping is exhaustively tabulated in
[`CHAPTER_MAPPING.md`](../CHAPTER_MAPPING.md) — use that file, don't
duplicate it here.

### 2. Practical action-controller program (H1-H9, Q1, "Strong Confirmatory")

**Goal**: can the framework act as a budgeted accept/review/block policy
that beats simple baselines, with honest disclosure of what's native vs.
surrogate vs. missing evidence?

**Path**: Full Empirical Validation (E1-E8, controlled, 2026-07-20) -> Q1
Empirical Remediation (real datasets: UCI Covertype, Fashion-MNIST, 20
Newsgroups, ElectricDevices) -> FXAI-Q1-FINAL-CLOSURE -> AI Pre-Review
blinding/leakage-remediation cycles (240 formative + 120 confirmatory cases,
tabular/image/text/time-series) -> Strong Confirmatory Formative ->
Final Practical Closure Formative -> **FXAI-FINAL-ONE-ZIP-PRACTICAL-CLOSURE**
(2026-07-21, `v1.3.0` technical-stable target, 17,404 sealed test objects
scored in one opening).

**Final locked scientific position** (from the one-zip closure — this
supersedes every earlier formative number in this program):

| Hypothesis | Status | Note |
| --- | --- | --- |
| H3 (action policy vs. baseline) | **Not supported** | at 20% review budget, P1 produced *more* invalid actions than a weighted-linear baseline (relative effect `-0.039`, CI crosses zero) |
| H3-P2 | Not estimable | no development operating point met the 0.05 risk ceiling |
| H5-A (route-contract validity) | **Supported**, controlled/compositional fault library only | F1 `1.0`, false cert `0.0` |
| H5-P (predictive error claim) | Not supported | never was |
| H6-A (planted-rule detectability) | **Supported**, synthetic planted-rule region only | H6-general not supported |
| H6-B | Not supported | required CI/Holm test never produced |
| H7-A (canonical hash preservation) | **Supported** | exact hash match, 17,404 artifacts |
| H7-B | Not supported | |
| H8, H9 | Bounded technical measurements only | explicitly disallowed as confirmatory claims by their own artifacts; H9 is operator-only, not end-to-end |

**Takeaway**: the action-*policy* hypothesis (H3) failed. The
route-*consistency* hypotheses (H5-A, H7-A) held, within their controlled
scope. This is why the framework's public claims center on contract
auditing/diagnosis, not on "this will catch more errors than a human
reviewer."

Human-facing gates (independent comprehension pilot, domain-language
review, expert-action review) were `planned_not_run` / `pending_external_review`
at closure and are not resolved in the ledger — see `PROJECT.md`.

### 3. Causal diagnostic program (H10, v16 → v23.3)

**Goal**: does the RouteGraph + minimal-cut diagnostic (a) work on
controlled route mutations (H10-C series), and (b) transfer to naturally
occurring software incidents, e.g. localizing a real bug in a real repo
(H10-C5/C5b/C5c/C7 series)?

**Methodology had to be corrected twice before a confirmatory lock could be
trusted** — both corrections are important precedent, see
[`DECISIONS.md#D010`](DECISIONS.md):
- **v19**: independent review found the mutation generator's source-node
  catalog exactly duplicated the evaluated H10 taxonomy — target leakage.
  Reported positive H10-L/H10-R effects were retroactively marked
  `invalid_not_evaluated_with_independent_source_truth` even though the
  numbers were reproducible.
- **v23.2**: the deterministic 240-template sealed set was found to be
  reconstructible from public source before it could be scored — thrown out
  before opening, replaced by an AES-256-GCM payload derived from an
  external secret (v23.3).

**Controlled-mutation results** (H10-C2/C3/C4/C6) — see Key Findings above
for the two that passed (C3, C4). H10-C2 (minimum-cut-membership
confirmatory) never reached lock: registered power analysis showed the
6-pipeline grid tops out at ~0.088 power against ~0.10 target; an honest
design estimate said ~222 pipelines / ~124K cases would be needed, and that
expanded design was never approved. `BLOCKED_POWER`, not evaluated.

**Natural-incident transfer (H10-C5 family) — never succeeded.** This is
the single largest concentration of negative/blocked results in the whole
project:
- H10-C5 (SWE-bench Lite, 26 incidents / 10 repos): `H10_C5_NOT_SUPPORTED`
  — no advantage over strongest greedy baseline on held-out repos.
- H10-C5b (BugsInPy, 24 held-out / 12 repos, one-time official scoring):
  `H10_C5B_NOT_SUPPORTED` — coverage `0.208` vs. required `0.70`, zero
  difference from `B_GREEDY` baseline.
- H10-C5c (evidence-retrieval redesign): development-gate failed before any
  held-out set was opened (`H10_C5C_DEVELOPMENT_GATE_FAIL`); candidate
  Recall@10 `0.567` vs. required `0.75`.
- H10-C7 family (BM25/dense/RRF/structural/causal-runtime retrieval
  variants R0 through R10M, ~10 development cycles): every variant that
  reached a go/no-go gate failed it. R5 came closest — held-out Recall@20
  `0.55` vs. required `0.80` (`H10_C7R_NOT_SUPPORTED`, 2026-07-30). R10M
  (final variant, richer causal-runtime evidence) still didn't beat frozen
  R9 symbol-recall or BM25 MRR on development data and never opened a
  held-out set.

**Bottom line**: the diagnostic engine is confirmed to work on *controlled,
registered* route mutations. Its transfer to *naturally occurring* code
faults was tested extensively and repeatedly failed to clear preregistered
gates. Treat any claim of "finds real bugs in real repos" as unsupported by
this repository's own evidence.

### 4. Engineering validation program (current release basis)

Distinct from the H10/H1-H9 confirmatory programs above — this is
*engineering* validation (does the shipped software correctly detect
registered, controlled consistency faults across independently built
pipelines), not a new scientific hypothesis test. This is what
`README.md` and `RELEASE_STATUS.md` currently lead with.

1. **ML Vertical v1** — one real sklearn+SHAP route end-to-end (REST, UI,
   MLflow, Docker), 10/10 acceptance scenarios.
2. **ML Pipeline v2** — extends with 8 more registered consistency
   contracts (S11-S18: leakage, split overlap, fit scope, feature order,
   convergence, artifact integrity, SHAP consistency, provenance). 18/18
   scenarios.
3. **ML Pipeline v2 Comparative** — 9 modes incl. strong local checks vs.
   full FuzzyXAI. Strong local checks falsely certified 6 cross-stage
   violations that full FuzzyXAI caught; paired-bootstrap CI for the
   difference was `[0.111, 0.556]` but the Holm-adjusted McNemar test was
   *not* significant (`p=0.5`) — descriptive result on a fixed registry, not
   a proven superiority claim.
4. **Cross-Pipeline Practical v1** — 5 packaged real pipelines, 200
   controlled cases / 1,000 mode decisions, 1,000 MLflow runs. Graph
   advantage = root-cause attribution + fewer redundant repairs, *not*
   superior pairwise detection (pairwise rules also hit contract recall
   `1.0`).
5. **External ML Pipeline Validation v1** (2026-08-01, current headline
   result) — 4 independently implemented public fixtures (sklearn
   ColumnTransformer, SHAP TreeExplainer, MLflow ElasticNet, LIME tabular)
   connected via **observation-only adapters that changed zero core files**.
   Full mode: violation recall `1.0`, stage/contract/root-cause accuracy
   `1.0`, repair/recertification `1.0`, zero false certifications/blocks.

## Rejected Approaches

- **Deterministic/plaintext sealed template banks for confirmatory
  scoring** — rejected twice (H10-C3 pre-v23.2 attempts) once found to be
  reconstructible from public inputs. Replaced by secret-derived encrypted
  payloads with an immutable single-opening record (`DECISIONS.md#D007`).
- **Mutation-generator-derived Gold/source truth** — rejected after v19
  leakage finding. Gold must now come from an independently implemented,
  import-isolated oracle that never shares a taxonomy with the evaluated
  solver.
- **BM25 / dense-retrieval / RRF / structural / causal-runtime retrieval for
  natural-code causal localization** (H10-C7, ~10 variants) — all rejected;
  none cleared preregistered held-out recall gates. Natural-code transfer
  remains an open problem, not a solved one.
- **Treating AI-generated review output as expert/human evidence** —
  explicitly rejected; AI pre-review is formative-only, always blinded, and
  never substitutes for the independent comprehension/domain/expert-action
  gates.
- **Local/simple/pairwise baselines as sufficient** — repeatedly measured
  as competitive on raw contract-recall (often reaching `1.0` too); the
  framework's actual advantage is root-cause localization and reduced
  redundant repair action, not raw detection recall. Don't oversell
  detection superiority — that claim doesn't hold.

## Architectural Evolution

```
DubnaXAI generated-site prototype  ─┐
                                     ├─ quarantined to historical archive branch (never on main)
                                     ┘
v1.0.0rc1  research framework completion
           unified FuzzyXAI.wrap(...).explain(...); evidence contracts; ExplanationGraph;
           operator manifest (30 entries)
v1.1.0rc1  Explanation Experience
           claim-centered ExplanationGraph; E0-E5 honesty levels; typed visual spec;
           comprehension study protocol frozen (planned_not_run)
v1.2.0rc3  Empirical Validation Gate
           object_85 real training experiment; 5-model capability matrix;
           comprehension pilot + domain review become blocking external gates
           (version path B chosen: no retroactive v1.2.0 stable tag)
v1.3.0rc1  Universal Model Integration
           ModelAdapterV2; sklearn/XGBoost/LightGBM/CatBoost/PyTorch/TensorFlow/ONNX adapters
     │
     ├── [parallel] Practical action-controller program (H1-H9, Q1) → FXAI-FINAL-ONE-ZIP-PRACTICAL-CLOSURE
     │              → v1.3.0 technical-stable target; H3 not supported, H5-A/H7-A supported
     │
     └── [parallel] Causal diagnostic program (H10, v16→v23.3)
                    → H10-C3/C4 confirmatory PASS; natural-incident transfer (C5/C5b/C5c/C7) not supported

v1.4.0a1   diagnostic-framework-v21
           H10's exploratory diagnostic components promoted to the public
           FuzzyXAI.diagnose() API (RouteGraph, contract validator, minimal
           cut, registered repair, recertification) — this became today's
           framework/fuzzyxai/fuzzyxai/diagnostics/ package
v1.4.0a2   Engineering validation program
           ML Vertical v1 → ML Pipeline v2 → v2 comparative → cross-pipeline
           practical v1 → external ML pipeline validation v1 (current)
```

## Current Scientific Position

- The mathematical core (chapters 2-3) is proof-checked and stable.
- The framework's *diagnostic* claims (contract violation detection, root
  cause via minimal cut, safe repair/recertification) are confirmed on
  controlled, registered route mutations across 5 independently built
  pipelines, including one with zero core-code changes on the external
  adapter side.
- The framework's *action-policy* claim (H3: catches more errors than a
  simple baseline under budget) is **not supported**.
- The framework's *natural-code-bug-localization* claim is **not
  supported** after ~10 development cycles across two protocol families
  (H10-C5, H10-C7).
- No human-utility, clinical-safety, or engineer-time-saved claim has ever
  been supported anywhere in the ledger; several are explicitly disabled by
  their own governing artifacts pending independent human studies that
  remain unresolved.

## Open Questions

See [`ROADMAP.md`](ROADMAP.md) for the actionable version of this list.

- Is there any retrieval/evidence strategy that can close the natural-code
  recall gap in H10-C7, or is natural-incident causal localization out of
  scope for this method family entirely?
- Have the comprehension pilot / domain-language review / expert-action
  review gates been run since the practical closure? Not evidenced in the
  ledger as of 2026-08-01.
- Is the framework alpha release track (`1.4.0aX`) intentionally decoupled
  from the dissertation's confirmatory human-study gates, or is that an
  undocumented gap?

## Next Experiments

Nothing is currently an active, scheduled next step in the ledger — the
last few entries are engineering/CI consolidation, not new protocol
openings. Do not assume a next study is "in flight" without checking
`git log` and `PROJECT_MEMORY.md` for entries after 2026-08-01.
