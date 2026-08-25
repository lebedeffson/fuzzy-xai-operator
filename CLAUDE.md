# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Before anything else: read `context/`

This file covers *how* to build/test/navigate the repo. For *what the
project is right now, why it's built this way, and what research has
already been done*, read the tool-agnostic memory layer instead — it's
shared with GPT/Codex sessions on this repo and is kept current on purpose:

- [`context/PROJECT.md`](context/PROJECT.md) — current state, version, known problems
- [`context/RESEARCH.md`](context/RESEARCH.md) — compressed history of every research program and its findings
- [`context/DECISIONS.md`](context/DECISIONS.md) — architectural/methodological decisions with rationale
- [`context/ROADMAP.md`](context/ROADMAP.md) — open threads
- [`context/USER_CONTEXT.md`](context/USER_CONTEXT.md) — how the user works

`PROJECT_MEMORY.md` at the repo root is the raw append-only ledger those
files are synthesized from — go there only for an exact commit hash, SHA256,
or statistic; do not re-derive history from it when `context/RESEARCH.md`
already has the synthesis.

## What this repository is

FuzzyXAI is an evidence-first framework for auditing, diagnosing, repairing, and
recertifying ML/XAI pipelines. It observes a route (dataset -> split ->
preprocessor -> model -> prediction -> explanation), validates contracts
between stages, computes a minimal diagnostic cut for failures, performs
registered/reversible repairs, and recertifies the route.

The repository doubles as the source tree for a doctoral dissertation: most
top-level directories (`experiments/`, `protocol/`, `results/`, `reports/`,
`dissertation_artifacts/`, `release_evidence/`, `study/`, `evidence/`,
`gold_oracle/`, `research_validation/`, per-hypothesis dirs like `h10_c*`,
`q1*`, `chapter*`) are locked evaluation protocols, generated evidence, or
one-off research scripts tied to specific defended experiments. **Do not edit
files under `protocol/` or already-scored `results/`/`reports/` artifacts
unless the task explicitly asks you to reproduce or extend that study** —
these are frozen for scoring/traceability. When in doubt about whether a
directory is "framework" or "dissertation study", check whether it's under
`framework/fuzzyxai/` (framework) or has a `PROTOCOL_LOCK.json`/`METHOD_LOCK.json`
(locked study).

The installable product — what almost all non-research tasks should touch —
lives at `framework/fuzzyxai/fuzzyxai/`.

## Core rules (from `AGENTS.md`)

- Public behavior must be evidence-first: unavailable evidence yields an
  explicit limitation or `insufficient_evidence`, never a fabricated metric.
- One canonical API: `FuzzyXAI.wrap(...).explain(...)` and the
  `fuzzyxai.visualization` namespace. `fuzzyxai.visual` and `fuzzyxai.viz` are
  compatibility shims only — new code must not target them.
- Every defended operator must stay mapped in
  `framework/fuzzyxai/operators_manifest.yaml` to a callable, schemas, tests,
  and a visualization policy.
- Confirmatory controller inputs must be observable before scoring; held-out
  labels may be targets but never feature channels.
- `PROJECT_MEMORY.md` records the release boundary and validated claims for
  past studies — read it before touching a specific hypothesis/study
  directory (e.g. `h10_c5b`, `ml_pipeline_v2`) to see what's already locked.
- Build shareable source archives only with `python scripts/build_framework_release.py`
  (never zip the dirty worktree).

## Commands

Install (editable, with dev extras):
```bash
python -m venv .venv && .venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"
```
Optional extras: `ml-vertical` (SHAP/REST/MLflow), `xgboost`, `lightgbm`, `catboost`.

Fast inner-loop check (run before/while editing framework code):
```bash
PYTHONPATH=framework/fuzzyxai:. python -m pytest -q tests/test_public_framework_api.py
python -m ruff check framework/fuzzyxai/fuzzyxai
python -m compileall -q framework/fuzzyxai/fuzzyxai
make operator-manifest-check
```

Run a single test file / test:
```bash
PYTHONPATH=framework/fuzzyxai:. python -m pytest -q tests/test_operator_trace.py
PYTHONPATH=framework/fuzzyxai:. python -m pytest -q tests/test_operator_trace.py::test_name
```
Note: `tests/conftest.py` already inserts the repo root and `framework/fuzzyxai`
onto `sys.path`, so `PYTHONPATH=...` is often redundant for pytest but is used
consistently across the Makefile/CI and is safe to keep.

Full regression suite (slow — hundreds of tests across the whole tree, `testpaths = tests`):
```bash
PYTHONPATH=framework/fuzzyxai:. python -m pytest -q
```

Type checking (strict, limited file list defined in `pyproject.toml` under `[tool.mypy]`):
```bash
mypy
```

Release/CI gate equivalent to `.github/workflows/fuzzyxai-readiness.yml`:
```bash
make framework-release-check   # operator-manifest-check + public API/evidence-first/release-contract tests
```

Focused REST/UI/MLflow checks:
```bash
PYTHONPATH=framework/fuzzyxai:. python -m pytest -q \
  tests/ml_pipeline_v2 tests/integration tests/external_ml_pipeline_v1
```

The `Makefile` has 100+ targets (`grep '^[a-zA-Z0-9_-]*:' Makefile`), but almost
all beyond the ones above are for specific dissertation chapters/studies
(`chapter3-*`, `q1-*`, `h10-*`) — don't invoke them unless the task is about
that study.

## Architecture

```
Model / dataset / training run
  -> ModelAdapter (facts and capabilities)
  -> evidence collectors
  -> chapter 2-3 core operators
  -> ExplanationEvidence
  -> ExplanationGraph
  -> ExplanationViewModel
  -> Matplotlib / HTML / MATLAB
```

The invariant operator route is `E -> T -> gamma -> Delta -> rDelta -> rho -> D -> chi -> action`
(evidence -> ... -> diagnosis -> action). Prediction, evidence quality,
diagnostics, and action are kept as separate values, never conflated.

Layering rule: `fuzzyxai.core` owns the mathematical implementation.
`fuzzyxai.operators` is a typed public facade that must only delegate to
`core` — no computation lives there. `fuzzyxai.evidence` observes data,
training, model knowledge, similar cases, and counterfactual tests.
`fuzzyxai.visualization` never computes scientific metrics, only renders.

Package layout (`framework/fuzzyxai/fuzzyxai/`):
- `runtime.py` — the public `FuzzyXAI` facade (`FuzzyXAI.wrap(...).explain(...)`)
- `adapters/` — capability-based model adapters (sklearn, XGBoost, etc.); a
  model being importable is not a support claim — adapters declare
  capabilities and return partial evidence for unsupported channels
- `evidence/` — claims, provenance, human-facing views
- `diagnostics/route_graph.py` — canonical route (nodes/edges/versions/schemas) construction
- `diagnostics/validator.py` — local and inter-stage contract checks
- `diagnostics/minimal_cut.py` — exact and approximate diagnostic cuts
- `diagnostics/repair_planner.py` / `repair_executor.py` — registered repair
  planning and execution; repair is never implicit — `repair_mode="execute"`
  requires an explicit `RepairExecutionContext`, satisfied preconditions,
  preserved source artifact, and a rollback path
- `diagnostics/recertification.py` — full-route re-verification after repair
- `pipelines/` — executable registered ML/XAI pipelines
- `external_adapters/` — observation-only adapters for external projects (must
  not mutate the registered core)
- `visualization/` — canonical visual spec + renderers (the authoritative
  namespace; `visual`/`viz` are shims)
- `operators_manifest.yaml` — traceability: every defended operator ->
  callable -> schemas -> tests -> visualization policy. Update this whenever
  you add/change a defended operator, then run `make operator-manifest-check`.

## Contributing conventions (from `CONTRIBUTING.md`)

1. Add scientific computations to `fuzzyxai.core` or `fuzzyxai.evidence`, never to presentation code.
2. Adapters declare capabilities and return partial evidence for unsupported channels.
3. Label surrogate rules and concepts explicitly.
4. Every new defended operator needs a typed contract, a test, a manifest row, and a reproducible artifact.
5. Do not add generated site cards, screenshots, caches, large datasets, or reproducible reports to the framework branch.
6. Before push: `git diff --check`, `python -m pytest`, `make doctorate-release-check`.

## Notes

- `ruff` line length is 160, target `py311`; `mypy` is `strict = true` but
  scoped to a fixed file list in `pyproject.toml`.
- The generated DubnaXAI site is quarantined to a historical archive branch —
  do not restore it to `main`.
