# FuzzyXAI — Project State

Compressed, tool-agnostic entry point for Claude Code, GPT/Codex, or anyone
picking up this repository cold. This file is the *current state* layer —
history and rationale live in [`RESEARCH.md`](RESEARCH.md) and
[`DECISIONS.md`](DECISIONS.md); don't duplicate them here, link to them.

## Purpose

FuzzyXAI is two things sharing one repository:

1. **An installable framework** (`fuzzyxai-operator` on PyPI-style packaging)
   that audits, diagnoses, repairs, and recertifies ML/XAI pipelines. Public
   entry point: `FuzzyXAI.wrap(model, ...).explain(...)`.
2. **The source tree for a doctoral dissertation** (Ministry of Science and
   Higher Education of the Russian Federation, theme No. 124112200072-2) —
   fuzzy-logic explanation theory (chapters 2-3), a risk-aware observer
   (chapter 5), and a large empirical/causal validation program (H1-H10, Q1).

These two things have different rules. The framework is live, editable
product code. The dissertation evidence (`protocol/`, scored `results/`,
scored `reports/`, `release_evidence/`, `dissertation_artifacts/`) is a
**frozen scientific record** — see [`DECISIONS.md#D007`](DECISIONS.md) for
why it must not be edited retroactively.

## Current Architecture

```
Model / dataset / training run
  -> ModelAdapter (facts and capabilities)
  -> evidence collectors
  -> core operators (chapter 2-3 math)
  -> ExplanationEvidence -> ExplanationGraph -> ExplanationViewModel
  -> Matplotlib / HTML / MATLAB

Route (data->split->preprocessor->model->prediction->explanation)
  -> RouteGraph -> contract validator -> minimal diagnostic cut
  -> registered repair (explicit context, preconditions, rollback)
  -> recertification
```

Full layer-by-layer breakdown, commands, and conventions are in
[`CLAUDE.md`](../CLAUDE.md) (and mirrored for non-Claude tools in
[`AGENTS.md`](../AGENTS.md)) — read that for "how do I build/test/run this."
This file only tracks *what's true right now*.

## Repository Map (state, not structure)

| Area | Status |
| --- | --- |
| `framework/fuzzyxai/fuzzyxai/` | Live product code. Edit freely, following [`CONTRIBUTING.md`](../CONTRIBUTING.md). |
| `framework/fuzzyxai/operators_manifest.yaml` | Traceability ledger — 41 registered operators as of `RELEASE_STATUS.md`. Must be updated with every new defended operator. |
| `protocol/*` | Locked experiment designs, one per study. Immutable once a study opens its sealed/held-out set. |
| `results/*`, scored `reports/*` | Machine-readable outcomes, one per study. Do not overwrite; a corrected result gets a new study/branch, not a rewrite. |
| `PROJECT_MEMORY.md` | Append-only chronological ledger of every milestone. Source of truth for [`RESEARCH.md`](RESEARCH.md); read directly only when you need a specific commit hash or exact statistic. |
| `RELEASE_STATUS.md` | Current release boundary in force. |
| generated DubnaXAI site | Quarantined to a historical archive branch. Never restore to `main` ([`DECISIONS.md#D008`](DECISIONS.md)). |

## Current Version

- Package `fuzzyxai-operator`, version `1.4.0a2`, branch `main`.
- Last substantive milestone in the ledger: `FUZZYXAI_EXTERNAL_ML_PIPELINE_VALIDATION_V1_SUPPORTED` (2026-08-01) — see [`RESEARCH.md`](RESEARCH.md#engineering-validation-program-current-release-basis).
- Last commits (through 2026-08-11) are CI/dependency cleanup and a README rewrite presenting FuzzyXAI as a standalone framework — no new research milestone since.
- Operator manifest: 41 registered operators mapped to callable/schema/tests/visualization.

## Known Problems / Needs Re-verification

Flagging these rather than asserting them, because sources disagree or are stale:

- [`docs/research_limitations.md`](../docs/research_limitations.md) says "Torch, Keras, and ONNX adapters are not implemented in this release," but `CHANGELOG.md` (1.3.0rc1) and `pyproject.toml` extras both say those adapters *were* added. Check current adapter state in `framework/fuzzyxai/fuzzyxai/adapters/` before relying on either doc.
- The three external gates that blocked a full stable release as of the last practical-closure entries — independent comprehension pilot, regulated-domain dictionary review, expert-action review — were still `planned_not_run` / `pending_external_review` at that time. `RELEASE_STATUS.md` no longer mentions them at all; it describes the alpha (`1.4.0aX`) release track with its own narrower engineering-validation boundary instead. **It's not established here whether the framework alpha track has been deliberately decoupled from the dissertation's human-study gates, or whether the gates are just undocumented in the newer file.** Confirm with the user before stating either as fact.

## Current Priority

Not stated anywhere in the repo as an active task list — see
[`ROADMAP.md`](ROADMAP.md) for the open threads the ledger itself flags as
unfinished. Ask the user what's active *this session*; don't infer it from
the dissertation backlog.
