# Project instructions

FuzzyXAI (`fuzzyxai-operator`) — evidence-first ML/XAI observation,
diagnosis, repair, and recertification framework, doubling as the source
tree for a doctoral dissertation. Read `context/PROJECT.md`,
`context/RESEARCH.md`, and `context/DECISIONS.md` first — they are the
compressed, tool-agnostic state/history/rationale layer for the
**dissertation-validation** track (H1-H10, Q1, sealed/held-out protocols).
`PROJECT_MEMORY.md` is the raw ledger they're synthesized from; consult it
directly only for exact hashes/statistics.

`context/` is the single canonical project-memory layer for both the
dissertation-validation and live framework/operator-semantics tracks. Keep
their documents scoped and concise; do not create a parallel context layer.

## Absolute constraints

- **No Git/VCS commands** (`git`, `gh`, `glab`, or any subcommand of them —
  including read-only ones like `status`/`diff`/`log`) unless the user
  explicitly asks for one in the current message. This is a standing rule
  across every session on this repository, not a one-off preference.
- Locked/scored dissertation artifacts (`protocol/`, scored `results/`,
  scored `reports/`, `release_evidence/`, `dissertation_artifacts/`,
  anything with a `PROTOCOL_LOCK.json`/`METHOD_LOCK.json`) are read-only.
  Full rule and rationale: `context/DECISIONS.md#D007`.
- Public behavior must be evidence-first: unavailable evidence yields an
  explicit limitation or `insufficient_evidence`, never a fabricated
  metric, explanation, or provenance value. `not_applicable` and `missing`
  are different states and must never be collapsed into one.
- Local-explanation semantics and global/systemic-composition semantics
  must never be mixed (e.g. a model's global `feature_importances_` is not
  a per-object local contribution just because it's multiplied by
  something). See `framework/fuzzyxai/AGENTS.md` for the concrete list of
  rejected shortcuts.
- Keep one canonical API: `FuzzyXAI.wrap(...).explain(...)`/`.explain_one(...)`
  and the `fuzzyxai.visualization` namespace. `visual`/`viz` are
  compatibility shims only.
- Every defended operator must remain mapped in
  `framework/fuzzyxai/operators_manifest.yaml` to a callable, schemas,
  tests, and a visualization policy (`make operator-manifest-check`).
- An LLM/SLM layer may verbalize/rephrase already-certified claims; it may
  never invent explanation evidence, a numeric value, or a claim not
  already present in the typed evidence it was given.
- Build shareable source archives with `python scripts/build_framework_release.py`;
  never zip the dirty worktree.
- Confirmatory controller inputs must be observable before scoring;
  held-out labels may be targets but never feature channels.

## Context bootstrap

At the start of a new task on the framework/operator-semantics track, read
only, in order:
1. `context/INDEX.md`
2. `context/CURRENT_STATE.md`
3. `context/ACTIVE_TASK.md`

Do not recursively read `context/`, old changelogs/audits, test
logs, or historical artifacts unless `INDEX.md` explicitly routes you there
for the problem you're actually working on. If the task is instead about
the dissertation-validation track (a specific `h10_*`/`q1_*`/`study/`
protocol), start from `context/PROJECT.md` instead — the two tracks have
different memory layers by design.

## Engineering loop

For each iteration:
1. State one concrete hypothesis/problem.
2. Inspect only the files relevant to it.
3. Make one logical change.
4. Run targeted tests (not the full suite).
5. Record the result in `context/ACTIVE_TASK.md`.
6. Continue only if the result justifies the next step.

Run the full regression only at a milestone/end of task — see
`context/CURRENT_STATE.md` for the current known-good baseline
to compare against.

## Context discipline

- Do not paste full test logs into the conversation. Report counts +
  failing test names + the relevant error fragment.
- Prefer targeted `rg`/`find`/`sed` inspection over opening large files
  wholesale.
- Do not reread files already summarized in `CURRENT_STATE.md`.
- Do not regenerate large validation bundles (`final_transparency_validation/`
  case outputs) after every small change — only at a milestone.
- Record decisions in `context/DECISIONS.md` instead of
  re-deriving them each session.

## Subagents

Off by default for routine engineering. Use only for genuinely independent,
bounded work (independent codebase mapping, isolated log analysis,
independent scientific review) — maximum two, and only if the task
explicitly calls for it. Return a distilled summary, not raw logs.

## Completion

Stop when the acceptance criterion in `ACTIVE_TASK.md` is met. Do not
opportunistically refactor unrelated code.

For an explicitly authorized multi-step implementation, an intermediate
incomplete state is not completion. Continue while the next scoped step is
unambiguous; end only at acceptance, a genuine blocker, a scientific choice,
or a safety restriction.

## Misc

- For H10-C5b container collection specifics, read
  `.codex/notes/h10-c5b-runtime.md`.
