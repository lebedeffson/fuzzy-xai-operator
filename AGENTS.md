# Project instructions

- Read `context/PROJECT.md`, `context/RESEARCH.md`, and `context/DECISIONS.md` first — they are the compressed, tool-agnostic state/history/rationale layer shared across Claude and GPT/Codex sessions. `PROJECT_MEMORY.md` is the raw ledger they're synthesized from; consult it directly only for exact hashes/statistics.
- The research framework is the primary product. The generated DubnaXAI site is quarantined under the historical archive branch and must not be restored to `main`.
- Public behavior must be evidence-first: unavailable evidence yields an explicit limitation or `insufficient_evidence`, never a fabricated metric.
- Keep one canonical API: `FuzzyXAI.wrap(...).explain(...)` and the `fuzzyxai.visualization` namespace. `visual` and `viz` are compatibility shims only.
- Every defended operator must remain mapped in `framework/fuzzyxai/operators_manifest.yaml` to a callable, schemas, tests, and visualization policy.
- Read `PROJECT_MEMORY.md` for the current release boundary and validated claims.
- Build shareable source archives with `python scripts/build_framework_release.py`; do not zip the dirty worktree.
- Confirmatory controller inputs must be observable before scoring; held-out labels may be targets but never feature channels.
- For H10-C5b container collection, read `.codex/notes/h10-c5b-runtime.md`.
