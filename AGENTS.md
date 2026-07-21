# Project instructions

- The research framework is the primary product. The generated DubnaXAI site is quarantined under the historical archive branch and must not be restored to `main`.
- Public behavior must be evidence-first: unavailable evidence yields an explicit limitation or `insufficient_evidence`, never a fabricated metric.
- Keep one canonical API: `FuzzyXAI.wrap(...).explain(...)` and the `fuzzyxai.visualization` namespace. `visual` and `viz` are compatibility shims only.
- Every defended operator must remain mapped in `framework/fuzzyxai/operators_manifest.yaml` to a callable, schemas, tests, and visualization policy.
- Read `PROJECT_MEMORY.md` for the current release boundary and validated claims.
- Build shareable source archives with `python scripts/build_framework_release.py`; do not zip the dirty worktree.
- Confirmatory controller inputs must be observable before scoring; held-out labels may be targets but never feature channels.
