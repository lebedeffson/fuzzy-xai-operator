---
name: fuzzyxai-engineering-loop
description: Implements or debugs one FuzzyXAI engineering issue with minimal context. Use for scoped framework changes, targeted tests, and ACTIVE_TASK checkpoints.
disable-model-invocation: true
---

# FuzzyXAI engineering loop

1. Read `context/ACTIVE_TASK.md`; read `CURRENT_STATE.md` only if needed.
2. State one hypothesis and acceptance criterion.
3. Inspect the smallest relevant code/test set.
4. Make one coherent change and run targeted tests.
5. Record command, compact result, and next step in `ACTIVE_TASK.md`.
6. Do not use Git/VCS, alter locked artifacts, or begin unrelated refactors.
