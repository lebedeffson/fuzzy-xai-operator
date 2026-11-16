# Active task

Milestone:
P19 — System Operator Closure

Current loop:
P19.2.1 — Final semantic micro-fix

Problem:
RF uncertainty depends on numeric class encoding; RF source evidence conflates
hard-vote proportion with model probability; chi_R and chi_R_critical are
collapsed; the public view may mark zero-weight missing risk inputs required.

Acceptance criterion:
Vote uncertainty is label-invariant; probability and vote proportion remain
separate typed signals; non-critical rupture contributes to rho without forcing
block; missing-required projection respects positive weights; targeted and one
full regression, wheel smoke and cache-free final ZIP pass.

Last result:
P19.2 baseline is closed and provides typed generic source evidence, strict
five-component rho, four action thresholds and wheel-only public API smoke.

Targeted tests:
57 passed, 87 warnings: P19 system/global/package, universal adapters and
public output layer. Added 0/1, -1/+1 and string-label RF invariance checks;
probability-vs-vote separation; non-critical and critical rupture routes; and
public zero-weight missing-component audit projection.
Full regression: 1261 passed, 11 skipped, 643 warnings in 288.12 s.
Fresh wheel-only smoke: PASS, 41 operators, local + RF + non-RF routes.

Next step:
P19.2.1 acceptance met; stop before P20.

Final bundle:
`final_transparency_validation/P19_FINAL_SCIENTIFIC_REVIEW.zip`

Blocked:
none

## Research branch: TISU typed F0 composition (2026-11-16)

Scope: opt-in `compose_checked` in the core SDK, controlled intervention
benchmark, focused tests, manifest registration, and public contract notes.
Do not modify the locked dissertation experiments or replace generic
`compose` defaults.

Local pre-publication evidence:
- 95 focused unit/parametrized checks passed, including four held-out samples
  from RandomForestClassifier and LogisticRegression.
- 143 held-out Random Forest predictions × 20 **predefined synthetic fault
  types** = 2,860 controlled checks.
- `compose_checked`: 2,860 / 2,860 expected statuses in these controlled
  tests only. The typed-only (without membership validation) ablation:
  2,438 / 2,860. No claim about unknown faults or field performance.
- The independent prediction record is supplied externally and is NOT signed;
  end-to-end authentic provenance is not certified.
- F0 scalar binary event contracts only; `compose()` remains unchanged.

Review gate: review PR CI / public SDK smoke before merging to main.
