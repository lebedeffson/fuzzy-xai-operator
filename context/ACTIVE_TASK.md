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
benchmark, focused tests, and public contract notes. It is NOT added to the
immutable doctoral H10-C5b canonical operator manifest.
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

TISU integration checkpoint: completed and merged to main via PR #19,
commit 56ffab33b79079980c799a60e0875393af27b01a (2026-11-16).
Focused Python 3.11/3.12 CI passed on the final PR branch: Ruff,
98 focused/legacy/manifest tests, wheel import and both deterministic
synthetic protocols (2860 RF cases and 1716 StandardScaler+LogReg cases).
These are correlated constructed perturbations, not independent patients,
real-world fault incidence or clinical performance claims.

Unrelated baseline CI debt observed before PR #19:
- framework readiness full pytest collection requires optional FastAPI/httpx
  packages not declared in the dev extra;
- broader H10/Q1 lints and a Q1 RuleFit dependency failure were already present
  or are external to the typed F0 interface. Locked dissertation study files
  must remain unchanged.

This follow-up restores the protected `framework/fuzzyxai/operators_manifest.yaml`
to the locked H10-C5b bytes (frozen SHA-256
`479fa55460426a1726f57a5ed5f745b0ad2e872eeefb21470dd2e3e865133949`)
and restores the doctoral-manifest test expectation of 41 entries.
The typed-F0 `compose_checked` stays public and opt-in, validated through
its independent test/benchmark/CI workflow, but is not a dissertation-sealed
operator manifest entry. This is deliberate: a new experimental method must
not silently mutate a frozen scientific protocol.

Dev dependencies for the full regression include the FastAPI/httpx TestClient
and NiceGUI UI projection imported by cross_pipeline_v1. Keep the dedicated
TISU validation on main pushes.

Next: validate method lock and targeted TISU suite together; retain all
dissertation locked artifacts unchanged.
Do not mix unrelated baseline workflow failures into TISU scientific evidence.
