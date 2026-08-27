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
