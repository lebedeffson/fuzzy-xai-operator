# Ablation report

A1 contract accuracy: `0.6667`.

A2 contract accuracy: `1.0000`.

A3 diagnostic cuts: `15/18`.

A4 full recertifications: `5/5`.

## Contributions

- A0, A1 and B2 remained at contract accuracy `0.6667`; object linkage alone
  did not diagnose a relation that was never checked.
- A2 added the preregistered inter-stage contracts and raised contract
  accuracy to `1.0000`, an absolute gain of `0.3333` over A1.
- A3 preserved A2 localization and produced diagnostic cuts for all 15
  intentionally violated scenarios, but executed no repair.
- A4 preserved the same localization, executed all five registered repairs,
  rechecked all 28 contracts each time, verified rollback, and introduced zero
  new critical violations.

The A2-to-A3 contribution is structural traceability, not an artificial change
in the already-correct top diagnosis. The A3-to-A4 contribution is executable
recovery and full-route recertification, not another localization gain.
