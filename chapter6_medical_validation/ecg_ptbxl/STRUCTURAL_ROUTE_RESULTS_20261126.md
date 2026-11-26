# ECG structural route experiment — completed 2026-11-26

Evidence: [successful GitHub Actions run 45804466688](https://github.com/lebedeffson/fuzzy-xai-operator/actions/runs/45804466688), artifact 15108907587, executed commit ae6262016bd2a24a8d446ca2bc30ff074ff8c6a0. Original classifier checkpoint SHA256: 7cbb2744448ff413434a2a2cfd4bf366f5cf7af85c151d874462f2df0f39cc9e. Artifact ZIP independently verified: c27bc255d3ed902b9e02c86fe6f50a6f55bdcec58e43c965a9f1da34c269824e.

48 distinct PTB-XL test patients, 16 controlled conditions per ECG (14 fault conditions and two benign conditions); 768 correlated routes, 672 faulted and 96 benign. Model prediction unchanged. No medical reference labels used for structural defect detection.

| Method | TP | FP | FN | TN |
|---|---:|---:|---:|---:|
| ECG adapter + native FuzzyXAI diagnostics | 672 | 0 | 0 | 96 |
| Direct checks with identical evidence | 672 | 0 | 0 | 96 |
| Confidence below 0.90 | 224 | 32 | 448 | 64 |

High-confidence subset: 448/448 controlled violations detected by both structural methods, 0/448 by confidence threshold. All 672 faulted routes localized to their affected explanation artifacts, blocked, and technically recertified after targeted restoration. All 96 benign conditions preserved the original action. 720 artifact replacements versus 1344 replacements in a full two-artifact rebuild; equal-information targeted direct checks also require 720. In 48 additional controls with an unavailable trusted artifact, zero invalid recertifications and 48 blocked routes. All 672 minimum cuts proven optimal under the declared experimental cost model. Independent archive audit passed (all array hashes, measured map distances, source/checkpoint identities, counts, metrics).

Scope: demonstrated ability to detect and repair registered controlled explanation-route integrity failures given independently recomputed reference artifacts and registered contracts. The new ECG adapter feeds actual detected issues to the existing system operator, not mutation labels. The FuzzyXAI core was not modified. No claim of superiority over direct checks with identical evidence, medical validity of attributions, actual diagnostic improvement, clinical efficacy, or fault prevalence. The earlier negative classifier-error ranking result remains valid and must not be erased.
