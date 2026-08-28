# Active task

Milestone:
Chapter 6 — Medical validation (ophthalmology, ECG, brain)

Current loop:
CH6-CLOSURE.2 — Finalized two-domain medical validation bundle

Problem:
The frozen CH6 baseline has two completed domains but stale reporting terms;
the Allen result is a small v1 pilot and IDRiD requires official interactive
access.

Hypothesis:
Chapter-facing projections can state system semantics correctly without
changing P19, and a separate pre-registered section-block v2 cohort can test
the same public route without using v1 metrics for protocol selection.

Acceptance criterion:
Reporting separates system Gamma/Delta from native-XAI diagnostics; brain v2
has an immutable protocol/manifest and three independent seed runs; strict SLM
uses certified public claims only; IDRiD remains explicit MISSING_DATA until
official access is supplied.

Last result:
Reporting QA now emits `system_Gamma` plus scope, keeps native-XAI disagreement
diagnostic-only, displays Delta as `—` when not applied, corrects PTB-XL to
Open Access / CC BY 4.0, and records the boundary between route consistency
and ground-truth correctness. v2 config SHA256:
09f564706b13b0457ba6cd82f4544a6d8e5a0d770f04fb5500831c9ec70dbcb0.
Its separate prepared cohort has 1,819 patches (1,231/280/308), with 10 test
section blocks. Seeds 2026/2027/2028 yielded held-out accuracy 0.9773/0.9838/
0.9935; the canonical seed 2027 was selected only by minimum validation loss.
Public v2 cases and controlled integrity faults use the corrected
pretrained=True replay contract. The same pinned local strict SLM
Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775 ran over
six ECG and five brain-v2 public human layers; every recorded strict output
has H=0 and preservation metrics 1.0. Full regression: 1262 passed, 11
skipped, 643 warnings in 276.67 s. Wheel-only local/RF/non-RF smoke passed.

Targeted tests:
CH6 domain tests: 27 passed, 1 skipped; v1/v2 checkpoint replay: 2 passed;
scoped compileall and ruff passed. CUDA smoke passed with torch/torchvision
2.11/0.26 cu128 in isolated overlay.

Next step:
Await authenticated official IDRiD access for the third empirical domain. Do
not claim three completed domains before that input is available. The frozen
baseline bundle remains unchanged; the new final bundle is versioned separately.

Blocked:
Ophthalmology final execution requires authenticated official IDRiD access;
the dataset is not available locally and no unofficial mirror will be used.
