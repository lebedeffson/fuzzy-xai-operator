# Independent ECG replication — 2026-11-26

Protocol: [retrain_independent_protocol.json](../configs/retrain_independent_protocol.json).
The old outputs and FuzzyXAI core are immutable. This is a new training run, not recovery of the historical best.pt. The public PTB-XL 1.0.3 cohort (DOI 10.13026/kfzx-aw45, CC BY 4.0) is verified against the official SHA256 manifest. The original configuration/optimization and patient folds are preserved; CPU execution differs from the historical GPU execution. Train three seeds in parallel; select by validation only.

The explanation job uses the newly saved best.pt, recomputes native and independently implemented IG and calibrated temporal occlusion on official waveforms. The review job computes the public lightweight system route on all test records; its missing local attribution channel remains explicit. Rank policies at the same review budget; report flagged errors and residual FN/FP, not a claim that review corrects them. This previously examined cohort is exploratory, not a new confirmatory test. No clinical outcomes or human adjudications are present.

Status before execution: protocol registered; new training and evaluation results **not yet available**. Workflow artifacts are the evidence; failed/unfinished jobs must not be represented as complete.
