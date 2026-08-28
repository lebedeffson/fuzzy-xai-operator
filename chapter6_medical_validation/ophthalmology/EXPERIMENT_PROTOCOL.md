# Frozen protocol: CH6-EYE-1

- Status before data access: `REGISTERED_NOT_RUN`.
- Scientific scope: engineering/scientific observation of an existing
  five-class CNN/XAI route; not clinical validation and not classifier-quality
  improvement.
- Primary dataset: labeled APTOS2019 training portion, expected N=3662.
- Split: stratified image-level 2564/549/549, seed 2026. Patient identifiers
  are not declared by the official CSV; no patient grouping is inferred from
  filenames. This is a limitation.
- External grading: IDRiD official 413/103 split; the 103 test images are not
  used for fitting, selection or temperature calibration.
- Lesion annotations: IDRiD official 54/27 segmentation split, post-hoc only.
- Architectures: VGG16 primary; EfficientNetB0 secondary architecture check.
- Seeds: 2026, 2027, 2028.
- Checkpoint selection: minimum validation loss; tie → maximum validation QWK;
  test metrics are never consulted.
- Loss: weighted cross entropy using train labels only.
- Calibration: optional temperature fitted on validation only and frozen.
- Native XAI: Grad-CAM required, fixed predicted target. Existing frozen
  TorchAdapter IG may provide the optional second channel.
- Case selection is algorithmic (`src/case_selection.py`), never manual.
- FuzzyXAI action thresholds and weights are technical demonstration policy,
  frozen in `configs/explain_plan_eye.yaml`; no clinical certification claim.
- Reduction of full spatial attribution is `not_applied` because frozen P19
  runtime does not expose an experiment-side spatial Pi/iota contract without
  core modification. No manual Delta is substituted.
- Controlled faults affect evidence/provenance only; no medical abnormality is
  fabricated.
- Test results are opened only after configs, split and selection rule hashes
  are fixed. Existing manifests are never overwritten silently.
