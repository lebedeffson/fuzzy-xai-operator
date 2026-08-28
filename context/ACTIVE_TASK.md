# Active task

Milestone:
Chapter 6 — Medical validation (ophthalmology, ECG, brain)

Current loop:
CH6-PAPILA.4 — Spatial, faithfulness and integrity-control closure

Problem:
The frozen two-domain Chapter 6 bundle has no reproducible fundus experiment.
IDRiD remains unavailable through its interactive gate; PAPILA v2 is now the
official open third-domain continuation and must be verified before training.

Hypothesis:
Official Figshare PAPILA metadata, labels and expert-1 contours can support a
frozen, patient-level healthy-vs-glaucoma protocol without leaking paired eyes
or treating suspect eyes as supervised binary examples.

Acceptance criterion:
The pre-registered fold-5 model provides LIME and Grad-CAM native evidence,
then a single public FuzzyXAI result for every selected case; spatial and
faithfulness diagnostics remain diagnostic-only and never become system Gamma.

Last result:
Official Figshare article 14798004 v2 was downloaded and verified outside the
repository. Its metadata reports GPL 3.0+. The verifier found 244 patients,
488 fundus JPEGs, labels healthy=333/glaucoma=87/suspect=68, and 1,963 contour
files (expert 1=977, expert 2=986). The pre-result split manifest
`papila_cv_folds_seed2026.json` has SHA256
`d2caccba592639f7a6f2d5e80b7ebb2473a399f6a987ad3ba356f502b6efa926`, with 210
clean binary patients and 34 suspect-associated patients excluded wholesale
from primary CV. Fixed seed 2026 completed all five outer folds: accuracy
0.7690±0.0343, balanced accuracy 0.5990±0.0718, AUROC 0.6874±0.0637. These are
limited classifier metrics, not clinical claims. Fold-5 extra seeds 2027 and
2028 were also saved; seed 2026 is canonical only because it has minimum
internal validation loss. Public fold-5 outputs exist for `RET028OD` (healthy)
and `RET014OD` (glaucoma), with LIME + Grad-CAM as separate native evidence and
strict SLM H=0. This is not yet the complete selected-case/control closure.
`PAPILA_MODEL_SANITY.md` now freezes the saved model protocol. The
programmatic selection found A=`RET038OS`, B=`RET098OS`, C=`RET119OS`,
D=`RET170OD` (FP), E=`RET265OS` (FN), G=`RET033OD`, H=`RET135OS`, and
F=`RET135OD` by complete fold-5 LIME/Grad-CAM sweep. Suspect predictions come
from the frozen canonical model only; A/D=`RET009OD`, B=`RET092OD`,
C=`RET067OD`, E=`RET093OS` by complete suspect sweep. Full public artifacts
now exist for all 12 distinct selected objects. The sweep distance is a native
positive-support diagnostic, explicitly not canonical system Gamma.
EYE_F diagnostic=1.0 was checked before freeze: the LIME positive map and raw
Grad-CAM map are both finite and non-zero (5,497 and 4,096 non-zero pixels),
so it is a valid maximum L1-distance result rather than a null-map artifact.

Targeted tests:
PAPILA verifier PASS; split freeze PASS; ROI tensor smoke PASS; new PAPILA
scripts compile under the CH6 overlay; 14 ophthalmology tests passed. CUDA
ResNet50 initialization is available in the isolated torch/torchvision overlay.

Next step:
Add spatial/faithfulness diagnostics and controlled faults; only after those
facts exist may the chapter-wide three-domain projections be updated.

Blocked:
None for PAPILA. IDRiD remains a separate MISSING_DATA line and must not be
represented as the executed ophthalmology experiment.
