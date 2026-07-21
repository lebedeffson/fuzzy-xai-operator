# FuzzyXAI Framework Release Status

## Final Practical Confirmatory Prelock

- Branch: `feat/final-confirmatory-and-chapter4`
- Frozen base: `68e6edcfa867b48684b89d98dd74b5fe4794ef55`
- Dataset/OOF implementation: `77e12c0e468fdff727fa1f89eda3b5e3e6aa19a4`
- Technical prelock check: `PASS_LOCAL`
- Public workflow: `29824352350`, Python 3.11/3.12 `PASS`
- Confirmatory datasets sealed: `5/5`
- OOF identities: `69825`
- Sealed-test identities: `17406`, overlap `0`
- P0 features: `PARTIAL`, disagreement and shift pending
- P1 route/explanation features: `PENDING_REAL_EVIDENCE`
- Confirmatory test: `UNOPENED`
- AI formative run 2: `INPUT_READY_SCORES_NOT_RUN`
- Final Chapter 4: `BLOCKED`
- Stable release: `BLOCKED`

Five independent UCI datasets are now prepared and sealed locally with workspace-only encrypted label vaults. The
tracked tree contains only licenses, manifests and hashed split identities. The 69,825-row OOF baseline did not load
sealed test and has zero identity overlap, but the confirmatory lock remains closed because P0/P1 evidence is
incomplete and the clean-session AI review has not been run.

The shareable technical artifact is
`release_artifacts/fuzzyxai-final-confirmatory-prelock-77e12c0e468f.zip` with SHA256
`ed4a87b01355d35c8c11b9172a3724c4aee477d850e09de0de778138ffae8289`. Its `BOUNDARY.json` explicitly disables
confirmatory and stable-release claims. `make final-seal-datasets`, `make final-controller-freeze`, final Chapter 4 and
`make final-release-archive` remain fail-closed until real external inputs pass.

The clean committed-tree source release is `release_artifacts/fuzzyxai-source-release-77e12c0e468f.zip` with 1,528
files and SHA256 `9e5a07f949b100418817c7ead4f464c46201897155e0692fc5147fff6a11d207`; the quarantined site is absent.

## Strong Confirmatory Formative Gate

- Branch: `feat/strong-confirmatory-closure`
- Implementation commit: `238194c113fc2f1677730318dfc0ba4d4d4426d5`
- Formative evidence: measured locally
- Public formative workflow: `29788608838`, PASS on evidence commit `fa6c40c`
- Confirmatory test: not opened
- External human gates: open
- Stable release: `BLOCKED`

The strong confirmatory code and formative evidence preserve all original negative statuses. The measured development
cycle does not support H3-v2 or overall H7. Controlled H5-A, semisynthetic H6-A, controlled H8, and operator-layer H9
meet their formative thresholds but cannot be described as independently confirmed. H6-B, independent datasets,
human comprehension, expert-action agreement, and domain-language approval remain unmeasured.

`make strong-confirmatory-lock` and `make chapter4-final` fail closed while their external prerequisites are absent.
The 17-section Chapter 4 output is a formative shell with explicit pending markers, not a final dissertation chapter.
No merge to `main` or stable tag is allowed from this state.

## AI Pre-review Final Blinding Gate

- Branch: `feat/ai-pre-review-final-closure`
- Implementation commit: `c1271db`
- Reviewer-packet hardening commit: `79f80a8`
- Technical status: `PASS_LOCAL`
- Public CI: `29767282573`, `PASS`
- Formative AI review: `planned_not_run`
- Confirmatory protocol: `BLOCKED_BY_FORMATIVE_STAGE`
- Human confirmation: `OPEN_EXTERNAL`
- Stable release: `BLOCKED`

The scoreable replacement bundle contains only the 240-case formative input. The prior `60ed5697` bundle is retained
for provenance but is invalid for evaluation because its reviewer inputs leaked outcome and expected-action fields.
The encrypted key and the 120-case confirmatory pool are excluded from public source and reviewer archives.

Local acceptance: `make ai-final-check` PASS; 720 reviewer variants, four modalities, 12 batches, 60 image assets,
claim-evidence coverage `1.0`, focused tests `10 passed`. These results establish packet integrity and blinding only;
they do not establish AI repeatability, human agreement, domain approval, comprehension benefit or expert-action
utility.

## AI Pre-review and Human Confirmation Candidate

Status: technical tooling PASS at `461d267a4a700a4aeff8251ef25f7da539d2bab1` on
`feat/ai-pre-review-human-confirmation`; external AI review and independent human confirmation remain open. Stable
release is `BLOCKED`.

- frozen evidence commits remain ancestors and unchanged;
- tracked design: 360 unique cases, 1,080 blind variants, four modalities, 18 bounded review batches;
- method identity: HMAC-randomized and encrypted with a secret excluded from repository and archives;
- local technical command: `make ai-pre-review-check` PASS;
- focused tests: `6 passed`;
- public focused workflow `29752785014`: PASS;
- negative gates: confirmatory lock and human-pack generation correctly reject missing external evidence;
- current claim states: `planned_not_run`, `pending_three_ai_runs`, and `external_gate`;
- offline technical DoD: `44/80 PASS`; public CI is separately PASS, while 35 items still require external AI sessions or humans;
- independent experts observed: 0;
- AI confirmatory runs observed: 0;
- human validation, domain approval and demonstrated comprehension: not claimed.

The next allowed transition is a real formative AI pre-review. A stable tag, expert-validation wording, and human
confirmation remain forbidden until the locked confirmatory and independent-human thresholds are actually measured.

## Two-Stage Selective Observer Formative Cycle

Status: formative method candidate on `feat/selective-observer-formative`; confirmatory claims and stable release are
`BLOCKED`.

- frozen predecessor: `e34e52fb8ae62ee1be043d6d5b26a0c9214a0572`;
- predecessor public CI: `29740096302`, 9/9 jobs PASS;
- predecessor negative results remain immutable: H3 full/hard not supported, H5 predictive not supported, H6
  confirmatory general effect not supported;
- new method: grouped out-of-fold four-action selective risk controller;
- H5 is separated into action-contract validity and an optional held-out predictive increment;
- H6 is separated into planted-rule method validation and held-out low-redundancy matched-control ablation;
- train/validation, formative participants and pilot experts are isolated from confirmatory data and participants;
- domain-language, comprehension and expert-action gates remain open;
- no confirmatory test has been opened and no participant or expert response has been generated.

The formative acceptance command is `make selective-observer-formative-check`. A positive confirmatory claim requires
a separately hashed protocol lock, independent data or participants, one test opening and unchanged frozen code.

## Q1 Final Closure Candidate

Status: technical closure at `e34e52fb8ae62ee1be043d6d5b26a0c9214a0572`; stable release is `BLOCKED`.

- frozen base: `41c32af25242164144fd907e4850fa9d4f426bd1`;
- local focused tests: `19 passed`;
- local lint and compile checks: PASS;
- local final smoke reproduction: PASS;
- public workflow `29740096302`: 9/9 jobs PASS, including native multiclass, Docker and aggregate verification;
- technical DoD: 161 PASS, 0 BLOCKED, 24 OPEN_EXTERNAL;
- domain-language review: open;
- comprehension study: open;
- expert-action review: open;
- stable tag and merge to `main`: forbidden until the applicable technical and external gates close.

The final candidate preserves all prior negative results. It distinguishes structural critical-rupture diagnosis from
predictive association, and it removes any general H6 rule-effect claim when the confirmatory result is null. The
external package contains frozen stimuli, blank response schemas and deterministic scorers only; it contains no
generated participant or reviewer responses.

## Q1 Empirical Remediation

Status: technical candidate on `feat/q1-empirical-remediation`; stable release remains `BLOCKED`.

- frozen predecessor commit: `cafe403c7d60e36b08f56a5325ba380718a5be35`;
- technical CI commit: `a9ca41ecc9bdfe74024214e5e4b1ed2c3eccefab`;
- public workflow run `29709914672`: fast, tabular, image, text, time-series, Docker and aggregate jobs PASS;
- real datasets: 4/4 above 10,000 objects, with source, license, preprocessing and hashes recorded;
- measured records: 16 model runs and 25 explainer runs;
- clean Docker command `docker run --rm fuzzyxai-q1 make reproduce-q1`: PASS;
- supported controlled claims: 5; not supported: 1; inconclusive: 1; external gate claim: 1;
- critical-rupture incremental AUPRC: `-0.0005609278151094133`; predictive/safety claim forbidden;
- matched rule ablation: context-dependent candidate only; general effect remains unconfirmed;
- comprehension pilot: `planned_not_run`;
- expert-action review: `planned_not_run`;
- domain-language review: `pending_external_review`;
- merge to `main`: blocked;
- stable release tag: forbidden.

The Q1 evidence supports technical reproducibility and measured execution across four modalities. It does not support
clinical effectiveness, production safety, human-comprehension improvement, domain validity or universal superiority.
The earlier negative empirical results remain frozen and are not replaced by this independent cycle.

## Full Empirical Validation

Status: technical candidate on `feat/full-empirical-validation`; stable release remains `BLOCKED`.

- four controlled modality datasets, 10,000 objects each: PASS;
- 10-fold by 5-seed rule ablation, 50 paired comparisons: PASS;
- SHAP, LIME, Anchors, RuleFit, confidence and FuzzyXAI baselines: measured;
- P1-P7 policy comparison, validation-only calibration and sensitivity analysis: PASS;
- adaptive hierarchy: `F0=1098`, `Fint=1140`, `NAS=2240`, `FML=5522`, no diagnostic refusals;
- adaptive FML fraction: `0.5522`; non-inferiority criterion: PASS;
- critical-rupture incremental AUPRC: `-0.12556024709557384`;
- safety claim: forbidden; interpretation: structural diagnostic only;
- 1k/5k/10k/50k scalability contour: PASS;
- full E1-E8 job, Docker reproduction, aggregation, and archive verification: PASS in public run `29707225733`;
- technical Definition of Done: `90/90 PASS`;
- independent expert/comprehension work: `planned_not_run`;
- regulated-domain language review: `pending_external_review`;
- merge to `main`: blocked;
- stable release tag: forbidden.

The datasets in this milestone are controlled protocol datasets. Their results support reproducibility and internal
contract claims only; they do not establish clinical validity, production safety, or external-domain superiority.

Status: `v1.3.0rc1` universal-model integration candidate on `feat/universal-model-integration`. The merged evidence
matrix contains 40/40 passing configurations: 34 deterministic core configurations and six optional runtime
integrations measured independently on Python 3.11 and Python 3.12. External comprehension and domain-language
review are not complete, so merge and release tags remain blocked.

## Universal Model Candidate

- 24 classification and 10 regression configurations;
- prediction parity, adapter conformance, and graph validation: `1.0` on the recorded core matrix;
- model-specific evidence for sklearn families and generic callable/probability/rule contracts;
- optional XGBoost, LightGBM, CatBoost, Torch, TensorFlow/Keras, and ONNX reports: 12/12 PASS;
- runtime reports including sklearn: 14/14 PASS with valid report checksums;
- public API verification: `explain_one`, `explain_batch`, `explain_global`, `why_not`, and `compare_models` PASS;
- explanation-quality matrix: 40/40 PASS; top-reason stability measured for 36 configurations;
- universal evidence CI: PASS on run `29700288413`;
- readiness Python 3.11/3.12, Octave, wheel, and full regression: PASS on run `29700288436`;
- no claim of identical evidence channels or universal arbitrary-model support.

## Version Sequence

The project follows version path B: the existing functional branches remain combined in the technical
`v1.3.0rc1` candidate. No separate stable `v1.2.0` tag will be created retroactively. After both external gates,
reviewed merge to `main`, and green `main` CI, the next allowed tag is `v1.3.0rc2`; stable `v1.3.0` requires a
subsequent release review. `v1.4.0` remains reserved for modality-specific anomaly, forecasting, text, and image
support.

## Release Decision

- computed Chapter 4 evidence: PASS;
- comprehension pilot: `planned_not_run`;
- regulated-domain review: `pending_external_review`;
- full current-branch regression and public CI: PASS;
- public `main` CI after merge: not run;
- release gate: `BLOCKED`;
- tag allowed: no.

---

## Previous v1.2.0rc3 Boundary

Status: `v1.2.0rc3` is an untagged Empirical Validation Gate candidate. The measured computational pipeline passes locally and on the public feature-branch CI. The independent comprehension pilot is `planned_not_run`, and the regulated-domain dictionary is awaiting external semantic review; therefore the release gate is `BLOCKED` and no tag is allowed.

## Measured Computational Gate

- dataset: Breast Cancer Wisconsin (Diagnostic), 569 objects, 30 features, CC BY 4.0;
- split: train 341, validation 114, test 114, seed 42;
- checkpoint model: 30 unique measured SGD states;
- selected case: `case_real_001`, chosen automatically from validation after training;
- forgetting event: epoch 9;
- rare subgroup: smallest of three train-only KMeans clusters, fixed before training/case selection;
- measured native tree rule: `tree_leaf_11`;
- target prediction after leaf suppression: `1 -> 0`;
- test accuracy: `0.947368 -> 0.903509`;
- validation subgroup recall: `0.923077 -> 0.615385`;
- cross-model contracts: logistic regression, decision tree, random forest, fitted Sugeno rules, callable black box;
- black box native rules: 0; tree and Sugeno native rules: present;
- similar-case evidence: one support and one counterexample;
- intervention mode: `sensitivity_analysis`, not an actionable recommendation.

## Controlled Boundary

- `object_85_controlled_story_fixture`: controlled contract and visualization fixture;
- `case_real_001`: measured checkpoint experiment;
- controlled and measured results have different run IDs, directories, and `result_origin` values;
- the research-only image fixture remains controlled and is not medical validation.

## Local Validation

- full Python 3.14 regression: `315 passed`, 409 third-party warnings;
- empirical-focused tests: `16 passed`;
- release-focused tests: `17 passed`;
- Ruff: PASS;
- strict MyPy: PASS;
- operator manifest: `30/30`, PASS;
- deterministic empirical rebuild: PASS;
- Chapter 4 empirical builder/verifier: PASS;
- Chapter 4 measured figures: `3/3` visually inspected;
- Chapter 4 empirical ZIP SHA256: `f80aa4ba799e91b492a10553e7f12c6ebe0e7572a226d6dd562cb8be3973b9e4`;
- public feature CI: PASS for Python 3.11, Python 3.12, and Octave ([run 29695395925](https://github.com/lebedeffson/fuzzy-xai-operator/actions/runs/29695395925));
- public main CI: not run for this candidate.

## Open External Gates

1. A/B comprehension pilot with at least six independent participants.
2. External subject-matter review of the regulated-domain dictionary.

No demonstrated-comprehensibility, clinical-validity, or release-readiness claim is made while either gate is open.

## Archive Policy

- source archive: clean allowlist from committed Git index;
- doctoral archive: separate full historical Git-index export;
- archives must be built only after the release documentation commit;
- no `v1.2.0rc3` tag before all external gates and green feature/main CI.
