# Project Memory

## FXAI-H10-C3-R4-SECURE-PREOPEN

- Date: 2026-07-24
- Branch: `fix/h10-c3-sealed-isolation-v23.3`
- Frozen readiness base: `feat/h10-c3-confirmatory-readiness-v23.2@f544e8bcb4b0c2fb31936af07f7087a60044d08e`
- Locked scientific implementation: `e729834c077ecb5c0011d9fb85d5f00b10129f18`
- Sealed opening count: `0`
- H10-C3a/H10-C3b: `NOT_EVALUATED_CONFIRMATORY`

The deterministic plaintext sealed set from v23.2 was invalidated before
scoring because its cases could be reconstructed from public source. Its
opening count remained zero. v23.3 changes only the operational isolation
boundary: the private template bank is derived from an external 256-bit
secret, stored only as an AES-256-GCM payload, and bound to the scientific and
operational locks by public commitments. Opening is atomically recorded before
decryption and cannot be reused after success or failure.

Open development and protocol-validation results, baselines, Gold, metrics,
margins, statistics, diagnostic algorithms and repair execution are unchanged.
The preopen handoff may contain only the encrypted payload, commitments,
design, status and protocol locks. Do not score without a separate protocol
owner authorization; ordinary reproduction must remain preopen-only.

## FXAI-H10-C3-R4-CONFIRMATORY-READINESS

- Date: 2026-07-23
- Branch: `feat/h10-c3-confirmatory-readiness-v23.2`
- Frozen predecessor: `fix/h10-c3-cost-stability-v23.1@15f3690347dae304c1c501ceb5af546de2ff7c4a`
- Locked implementation: `e729834c077ecb5c0011d9fb85d5f00b10129f18`
- Package version: `1.4.0a3`
- Sealed opening count: `0`
- Scientific status: `SUPERSEDED_BY_SECURE_PREOPEN_V23.3`

R4 uses three structurally disjoint 660-template banks and six genuinely
different route families. Canonical overlap is zero for graphs, mutation
structure, coverage/cost combinations and repair dependencies. H10-C3b now
executes repairs against a copied `RouteGraph` through the production
`RepairExecutor` and requires strict `RouteRecertifier` success.

Open development and independent protocol-validation effects pass their
registered margins for H10-C3a and H10-C3b, with positive direction in all six
pipeline families. These remain preconfirmatory results. Template-level power
selected 40 templates per family, stratified across S2-S5; the lower power
bound exceeds 0.80 for both claims. The original deterministic 240-template
set was never opened, but was later found reconstructible from public inputs
and therefore cannot be used for confirmatory scoring.

Two pre-opening attempts are retained as invalid audit evidence: one selected
an unstratified S2-only sealed subset, and one stored a plaintext private
mutation log. Neither attempt opened or scored sealed outcomes. H10-C3a and
H10-C3b remain `NOT_EVALUATED_CONFIRMATORY`; no prior negative claim was
changed.

## FXAI-DIAGNOSTIC-FRAMEWORK-V21

- Date: 2026-07-23
- Branch: `feat/diagnostic-framework-v21`
- Frozen predecessor: `feat/h10-final-gold-v20@13f82805c69fb974a236df6d8990eea115251c23`
- Implementation commit: `91e0dde54a2edf64ac1dd1efcc67eab4af6a0fe5`
- Package version: `1.4.0a1`
- Software alpha status: PASS
- Scientific status: `BLOCKED_PRECONFIRMATORY`
- Sealed opening count: `0`

The v21 alpha turns the exploratory H10 components into a public structural
diagnostic framework rather than a decision policy. `FuzzyXAI.diagnose()` now
builds a complete registered route graph, validates node and edge contracts,
separates symptom, contract violation, proximate cause and source component,
finds exact or explicitly approximate weighted diagnostic cuts, proposes
provider-bound repair steps, and reruns all contracts after explicitly
authorized external changes.

Production diagnostics do not import `gold_oracle`, read mutation logs, use
source/repair truth, or copy `expected` values into observations. Repair
execution is fail-closed without an explicit `RepairExecutionContext`; batch
execution is disabled. The diagnostic package has 95% measured line coverage,
40 focused tests, and a six-pipeline operator-only benchmark over 1,200 routes
with p95 below 1 ms in the recorded local environment.

No scientific H10 result was upgraded. The prior H10-C difference remains
exploratory, and the draft `FXAI-H10-C2-DIAGNOSTIC-CUT` protocol keeps
confirmatory scoring disabled until power analysis and independent
two-reviewer adjudication are complete. No old sealed cases or frozen
v16/v18/v19/Gold artifacts were changed.

## FXAI-H10-FINAL-GOLD-PRECONFIRMATORY

- Date: 2026-07-23
- Branch: `feat/h10-final-gold-v20`
- Frozen base: `feat/h10-audit-confirmatory-v19@1713434980d4f4c3fed67be163ae8070d6388cdb`
- Gold implementation commit: `1145c395ced391cce6ea60ba3a8131121d03cfd3`
- Study: `FXAI-H10-FINAL-GOLD`
- Phase: preconfirmatory development
- Sealed opening count: `0`
- Scientific release: blocked

The independent Gold implementation derives source truth from nodes and edges
actually changed by low-level transactions, repair truth from inverse
transactions, and broken paths from graph differences. Its exhaustive cut
oracle is separate from the evaluated H10 solver and has no H10 imports. The
generated corpus contains 4,500 cases across six modality-specific route
pipelines: 900 clean, 900 single, 1,800 composite, and 900 unknown/ambiguous.
Two real reviewers must still complete the 200-case blind adjudication package;
the pipeline does not generate reviewer answers.

The first development run exposed an unfairly weak baseline because propagated
`derived_status` symptoms were treated as independent repair targets. Before
protocol lock, both independent baselines were strengthened to ignore that
explicitly derived field. After this correction, Full H10 and the best strong
baseline both reached source F1 `1.0` and repair F1 `1.0` on the composite
development subset. The expected H10-L and H10-R effects are therefore `0.0`,
below the registered `0.04` practical margin; increasing sample size cannot
recover an absent effect. Full H10 retains a development-only secondary
minimal-cut exact-match result (`1.0` versus `0.286111...`), but H10-C was not a
primary confirmatory claim and must not be promoted after observing results.

Protocol lock and sealed scoring fail closed for two reasons: manual
adjudication is incomplete and the primary power gate is blocked. No sealed
truth was read, no opening record exists, and v16/v18/v19 remain unchanged.

## FXAI-H10-V19-REPOSITORY-INTEGRATION

- Date: 2026-07-22
- Branch: `feat/h10-audit-confirmatory-v19`
- Base: `origin/main@e678c047896b81ed796ec6be9cdc4370cf12ddca`
- Input handoff SHA256: `c7060ab1f88bb853cd73435bdd5a0e95e4ce9f0cbab2554f77c2a79d854443c4`
- Sealed scoring openings: `1` in the imported handoff; no repository re-opening
- Technical integration: complete
- Scientific release: blocked

The v19 handoff reports positive H10-L and H10-R effects, but repository-level
methodology review found semantic target coupling. The mutation generator
assigns source nodes from the oracle's static mutation catalog, and that
leaf-to-source mapping exactly duplicates the evaluated H10 taxonomy. Repair
truth is copied from the same source-node catalog. The oracle has import-level
independence and its cut solver is separately implemented, but the primary
localization and repair targets do not arise from independently mutated graph
nodes or edges. Consequently H10-L and H10-R are
`invalid_not_evaluated_with_independent_source_truth` even though their frozen
numerical effects remain reproducible. H10-C remains secondary descriptive,
H10-U remains descriptive, and H10-T remains a byte-identical trace result.

The original handoff reports and positive claim registry are preserved under
`artifacts/h10_v19/imported_handoff/`; repository closure files carry the
fail-closed status. The invalid v18 closure is retained under
`artifacts/h10_history/v18/`. No v16/v18 branch or tag was changed, no old
sealed test was rescored, and no stable tag may be created from this branch.

## FXAI-FINAL-ONE-ZIP-PRACTICAL-CLOSURE

- Date: 2026-07-21
- Release target: `v1.3.0` technical stable
- Branch: `feat/one-zip-final-practical-closure`, merged to `main`
- Green pre-release main CI: readiness `29846598207`; universal integration `29846598714`
- Original protocol source: `3a7ad13ae33f2d84f0384c34c0a417c0f1a34def`
- Declared scoring-recovery source: `13a29170cd96cc7e94f86a4fa2c129c7e75f295e`
- Protocol identifier: `FXAI-FINAL-ONE-ZIP-PRACTICAL-CLOSURE`
- Sealed objects scored: `17,404`
- Post-open tuning: `false`
- Human claims: disabled and out of scope

The original confirmatory scoring attempt is preserved as `invalid_after_label_opening`. The failure occurred before
outcomes were scored: the runner treated the outer `{"labels": ...}` vault envelope as the identity map and detected
the repeated service key `labels`. All test predictions, explanations, canonical evidence, P0/P1 features and policy
actions had already been written and hashed without labels. A separate scoring-only recovery commit unwrapped that
existing envelope and scored the immutable pre-score artifacts. The original opening record, invalid marker, protocol
lock and recovery lock remain part of the final evidence. This is a declared protocol deviation and is not represented
as an untouched original confirmatory run.

Final scientific boundary:

- H3-P1 is not supported: at the frozen 20% review budget P1 produced 2,828 invalid automatic actions versus 2,722
  for `weighted_linear_score`; relative reduction `-0.0389419544`, 95% CI for the absolute effect
  `[-0.0271225273, 0.0198597763]`, Holm-adjusted p `0.9999990968`;
- H3-P2 is not estimable because no development operating point met the frozen 0.05 risk ceiling;
- H3-P3 and H3-P4 are not supported; H3-original remains not supported;
- H5-A is supported only for the registered controlled/compositional fault library: F1 `1.0`, false certification
  `0.0`, source localization `1.0`; H5-P-original remains not supported;
- H6-A is supported only in the registered synthetic planted-rule eligible region. H6-B is not supported because the
  required locked confidence interval and Holm test were not produced; H6-general remains not supported;
- H7-A is supported with exact hash preservation for 17,404 canonical artifacts. H7-B is not supported;
- H8 and H9 retain bounded technical measurements but are not confirmatory claims because their own artifacts mark
  confirmatory claims as disallowed; H9 is operator-only and not an end-to-end explainer benchmark;
- AI text review was not run. Domain-language, comprehension and expert-action claims remain disabled.

The generated Chapter 4 package contains 17 sections, 31 machine-generated tables and 23 figures. The rendered output
is 19 A4 pages with no placeholders; DOCX and PDF are generated from the claim registry and final statistics. The one
external ZIP is built from the committed-tree source allowlist plus immutable evidence, and excludes raw data, label
vaults, the decryption key, caches and the quarantined site. The stable tag is created only after the final version
commit passes the same public readiness and universal-integration workflows; this release operation does not upgrade
unsupported hypotheses or absent human evidence.

## FXAI-FINAL-ONE-ZIP-PRACTICAL-PRELOCK

- Date: 2026-07-21
- Branch: `feat/one-zip-final-practical-closure`
- Real multimodal implementation: `1160c77b739fa54e7a433e1aeec24c7d9440c978`
- One-shot confirmatory and final artifact implementation: `418e05c7c0516b7ccf6e33a7de3fb48a22fce93c`
- Fixed-risk operating-point protocol: `9def80e4871f49f0abb18fb8b7972b60a03be720`
- Protocol identifier: `FXAI-FINAL-ONE-ZIP-PRACTICAL-CLOSURE`
- Confirmatory protocol: prelock PASS, not locked
- Confirmatory labels: unopened
- Technical stable release: blocked pending the one-shot run and final evidence

The final cycle now uses five real independent datasets with content-aware split isolation. Exact feature duplicates,
image perceptual-near-duplicates, normalized/MinHash text duplicates and subject-level time-series overlap are all
audited before OOF generation. The current split contains 69,827 train/development OOF identities and 17,404 sealed
test identities with zero identity, group, exact-content or registered near-duplicate violations. Label vaults remain
encrypted and are not feature channels.

Measured prelock evidence:

- real model and component-occlusion pipelines run for two tabular datasets, shoulder radiographs, SMS text and UCI
  HAR sensor windows; every development row is predicted by a model that did not train on that row;
- P0 has 10 predictive channels and P1 adds 13 route/explanation channels with explicit `not_applicable` masks;
- real formative H3 at the frozen 20% review budget records 6,283 invalid accepts for P1 versus 8,356 for the best
  simple matched-budget policy, a 24.8085% development-only reduction; this is not confirmatory evidence;
- post-hoc and glass-box comparisons are separated. SHAP, LIME, Anchors and component occlusion use the same frozen
  predictors; GAM, EBM, RuleFit, sparse tree and rule list are evaluated as predictors. FXAM is pinned to arXiv
  2111.08255 but excluded because no pinned reproducible implementation is registered;
- H7-A preserves canonical hashes for all 69,827 OOF explanation artifacts. H7-B remains blocked because projection
  stability and fidelity trade-off are not yet measured;
- the historical controlled H5-A, H6-A, H8 and H9 evidence is retained as formative only. H6-A did not meet its prior
  formative target, H6-B has not been scored, and operator-only H9 must not be described as end-to-end scaling;
- the 240-case/720-variant blind AI text-review input is available, but no review was run. The technical cycle may
  continue only because all human-comprehension, expert-confirmation and domain-safety claims are disabled.

Implemented one-shot boundary:

- pre-score builds test predictions, canonical evidence, P0/P1 and all policy actions without opening labels;
- an immutable opening marker is written before vault decryption, and any post-opening failure forbids reuse of the
  run as the original confirmation;
- H5-A, planted H6-A, train/development-selected H6-B candidates, H8 and operator-only H9 are part of the frozen
  orchestrator; H6-B labels are used only during scoring;
- final statistics require effect sizes, hierarchical intervals and Holm adjustment; claim status has no manual
  positive override;
- Chapter 4 and `fuzzyxai-final-practical-closure-<commit12>.zip` fail closed until completion, statistics and claims
  exist. The ZIP builder applies the clean-source allowlist and excludes old ZIP files, raw data, vaults, keys and the
  quarantined site.

Next step: review the prelock code and measured limitations, commit the refreshed manifests, then create the protocol
lock once. Run the sealed confirmatory command only from the exact locked HEAD. Do not tune the image model, risk
ceiling, fixed budgets, features, candidate rules or success thresholds after lock.

## FXAI-FINAL-PRACTICAL-CONFIRMATORY-PRELOCK

- Date: 2026-07-21
- Branch: `feat/final-confirmatory-and-chapter4`
- Frozen base: `68e6edcfa867b48684b89d98dd74b5fe4794ef55`
- Dataset sealing and OOF implementation: `77e12c0e468fdff727fa1f89eda3b5e3e6aa19a4`
- Public sealed-prelock workflow: `29827701682`, Python 3.11/3.12 PASS on
  `613e35d00dedd4b5c9368861734a79d0a958df11`
- Research phase: formative iteration 2 of at most 3
- Confirmatory test: unopened
- Stable technical release: blocked

The five independent datasets are now downloaded, prepared and sealed locally. Public Git evidence contains source,
license, preprocessing and split manifests plus hashed identities; raw/processed data, encrypted labels and OOF rows
with development targets remain workspace-local. The practical controller and frozen hypotheses were not changed.

Measured technical prelock evidence:

- focused final/practical regression: `20 passed`; Ruff and prelock verification: PASS;
- independent registry: Bank Marketing, Default of Credit Card Clients, Shoulder Implant X-Ray, SMS Spam and UCI HAR;
  all five are `sealed`, with zero formative identity overlap;
- split evidence: 69,825 train/development OOF identities, 17,406 sealed-test identities and zero overlap;
- OOF baseline artifacts: 69,825 rows, generated without loading sealed test; per-object held-out labels are not used
  as feature channels;
- route-fault library: 40 distinct templates and 14 double/triple compositions;
- blind AI run-2 input: `release_artifacts/fuzzyxai-ai-formative-run2-input-77e12c0e468f.zip`, 240 cases,
  720 variants, 12 batches, rubric, schemas and 60 image assets; scores are not run and AI is not external validation;
  SHA256 `2451660d981983ea18a5dc33940f0c022139517a1e0d4ad3891083c866ea1438`;
- controlled formative shadow replay: 100,000 events across clean, shift, schema, calibration, update and recovery
  phases; it is explicitly not confirmatory or observed-production evidence;
- deterministic prelock evidence archive:
  `release_artifacts/fuzzyxai-final-confirmatory-prelock-77e12c0e468f.zip`, SHA256
  `ed4a87b01355d35c8c11b9172a3724c4aee477d850e09de0de778138ffae8289`.
- clean committed-tree source archive: `release_artifacts/fuzzyxai-source-release-613e35d00ded.zip`, 1,539 files,
  SHA256 `ff7f54485d7977754b93a164767363ce95d68856aaf3348708f4f844a469c4ea`; quarantined site absent.

Fail-closed boundary:

- predictive P0 remains partial because real model-disagreement and shift channels are absent;
- route/explanation P1 remains pending and contains no invented values;
- protocol lock is blocked until P0/P1 are complete and a real clean-session AI formative run 2 passes;
- H3-P1-P4, H5-A, H6-A/B, H7-A/B, H8 and H9 remain `blocked_pending_sealed_confirmation` in the new claim file;
- H3-original, H5-P-original and H6-general remain `not_supported` and must not be relabeled;
- final statistics, final Chapter 4 DOCX/PDF, final archive, merge and stable tag remain forbidden.

Next step: run the supplied 240-case blind ZIP in a temporary clean chat, import all 720 immutable raw reviews, then
populate the missing real P0/P1 channels from train/development only. Create the protocol lock once after both gates
pass. Do not open sealed-test labels or tune datasets, costs, thresholds, strata or primary metrics before that point.

## FXAI-FINAL-PRACTICAL-CLOSURE-FORMATIVE

- Date: 2026-07-21
- Branch: `feat/final-practical-closure`
- Base: `63cef7578d28a28dac63654f24642a980b49bc90`
- Practical-controller implementation: `699662e01a7bb40c437717455b55896ebd011d5d`
- Release-gate implementation: `2f70d167414c170ddfc7559d59e287bcb4ea3f52`
- Formative evidence commit: `d5ac4d5`
- Research phase: formative development
- Confirmatory protocol: blocked, test unopened
- Stable technical release: blocked

This milestone implements the production-facing, budgeted FuzzyXAI action controller without changing prior result
statuses. `H3-original`, `H5-P-original`, and `H6-general` remain `not_supported`. The public API now exposes typed
`assess_action`, batch and streaming assessment, formal hard guards, OOF predictive/route risk estimators, calibrated
budget allocation, exact canonical explanation preservation, deterministic replay, shadow/canary monitoring, and
rollback thresholds. Low confidence alone cannot produce `block`; only a frozen formal contract violation can.

Measured full-profile formative evidence:

- H3 practical development comparison at a 20% review budget: the controller recorded `0` wrong or invalid automatic
  actions at automatic coverage `0.7785588752196837`; the best simple baseline recorded `12` at coverage
  `0.8014059753954306`. This is a development result only and permits no confirmatory superiority claim;
- H5-A controlled route validity: F1 `0.9990900818926297`, false certification `0.0018181818181818182`, source
  localization `0.9981818181818182`; naturally occurring failures remain unobserved in a sealed pipeline;
- H6-A detectability envelope: `81` configurations, detection rate `0.2962962962962963`, formative target not met;
  H6-B remains not run and requires two independent sealed tabular datasets;
- H7-A exact canonical source-hash preservation rate `1.0`; H7-B remains a presentation trade-off requiring
  confirmation;
- H8 controlled grid sensitivity formative target met;
- H9 cached operator layer measured through `5,000,000` objects with empirical exponent `0.928977185286909`; local-explainer
  cost is excluded and separately disclosed.

Validated technical boundary:

- focused public/practical/strong-confirmatory regression: `23 passed`; practical suite: `12 passed`;
- Ruff, compile, public import, evidence checksums and Parquet reads: PASS;
- machine-generated formative Chapter 4 shell: 17 sections, 6 tables, 6 figures, zero placeholders;
- claim registry 3.0 removes human-comprehension, expert-confirmation, domain-safety, and specialist-superiority claims
  from technical release scope rather than pretending those gates passed;
- protocol lock fails closed until two independent tabular datasets plus image, text, and time-series datasets are
  sealed, all controller features are OOF, and a real blinded AI formative run 2 passes;
- final DOCX/PDF generation fails closed until every enabled computational claim has sealed confirmatory evidence.

Next step: import a real 240-case blinded AI formative run 2, seal independent confirmatory datasets and splits, lock
the protocol once, and run the untouched result packages. Do not tune thresholds, datasets, costs, strata, or primary
metrics after lock. Do not report the formative H3 advantage as confirmed.

## FXAI-STRONG-CONFIRMATORY-FORMATIVE

- Date: 2026-07-21
- Branch: `feat/strong-confirmatory-closure`
- Implementation commit: `238194c113fc2f1677730318dfc0ba4d4d4426d5`
- Frozen ancestors: `e34e52f`, `bd48a9c`, `1f5fd77`
- Research phase: formative development
- Public formative workflow: `29788608838`, PASS on evidence commit `fa6c40c`
- Confirmatory protocol: blocked
- Stable release: blocked

This milestone implements the fail-closed strong confirmatory program without rewriting or relabeling prior evidence.
The immutable statuses remain `H3-original = not_supported`, `H5-P-original = not_supported`, and
`H6-general = not_supported`. EBM, GAM, RuleFit and rule lists are registered as interpretable predictors rather than
post-hoc explainers; modality-specific post-hoc methods and action policies are separate comparator families.

Measured formative evidence:

- H3-v2 selective observer: formative target not met; no positive policy claim is allowed;
- H5-A controlled route validity: F1 `0.998998998998999`, false certification `0.002`, source localization `0.998`,
  invalid-action recall `0.998`; this is a route-contract result, not an error-prediction claim;
- H6-A semisynthetic planted rules on real tabular features: `24` configurations, detection rate
  `0.8333333333333334`, mean specific effect `0.21666666666666667`; H6-B remains not run;
- H7 stability: image and controlled time-series profiles meet the formative fidelity/stability gate, while tabular
  and cached 20 Newsgroups profiles fail fidelity non-inferiority; the overall H7 formative target is not met;
- H8 controlled component-grid action/representation stability: formative target met;
- H9 cached streaming operator layer: measured to `1,000,000` objects, deterministic repeat PASS, empirical exponent
  `0.9220263168835`; local-explainer cost is explicitly excluded.

The protocol lock deletes/refuses itself until sealed independent dataset manifests and a real formative AI-review
acceptance file exist. Final Chapter 4 generation additionally requires locked confirmatory claims plus real
domain-language, comprehension, and expert-action gates. The current output is a checksummed formative evidence
package and a 17-section chapter shell only; no new strong dissertation conclusion, merge, stable tag, or external
validation claim is allowed.

Next step: complete real formative AI review and seal independent confirmatory datasets. Then lock the protocol once,
run the untouched confirmatory split, and invite independent humans. Do not tune thresholds after test opening.

## FXAI-AI-PRE-REVIEW-FINAL-BLINDING

- Date: 2026-07-20
- Branch: `feat/ai-pre-review-final-closure`
- Frozen invalid-for-scoring prototype: `60ed5697d4d607df59556ea82de63527905f0f4f`
- Leakage-remediation implementation: `c1271db`
- Reviewer-packet hardening commit: `79f80a8`
- Public workflow: `29767282573`, PASS
- Stable release: blocked

The previous 360-case analysis bundle remains a technical prototype but must not be scored: it disclosed outcome,
expected-action and structural answer-key fields. The replacement freezes a reviewer-visible formative input with 240
cases and 720 randomized variants across tabular, image, text and time-series modalities. It includes human-readable
measured features, image regions, text phrases and time intervals, plus direction, normalized magnitude, rank,
stability, source agreement, limitations and complete claim-evidence links.

Validated technical boundary:

- automated leakage and evidence audit: PASS, `240/240` cases, `720/720` variants, claim coverage `1.0`;
- public formative archive: 12 batches and 60 referenced image assets;
- outcome, true label, original stratum, expected action, hidden rupture type, answer-key notes and method identity are
  absent from reviewer-visible records;
- the public ZIP excludes the encrypted scoring key, private paths and all confirmatory records;
- reviewer records contain no internal semantic-block labels; technical audit and claim-registry files are kept out of
  the reviewer ZIP to avoid priming;
- negative tests fail closed for injected outcome, action, stratum and method fields;
- focused test suite: `10 passed`; Ruff and compile checks: PASS;
- confirmatory lock fails closed until at least two real formative runs meet the frozen acceptance contract;
- claim registry remains evidence-first: predecessor negative results are preserved and seven external claims remain
  `open_external`.

Next step: run the 240-case formative AI pre-review using the new public archive. Do not expose the 120 confirmatory
cases, create a protocol lock, invite human reviewers or claim AI-human agreement before real formative results are
imported and accepted. The focused public technical workflow is green; it does not replace the open external gates.

## FXAI-AI-PRE-REVIEW-HUMAN-CONFIRMATION

- Date: 2026-07-20
- Branch: `feat/ai-pre-review-human-confirmation`
- Technical implementation commit: `5c8d0fe`
- Public technical validation commit: `461d267a4a700a4aeff8251ef25f7da539d2bab1`
- Public workflow: `29752785014`, PASS
- Frozen Q1 evidence: `e34e52fb8ae62ee1be043d6d5b26a0c9214a0572`
- Formative observer base: `bd48a9ca3795e2665e0e6a4f1ab4f4e981774c2b`
- Technical status: PASS; AI and human gates open
- Stable release: blocked

This milestone prepares a blind AI pre-review followed by independent human confirmation without treating AI output
as expert evidence. The tracked source snapshot contains 360 unique measured cases: 240 formative and 120
confirmatory, split equally across tabular, image, text and time-series modalities. Three explanation variants per
case are randomized with an out-of-repository HMAC secret; the reversible method map is AES-256 encrypted.

Validated technical boundary:

- master log: 1,080 blind variants;
- review packets: 18 batches, at most 20 cases and 60 variants per batch;
- rubric: R1-R10 plus 12 critical-defect flags;
- AI importer requires complete batches, immutable hashes, exact commit identity and independent `AI_RUN_1..3`;
- confirmatory lock fails closed until a real formative acceptance record exists;
- human packet construction fails closed until protocol lock, three AI runs and score commitment exist;
- human importer requires at least three independently hashed reviewers and rejects AI reviewer records;
- technical DoD after memory/release documentation: 44 PASS, 36 external/public-CI items open;
- focused local tests: 6 passed; new-code Ruff and compile checks: PASS;
- deterministic analysis-input archive: PASS for repeated same-commit builds.
- public focused workflow: PASS on the exact technical validation commit.

Frozen evidence limitation: the Q1 Fashion-MNIST protocol labels every evaluated image class as rare, so the image
modality has no measured `common_class` stratum. The study records this limitation rather than fabricating a class.

Next step: run the formative AI pre-review externally, import the raw JSONL without rewriting it, repair only observed
defects, and lock the confirmatory protocol. Human confirmation remains forbidden until three confirmatory AI runs
are committed. No human-comprehension, domain-approval, expert-validation or stable-release claim is allowed yet.

## Q1 Empirical Remediation

- Date: 2026-07-20
- Branch: `feat/q1-empirical-remediation`
- Frozen predecessor: `cafe403c7d60e36b08f56a5325ba380718a5be35`
- Technical CI commit: `a9ca41ecc9bdfe74024214e5e4b1ed2c3eccefab`
- Public Q1 workflow: `29709914672`
- Candidate status: technical remediation PASS; stable release remains blocked

This independent cycle preserves the earlier E1-E8 negative results and evaluates new preregistered hypotheses
without changing their criteria after measurement. The public workflow passed the focused contracts, four independent
real-modality jobs, a clean Docker reproduction and aggregate evidence verification.

Measured technical evidence:

- real datasets: UCI Covertype `581012`, Fashion-MNIST `70000`, 20 Newsgroups `18846`, and ElectricDevices `16637` objects;
- real benchmark matrix: 16 measured model records and 25 measured explainer records;
- model channels include linear, tree, boosting, CNN, sequence and ONNX execution;
- explainer channels include SHAP, LIME, Anchors, RuleFit, Grad-CAM, Integrated Gradients, and text/time-series masking;
- H1 controlled paired fidelity: 240 pairs, mean delta `0.0`, lower 95% bound `0.0`, non-inferior at margin `-0.02`;
- H2 controlled traceability: `K_trace 0.0 -> 1.0`, missingness F1 `1.0`, false-certification rate `0.0`;
- H3 adaptive cascade controlled cost fraction `0.36666666666666664`, with the preregistered risk-cost criterion met;
- H4 adaptive hierarchy uses FML for `0.5641666666666667` of objects and reduces mean representation complexity by `2.3258333333333336` versus always-FML while meeting non-inferiority;
- H5 structural diagnosis F1 `1.0`, but predictive incremental AUPRC `-0.0005609278151094133`; predictive and safety claims remain forbidden;
- H6 matched rule-ablation candidate remains `inconclusive`; its exploratory conditional model has holdout R2 `-0.06296668096584646` and does not support a general rule-removal claim;
- all datasets, licenses, preprocessing declarations and raw/processed hashes are recorded in the aggregate evidence;
- clean command `docker run --rm fuzzyxai-q1 make reproduce-q1`: PASS;
- technical DoD after archive closure: 105 PASS, 0 BLOCKED, 3 OPEN_EXTERNAL.

Required boundaries:

- the real benchmarks demonstrate reproducible framework execution, not clinical, production or external-domain validity;
- the critical-rupture result is a structural diagnostic result only;
- the rule-ablation result is a controlled context-dependent candidate, not a confirmed general effect;
- independent comprehension, expert-action and domain-language reviews are still not run;
- merge to `main` and a stable release tag remain forbidden while any external gate is open.

Next step: run the three independent external protocols. Do not reopen or overwrite the frozen predecessor results.

## Full Empirical Validation

- Date: 2026-07-20
- Branch: `feat/full-empirical-validation`
- Final technical closure commit: `fc8705ececdcc58f6cc171017984d9c6af05dd45`
- Final full empirical CI run: `29707225733`
- Candidate status: full E1-E8 technical validation; stable release remains blocked

The current milestone adds a controlled, evidence-first validation layer without changing the defended chapter 2-3
operator semantics. Four deterministic modality contours contain 10,000 objects each. The full protocol measures
repeated rule ablation, explanation baselines, P1-P7 action policies, calibration, sensitivity, adaptive uncertainty
representations, critical-rupture association, and scalability. The Docker image runs the same one-command pipeline.

Measured boundaries must remain explicit:

- SHAP, LIME, Anchors, RuleFit, confidence, disagreement, reduced-history FuzzyXAI, and full FuzzyXAI are measured;
- the 50-pair rule-ablation experiment does not confirm a general subgroup-recall effect;
- adaptive selection uses `FML` for 55.22% of controlled objects and is non-inferior to always-FML in the declared risk metric;
- the critical-rupture indicator has incremental AUPRC `-0.12556024709557384`, so no safety or predictive-gain claim is allowed;
- the critical rupture remains a structural diagnostic indicator only;
- controlled multimodal results do not establish external-domain generalization;
- expert review and comprehension pilot remain `planned_not_run`; domain semantic review remains pending;
- merge to `main` and a stable tag remain forbidden while those external gates are open.

Public run `29707225733` passed protocol smoke, full E1-E8, checkout-independent Docker reproduction, final evidence
aggregation, archive verification, and the 90-item technical Definition of Done. Stable release remains blocked by
the three explicitly recorded external gates, not by a computational failure.

- Date: 2026-07-19
- Branch: `feat/universal-model-integration`
- Base: `79d39f61f02df1cf1d63fc92dd2a1ebebfef7de7`
- Universal API commit: `3dda3e3`
- Model evidence commit: `a76677a`
- CI correction commit: `fbfcdd3`
- Cross-runtime evidence commit: `39067b3`
- Candidate version: `1.3.0rc1`
- Release tag: not created

## Current Focus

The installable research framework remains the primary product. The generated DubnaXAI site stays quarantined.
This milestone adds capability-based model integration and measured explanation-quality disclosure without changing
the defended chapter 2-3 operator semantics.

## Implemented

- `ModelAdapterV2`, task/input/output contracts, capability descriptors, and native/derived/surrogate/external origin;
- priority adapter registry, explicit resolution report, lazy optional imports, and plugin entry-point discovery;
- sklearn linear, tree, ensemble, SVM, KNN, Naive Bayes, regression, and Pipeline adapters;
- optional XGBoost, LightGBM, CatBoost, PyTorch, TensorFlow/Keras, and ONNX adapters with separate tests;
- `ExplanationPlanner`, surrogate-fidelity blocking, adapter conformance, and quality reports;
- `explain_batch()`, `explain_global()`, `why_not()`, `compare_models()`, and capability reporting;
- deterministic benchmark with 24 classification and 10 regression configurations;
- checksummed runtime reports for sklearn, XGBoost, LightGBM, CatBoost, PyTorch, TensorFlow/Keras, and ONNX;
- automatic report merge, cross-version conflict detection, unified support matrix, API report, and quality report;
- frozen external A/B pilot package and external domain-language review package;
- aggregate Chapter 4 candidate evidence package.

## Measured Validation

- 34 core model configurations and six optional runtime integrations verified;
- unified model support matrix: `40/40 pass`;
- runtime reports: `14/14 pass` across Python 3.11 and Python 3.12;
- prediction parity rate: `1.0`;
- adapter conformance rate: `1.0`;
- explanation graph validation rate: `1.0`;
- strict MyPy: PASS;
- Ruff and compile checks: PASS;
- focused local compatibility/model/external-gate suite: `48 passed`, `6 skipped` optional runtimes;
- public readiness CI: `338 passed`, `6 skipped` on Python 3.11 and Python 3.12;
- public Octave JSON/dashboard smoke: PASS;
- public optional runtime CI: XGBoost, LightGBM, CatBoost, PyTorch, TensorFlow/Keras, and ONNX PASS on both Python versions;
- five public explanation APIs: PASS in all runtime reports;
- explanation-quality matrix: `40/40 pass`; measured top-reason stability in 36 configurations;
- readiness run: `29700288436`;
- universal model run: `29700288413`;
- existing measured checkpoint experiment and native rule ablation retained;
- model, pilot, domain-review, and Chapter 4 manifests: PASS.

The full regression suite was moved from the workstation to GitHub runners after the user reported Cursor
instability. Both supported Python jobs passed. This validates the feature branch, not `main` after merge.

## External Gates

- independent A/B comprehension pilot: `planned_not_run`, zero participants;
- regulated-domain dictionary review: `pending_external_review`;
- Chapter 4 computed evidence: PASS;
- demonstrated human comprehension: not claimed;
- release gate: `BLOCKED`;
- `v1.2.0` tag and merge to `main`: forbidden until external gates and main CI pass.

## Version Boundary

- `v1.2.0`: Human Explanation and empirical computational gate implemented; final tag blocked externally.
- `v1.3.0rc1`: universal tabular model integration and merged cross-runtime evidence candidate implemented on this branch.
- version path B selected: no retroactive stable `v1.2.0`; next allowed tag after external gates, main merge, and main CI is `v1.3.0rc2`.
- `v1.4.0`: anomaly, forecasting, text, image, and richer checkpoint integrations remain future scope.

## Archive Policy

- source archive is built only from a committed Git index with `python scripts/build_framework_release.py`;
- Chapter 4 candidate ZIP is reproducible but not a final-release claim;
- doctoral historical archive remains separate;
- generated site content and unrelated dirty-worktree artifacts must not enter the source archive.

## Next Step

Run the real external pilot and domain review. After both pass, review merge to `main`, require green main CI, rebuild
committed-index archives, and only then create a release tag. Do not claim `v1.4.0` support before modality-specific
adapters and benchmarks exist.
## FXAI-SELECTIVE-OBSERVER-FORMATIVE

- Date: 2026-07-20
- Branch: `feat/selective-observer-formative`
- Frozen predecessor: `e34e52fb8ae62ee1be043d6d5b26a0c9214a0572`
- Research phase: formative development
- Stable release: blocked

This milestone starts a new two-stage research cycle without modifying the predecessor evidence. It adds a grouped
out-of-fold selective risk controller with `accept`, `short_review`, `full_review` and `block`; preregisterable
confidence, uncertainty, explainer-disagreement and selective-risk baselines; separate risk-coverage and cost-review
comparisons; H5 action-contract validation separated from held-out error prediction; H6 planted-rule method checks
separated from low-redundancy matched-control confirmation; and an immutable stage-B protocol-lock contract.

The development package is intentionally claim-safe:

- controller source channels and controller selection predictions are out-of-fold;
- test records cannot be passed to development contracts;
- formative datasets and participants cannot overlap confirmatory identities;
- formative operating-point gains never enable a confirmatory claim;
- external-study files contain protocols and empty record lists only;
- the old H3, H5 predictive and H6 null results remain visible;
- no stable tag or merge is allowed during formative development.

Acceptance command: `make selective-observer-formative-check`. The next scientific action is formative usability and
expert-task piloting, followed by a signed/timestamped protocol lock. Only then may new datasets, participants and
experts be opened once for confirmation.

## FXAI-Q1-FINAL-CLOSURE

- Date: 2026-07-20
- Branch: `feat/q1-final-closure`
- Frozen base: `41c32af25242164144fd907e4850fa9d4f426bd1`
- Candidate status: technical closure PASS at `e34e52fb8ae62ee1be043d6d5b26a0c9214a0572`; external gates open
- Stable release: blocked

This milestone adds a final evidence-first validation boundary rather than replacing the defended operators or the
frozen predecessor experiments. It defines native multiclass tasks for Covertype, Fashion-MNIST, 20 Newsgroups and
ElectricDevices; measured explainer cohorts; separate H1-H5 results; a leakage-resistant two-dataset confirmatory H6
protocol; end-to-end scalability; claim registry 2.0; deterministic archive identity; and protected external-study
scorers. The generated DubnaXAI site remains quarantined.

Local validation at implementation time:

- Q1 final focused suite: `19 passed`;
- Ruff for the Q1 final package, scripts and tests: PASS;
- compileall: PASS;
- `scripts/q1_final/reproduce_all.py --profile smoke`: PASS;
- strict external verifier: expected FAIL with `domain_language_review`, `comprehension`, and `expert_action_review` open;
- public workflow `29740096302`: 9/9 jobs PASS, including four native modalities, rule ablation, scalability, Docker and aggregate verification;
- technical DoD: 161 PASS, 0 BLOCKED, 24 OPEN_EXTERNAL.

Release boundary:

- the four native multiclass jobs, required explainers, H1-H6, full scalability, Docker reproduction and archive
  verification must pass in `.github/workflows/q1-final-validation.yml` before technical closure;
- external scorers never generate participant or reviewer records;
- an approved or exempt ethics record, genuine anonymized responses and signed records are required to close an
  external gate;
- null or negative H3, H5 predictive, H6, comprehension or expert-action results must remain visible and must remove
  the corresponding positive claim;
- no merge to `main`, stable `v1.3.0` tag or stable GitHub release is allowed while mandatory gates are open;
- archive commit and CI run IDs are written at build time to `release_evidence/q1_final/run_identity.json`.

Next step: preserve the technical result and conduct the three independent external protocols. Do not manufacture
responses or promote formative evidence to a confirmatory claim.

## FXAI-H10-C2-PRECONFIRMATORY

- Date: 2026-07-23
- Branch: `feat/h10-c2-preconfirmatory-v22`
- Frozen dependency: `a8f150b1ef3b5c6041c28098a5cc90d0e8e20ae5` (`v21`, `1.4.0a1`)
- Confirmatory claims: `H10-C2a NOT_EVALUATED`, `H10-C2b NOT_EVALUATED`
- Sealed opening count: `0`
- Scientific status: `BLOCKED_POWER`

The new package separates minimum-cut membership from repair recertification, derives Gold from low-level
transactions with an independent exhaustive oracle, enumerates equivalent optimal cuts, isolates seven baseline
implementations, exports two blank blinded reviewer packages, and blocks sealed scoring behind protocol,
adjudication, leakage, and single-opening gates. The FuzzyXAI v21 core is unchanged.

The registered six-pipeline power grid did not reach the target: the highest observed Monte Carlo power in the
bounded grid was approximately `0.0875` for H10-C2a and `0.0800` for H10-C2b. Under the registered moderate effects
and ICC, the first analytical expanded-cluster estimate reaching both targets requires approximately 222 independent
pipelines and 124,320 cases. This is an unapproved design estimate, not a new protocol or scientific result.

Do not generate or score a sealed H10-C2 set until the power assumptions and expanded design are externally reviewed,
approved, and locked. Do not fill reviewer forms programmatically. Development and protocol-validation outputs remain
nonconfirmatory.
