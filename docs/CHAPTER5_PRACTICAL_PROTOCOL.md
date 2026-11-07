# Chapter 5: reproducible practical validation

## What this implementation fixes

The former `medical_ecg_signal` route returned `defer_to_human` for every
input regardless of signal quality. This prevented an actual experiment with
new ECG records. The new technical quality gate computes a deterministic
feature vector from raw samples and chooses one of three **audit** actions:

- `audit_report`: technical QC passed; **not** automatic clinical acceptance.
- `request_more_data`: signal quality uncertain or weak degradation.
- `defer_to_human`: severe signal quality defect, including short recordings.

All decisions are made through the public FuzzyXAI route/proof API, not a
manually populated output file. The original ten-row sample remains a **demo
control artifact**, not a sample of clinical validation.

## Reproducible local commands

Run in the repository root, using Python 3.11 and `pip install -e ".[dev]"`:

```bash
python -m pytest -q tests/test_chapter5_ecg_quality.py tests/test_chapter5_evidence_integrity.py tests/test_chapter5_hybrid_counterfactual.py
python -m experiments.chapter5_ecg_batch \
  --csv data/raw/ecg/input_ecg_sample_signal.csv \
  --out reports/chapter5/ecg_quality_batch --source-type fixture
python -m experiments.chapter5_evidence_integrity \
  --out reports/chapter5/evidence_integrity --recompute
python -m experiments.chapter5_hybrid_counterfactual \
  --out reports/chapter5/hybrid_counterfactual
```

To process **external, locally available** real ECG records, convert your data
to long-form CSV with `record_id,sample,signal,sampling_rate`. Optional
`noise_mask` and `missing_mask` fields contain per-sample 0/1 annotations.
For a 12-lead recording, use one file per lead or convert a lead to a separate
record identifier; the current protocol is single-lead QC, not a multilead
cardiac classifier. Use:

```bash
python -m experiments.chapter5_ecg_batch \
  --csv /secure/location/deidentified_ecg.csv \
  --out reports/chapter5/ecg_external_audit \
  --source-type external_unverified
```

Do **not** commit personal ECG records. The exporter emits pseudonymous record references
and checksum provenance rather than direct clinical identities or raw waveforms.
Hashes are NOT a guarantee of anonymization.
The source data remain in the private local file.

## Experimental contract

1. Input waveform is loaded from CSV; quality score is recomputed, not trusted
   from a `quality_score` field in the input.
2. The score is a **noncalibrated engineering heuristic**: `clip(1 - 1.4*m -
   0.8*n - 0.65*f, 0, 1)`, where `m` is missing fraction, `n` is an externally
   marked noisy fraction, and `f` is the adjacent-value flatline fraction.
   This is a prototype protocol, not a validated signal quality index.
3. Fixed thresholds select a technical audit action. No external training or
   medical reference ground truth is used to pick these values.
4. Each route exports `features.json`, `route.json`, and `proof_trace.json`.
   Proof/route consistency is verified before a case is counted.
5. `cases.csv` and `summary.json` contain transparent counts, integrity
   evidence, and explicit `clinical_metrics: null`.
6. The multi-domain integrity audit checks the existing five application
   route/proof/site artifacts for discrepancies, then independently recomputes the five control routes.
   It does **not** verify
   independent accuracy of the external XAI/ML models.

## Required additional work before clinical or external quantitative claims

- Choose a licensed public ECG corpus (e.g. PhysioNet PTB-XL) and pin its
  release, hashes, inclusion criteria, sampling frequency, and patient IDs.
- For clinical prediction, separately train/evaluate a named model using
  patient-disjoint train/validation/test splits and clinically validated labels.
- For technical signal QC, use independently annotated clean/noisy/missing
  segments; compare with established signal quality indices at matched costs.
- Prespecify a baseline and thresholds on validation data; freeze evaluation
  on a held-out test partition; report confidence intervals and subgroup
  behavior. A synthetic signal defect is not an external ground-truth label.
- Run real GD-ANFIS/SHAP, GIS, BEACON, HYBRID, and vessel-segmentation outputs
  through pinned adapters before claiming quantitative superiority.

Existing hard-coded HYBRID-XIRIS batch actions and other fixture studies
remain synthetic and **must not** be reported as independent evaluation.

## HYBRID-XIRIS guardrail grid

`chapter5_hybrid_counterfactual.py` generates 64 input combinations and sends
each through `build_route -> build_proof_trace -> verify_proof_trace`.
The old 168/1000 fixed action sequence is **not** used in this protocol.
The critical reference uses the same explicit conflict predicate as the
operator, so zero misses are a regression/safety invariant, NOT independent
recognition evidence or a predictive performance result.

## Practical study v2: real WDBC holdout + comparable baselines

New one-command executable study:

```bash
python -m pytest -q tests/test_chapter5_real_heldout.py tests/test_chapter5_ptbxl_qc.py tests/test_chapter5_practical_bundle.py
python -m experiments.chapter5_practical_bundle --out reports/chapter5/practical_v2
```

The WDBC experiment trains a new Random Forest on 60% of its 569 rows and
reserves independent 20%/20% validation/test portions for each of three
predeclared random seeds. All preprocessing parameters come from training;
acceptance thresholds use validation inputs/predictions, not labels. Results
are reported by condition on the held-out test records, with per-seed paired
bootstrap intervals for the controlled 40%-missing corruption.

Three policies are compared:

1. `confidence_only`: probability-threshold baseline with no source-quality input.
2. `confidence_equal_guardrail`: the *same* baseline, now given the identical
   critical input-quality flag as the proposed observer.
3. `canonical_risk_with_qc`: calls `fuzzyxai.core.risk_observer.observe_risk`
   using five P19 components and the same critical override.

The binary output of the predictive model is never passed as the expected
control action, and the corruption condition is used only for evaluation.
The low-dimensional `one_minus_I_pre` and `Delta` inputs here are **technical
missingness proxies**, not measured semantic interpretability or the full
mathematical reduction operator; this distinction is part of the reproducible
manifest and the manuscript insert. Results cannot establish an isolated
advantage of FuzzyXAI over an equally informed safety baseline.

Figures are strict monochrome PNG plus vector PDF from the result CSV, not
manually assembled charts. `chapter5_practical_bundle.py` writes SHA256 for
every generated report, case CSV and figure.

## PTB-XL: real ECG protocol, optional data

Source: PTB-XL 1.0.3, PhysioNet, DOI
[10.13026/kfzx-aw45](https://doi.org/10.13026/kfzx-aw45), Creative Commons
Attribution 4.0. Official metadata `ptbxl_database.csv` contain `patient_id`,
`strat_fold`, `filename_lr`, and `filename_hr`; `records100/` are 100-Hz
12-lead WFDB recordings. Per source documentation, folds 1-8/9/10 are the
recommended train/validation/test partition and keep patient IDs in one fold.

No files are bundled or silently downloaded. Obtain the public dataset from
https://physionet.org/content/ptb-xl/1.0.3/ and install optional `wfdb`.
Then run:

```bash
python -m pip install wfdb
python -m experiments.chapter5_ptbxl_qc \
  --dataset /data/ptb-xl-1.0.3 --max-records 300 \
  --out reports/chapter5/ptbxl_technical_qc
# Or build the combined evidence package:
python -m experiments.chapter5_practical_bundle \
  --ptbxl /data/ptb-xl-1.0.3 --ptbxl-limit 300
```

The test-only protocol selects fold-10 rows without label conditioning,
verifies patient-fold isolation across all metadata rows, and loads the
original 12-channel waveform. The technical QC checks each lead directly;
controlled lead removal/flatline are added **after** loading the real source
waveform. Thus the study can measure detection of **injected technical faults**;
it cannot estimate ECG disease sensitivity, stenosis diagnosis, or real-world
clinical safety. Dataset metadata about artifacts are not used as ground
truth or passed into the QC gate. If waveforms are absent, the tool fails
explicitly instead of inventing a real-data result.

The separately exported `CHAPTER5_EMPIRICAL_RESULTS.md` has exact numerical
results, the source boundaries, and a direct translation guide for chapter 5.
No chapter 1-4 files or the introduction are edited by this protocol.
