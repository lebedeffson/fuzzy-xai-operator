# Dissertation demo summary

## Dataset modes

| dataset_mode | status | rows | domain |
|---|---:|---:|---|
| breast_cancer | READY | 569 | medical |
| diabetes_binary | READY | 442 | medical |
| wine_risk | READY | 178 | general |
| synthetic_ruptures | READY | 900 | controlled |
| registry_programs | READY | 10007 | tabular_text |
| registry_mosmed_doctor_analysis | READY | 10 | medical_audit |
| registry_steel_ir | READY | 8080 | industrial_cv |

## Quantitative validation (built-in modes)

| dataset | acc | roc_auc | agreement_proxy | rupture_rate | critical_rupture_rate |
|---|---:|---:|---:|---:|---:|
| breast_cancer | 0.972027972027972 | 0.9949685534591195 | 0.5594405594405595 | 0.0 | 0.0 |
| diabetes_binary | 0.7567567567567568 | 0.8298701298701298 | 0.0990990990990991 | 0.0 | 0.0 |
| wine_risk | 0.9777777777777777 | 1.0 | 0.6222222222222222 | 0.0 | 0.0 |
| synthetic_ruptures | 0.9511111111111111 | 0.9955908289241623 | 0.22666666666666666 | 0.3466666666666667 | 0.2222222222222222 |

## Registry modes (readiness and limitations)

| dataset | pipeline_completed | agreement_proxy_applicable | agreement_proxy | note |
|---|---:|---:|---:|---|
| registry_programs | True | False | N/A | Prototype measurements per object; no I/O timing. Registry mode validates readiness/portability of the pipeline; action quality metric may be N/A. No expert action labels: agreement_proxy is not applicable. |
| registry_mosmed_doctor_analysis | True | False | N/A | Prototype measurements per object; no I/O timing. Registry mode validates readiness/portability of the pipeline; action quality metric may be N/A. Small audit slice (n=10): not for statistical validation. |
| registry_steel_ir | True | False | N/A | Prototype measurements per object; no I/O timing. Registry mode validates readiness/portability of the pipeline; action quality metric may be N/A. ROC AUC must not be interpreted as quality here; use this mode for industrial contour portability. |

## Real reduction example

- object: `sample_113`
- selected_class: `F_N_src`
- reduction_loss: `0.106811`
- action: `block`

## Notes

- Registry modes may remain `MISSING` until local files are connected.
- Benchmark timing is prototype-level per object and excludes I/O.
- `agreement_proxy = N/A` means no expert action labels for that dataset mode.