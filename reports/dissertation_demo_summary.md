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

## Real reduction example

- object: `sample_113`
- selected_class: `F_N_src`
- Delta: `0.106811`
- action: `block`

## Notes

- Registry modes may remain `MISSING` until local files are connected.
- Benchmark timing is prototype-level per object and excludes I/O.