# Proof guide

Run all checks:

```bash
pytest -q
python proofs/run_all_proofs.py
```

The generated reports show the computational counterpart of the dissertation route:

```text
Data -> ExplainPlan -> P_sit -> F* -> A_k^F -> Delta -> E_k_ext -> d_E_ext -> I(E_G) or diagnostics
```

## Reports

- `reports/chapter2_proof.json`: construction, composition, `d_E`, `L`, `I`, and `D_ij`.
- `reports/chapter2_calibration_report.json`: cross-validated beta calibration.
- `reports/chapter3_proof.json`: hierarchy selection and reduction losses.
- `reports/chapter3_end_to_end_report.json`: cross-chapter route with `E_k_ext`.
- `reports/breast_cancer_benchmark.json`: minimal open medical-like benchmark.


## Final dissertation validation

Run the fixed numerical validation report:

```bash
python proofs/validate_thesis_examples.py
```

Generated artifacts:

- `reports/thesis_validation.json`
- `reports/thesis_validation.md`

Run the one-click thesis demo route:

```bash
python examples/thesis_demo.py
```

Generated artifacts:

- `reports/thesis_demo_report.json`
- `reports/thesis_demo_report.md`
- `reports/thesis_demo_composition_graph.html`

The NiceGUI app also exposes these operations on the **Thesis Demo** page.
