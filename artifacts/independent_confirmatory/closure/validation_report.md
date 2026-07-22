# Independent Confirmatory Closure Validation Report

- Scientific release status: `experimental_not_stable`
- Merge to `main`: `false`
- Sealed scoring objects: `48072`
- Frozen primary baseline: `route_only`
- H3-R1 relative reduction: `0.01597582` (required: `>= 0.15`)
- H3-R1 hierarchical 95% CI for full minus baseline: `[-0.005072463768115942, 0.00728983817320439]`
- H3-R1 Holm-adjusted p: `0.023427267`
- H3 hard-block rate: `0.00488850`
- H3 false-block rate: `0.00000000`
- H3-R2 coverage gain: `-0.00107701`
- H5 unknown recall / AUROC: `1.00000000` / `1.00000000`
- H5 known-type macro-F1: `0.46565387`
- H5 source localization: `0.68402778`
- H6 confirmatory opening: `false`
- Replay incident recall: `1.00000000`
- Replay hard-block rate: `0.00372800`

The replay incident-level table was recomputed after a declared aggregation-only defect. Controller actions,
policy thresholds and sealed H3/H5 results were not changed. The full details are in
`artifacts/independent_confirmatory/replay/scoring_recovery_deviation.json`.
