# Dataset card: Breast Cancer Wisconsin (Diagnostic)

- Source: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- DOI: 10.24432/C5DW2B
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Bundled loader: `sklearn.datasets.load_breast_cancer`
- Version anchor: scikit-learn bundled copy used by this environment
- Objects: 569
- Features: 30
- Target: UCI diagnostic dataset label, recoded as `1=malignant`, `0=benign`
- Missing values in bundled matrix: 0
- Split: train=341, validation=114, test=114
- Split seed: 42
- Preprocessing: `StandardScaler` fitted on train only
- Rare subgroup: smallest of three KMeans clusters fitted on standardized train features before model training
- Subgroup definition hash: `692b0a5a713f834ec620b2c50d6744643626b427236c9ebadf407437bfe03077`
- Snapshot SHA256: `9f4874774e7b44c492bed00e0b09dfc354cbdbdb5d88eedc4ea12e9f0312104c`
- Download/access date recorded by protocol: 2026-07-19

## Limitations

This is a methodological classification benchmark. The experiment does not establish clinical validity,
diagnostic utility, fairness, or deployment readiness. Human-readable domain wording remains unavailable
until an independent subject-matter reviewer signs the versioned dictionary.
