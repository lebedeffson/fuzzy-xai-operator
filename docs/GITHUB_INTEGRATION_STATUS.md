# FuzzyXAI doctoral integration status (2026-11-07)

Branch: `chapter6-ophthalmology` (NOT `main`).

- 2026-11-07: new docs-only integration contract committed at `7ead2c2a8cf94581648898a11bfd6d8238b99487`.
- 2026-11-07: 56 FuzzyXAI v6 overlay sources, experiments, documentation, and tests committed at `2c697d73778fdd6af06897033df4712f375d413d`.
- 2026-11-07: eight preflight-checked upstream edits, including runtime certification and risk protections, committed at `e4e770a1753702b1b14f1c36e726c0fa813eb43c`.
- CI workflows now run on pushes to this branch. The *actual* CI status must be read from GitHub Actions. A commit alone is not evidence of passing the full repository suite.

Existing `RELEASE_MANIFEST.json` is the historical v5 overlay manifest; it is NOT a hash of this merged GitHub checkout. Earlier README phrases describing GitHub write error 403 were accurate at packaging time and are superseded by this status document.

## Scientific limits

IT2 uncertainty intervals are **footprints of membership uncertainty**; they are not predictive confidence intervals or medically validated coverage statements. WDBC is an exploratory benchmark; PTB-XL waveforms and ophthalmology clinical studies were not independently verified in this integration.

Do not claim full conformity until GitHub tests, integrity checks, and the actual runtime are verified on this branch. See `install_fuzzyxai_upgrade.py --check` for the eight-file integration preflight.
