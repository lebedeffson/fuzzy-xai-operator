# Existing-branch integration: chapter6-ophthalmology

Repository: https://github.com/lebedeffson/fuzzy-xai-operator

Target branch: `chapter6-ophthalmology`, observed HEAD:
`8e45ba8ca9ab2e1217e12d98a3ec9a06d7cd413a`.

The default branch `main` is older (`aae7aa717bff913240cbb0026807cd1ccfa67160`).
No changes from this overlay were pushed or committed to either remote branch.
GitHub connector returned HTTP 403 on branch creation AND direct file creation
in `chapter6-ophthalmology`; container git could not resolve github.com.

On a workstation with authorized GitHub access:

```bash
git fetch origin
git switch chapter6-ophthalmology
git pull --ff-only origin chapter6-ophthalmology
# Extract the provided overlay ZIP into the repository root (preserve existing files).
python install_fuzzyxai_upgrade.py --check
python install_fuzzyxai_upgrade.py
python -m pip install -e '.[dev]'
python -m pytest -q tests/test_doctoral_interval_type2.py
python -m pytest -q tests/test_doctoral_runtime_integration.py
python -m experiments.doctoral_interval_type2 --out reports/chapter5/interval_type2 --bootstrap 24
python -m pytest -q
# Review git diff and generated outputs before committing.
git add framework/fuzzyxai/fuzzyxai/interval_type2.py experiments/doctoral_interval_type2.py tests/test_doctoral_interval_type2.py .github/workflows/doctoral-interval-type2.yml
# Include other overlay files only after review of the whole working-tree diff.
git commit -m 'feat: add verified interval type-2 fuzzy inference and heldout study'
git push origin chapter6-ophthalmology
```

`install_fuzzyxai_upgrade.py` edits eight existing project files, so inspect their
diff before staging. The installer was validated against the pinned `main`
snapshot locally, not against the latest existing branch. Preflight might reject
newer branch changes: stop rather than forcibly overwriting them.

Scientific scope: train-only bootstrap footprint of uncertainty (FOU) with
coupled low/high memberships and analytic additive logit bounds. This is not a
calibrated prediction interval or a type-3 model. Matched Type-1 baselines
are included; no claim of IT2 performance superiority is justified.
