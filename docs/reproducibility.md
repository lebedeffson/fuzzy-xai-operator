# Reproducibility

Minimum local gate:

```bash
python -m pytest
make operator-manifest-check
make doctorate-release-check
python -m build
```

The main suite is offline and uses fixed seeds. A clean source archive is built by `scripts/build_framework_release.py`; it excludes the site prototype, virtual environments, caches, large data, generated experiment trees, and local paths. The archive includes source, tests, docs, small fixtures, release evidence, repository tree, test report, status, Project Memory, and SHA256.

MATLAB/Octave execution status must be recorded separately. Packaging MATLAB files is not equivalent to running them.
