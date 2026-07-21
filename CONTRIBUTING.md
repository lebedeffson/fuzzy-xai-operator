# Contributing

1. Add scientific computations to `fuzzyxai.core` or `fuzzyxai.evidence`, never to presentation code.
2. Declare adapter capabilities and return partial evidence for unsupported channels.
3. Label surrogate rules and concepts explicitly.
4. Add a typed contract, test, manifest row, and reproducible artifact for every new defended operator.
5. Do not add generated site cards, screenshots, caches, large datasets, or reproducible reports to the framework branch.
6. Run `git diff --check`, `python -m pytest`, and `make doctorate-release-check` before push.
