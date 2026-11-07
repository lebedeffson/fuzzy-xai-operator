# FuzzyXAI interval type-2 extension

Target branch: `chapter6-ophthalmology`.

The IT2 model uses lower and upper membership functions, interval-valued rule firing strengths, and an interval output envelope computed from the same inference parameters. This is an *epistemic parameter envelope*, **not** a calibrated statistical confidence interval.

A valid proof must disclose membership parameters, interval arithmetic, the actual model run, source identifiers, and any reduction loss. Crossing a policy decision threshold must not be mislabeled a safe automatic acceptance. Compare against matched type-1 and risk-only baselines using a held-out test set.

The research implementation is distributed in the FuzzyXAI v6 overlay until the full integration suite is run on this branch.
