# Visualization

The canonical dashboard answers five questions:

1. Are the data complete and atypical?
2. When was the object learned or forgotten?
3. Which native rules, surrogate statements, and class concepts exist?
4. Which evidence supports this prediction and what changes it?
5. Which conflict, loss, or missing trace determines the action?

Titles describe meaning, not only symbols. For example, `gamma=0.351` is displayed as disagreement between evidence channels and explicitly is not a probability of error.

Python uses `result.plot(kind="dashboard")`. MATLAB reads the same JSON through `fuzzyxai.loadResult` and provides dashboard, rule, membership, training, similar-case, and provenance plots. Presentation code does not recompute evidence.
