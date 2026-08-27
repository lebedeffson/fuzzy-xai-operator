# FuzzyXAI framework rules

FuzzyXAI is an evidence-first observing, composition, and orchestration layer,
not another SHAP/LIME/IG attribution algorithm. Preserve architecture-specific
evidence semantics.

Never convert RandomForest `feature_importances_` into local contribution;
linear coefficients into rules; similarity into causality; or opposite-class
neighbors into supporting evidence. Never fabricate missing values, substitute
raw U_model for u_M, use presentation loss as dissertation Delta, or compute
Gamma without the required executed T_ij transformation.

`not_applicable`, `missing_required`, and measured values are distinct states.
Mathematics belongs in core/evidence; visualization only renders it.
