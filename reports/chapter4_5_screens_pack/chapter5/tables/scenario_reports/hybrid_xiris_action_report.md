# Scenario action report: hybrid_xiris

- module: `HYBRID-XIRIS biometric pipeline`
- adapter_called: `True`
- output_type: `ExplanationArtifact + report`
- status: `real-output-compatible`
- evidence_level: `repository-output-level`
- action: `audit_report`
- claim_scope: adapter compatibility and evidence routing, not local model retraining

Численные `chi_R`, `chi_Auto` и `rho` для внешнего модуля не подставляются искусственно. Если адаптер не предоставляет полный структурный контур, сценарий фиксируется как audit/report-only.