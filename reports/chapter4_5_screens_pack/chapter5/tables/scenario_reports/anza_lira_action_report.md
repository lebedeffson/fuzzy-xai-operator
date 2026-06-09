# Scenario action report: anza_lira

- module: `ANZA-LIRA vessel segmentation`
- adapter_called: `True`
- output_type: `ExplanationArtifact + report`
- status: `fixture-certified`
- evidence_level: `repository-output-level`
- action: `audit_report`
- claim_scope: vessel-segmentation adapter route and report generation, not local model retraining

Численные `chi_R`, `chi_Auto` и `rho` для внешнего модуля не подставляются искусственно. Если адаптер не предоставляет полный структурный контур, сценарий фиксируется как audit/report-only.