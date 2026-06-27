# Scenario action report: synt_isic_timeshap

- module: `SYNT-ISIC / Time-SHAP`
- adapter_called: `True`
- output_type: `ExplanationArtifact + report`
- status: `fixture-certified`
- evidence_level: `repository-output-level`
- action: `audit_report`
- claim_scope: temporal explanation route and report generation, not local dermatology validation

Численные `chi_R`, `chi_Auto` и `rho` для внешнего модуля не подставляются искусственно. Если адаптер не предоставляет полный структурный контур, сценарий фиксируется как audit/report-only.