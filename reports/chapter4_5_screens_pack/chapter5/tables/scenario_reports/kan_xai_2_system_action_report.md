# Scenario action report: kan_xai_2_system

- module: `KAN-XAI 2.0 System`
- adapter_called: `True`
- output_type: `ExplanationArtifact + report`
- status: `fixture-certified`
- evidence_level: `repository-output-level`
- action: `audit_report`
- claim_scope: medical image explanation adapter route, not local model retraining

Численные `chi_R`, `chi_Auto` и `rho` для внешнего модуля не подставляются искусственно. Если адаптер не предоставляет полный структурный контур, сценарий фиксируется как audit/report-only.