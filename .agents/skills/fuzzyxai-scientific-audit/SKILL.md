---
name: fuzzyxai-scientific-audit
description: Audits FuzzyXAI Gamma, Delta, uncertainty, I_pre, rho, action, provenance, and model-specific evidence semantics. Use only for semantic verification of scientific operators.
disable-model-invocation: true
---

# FuzzyXAI scientific audit

For every audited number, verify computational source, formula, input values,
ExplainPlan parameters, semantic meaning, provenance node, human wording, and
missing/not-applicable handling. Reject fabricated values, direct Gamma without
T_ij, presentation loss presented as Delta, raw U_model presented as u_M, and
partial risk presented as full rho.
