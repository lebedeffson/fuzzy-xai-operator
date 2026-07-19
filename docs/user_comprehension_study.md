# User comprehension study

Status: `planned_not_run`.

The release candidate contains the protocol and scoring template, not fabricated participant results. At least six independent participants are required: three model integrators and three domain specialists who did not implement FuzzyXAI.

For object 85, ANFIS, and the medical research fixture, participants first see only `HumanExplanation` cards and identify the decision, reasons, concern, reliability, and action. Technical disclosure is tested separately: model integrators trace one card to claims/evidence and identify native, surrogate, and missing channels. The medical task explicitly checks that mask IoU is not interpreted as diagnosis probability.

Acceptance requires archived anonymized raw responses, independent scoring, per-question correctness, task time, unsupported-inference count, qualitative failure categories, and documented UI/text revisions. Until then no claim of demonstrated comprehensibility is allowed.
