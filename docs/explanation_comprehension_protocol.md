# Explanation comprehension protocol

Status: `planned_not_run`.

This protocol does not report user-study findings. It defines the gate that must be run before claiming that FuzzyXAI explanations are understandable to domain users.

## Participants

- 3-5 software developers integrating predictive models;
- 3-5 domain specialists who did not implement FuzzyXAI;
- optional independent auditor familiar with model governance.

## Domain-user tasks

After reading only the first-level cards for controlled object 85, the ANFIS fixture, and the research-only medical similarity fixture, each participant must identify:

1. what the model decided;
2. why it made that decision;
3. what creates doubt or limits trust;
4. whether the result is suitable for automatic use;
5. what should be done next.

Participants must not need to decode rule IDs, claim IDs, E0-E5, operator symbols, or raw action codes. For the medical fixture they must additionally state what the 89% value compares and what it does not mean.

## Evidence-disclosure tasks

After opening technical details, model integrators and auditors identify native, surrogate, and missing channels; trace one card to its claims and evidence; and locate the relevant model/checkpoint provenance. These tasks evaluate disclosure, not first-level domain comprehension.

## Measured outcomes

- task correctness, reported separately for every question;
- time to locate supporting evidence;
- unsupported-inference count;
- System Usability Scale as a descriptive secondary measure;
- qualitative failure categories.

Record separately whether a participant introduced an unsupported causal, diagnostic, or probability interpretation.

## Acceptance gate

No claim of demonstrated comprehensibility is allowed until the study is run, raw responses are archived, scoring is independently checked, and the status changes from `planned_not_run` to a dated result. At minimum, every participant must identify the decision and action correctly, and at least 80% must answer all five domain-user questions without an unsupported inference.
