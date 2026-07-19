# Explanation comprehension protocol

Status: `planned_not_run`.

This protocol does not report user-study findings. It defines the gate that must be run before claiming that FuzzyXAI explanations are understandable to domain users.

## Participants

- 3-5 software developers integrating predictive models;
- 3-5 domain specialists who did not implement FuzzyXAI;
- optional independent auditor familiar with model governance.

## Tasks

For controlled object 85, the ANFIS fixture, and the research-only medical similarity fixture, each participant must identify:

1. the model decision and recommended action;
2. the strongest supporting and contradicting claims;
3. which channels are native, surrogate, and unavailable;
4. what changed during training, if training evidence exists;
5. what the similarity score compares and what it does not mean;
6. the evidence source for one selected claim;
7. one limitation that prevents automatic use.

## Measured outcomes

- task correctness, reported separately for every question;
- time to locate supporting evidence;
- unsupported-inference count;
- System Usability Scale as a descriptive secondary measure;
- qualitative failure categories.

## Acceptance gate

No claim of demonstrated comprehensibility is allowed until the study is run, raw responses are archived, scoring is independently checked, and the status changes from `planned_not_run` to a dated result.
