# H10-C7-R5V audit report

Status: `H10_C7_R5V_BLOCKED_AUDIT`

Scientific result: `NOT_EVALUATED`.

## Target alignment

The registered pair target requires an exact file, symbol, and evaluation
contract-family match for the same candidate. The audit did not reproduce the
preregistered control counts.

| Quantity | Expected | Observed |
| --- | ---: | ---: |
| Correct top-1 pairs | 11 | 10 |
| Correct pairs in top-3 | 14 | 12 |
| Misaligned old positive labels | 3 | 4 |

The four old labels that differ from the strict top-1 pair target are:

- `bugsinpy-black-23`
- `bugsinpy-pandas-73`
- `bugsinpy-thefuck-22`
- `bugsinpy-tornado-8`

For `bugsinpy-black-23`, rank 3 contains `black.py::whitespace`, but its
predicted evaluation contract is `CONFIGURATION`; the Gold atom for that
symbol is `DATA_CONTRACT`. For `bugsinpy-pandas-73`, rank 1 contains
`pandas/core/frame.py::DataFrame`, but the predicted evaluation contract is
`DATA_CONTRACT`; the Gold atom for that symbol is `CONFIGURATION`. Neither is
a correct candidate-contract pair under the locked target.

## Evidence specificity

The 30 replay incidents contain 25,596 dynamic events:

| Event kind | Count |
| --- | ---: |
| `call` | 12,724 |
| `coverage` | 12,768 |
| `traceback_frame` | 104 |

No `argument_value`, `return_value`, `assertion_operand`, `last_writer`, or
`value_flow` event is present. Candidate-specific value provenance is
therefore available for `0/90` top-3 pairs.

Assertion text, exception text, and unbound direct contract observations are
recorded as incident-level context. They are not counted as evidence for every
candidate.

## Decision

The protocol requires the verifier run to stop when the target-alignment
counts differ or when candidate-specific value provenance is unavailable for
the majority of pairs. Both stop conditions hold.

V0 and V1 were not executed. No threshold was selected. R5 retrieval was not
modified, no held-out set was created, and no scientific result was formed.

The next admissible operation is a one-time candidate-specific trace
collection that records assertion operands, argument and return values, last
writers, and exception origins. A verifier replay is allowed only after a new
trace-availability audit and an explicit protocol amendment that resolves the
incorrect expected pair counts without changing the strict pair target.
