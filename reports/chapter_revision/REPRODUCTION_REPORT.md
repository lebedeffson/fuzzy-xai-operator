# Chapter 4 Q1 Evidence Expansion Reproduction

## Frozen boundaries

- Protocol lock commit: `989502f`
- Scientific implementation commit: `91eae88`
- Telemetry-only follow-up: `2e1e2ea`
- Evidence commit: `5066d44`
- Final verified code commit before packaging: `b317c6f`
- Chapter document modified: `false`

## Results

- H10-C5: `H10_C5_NOT_SUPPORTED`
- H10-C6: `H10_C6_SUPPORTED`
- Multimodal route validation: `MULTIMODAL_ROUTE_VALIDATION_PASS`
- H9 end-to-end target: `H9_E2E_TARGET_NOT_MET`
- H10-C3/H10-C4 legacy evidence integrity: `PASS`
- Claim lint: `PASS`
- Operator manifest: `PASS` (`39` registered operators)

H10-C5 screened 300 SWE-bench Lite candidates and selected 26 incidents from
10 repositories. The held-out repository comparison against the strongest
greedy baseline had difference `0.0` with a confidence interval crossing zero.
Local project execution was not completed, so natural recovery claims are
disabled.

## Verification

Executed from detached clean worktree at `b317c6f`:

```text
580 passed, 5 skipped, 491 warnings in 220.82s
```

Focused prospective tests:

```text
17 passed, 1 skipped
```

The skipped focused test requires the separately locked SWE-bench parquet. Its
actual source run passed SHA256 verification:

```text
7a21f37b8bc179c7db5beeb14e88ac538ba283455c776e6b2535bbfb6e3551b4
```

No human-time, user-comprehension, industrial-effectiveness, or natural
recovery advantage is claimed.
