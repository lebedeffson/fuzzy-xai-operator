# H10-C7R runtime evidence

- Status: `H10_C7R_RUNTIME_EVIDENCE_COMPLETE`
- Complete incidents: 40
- Complete repositories: 30
- Failed or unavailable reserve candidates: 9
- Network during execution: none
- Runtime manifest SHA256:
  `67925f6a974abbf9ee6fb575619e79a315b42f7398a19a263bd584d52dbec2b4`
- Runtime ledger SHA256:
  `9a6ae278abac8037506fb809c85bd64bffe9b66fb3e231b3eb24851cfc2478f6`
- Runtime report SHA256:
  `10f14b9f7935574907c7465350e498bcef2900dfbff8e5982d32c2e415bc350d`
- Image-availability lock SHA256:
  `1350607e09923092b9b4aa459b28a76c748201e44021b70c23a175bb7ac9597c`
- Sealed Gold SHA256:
  `3bf81a63b1bfaad84162744b73b3126e841ce7233f51328b8127061a1046a83e`
- SWE-bench-Live source SHA256:
  `1202acd70b011211ab552087ecc69d3c85fccccbfabeb19895a7f20c72c6ca4f`

The collector used digest-pinned images, disabled container networking, and
kept observable runtime evidence physically separate from Gold until scoring
authorization. Scoring was opened once and was not repeated during reporting,
testing, or packaging.
