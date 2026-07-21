# Strong confirmatory closure

This directory defines the research boundary for `feat/strong-confirmatory-closure`.

The original statuses `H3-original = not_supported`, `H5-P-original = not_supported`, and `H6-general = not_supported` are immutable. New H3-v2, H5-A, H6-A/B, H7, H8, and H9 measurements begin as formative evidence and cannot replace those results.

`confirmatory_protocol_lock.json` must not exist until all of the following are present and verified:

- sealed manifests for at least two tabular, one image, one text, and one time-series confirmatory dataset;
- a completed formative AI pre-review gate that is explicitly not labeled external validation;
- frozen protocol, endpoints, baselines, exclusions, power analysis, and test hashes.

Commands:

```bash
make strong-confirmatory-smoke
make strong-confirmatory-formative
make chapter4-formative-shell
make strong-confirmatory-lock   # expected to exit 2 while prerequisites are absent
make chapter4-final             # expected to exit 2 while external gates are open
```

No stable release or final Chapter 4 may be produced from formative evidence.
