# H10-C3 v23.1 cost stability validation

## Scope

- Open development and protocol-validation evidence only.
- No sealed dataset was created or opened.
- H10-C3a and H10-C3b remain `NOT_EVALUATED_CONFIRMATORY`.

## Root cause

The historical sensitivity helper scaled only candidate IDs with `greedy-` or `direct-` prefixes. It therefore performed a heterogeneous perturbation while reporting a global multiplier.

## Corrected global-scale audit

- Status: `PASS`.
- Case-scale checks: `5040`.
- Changed Gold cut sets: `0`.
- Changed Full H10 cuts: `0`.
- Changed frozen-baseline cuts: `0`.
- Changed membership values: `0`.
- Changed normalized regret values: `0`.
- Cache-key collisions: `0`.
- Registered factors: `0.01, 0.1, 0.8, 1.0, 1.2, 10.0, 100.0`.
- H10-C3a effect at every global factor: `0.76699029126214`.
- H10-C3b effect at every global factor: `0.33980582524272`.

## Non-uniform sensitivity

- Status: `PASS`.
- `non_uniform:node_down_edge_up`: H10-C3a `0.23300970873786`, H10-C3b `0.33980582524272`, false certification `0.0`, new critical violations `0.0`.
- `non_uniform:node_up_edge_down_human_up`: H10-C3a `0.23300970873786`, H10-C3b `0.33980582524272`, false certification `0.0`, new critical violations `0.0`.

## Reproduction boundary

- Base effect H10-C3a: `0.76699029126214`.
- Base effect H10-C3b: `0.33980582524272`.
- Frozen baseline selection was reused without reselection.
- Historical `artifacts/h10_c3_v23` files were not modified.
- The historical blocked gate remains preserved as historical evidence.

## Quality checks

- Focused H10/diagnostics tests: `62 passed`.
- Full regression: `533 passed, 4 skipped`.
- Changed-scope Ruff: `PASS`.
- Repository-wide Ruff baseline: `313` historical findings.
