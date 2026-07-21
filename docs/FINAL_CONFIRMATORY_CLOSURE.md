# Final confirmatory closure

This cycle starts from commit `68e6edcfa867b48684b89d98dd74b5fe4794ef55` and does not rewrite frozen
H1-H6 results. It adds the contracts and gates required to move from practical-controller formative evidence to a
sealed independent confirmation.

## Current boundary

- The practical controller and previous formative evidence remain unchanged.
- The new protocol fixes the primary endpoint, 20% review budget, cost profile, false-block ceiling and a maximum of
  three formative iterations.
- Five independent datasets are sealed locally. Raw/processed data, encrypted labels and OOF rows with development
  targets are excluded from Git; licenses, manifests and hashed split identities are tracked.
- The 100,000-event shadow stream is a controlled formative replay of realistic incidents, not observed production
  failures.
- The AI run-2 archive is reviewer input only. It contains no scores and is not human or expert validation.
- Final lock, confirmatory statistics, final Chapter 4 and stable release fail closed while external inputs are absent.

## Sealed inputs

The preparation command writes these files under `study/final_confirmatory_closure/`:

```text
confirmatory_dataset_manifest.input.json
confirmatory_split_manifest.input.json
```

Each dataset row must satisfy `fuzzyxai.final_closure.SealedDataset` and reference a workspace-local `.enc` label vault
whose SHA256 matches `label_vault_sha256`. The split manifest must declare:

```json
{
  "tuning_runner_can_read_test_labels": false,
  "test_labels_loaded_by_tuning": false,
  "controller_feature_source": "out_of_fold_train_development_only",
  "test_identity_visibility_during_tuning": "hash_only",
  "oof_object_hashes": ["<sha256>"],
  "sealed_test_object_hashes": ["<sha256>"]
}
```

Run `make final-local-data-check` on the controlled workstation. The current manifest deliberately leaves model disagreement, shift and every real
route/explanation channel missing, so the protocol lock remains closed rather than substituting synthetic values.

Public CI runs `make final-release-check` against the committed manifests and audits. It must not require excluded
raw data, encrypted vaults or development-label OOF rows.

Run the self-contained blind ZIP in a temporary clean chat, save all 720 raw rows as `reviews.jsonl` together with
`session_metadata.json`, and import them with:

```bash
make final-ai-run2-import AI_RUN2_INPUT=<directory>
make final-ai-run2-report
```

Only after run 2 passes and P0/P1 contain real OOF evidence may `make final-controller-freeze` create the one-way lock.

## Technical prelock check

```bash
make final-release-check
make final-prelock-archive
```

The prelock archive is a shareable technical artifact. Its `BOUNDARY.json` explicitly forbids confirmatory and stable
release claims.
