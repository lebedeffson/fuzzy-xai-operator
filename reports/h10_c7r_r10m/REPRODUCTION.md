# H10-C7R R10M reproduction

1. Extract the locked causal recollection artifact.
2. Mount the two snapshots at the paths recorded in `R10M_MODEL_LOCK.json`.
3. Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.
4. Run `make h10-c7r-r10m-test`.
5. Run `make h10-c7r-r10m-development` with the extracted root and a writable model-score cache.

The scorer checkpoints each incident and never passes Gold to a retrieval or model feature channel.
