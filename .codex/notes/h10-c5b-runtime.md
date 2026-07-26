# H10-C5b runtime collection

- Pin every runtime image by manifest digest.
- Use an `incident_id` image key when a split contains multiple incidents from one repository; a repository key is safe only for a single incident.
- SWE-bench instance images run tests with `/opt/miniconda3/envs/testbed/bin/python`.
- On an Arch self-hosted runner, provision host Python 3.11 with `uv --seed`; `actions/setup-python` does not publish Arch rolling builds. Keep uv cache and interpreter storage under the writable runtime source directory because the runner service protects `$HOME` as read-only.
- Pass the registered split parquet as `H10_C5B_RUNTIME_SOURCE_DATASET` so the FAIL_TO_PASS patch is checksum-bound before execution.
- Runtime collection may change infrastructure only; the 11 files in `METHOD_LOCK.json` must remain byte-identical.
- Upload only `runtime-development` and `runtime-held_out`; a broad `runtime-*` glob also captures the local Python toolchain.
- With `PrivateTmp=true`, Docker cannot bind a collector sandbox created under the runner's private `/tmp`; set `TMPDIR` to a mode-0700 directory under the Docker-visible runtime source root.
