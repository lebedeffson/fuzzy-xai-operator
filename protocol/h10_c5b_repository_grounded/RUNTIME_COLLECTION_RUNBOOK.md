# H10-C5b runtime collection runbook

Scientific method commit:
`7aa72a19a70bdb5eedea520742f269bc6c26aeea`.

Runtime collection must use an ephemeral Linux runner with a working Docker
daemon and no unrelated credentials. Every repository must have a prebuilt
image pinned by `@sha256:<digest>` in:

`/work/h10-c5b/H10_C5B_CONTAINER_IMAGES.json`.

The runner must provide Python 3.11 directly or `uv` for provisioning a
dedicated Python 3.11 virtual environment. GitHub's `setup-python` toolcache is
not assumed on non-Ubuntu self-hosted distributions.

The image must contain the buggy project's registered dependencies. Runtime
containers have no network, no added capabilities, `no-new-privileges`, a PID
limit, a memory limit, and a CPU limit. A Docker infrastructure exit code,
timeout, missing traceback, or pytest collection error is not a reproduced
incident.

The registered SWE-bench images execute tests with
`/opt/miniconda3/envs/testbed/bin/python`. Calling the base Conda interpreter
is an infrastructure error because it does not contain the incident's pytest
environment.

The runtime sandbox places Hypothesis state in writable `/tmp` and disables
pytest's optional cache. A failure while importing `conftest.py` or during test
collection is classified as infrastructure, even when pytest returns code 1
and emits Python frames. See
`H10_C5B_RUNTIME_PYTEST_ENVIRONMENT_AMENDMENT.json`.

ANSI terminal sequences are removed only from the text used to classify trace
completeness. Raw stdout, stderr, and traceback evidence remain byte-for-byte
unchanged. This rule is locked in
`H10_C5B_RUNTIME_TRACE_NORMALIZATION_AMENDMENT.json`.

If an immutable upstream image fails before pytest collection because its
dependency environment is no longer internally compatible, apply only the
prospectively recorded runtime correction in
`H10_C5B_RUNTIME_ENVIRONMENT_AMENDMENT.json`. The correction must be applied
uniformly to every development candidate from the affected repository, on top
of each exact base-image digest, using a SHA256-verified local wheel. Publish
and execute only the resulting manifest-digest image references. Preserve the
failed evidence and do not inspect Gold or method predictions before retrying.

Before execution, the collector applies the dataset's registered public
`test_patch` to the disposable sandbox and verifies its SHA256. The test patch
is a harness input needed to materialize `FAIL_TO_PASS`; it is not the Gold fix
patch and is never copied into the RouteAuditor manifest or scoring features.

## Development

Prepare the registered sources:

```bash
make h10-c5b-prepare \
  H10_C5B_SOURCE_DIR=/work/h10-c5b
```

Create the Gold-free runtime channel:

```bash
make h10-c5b-prepare-runtime-inputs \
  H10_C5B_MANIFEST=/work/h10-c5b/H10_C5B_DEVELOPMENT_MANIFEST.jsonl \
  H10_C5B_RUNTIME_DIR=/work/h10-c5b/runtime-development/inputs \
  H10_C5B_CONTAINER_IMAGES=/work/h10-c5b/H10_C5B_CONTAINER_IMAGES.json \
  H10_C5B_RUNTIME_SOURCE_DATASET=/work/h10-c5b/sources/development.parquet
```

Collect evidence:

```bash
make h10-c5b-collect-runtime \
  H10_C5B_MANIFEST=/work/h10-c5b/runtime-development/inputs/H10_C5B_RUNTIME_COLLECTION_MANIFEST.jsonl \
  H10_C5B_RUNTIME_COMMANDS=/work/h10-c5b/runtime-development/inputs/H10_C5B_RUNTIME_COMMANDS.json \
  H10_C5B_RUNTIME_DIR=/work/h10-c5b/runtime-development/evidence
```

The next gate requires `trace_complete_count == incident_count` and every
incident status to be `BUG_REPRODUCED_WITH_TRACE`. Otherwise generate the
replacement ledger and stop. The next candidate is selected from the same
repository by the locked SHA256 order without viewing Gold or method output.

After complete evidence, merge runtime paths into the full development
manifest, run development scoring, and create
`DEVELOPMENT_RUNTIME_LOCK.json`.

## Held-out

Held-out runtime collection is unavailable until the development runtime lock
exists and verifies against the frozen method commit. The held-out manifest
must contain at least 24 incidents from at least eight repositories.

To keep bounded storage on the dedicated runner, the collector removes the
exact held-out image reference after stdout, stderr, traceback, and their
digests have been recorded. This storage-only behavior is locked in
`H10_C5B_RUNTIME_STORAGE_AMENDMENT.json`; it never removes unrelated images or
containers and does not alter the runtime sandbox or method inputs.

The operational workflow can collect held-out runtime evidence, but it does
not invoke held-out scoring. Official held-out scoring requires a separate
authorization and a subsequently locked enriched manifest. No such
authorization or scoring command is part of this branch.
