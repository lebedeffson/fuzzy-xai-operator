# H10-C5b runtime collection runbook

Scientific method commit:
`7aa72a19a70bdb5eedea520742f269bc6c26aeea`.

Runtime collection must use an ephemeral Linux runner with a working Docker
daemon and no unrelated credentials. Every repository must have a prebuilt
image pinned by `@sha256:<digest>` in:

`/work/h10-c5b/H10_C5B_CONTAINER_IMAGES.json`.

The image must contain the buggy project's registered dependencies. Runtime
containers have no network, no added capabilities, `no-new-privileges`, a PID
limit, a memory limit, and a CPU limit. A Docker infrastructure exit code,
timeout, missing traceback, or pytest collection error is not a reproduced
incident.

The registered SWE-bench images execute tests with
`/opt/miniconda3/envs/testbed/bin/python`. Calling the base Conda interpreter
is an infrastructure error because it does not contain the incident's pytest
environment.

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

The operational workflow can collect held-out runtime evidence, but it does
not invoke held-out scoring. Official held-out scoring requires a separate
authorization and a subsequently locked enriched manifest. No such
authorization or scoring command is part of this branch.
