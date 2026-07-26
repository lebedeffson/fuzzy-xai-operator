# H10-C5c development data collection

H10-C5c remains a prospective development cycle. The collection code in this
release does not create a scientific result by itself.

## Source boundary

The development source is a locally checked-out, commit-pinned copy of
BugsInPy. The prospective lock pins
`11c5f1eea954a42132cfd06bf257766a7963e0fd`; materialization rejects any other
commit and any dirty benchmark checkout. The preparer reads `project.info`,
`bug.info`, `bug_patch.txt` and
`run_test.sh` without evaluating shell assignments. It excludes all official
H10-C5b held-out repositories and all non-Python-3 incidents. Selection is a
locked deterministic balanced round-robin: 30 incidents, at least eight
repositories and no more than four incidents per repository.

The benchmark checkout commit, each upstream project commit, the buggy and
fixed revisions and metadata-file SHA256 values are written to the source
registry. Local checkout paths are not serialized into the source or selection
registries. Gold patches and fixed-source snapshots remain outside the public
incident object and are opened only by the development scorer after the
prediction has been produced.

## Materialization

```bash
make h10-c5c-prepare-bugsinpy \
  H10_C5C_BUGSINPY_ROOT=/absolute/path/to/BugsInPy \
  H10_C5C_SOURCE_DIR=/absolute/path/to/h10-c5c-development \
  H10_C5C_REPOSITORY_CACHE=/absolute/path/to/repository-cache \
  H10_C5C_ALLOW_NETWORK=1
```

The materializer follows the benchmark checkout boundary: registered exposing
tests are taken from the fixed revision and overlaid onto the buggy source snapshot
before runtime collection. Their paths and SHA256 values are recorded separately;
the production fix patch remains unavailable to the diagnostic method until scoring.

The command produces:

- `H10_C5C_DEVELOPMENT_UNCOLLECTED.jsonl`;
- `H10_C5C_COMMAND_REGISTRY.json`;
- `H10_C5C_SOURCE_REGISTRY.json`;
- `H10_C5C_SELECTION_REPORT.json`;
- commit-pinned buggy source snapshots and post-prediction Gold files.

The uncollected manifest is not scoreable. Every row has
`runtime_evidence_status=PENDING_COLLECTION`.

## Runtime evidence

Runtime collection must be executed inside a compatible project environment.
The collector copies each buggy snapshot to a temporary sandbox, instruments
the registered Python or pytest command and records events with a test ID:
function coverage, project-internal calls and exact project traceback frames.
It also stores stdout, stderr and assertion differences. Gold fields are never
written to runtime event streams.

```bash
make h10-c5c-collect-runtime \
  H10_C5C_UNCOLLECTED_MANIFEST=/absolute/path/to/H10_C5C_DEVELOPMENT_UNCOLLECTED.jsonl \
  H10_C5C_COMMAND_REGISTRY=/absolute/path/to/H10_C5C_COMMAND_REGISTRY.json \
  H10_C5C_RUNTIME_DIR=/absolute/path/to/h10-c5c-runtime
```

`H10_C5C_ALLOW_SETUP=1` creates an isolated virtual environment with the exact
registered Python major.minor, installs the registered requirements file and then
runs the registered BugsInPy setup script. It is disabled by default because this
stage may download and execute external project dependencies. A failed virtual
environment, dependency installation or setup script produces
`ENVIRONMENT_SETUP_FAILED`; tests are not executed in a partially prepared
environment.

`H10_C5C_RUNTIME_COMPATIBILITY_AMENDMENT_002.json` was locked after three
failed infrastructure attempts and before any successful readiness or
development scoring. It registers `pip==20.3.4`, `setuptools==44.1.1` and
`wheel==0.36.2`, plus pip's legacy resolver and disabled build isolation, for
the same fixed 30 incidents. These compatibility settings address historical
2019-2020 package metadata; they do not change incident selection, diagnostic
inputs, Gold, method weights, abstention, endpoints or scientific gates. The
readiness verifier requires every command record to match the amendment
exactly.

`H10_C5C_RUNTIME_INSTALLATION_AMENDMENT_003.json` records a second
infrastructure-only correction after run `30221595604`: requirements are
installed independently, the registered setup script is executed, and the
independent installation pass is repeated. This is the order implemented by
the locked BugsInPy `bugsinpy-compile` script. Unavailable optional or
platform-specific requirements are retained as `PARTIAL` evidence rather than
preventing the exposing test from running. Setup-script failures remain
fail-closed. The selected incidents, source snapshots, tests, diagnostic
method and gates are unchanged.

An incident is marked `BUG_REPRODUCED_WITH_TRACE` only when its registered test
fails and the event stream contains both a project executed slice and a
project traceback frame. The development scorer rejects any other status.

## Development readiness

Before any development scoring, the enriched manifest is bound to the command
registry, source registry and runtime evidence report. The verifier checks the
locked incident/repository counts, the per-repository cap, exclusion of H10-C5b
held-out repositories, complete per-test coverage and traceback evidence,
source-patch hashes and zero forbidden Gold keys in runtime streams.

```bash
make h10-c5c-development-readiness \
  H10_C5C_DEVELOPMENT_MANIFEST=/absolute/path/to/H10_C5C_DEVELOPMENT_RUNTIME_ENRICHED.jsonl \
  H10_C5C_COMMAND_REGISTRY=/absolute/path/to/H10_C5C_COMMAND_REGISTRY.json \
  H10_C5C_SOURCE_REGISTRY=/absolute/path/to/H10_C5C_SOURCE_REGISTRY.json \
  H10_C5C_RUNTIME_REPORT=/absolute/path/to/H10_C5C_RUNTIME_EVIDENCE_REPORT.json \
  H10_C5C_READINESS_REPORT=/absolute/path/to/H10_C5C_DEVELOPMENT_READINESS.json
```

The readiness report remains `scientific_result=NOT_EVALUATED`. A failed check
returns a non-zero exit code and blocks the development runner.

## Development scoring

Only a complete enriched manifest and its passing readiness report may be
passed to:

```bash
make h10-c5c-run-development \
  H10_C5C_DEVELOPMENT_MANIFEST=/absolute/path/to/H10_C5C_DEVELOPMENT_RUNTIME_ENRICHED.jsonl \
  H10_C5C_READINESS_REPORT=/absolute/path/to/H10_C5C_DEVELOPMENT_READINESS.json
```

The result may be `H10_C5C_DEVELOPMENT_GATE_PASS` or
`H10_C5C_DEVELOPMENT_GATE_FAIL`. Neither state is a held-out scientific result.
The development scorer creates a repair plan but does not execute it; rows are
marked `repair_execution_status=NOT_RUN_DEVELOPMENT_SCORING_ONLY`, while
regression and recertification remain `NOT_EVALUATED`. A disjoint held-out
protocol is forbidden until every locked development gate passes.

## Exact Python runtime and remote collection

`H10_C5C_DATA_COLLECTION_AMENDMENT_001.json` requires each instrumented
command to run under the registered Python major.minor version. The collector
accepts an optional JSON interpreter map:

```json
{
  "3.8": "/absolute/path/to/python3.8",
  "3.11": "/absolute/path/to/python3.11"
}
```

Pass it with `H10_C5C_INTERPRETER_MAP` to `make h10-c5c-collect-runtime`.
A missing or mismatched interpreter produces `PYTHON_RUNTIME_UNAVAILABLE`;
the readiness gate then fails. Silent fallback to the controller interpreter
is prohibited. Collection may use a bounded worker pool through
`H10_C5C_WORKERS`; output order remains the locked manifest order and every
incident writes to an independent evidence directory.

The `H10-C5c prospective development implementation` GitHub Actions workflow
has a manual `collect-development` operation. It accepts only the exact
40-character BugsInPy commit from the prospective lock, materializes the locked
30-incident development selection,
reads the exact Python versions required by that selection, reuses explicitly
provided interpreters or provisions missing versions in isolated Conda prefixes,
collects runtime evidence, runs readiness and, only after readiness PASS, executes
development-only scoring. The workflow always uploads an operational evidence
archive, including failed readiness runs. It never creates or scores an
H10-C5c held-out set, and the scientific result remains `NOT_EVALUATED`.


## Evidence binding

The readiness gate verifies that the runtime report SHA256 bindings match the exact
enriched manifest and command registry supplied for scoring. It also requires an
exact one-to-one incident set across the manifest, command registry, source registry
and runtime report, and rechecks the materialized exposing-test hashes. Therefore an
evidence report from another collection cannot be substituted silently.
