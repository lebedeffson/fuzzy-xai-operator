# Full Regression Report

Date: 2026-07-25

## Configuration

The repository contains 565 collected tests in 182 test files. The canonical
import path used for the run was:

```text
PYTHONPATH=experiments/h10_c3/src:framework/fuzzyxai:.
```

The monolithic process was unsuitable on this workstation because the kernel
OOM killer terminated it at a 3.9 GiB process peak while other user workloads
were active. The suite was therefore executed file by file in fresh Python
processes. This is an execution-environment workaround only; no test selection,
assertion, fixture, or scientific implementation was weakened.

## Result

```text
collected: 565
passed: 560
skipped: 4
environment-blocked: 1
failed after fixes: 0
```

The environment-blocked test is:

```text
tests/test_dataset_benchmark.py::
test_registry_programs_observer_accuracy_not_applicable
```

Its legacy implementation applies dense `pandas.get_dummies()` expansion to a
local 10,007-row registry table containing several nearly unique text columns.
The isolated test was killed by the kernel at approximately 4 GiB. The two
other tests in the same file passed in isolated processes. This test is outside
H10-C4 and does not read or validate H10-C4 evidence.

The sharded regression exposed one stale contract assertion: the public
operator-manifest test expected 33 entries although H10-C4 registered two new
defended operators. The expected count was corrected to 35, and
`tests/test_public_framework_api.py` then passed with `6 passed`.

## H10-C4 Gates

```text
make h10-c4-test: 30 passed, Ruff PASS
make h10-c4-verify: PASS
H10-C3 immutable artifacts: 54/54 PASS
manuscript claim lint: PASS
SHA256 manifest: PASS
```

No scientific result, cost model, held-out scenario, baseline, or decision rule
was changed in response to regression testing.
