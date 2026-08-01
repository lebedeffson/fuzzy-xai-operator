# H10-C7R R9 reproduction

The development replay uses stored graphs and runtime events only. It does not
start containers, install project dependencies, rerun failing tests, download
models, or recalculate official H10-C7R-v1 scoring.

```bash
make h10-c7r-r9-test \
  H10_C3_PYTHON=/home/lebedeffson/Code/venv/bin/python

make h10-c7r-r9-development \
  H10_C3_PYTHON=/home/lebedeffson/Code/venv/bin/python \
  H10_C7R_OPERATION_ROOT=/home/lebedeffson/.local/share/fuzzyxai/h10-c7r
```

The second command forces `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`. The operation root supplies the previously collected
observable manifest and the now-disclosed development Gold.

Expected final status:

```text
H10_C7R_R9_DEVELOPMENT_NO_GO
Scientific result: NOT_EVALUATED
New held-out created/scored: false/false
```
