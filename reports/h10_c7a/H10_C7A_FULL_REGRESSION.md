# H10-C7A full regression

Command:

```bash
PYTHONPATH=experiments/h10_c3/src:framework/fuzzyxai:. \
  /home/lebedeffson/Code/venv/bin/python -m pytest -q
```

Result:

```text
777 passed, 6 skipped, 491 warnings in 239.55s
```

The warnings are existing scikit-learn feature-name warnings from optional
adapter and demonstration tests. No test failed.

The regression suite regenerates historical demonstration outputs as a test
side effect. Those generated differences were removed after the PASS result;
they are not part of the H10-C7A change set.
