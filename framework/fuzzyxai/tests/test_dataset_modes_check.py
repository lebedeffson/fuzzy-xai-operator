from __future__ import annotations

import subprocess
import sys


def test_dataset_modes_check_has_expected_columns() -> None:
    out = subprocess.check_output([sys.executable, 'examples/check_dataset_modes.py'], text=True)
    lines = [line for line in out.strip().splitlines() if line.strip()]
    assert lines
    assert lines[0] == 'dataset_mode\tstatus\trows\tdomain\tvalidates\tdetails'
    assert any(line.startswith('breast_cancer\tREADY\t') for line in lines[1:])
