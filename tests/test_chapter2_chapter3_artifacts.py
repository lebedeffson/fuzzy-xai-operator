from __future__ import annotations

import csv
import json

from fuzzyxai.experiments.chapter2_calibration import run as run_calibration
from fuzzyxai.experiments.chapter2_equal_raw_structure import run as run_equal_raw
from scripts.build_chapter3_artifacts import run as run_chapter3


def test_chapter2_calibration_outputs_constants(tmp_path) -> None:
    payload = run_calibration(tmp_path / 'chapter2', n_pairs=12)
    assert payload['status'] == 'ok'
    rows = list(csv.DictReader((tmp_path / 'chapter2/calibration_constants.csv').open(encoding='utf-8')))
    assert {r['constant_name'] for r in rows} == {'c_H_cal', 'c_O_cal', 'c_K_cal'}


def test_equal_raw_structure_has_expected_modes(tmp_path) -> None:
    payload = run_equal_raw(tmp_path / 'chapter2')
    assert 'equal_raw_structure' in payload['modes']
    rows = payload['rows']
    fx = [r for r in rows if r['mode'] == 'equal_raw_structure' and r['policy'] == 'fuzzyxai_observer'][0]
    assert fx['missed_critical'] == 0
    assert fx['certified_path_access'] is True


def test_chapter3_artifacts_outputs_specs(tmp_path) -> None:
    payload = run_chapter3(tmp_path / 'chapter3')
    assert payload['status'] == 'ok'
    assert (tmp_path / 'chapter3/dataset_roles_summary.csv').exists()
    spec = json.loads((__import__('pathlib').Path('dissertation_artifacts/diagram_specs/chapter3/fig_3_8_chi_auto_sample113.json')).read_text(encoding='utf-8'))
    assert spec['figure_id'] == 'fig_3_8'
