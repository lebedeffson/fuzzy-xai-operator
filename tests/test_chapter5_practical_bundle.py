from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.chapter5_practical_bundle import generate


def test_bundle_evidence_is_auditable_and_does_not_invent_ptbxl(tmp_path: Path) -> None:
    output = tmp_path / 'reproducible'
    report = generate(output, seeds=(42,), n_boot=100)
    assert report['status'] == 'done'
    assert report['ptbxl_executed'] is False
    assert report['wdbc_splits'] == 1
    assert report['wdbc_heldout_cases'] == 456
    md = (output / 'CHAPTER5_EMPIRICAL_RESULTS.md').read_text(encoding='utf-8')
    assert 'PTB-XL real-waveform run: NOT EXECUTED' in md
    ru = (output / 'CHAPTER5_EMPIRICAL_RESULTS_RU.md').read_text(encoding='utf-8')
    assert '342' not in ru  # this one-seed test contains 114 test rows, not the three-seed publication
    assert 'канонический' in ru.lower()
    assert 'equal-guardrail baseline' in md.lower()
    assert (output / 'figures/fig_5_wdbc_auto_accept_coverage.png').is_file()
    assert (output / 'figures/fig_5_wdbc_critical_flag_accept.pdf').is_file()
    manifest = json.loads((output / 'manifest_sha256.json').read_text(encoding='utf-8'))
    assert manifest['ptbxl_executed'] is False
    for name, expected in manifest['artifacts'].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
