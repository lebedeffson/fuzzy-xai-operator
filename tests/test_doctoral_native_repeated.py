"""Scientific reporting contracts for repeated exploratory split protocol."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.doctoral_native_repeated import run


def test_two_seed_protocol_keeps_source_artifacts_and_checksums(tmp_path):
    folder=tmp_path/'reports'
    result=run(folder,seeds=(17,42),features=5)
    assert result['seed_list']==[17,42]
    assert result['critical_input_stress_cases_total']==48
    assert result['critical_input_stress_all_blocked']
    assert len(result['per_seed_model_metrics'])==6
    assert len(result['certificate_valid_fraction_per_split'])==2
    assert 'not independent' in ' '.join(result['limitations']).lower()
    assert (folder/'heldout_roc.png').is_file()
    assert (folder/'roc_auc_per_seed.pdf').is_file()
    assert 'Нечёткая' in (folder/'CHAPTER5_NATIVE_FUZZY_RU.md').read_text(encoding='utf-8')
    manifest=json.loads((folder/'manifest_sha256.json').read_text(encoding='utf-8'))
    assert len(manifest['files']) > 12
    for relative, expected in manifest['files'].items():
        assert hashlib.sha256((folder/relative).read_bytes()).hexdigest()==expected


def test_seed_validation_rejects_duplicate_seed_or_one_seed(tmp_path):
    with pytest.raises(ValueError,match='two distinct'):
        run(tmp_path/'a',seeds=(42,42))
    with pytest.raises(ValueError,match='two distinct'):
        run(tmp_path/'b',seeds=(42,))
