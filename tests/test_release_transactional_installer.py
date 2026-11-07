"""Verify the combined chapter 1-5 source edit is fail-closed and atomic."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.test_doctoral_patcher import source_snapshots
import install_fuzzyxai_upgrade as installer

REGISTRY_CONTENT = '''from __future__ import annotations
def build_medical_ecg_signal_route(adapted: AdaptedInput) -> OperatorRoute:
    return _route("control")


def build_gd_anfis_shap_route(adapted: AdaptedInput) -> OperatorRoute:
    return _route("control")
'''
ADAPTER_CONTENT = '''from __future__ import annotations
class SignalQualityAdapter(BaseAdapter):
    required_fields = ("sampling_rate", "quality_score", "noise_segments", "missing_fragments", "baseline_instability")
'''


def setup_root(root: Path) -> dict[str, str]:
    source = {**source_snapshots(), installer.REGISTRY: REGISTRY_CONTENT, installer.ADAPTER: ADAPTER_CONTENT}
    for name, text in source.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return source


def test_combined_preflight_never_writes_and_apply_is_idempotent(tmp_path):
    originals = setup_root(tmp_path)
    before, changed, report = installer.stage(tmp_path)
    assert report["original_files_checked"] == 8
    assert report["files_to_update"] == 8
    assert len(before) == len(changed) == 8
    assert {name: (tmp_path/name).read_text(encoding="utf-8") for name in originals} == originals
    result = installer.apply(tmp_path)
    assert result["status"] == "applied"
    assert len(result["applied"]) == 8
    assert installer.stage(tmp_path)[2]["files_to_update"] == 0
    assert installer.apply(tmp_path)["status"] == "already_applied"
    for name in originals:
        if name.endswith('.py'):
            ast.parse((tmp_path / name).read_text(encoding='utf-8'))


def test_missing_target_rejects_transaction_before_any_write(tmp_path):
    originals = setup_root(tmp_path)
    (tmp_path / installer.ADAPTER).unlink()
    with pytest.raises(FileNotFoundError):
        installer.apply(tmp_path)
    for name, text in originals.items():
        if name != installer.ADAPTER:
            assert (tmp_path / name).read_text(encoding="utf-8") == text


def test_unexpected_old_route_rejects_before_any_write(tmp_path):
    originals = setup_root(tmp_path)
    bad = tmp_path / installer.REGISTRY
    bad.write_text('def unexpected():\n    pass\n', encoding='utf-8')
    with pytest.raises(RuntimeError, match='ECG scenario route changed'):
        installer.apply(tmp_path)
    for name, original in originals.items():
        if name != installer.REGISTRY:
            assert (tmp_path / name).read_text(encoding='utf-8') == original


def test_post_write_failure_rolls_back_all_writes(tmp_path, monkeypatch):
    originals = setup_root(tmp_path)
    original_replace = installer._replace_bytes
    calls = 0
    def fail_on_fourth(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise PermissionError('injected failure')
        return original_replace(*args, **kwargs)
    monkeypatch.setattr(installer, '_replace_bytes', fail_on_fourth)
    with pytest.raises(PermissionError, match='injected failure'):
        installer.apply(tmp_path)
    assert {name: (tmp_path/name).read_text(encoding='utf-8') for name in originals} == originals


def test_check_cli_writes_nothing_even_when_receipt_path_given(tmp_path, capsys):
    originals = setup_root(tmp_path)
    destination = tmp_path / 'should-not-exist.json'
    assert installer.main(['--root', str(tmp_path), '--check', '--receipt', str(destination)]) == 0
    assert not destination.exists()
    assert {name: (tmp_path/name).read_text(encoding='utf-8') for name in originals} == originals
