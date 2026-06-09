from __future__ import annotations

from fuzzyxai.adapters.beacon_xai import BeaconXAIAdapter, DEFAULT_FIXTURE, sha256_file


def test_beacon_xai_adapter_is_executable_with_pinned_source() -> None:
    raw = BeaconXAIAdapter().explain()
    assert raw['status'] == 'ok'
    assert raw['source_repo'] == 'https://github.com/fims9000/BeaconXAI'
    assert raw['source_commit'] == '660366759fb0b5045491a9f7b9fa50745afe44db'
    assert raw['fixture_sha256'] == sha256_file(DEFAULT_FIXTURE)
    channels = raw['channels']
    assert channels['R_k']
    assert channels['alpha_k']
    assert channels['eta_k']
    assert channels['u_k']
    assert channels['tau_k']
    assert channels['quantitative_claims'] is False
