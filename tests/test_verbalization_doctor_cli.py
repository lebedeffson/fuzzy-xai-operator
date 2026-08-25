from __future__ import annotations

from fuzzyxai.verbalization.__main__ import main


def test_doctor_reports_unreachable_ollama_without_installing_anything(capsys) -> None:
    exit_code = main(["doctor", "--host", "http://127.0.0.1:1"])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "[deterministic path] OK" in output
    assert "UNREACHABLE" in output
    assert "ollama serve" in output


def test_doctor_never_calls_pull_or_serve(monkeypatch, capsys) -> None:
    import subprocess

    def _forbidden(*args, **kwargs):
        raise AssertionError("doctor must never execute an install/pull/serve command itself")

    monkeypatch.setattr(subprocess, "run", _forbidden)
    monkeypatch.setattr(subprocess, "Popen", _forbidden)
    main(["doctor", "--host", "http://127.0.0.1:1"])


def test_main_requires_a_subcommand() -> None:
    import pytest

    with pytest.raises(SystemExit):
        main([])
