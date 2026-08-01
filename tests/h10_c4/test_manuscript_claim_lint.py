import subprocess
import sys


def _run(path):
    return subprocess.run(
        [sys.executable, "scripts/manuscript_claim_lint.py", str(path)],
        text=True,
        capture_output=True,
        check=False,
    )


def test_claim_lint_accepts_scoped_h10_c3_statement(tmp_path) -> None:
    manuscript = tmp_path / "valid.txt"
    manuscript.write_text(
        "H10-C3 подтверждено в пределах контролируемых структурных мутаций.\n"
        "Проверка 5 млн записей выполнена при заранее рассчитанных объяснениях.\n",
        encoding="utf-8",
    )

    assert _run(manuscript).returncode == 0


def test_claim_lint_rejects_unscoped_or_human_utility_claims(tmp_path) -> None:
    manuscript = tmp_path / "invalid.txt"
    manuscript.write_text(
        "H10-C3 подтверждено.\n"
        "Доказана практическая полезность и снижает трудозатраты.\n",
        encoding="utf-8",
    )

    result = _run(manuscript)
    assert result.returncode == 1
    assert "H10_C3_SCOPE_MISSING" in result.stderr
    assert "UNSUPPORTED_CLAIM" in result.stderr
