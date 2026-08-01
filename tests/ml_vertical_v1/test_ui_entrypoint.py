from __future__ import annotations

import sys

from apps import layered_demo


def test_ml_vertical_only_dispatches_without_loading_historical_reports(monkeypatch) -> None:
    called: dict[str, int] = {}

    monkeypatch.setattr(layered_demo, "run_ml_vertical_ui", lambda port: called.setdefault("port", port))
    monkeypatch.setattr(sys, "argv", ["layered_demo", "--port", "8123", "--ml-vertical-only"])

    layered_demo.main()

    assert called == {"port": 8123}
