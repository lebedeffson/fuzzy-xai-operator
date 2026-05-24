from __future__ import annotations

import json
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fuzzyxai.risk.observer_pipeline import build_full_observer_pipeline_report, write_full_observer_pipeline_report


def main() -> int:
    report = build_full_observer_pipeline_report()
    paths = write_full_observer_pipeline_report(report)
    if '--open' in sys.argv:
        webbrowser.open(Path(paths['html']).resolve().as_uri())
    print(json.dumps({
        'status': report['status'],
        'safe_action': report['with_observer']['safe_action'],
        'application_risk': report['with_observer']['application_risk'],
        'I_pre': report['with_observer']['I_pre'],
        'I_final': report['with_observer']['I_final'],
        **paths,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
