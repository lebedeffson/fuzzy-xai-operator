from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fuzzyxai.calibration import synthetic_calibration_pairs, cross_validate_beta

OUT = ROOT / 'reports'
OUT.mkdir(exist_ok=True)


def main():
    pairs = synthetic_calibration_pairs(n=75, seed=42)
    report = cross_validate_beta(pairs, folds=5, seed=42)
    report['chapter'] = 2
    report['purpose'] = 'cross-validated calibration of beta weights for d_E'
    (OUT / 'chapter2_calibration_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
