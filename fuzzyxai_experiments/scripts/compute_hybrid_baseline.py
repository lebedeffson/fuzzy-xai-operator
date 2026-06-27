from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/generated/hybrid_xiris_objects.csv'


def main() -> None:
    rows = list(csv.DictReader(DATA.open(encoding='utf-8')))
    missed = sum(1 for r in rows if r['is_critical'] == 'true' and r['baseline_action'] == 'accept')
    print(f'hybrid_baseline_missed={missed}')

if __name__ == '__main__':
    main()
