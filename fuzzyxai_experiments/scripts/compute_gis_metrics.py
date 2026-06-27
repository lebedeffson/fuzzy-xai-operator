from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/fixtures/gis_integro_fixture.csv'


def main() -> None:
    row = next(csv.DictReader(DATA.open(encoding='utf-8')))
    p = float(row['probability'])
    mean_alpha = (float(row['alpha_spatial_risk']) + float(row['alpha_shap_regularized'])) / 2
    positive_shap = float(row['shap_region_density']) + float(row['shap_route_connectivity'])
    gamma = max(abs(p - mean_alpha), abs(p - positive_shap))
    delta = float(row['reduction_loss'])
    print(f'gamma_route={gamma:.2f} Delta={delta:.2f}')

if __name__ == '__main__':
    main()
