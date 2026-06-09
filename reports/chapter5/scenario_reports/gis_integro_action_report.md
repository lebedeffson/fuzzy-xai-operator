# GIS INTEGRO action report

- registry_id: `gis_integro`
- adapter_called: `True`
- under_the_hood: `GD-ANFIS rules + SHAP regularization`
- fixture_sha256: `dd18815b5e138f23573de1e099968e8393a33bf4356186adae764e2f3950c92b`
- alpha_k: `{'r_spatial_risk': 0.71, 'r_shap_regularized': 0.73}`
- eta_k / SHAP: `{'region_density': 0.29, 'route_connectivity': 0.18, 'noise_regularizer': -0.04}`
- action: `audit_report`

GIS INTEGRO исполняется как fixture-маршрут через GD-ANFIS/SHAP-каналы, но остаётся `source-pending` до закрепления внешнего источника.