# GIS INTEGRO action report

- registry_id: `gis_integro`
- adapter_called: `True`
- under_the_hood: `GD-ANFIS rules + SHAP regularization`
- fixture_sha256: `ae41cb6eb0d6b21b291eace74449f808f48cf82e643f16fc6669b59b0d88a262`
- alpha_k: `{'r_spatial_risk': 0.71, 'r_shap_regularized': 0.73}`
- eta_k / SHAP: `{'region_density': 0.29, 'route_connectivity': 0.18, 'noise_regularizer': -0.04}`
- gamma_route: `0.2000`
- Delta: `0.0800`
- action: `audit_report`

GIS INTEGRO исполняется как контрольный fixture-маршрут через GD-ANFIS/SHAP-каналы. Числа gamma_route и Delta вычислены из fixture, но сценарий остаётся `source-pending` до закрепления внешнего источника.