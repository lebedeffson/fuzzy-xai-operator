from __future__ import annotations

from .base import BaseAdapter


class GeospatialRouteAdapter(BaseAdapter):
    repo_id = "control/gis-integro"
    scenario_id = "gis_integro"
    required_fields = ("p", "alpha_mean", "s")


def adapt_gis_integro(payload: dict) -> object:
    return GeospatialRouteAdapter().to_adapted_input(payload)
