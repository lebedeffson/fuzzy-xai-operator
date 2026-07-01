from __future__ import annotations

from fuzzyxai.adapters.geospatial_route import adapt_gis_integro


def get_input():
    return adapt_gis_integro({"p": 0.67, "alpha_mean": 0.72, "s": 0.47})
