"""Country membership for grid cells, from Natural Earth boundaries.

Natural Earth 1:50m admin-0 countries (public domain,
https://www.naturalearthdata.com/). Point-in-polygon uses matplotlib's Path,
so no GIS stack is needed. At 1:50m a few coastal cells may be assigned to
the sea; that does not affect inland analog candidates.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
from matplotlib.path import Path as PolygonPath

BOUNDARIES = (
    Path(__file__).resolve().parent.parent
    / "data" / "raw" / "naturalearth" / "ne_50m_admin_0_countries.geojson"
)
SOURCE_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
    "geojson/ne_50m_admin_0_countries.geojson"
)


def _rings(geometry: dict) -> List[np.ndarray]:
    if geometry["type"] == "Polygon":
        polygons = [geometry["coordinates"]]
    elif geometry["type"] == "MultiPolygon":
        polygons = geometry["coordinates"]
    else:
        return []
    # Outer rings only: no country used here has an inland enclave that
    # matters at 18 km resolution.
    return [np.asarray(poly[0], dtype=float) for poly in polygons]


def outline(iso_a3: str) -> List[np.ndarray]:
    """Outer boundary rings of a country as arrays of (lon, lat)."""
    if not BOUNDARIES.exists():
        raise FileNotFoundError(
            "Natural Earth boundaries missing. Run scripts/download_data.py."
        )
    data = json.loads(BOUNDARIES.read_text(encoding="utf-8"))
    for feature in data["features"]:
        if feature["properties"].get("ADM0_A3") == iso_a3:
            return _rings(feature["geometry"])
    raise KeyError("Country not found: " + iso_a3)


def contains(iso_a3: str, latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    """True where (latitude, longitude) falls inside the country."""
    points = np.column_stack([longitude, latitude])
    inside = np.zeros(len(points), dtype=bool)
    for ring in outline(iso_a3):
        lon_min, lat_min = ring.min(axis=0)
        lon_max, lat_max = ring.max(axis=0)
        box = (
            (points[:, 0] >= lon_min) & (points[:, 0] <= lon_max)
            & (points[:, 1] >= lat_min) & (points[:, 1] <= lat_max)
        )
        if box.any():
            inside[box] |= PolygonPath(ring).contains_points(points[box])
    return inside
