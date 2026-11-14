"""The analog search engine.

Scoring is deliberately simple and inspectable. Every cell is converted to
its position in Earth's own distribution for each variable, then compared
against the profile's target position. A judge can follow the arithmetic,
and every sub-score is reported, so the engine can say *where* an analogy
holds and *where it breaks down* - which is what the challenge asks for.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from known_sites import EXCLUSION_RADIUS_KM, KNOWN_SITES
from planet_profiles import PlanetProfile

EARTH_RADIUS_KM = 6371.0

# Minimum separation between reported candidates. Adjacent grid cells are
# the same place; without this the top ten would be one valley ten times.
DEFAULT_SEPARATION_KM = 400.0


# A cell is scored only if the dimensions measured there carry at least this
# share of the profile's total weight. Below it, too much of the analogy is
# unknown to rank the cell at all. Missing values are mostly cloud-bound
# tropical coasts and small islands, where no candidate would be anyway.
MIN_WEIGHT_COVERAGE = 0.75


def to_percentiles(grid: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Rank-transform each feature onto 0..1 across all land cells.

    Rank rather than z-score because several variables are strongly skewed
    (precipitation especially) and a rank is robust to that without
    assuming any distribution. Missing values stay missing.
    """
    out = {}
    for feature in features:
        column = grid[feature]
        n = int(column.notna().sum())
        ranks = column.rank(method="average").to_numpy()
        out[feature] = (ranks - 1.0) / max(n - 1, 1)
    return pd.DataFrame(out, index=grid.index)


def _closeness(percentile: np.ndarray, target: float) -> np.ndarray:
    """1.0 at the target, 0.0 at the farthest point Earth can offer.

    Normalising by the maximum reachable distance keeps dimensions
    comparable: a target of 0.0 and a target of 0.5 are both scored on
    their own achievable range.
    """
    reach = max(target, 1.0 - target)
    if reach <= 0:
        return np.ones_like(percentile)
    return np.clip(1.0 - np.abs(percentile - target) / reach, 0.0, 1.0)


def closeness_matrix(
    percentiles: pd.DataFrame,
    features: List[str],
    targets: List[float],
) -> np.ndarray:
    """Per-dimension closeness for every cell, shape (cells, dimensions)."""
    return np.column_stack(
        [_closeness(percentiles[f].to_numpy(), t) for f, t in zip(features, targets)]
    )


def weighted_score(closeness: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted mean over the dimensions that were measured at each cell."""
    measured = np.isfinite(closeness)
    filled = np.where(measured, closeness, 0.0)
    weight_present = measured @ weights
    total = filled @ weights
    with np.errstate(invalid="ignore", divide="ignore"):
        scores = total / weight_present
    scores[weight_present < MIN_WEIGHT_COVERAGE * weights.sum()] = np.nan
    return scores


def score(
    grid: pd.DataFrame,
    percentiles: pd.DataFrame,
    profile: PlanetProfile,
) -> pd.DataFrame:
    """Score every land cell against one planetary profile."""
    features = [d.feature for d in profile.dimensions]
    targets = [d.target for d in profile.dimensions]
    weights = np.array([d.weight for d in profile.dimensions], dtype="float64")

    closeness = closeness_matrix(percentiles, features, targets)
    result = pd.DataFrame(
        {"dim_" + f: closeness[:, i] for i, f in enumerate(features)},
        index=grid.index,
    )
    result["analog_score"] = weighted_score(closeness, weights)
    return result


def _haversine_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float
) -> np.ndarray:
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def known_site_distance_km(grid: pd.DataFrame) -> np.ndarray:
    """Distance from every cell to the nearest established analog site."""
    lat = grid["latitude"].to_numpy()
    lon = grid["longitude"].to_numpy()
    nearest = np.full(len(grid), np.inf)
    for site in KNOWN_SITES:
        nearest = np.minimum(
            nearest, _haversine_km(lat, lon, site.latitude, site.longitude)
        )
    return nearest


# The continental ice sheets. Antarctica is named in the challenge text as
# already widely studied, and its interior is among the least operationally
# accessible terrain on Earth. Both ice sheets also dominate any cold-and-dry
# ranking so completely that they bury everything else, so they are masked
# when the search is asked for less-recognized analogs.
ANTARCTICA_MAX_LAT = -60.0
GREENLAND_BOUNDS = {"lat_min": 59.0, "lat_max": 84.0, "lon_min": -73.0, "lon_max": -11.0}


def ice_sheet_mask(frame: pd.DataFrame) -> np.ndarray:
    """True for cells on the Antarctic or Greenland ice sheets."""
    lat = frame["latitude"].to_numpy()
    lon = frame["longitude"].to_numpy()
    antarctica = lat < ANTARCTICA_MAX_LAT
    greenland = (
        (lat >= GREENLAND_BOUNDS["lat_min"])
        & (lat <= GREENLAND_BOUNDS["lat_max"])
        & (lon >= GREENLAND_BOUNDS["lon_min"])
        & (lon <= GREENLAND_BOUNDS["lon_max"])
    )
    return antarctica | greenland


def top_candidates(
    scored: pd.DataFrame,
    n: int = 10,
    separation_km: float = DEFAULT_SEPARATION_KM,
    exclude_known: bool = True,
    exclusion_km: float = EXCLUSION_RADIUS_KM,
    exclude_ice_sheets: bool = True,
) -> pd.DataFrame:
    """Pick the best cells, spatially separated so each is a distinct place.

    Greedy non-maximum suppression: take the best remaining cell, then drop
    everything within separation_km of it, repeat.
    """
    work = scored[scored["analog_score"].notna()]
    if exclude_known:
        work = work[work["known_site_distance_km"] > exclusion_km]
    if exclude_ice_sheets:
        work = work[~ice_sheet_mask(work)]

    work = work.sort_values("analog_score", ascending=False)
    if work.empty:
        return work

    # Only the strongest slice can possibly survive suppression, and
    # scanning all 800k rows repeatedly is wasted effort.
    pool = work.head(max(n * 2000, 20000)).reset_index(drop=True)

    lat = pool["latitude"].to_numpy()
    lon = pool["longitude"].to_numpy()
    available = np.ones(len(pool), dtype=bool)
    chosen: List[int] = []

    for idx in range(len(pool)):
        if not available[idx]:
            continue
        chosen.append(idx)
        if len(chosen) >= n:
            break
        distance = _haversine_km(lat, lon, lat[idx], lon[idx])
        available &= distance > separation_km

    return pool.iloc[chosen].reset_index(drop=True)


def nearest_scored_cell(
    grid: pd.DataFrame, scores: np.ndarray, latitude: float, longitude: float
) -> int:
    """Index of the nearest land cell that has a score.

    The nearest cell, not the best cell nearby: taking the best of a
    neighbourhood would flatter every site and inflate the validation.
    """
    distance = _haversine_km(
        grid["latitude"].to_numpy(), grid["longitude"].to_numpy(), latitude, longitude
    )
    distance = np.where(np.isfinite(scores), distance, np.inf)
    return int(np.argmin(distance))


def top_percent(scores: np.ndarray, value: float) -> float:
    """Share of scored land cells that score at least this high, in %."""
    valid = scores[np.isfinite(scores)]
    return 100.0 * float((valid >= value).sum()) / max(len(valid), 1)


def validate_against_known(
    grid: pd.DataFrame,
    scored: pd.DataFrame,
    profile: PlanetProfile,
) -> pd.DataFrame:
    """Where each established analog site for this profile lands.

    This is the honesty check. If sites chosen for a Mars-like environment
    do not score well for the Mars profiles, the method is wrong.
    """
    from known_sites import sites_for

    scores = scored["analog_score"].to_numpy()
    rows = []
    for site in sites_for(profile.site_key()):
        idx = nearest_scored_cell(grid, scores, site.latitude, site.longitude)
        rows.append(
            {
                "site": site.name,
                "country": site.country,
                "basis": site.basis,
                "used_for": site.used_for,
                "analog_score": round(float(scores[idx]), 3),
                "top_percent": round(top_percent(scores, scores[idx]), 1),
                "reference": site.reference,
                "latitude": site.latitude,
                "longitude": site.longitude,
            }
        )
    frame = pd.DataFrame(rows).sort_values("analog_score", ascending=False)
    return frame.reset_index(drop=True)


def characterize(
    row: pd.Series, profile: PlanetProfile, top_k: int = 3
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """Split a candidate into its strongest and weakest dimensions.

    The challenge asks participants to say what is similar, how similar, and
    what is not. This returns exactly that, per site.
    """
    scores = []
    for dim in profile.dimensions:
        value = float(row["dim_" + dim.feature])
        if np.isfinite(value):
            scores.append((dim.label, value))
    scores.sort(key=lambda item: item[1], reverse=True)
    return scores[:top_k], scores[-top_k:][::-1]


def build_scored_frame(
    grid: pd.DataFrame,
    percentiles: pd.DataFrame,
    profile: PlanetProfile,
    known_distance: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Grid plus scores plus known-site distance, ready for the UI."""
    scored = score(grid, percentiles, profile)
    out = pd.concat([grid, scored], axis=1)
    out["known_site_distance_km"] = (
        known_distance if known_distance is not None else known_site_distance_km(grid)
    )
    return out
