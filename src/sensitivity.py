"""How much do the candidates depend on the weights we chose?

The weights in planet_profiles.py are judgement calls. A fair objection is
that a different but equally defensible set would produce a different list.
This module tests that directly: every weight is multiplied by an
independent random factor between 0.5 and 1.5, the search is rerun, and
this is repeated a few hundred times.

A candidate's **stability** is the share of runs in which some top-ten pick
lands within STABILITY_RADIUS_KM of it. A candidate at 95% is a property of
the data; one at 30% is a property of our weights, and the interface says
so.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

import analog
from planet_profiles import PlanetProfile

RUNS = 300
WEIGHT_SPREAD = 0.5          # each weight scaled by U(1 - spread, 1 + spread)
TOP_N = 10
STABILITY_RADIUS_KM = 250.0
POOL_SIZE = 20000
RANDOM_SEED = 7


def _pick(lat: np.ndarray, lon: np.ndarray, scores: np.ndarray,
          eligible: np.ndarray, n: int, separation_km: float) -> np.ndarray:
    """Indices of the top-n spatially separated cells (numpy-only, fast)."""
    candidates = np.flatnonzero(eligible & np.isfinite(scores))
    if len(candidates) > POOL_SIZE:
        top = np.argpartition(-scores[candidates], POOL_SIZE)[:POOL_SIZE]
        candidates = candidates[top]
    candidates = candidates[np.argsort(-scores[candidates])]

    available = np.ones(len(candidates), dtype=bool)
    chosen: List[int] = []
    for i, idx in enumerate(candidates):
        if not available[i]:
            continue
        chosen.append(idx)
        if len(chosen) >= n:
            break
        d = analog._haversine_km(lat[candidates], lon[candidates], lat[idx], lon[idx])
        available &= d > separation_km
    return np.array(chosen, dtype=int)


def eligible_mask(scored: pd.DataFrame, exclude_ice_sheets: bool = True) -> np.ndarray:
    mask = scored["known_site_distance_km"].to_numpy() > analog.EXCLUSION_RADIUS_KM
    if exclude_ice_sheets:
        mask &= ~analog.ice_sheet_mask(scored)
    return mask


def stability(
    scored: pd.DataFrame,
    percentiles: pd.DataFrame,
    profile: PlanetProfile,
    candidates: pd.DataFrame,
    runs: int = RUNS,
    exclude_ice_sheets: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stability of each candidate, plus the run-by-run picks.

    Returns (stability per candidate in 0..1, array of all picked indices).
    """
    rng = np.random.default_rng(RANDOM_SEED)
    features = [d.feature for d in profile.dimensions]
    targets = [d.target for d in profile.dimensions]
    base = np.array([d.weight for d in profile.dimensions], dtype="float64")
    closeness = analog.closeness_matrix(percentiles, features, targets)

    lat = scored["latitude"].to_numpy()
    lon = scored["longitude"].to_numpy()
    eligible = eligible_mask(scored, exclude_ice_sheets)

    picks = []
    for _ in range(runs):
        factors = rng.uniform(1.0 - WEIGHT_SPREAD, 1.0 + WEIGHT_SPREAD, size=len(base))
        scores = analog.weighted_score(closeness, base * factors)
        picks.append(_pick(lat, lon, scores, eligible, TOP_N, analog.DEFAULT_SEPARATION_KM))

    hits = np.zeros(len(candidates))
    for chosen in picks:
        for j, (clat, clon) in enumerate(zip(candidates["latitude"], candidates["longitude"])):
            d = analog._haversine_km(lat[chosen], lon[chosen], clat, clon)
            if (d <= STABILITY_RADIUS_KM).any():
                hits[j] += 1
    return hits / runs, np.concatenate(picks) if picks else np.array([], dtype=int)
