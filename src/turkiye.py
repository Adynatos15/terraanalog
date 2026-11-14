"""Turkiye case study: what the global search says about one country.

Two questions:

1. Ranked purely by the data, where are Turkiye's best Moon and Mars
   analogs, and how do they compare with the rest of the world?
2. How does the method score the Turkish places that already come up in
   planetary analog discussion, chiefly Salda Lake, and the volcanic fields
   a geologist would point to first?

Coordinates of named places are approximate centres, good to a few km,
which is well inside one 18 km grid cell.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

import analog
import countries
from planet_profiles import PlanetProfile

ISO = "TUR"
LOCAL_SEPARATION_KM = 120.0


@dataclass(frozen=True)
class Place:
    name: str
    latitude: float
    longitude: float
    why: str


PLACES: List[Place] = [
    Place("Salda Lake", 37.55, 29.68,
          "Established Mars analog: hydromagnesite carbonates compared with Jezero crater."),
    Place("Meke Maar and Karapinar volcanic field", 37.68, 33.64,
          "Maar crater and cinder cones in the driest basin in Turkiye."),
    Place("Tuz Golu (Salt Lake)", 38.80, 33.35,
          "Large evaporite flat in the Central Anatolian closed basin."),
    Place("Kula", 38.55, 28.65,
          "Young basaltic lava flows and cinder cones."),
    Place("Cappadocia (Goreme)", 38.64, 34.83,
          "Eroded ignimbrite landscape; aeolian and fluvial erosion forms."),
    Place("Karacadag", 37.75, 39.85,
          "Basaltic shield volcano with extensive lava plains."),
    Place("Nemrut", 38.62, 42.23,
          "High-altitude caldera with crater lakes."),
    Place("Erzurum-Kars", 40.20, 42.20,
          "High, cold eastern plateau with long, hard winters."),
]


def turkey_mask(grid: pd.DataFrame) -> np.ndarray:
    return countries.contains(ISO, grid["latitude"].to_numpy(), grid["longitude"].to_numpy())


def local_ranking(scored: pd.DataFrame, inside: np.ndarray, n: int = 5) -> pd.DataFrame:
    """Best spatially separated cells inside Turkiye for one profile."""
    local = scored[inside & scored["analog_score"].notna().to_numpy()]
    local = local.sort_values("analog_score", ascending=False).reset_index(drop=True)
    lat = local["latitude"].to_numpy()
    lon = local["longitude"].to_numpy()
    keep, available = [], np.ones(len(local), dtype=bool)
    for i in range(len(local)):
        if not available[i]:
            continue
        keep.append(i)
        if len(keep) >= n:
            break
        available &= analog._haversine_km(lat, lon, lat[i], lon[i]) > LOCAL_SEPARATION_KM
    return local.iloc[keep].reset_index(drop=True)


def score_places(
    grid: pd.DataFrame, scored: pd.DataFrame, profile: PlanetProfile, inside: np.ndarray
) -> pd.DataFrame:
    """Each named place: global top %, rank within Turkiye, best and worst axes."""
    scores = scored["analog_score"].to_numpy()
    local_scores = np.sort(scores[inside & np.isfinite(scores)])[::-1]
    rows = []
    for place in PLACES:
        idx = analog.nearest_scored_cell(grid, scores, place.latitude, place.longitude)
        value = float(scores[idx])
        strong, weak = analog.characterize(scored.iloc[idx], profile, top_k=2)
        rows.append({
            "profile": profile.key,
            "place": place.name,
            "analog_score": round(value, 3),
            "global_top_percent": round(analog.top_percent(scores, value), 1),
            "turkiye_rank": int((local_scores > value).sum()) + 1,
            "turkiye_cells": int(len(local_scores)),
            "strongest": "; ".join("%s %.2f" % s for s in strong),
            "weakest": "; ".join("%s %.2f" % s for s in weak),
            "why": place.why,
        })
    return pd.DataFrame(rows)
