"""Tests for the arithmetic every reported number depends on."""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import analog  # noqa: E402
import modis_gibs  # noqa: E402
import validation  # noqa: E402
from planet_profiles import PROFILES, get_profile  # noqa: E402


# ------------------------------------------------------------ closeness
def test_closeness_is_one_at_target_and_zero_at_farthest_point():
    p = np.array([0.0, 0.5, 1.0])
    assert analog._closeness(p, 0.0).tolist() == [1.0, 0.5, 0.0]
    assert analog._closeness(p, 1.0).tolist() == [0.0, 0.5, 1.0]


def test_closeness_for_mid_target_uses_its_own_reachable_range():
    # Target 0.25: the farthest reachable point is 1.0, 0.75 away.
    c = analog._closeness(np.array([0.25, 1.0]), 0.25)
    assert c[0] == pytest.approx(1.0)
    assert c[1] == pytest.approx(0.0)


def test_closeness_keeps_missing_values_missing():
    c = analog._closeness(np.array([np.nan, 0.2]), 0.0)
    assert np.isnan(c[0]) and c[1] == pytest.approx(0.8)


# ------------------------------------------------------------ weighting
def test_weighted_score_skips_missing_dimensions():
    closeness = np.array([[1.0, np.nan, 0.0]])
    weights = np.array([1.0, 0.1, 1.0])
    # Missing dimension carries under 25% of the weight, so the cell is
    # scored on the other two: (1*1 + 0*1) / 2.
    assert analog.weighted_score(closeness, weights)[0] == pytest.approx(0.5)


def test_weighted_score_refuses_cells_missing_too_much_weight():
    closeness = np.array([[1.0, np.nan, np.nan]])
    weights = np.array([1.0, 1.0, 1.0])
    assert np.isnan(analog.weighted_score(closeness, weights)[0])


def test_percentiles_ignore_missing_values():
    grid = pd.DataFrame({"x": [10.0, np.nan, 30.0, 20.0]})
    pct = analog.to_percentiles(grid, ["x"])["x"].to_numpy()
    assert pct[0] == pytest.approx(0.0)
    assert np.isnan(pct[1])
    assert pct[2] == pytest.approx(1.0)
    assert pct[3] == pytest.approx(0.5)


# ------------------------------------------------------------ geography
def test_haversine_matches_a_known_distance():
    # One degree of latitude is about 111.2 km.
    d = analog._haversine_km(np.array([0.0]), np.array([0.0]), 1.0, 0.0)
    assert d[0] == pytest.approx(111.19, abs=0.1)


def test_ice_sheet_mask_covers_antarctica_and_greenland_only():
    frame = pd.DataFrame({
        "latitude": [-75.0, 72.0, 37.75, 64.0],
        "longitude": [100.0, -40.0, 33.75, -150.0],
    })
    assert analog.ice_sheet_mask(frame).tolist() == [True, True, False, False]


def test_top_candidates_are_spatially_separated():
    lat = np.array([0.0, 0.1, 10.0, 20.0])
    frame = pd.DataFrame({
        "latitude": lat, "longitude": np.zeros(4),
        "analog_score": [0.9, 0.89, 0.5, 0.4],
        "known_site_distance_km": np.full(4, 1e6),
    })
    top = analog.top_candidates(frame, n=3, separation_km=100.0, exclude_ice_sheets=False)
    # The 0.89 cell is 11 km from the 0.9 cell and must be suppressed.
    assert top["latitude"].tolist() == [0.0, 10.0, 20.0]


# ------------------------------------------------------------ validation
def test_auc_is_one_half_and_zero_in_the_obvious_cases():
    neg = np.array([0.1, 0.2, 0.3])
    assert validation._auc(np.array([0.9]), neg) == pytest.approx(1.0)
    assert validation._auc(np.array([0.0]), neg) == pytest.approx(0.0)
    assert validation._auc(np.array([0.2]), neg) == pytest.approx(0.5)


# ------------------------------------------------------------ colour decoding
def test_gibs_decode_recovers_values_and_rejects_unknown_colours():
    table = {(255, 0, 0): 300.0, (0, 0, 255): 250.0}
    pixels = np.array([[[255, 0, 0, 255], [0, 0, 255, 255],
                        [1, 2, 3, 255], [255, 0, 0, 0]]], dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    values, rate = modis_gibs._decode(buffer.getvalue(), table)
    assert values[0, 0] == pytest.approx(300.0)
    assert values[0, 1] == pytest.approx(250.0)
    assert np.isnan(values[0, 2])      # colour not in the table: missing, never guessed
    assert np.isnan(values[0, 3])      # transparent: no data
    assert rate == pytest.approx(2 / 3)


# ------------------------------------------------------------ profiles
@pytest.mark.parametrize("key", sorted(PROFILES))
def test_every_profile_is_well_formed(key):
    profile = get_profile(key)
    assert profile.dimensions
    for d in profile.dimensions:
        assert 0.0 <= d.target <= 1.0
        assert d.weight > 0
        assert d.rationale.strip()


def test_practice_profile_is_the_physics_profile_without_temperature():
    physics = get_profile("mars_midlat")
    practice = get_profile("mars_midlat_practice")
    assert [d.feature for d in practice.dimensions] == [
        d.feature for d in physics.dimensions if d.feature != "lst_mean_c"
    ]
    assert "post hoc" in practice.provenance
