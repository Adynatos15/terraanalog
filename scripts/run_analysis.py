"""Run the full analysis once and write every reported number to disk.

The README, the demo and the app all read from docs/results/, so no figure
quoted anywhere can drift from what the code actually produces.

Outputs (docs/results/):
  validation.csv        AUC per profile and site group, with baseline
  dimension_auc.csv     AUC of each single dimension against known sites
  ablation.csv          AUC with each dimension removed
  candidates.csv        top candidates per profile, with stability
  known_sites.csv       every established site's score and rank
  summary.json          headline numbers
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import analog  # noqa: E402
import placenames  # noqa: E402
import sensitivity  # noqa: E402
import validation  # noqa: E402
from known_sites import sites_for  # noqa: E402
from planet_profiles import PROFILE_ORDER, get_profile  # noqa: E402

GRID_PATH = ROOT / "data" / "processed" / "earth_grid.parquet"
OUT_DIR = ROOT / "docs" / "results"
N_CANDIDATES = 10
NEGATIVE_SAMPLE = 50000
SEED = 1
MIN_GROUP = 3


def dimension_tables(grid, percentiles, profile, rng):
    """Single-dimension AUC and drop-one ablation.

    Uses the sites chosen for their environment when there are at least
    MIN_GROUP of them; otherwise all sites for the profile, because an AUC
    over one or two sites says nothing.
    """
    env = sites_for(profile.site_key(), "environment")
    if len(env) >= MIN_GROUP:
        sites, group = env, "environment"
    else:
        sites, group = sites_for(profile.site_key()), "all"
    features = [d.feature for d in profile.dimensions]
    targets = [d.target for d in profile.dimensions]
    weights = np.array([d.weight for d in profile.dimensions], dtype="float64")
    closeness = analog.closeness_matrix(percentiles, features, targets)
    full = analog.weighted_score(closeness, weights)
    valid = np.flatnonzero(np.isfinite(full))
    sample = rng.choice(valid, size=min(NEGATIVE_SAMPLE, len(valid)), replace=False)

    def auc(scores):
        return validation._auc(validation.site_scores(grid, scores, sites), scores[sample])

    full_auc = auc(full)
    dim_rows, ablation_rows = [], []
    for i, d in enumerate(profile.dimensions):
        dim_rows.append({
            "profile": profile.key, "sites": group, "n": len(sites),
            "dimension": d.label, "feature": d.feature, "target": d.target,
            "weight": d.weight, "auc": round(auc(closeness[:, i]), 3),
        })
        w = weights.copy()
        w[i] = 0.0
        dropped = auc(analog.weighted_score(closeness, w))
        ablation_rows.append({
            "profile": profile.key, "sites": group, "n": len(sites),
            "removed": d.label, "auc": round(dropped, 3),
            "change": round(dropped - full_auc, 3),
        })
    return dim_rows, ablation_rows


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grid = pd.read_parquet(GRID_PATH)
    features = sorted({d.feature for k in PROFILE_ORDER for d in get_profile(k).dimensions}
                      | {"annual_precip_mm"})
    percentiles = analog.to_percentiles(grid, features)
    known_distance = analog.known_site_distance_km(grid)
    rng = np.random.default_rng(SEED)

    validation_rows, dim_rows, ablation_rows = [], [], []
    candidate_rows, known_rows = [], []

    for key in PROFILE_ORDER:
        profile = get_profile(key)
        print("profile:", profile.name, flush=True)
        scored = analog.build_scored_frame(grid, percentiles, profile, known_distance)
        scores = scored["analog_score"].to_numpy()

        table = validation.evaluate(grid, percentiles, profile, scores)
        table["provenance"] = profile.provenance
        validation_rows.extend(table.to_dict("records"))

        d_rows, a_rows = dimension_tables(grid, percentiles, profile, rng)
        dim_rows.extend(d_rows)
        ablation_rows.extend(a_rows)

        known = analog.validate_against_known(grid, scored, profile)
        known.insert(0, "profile", key)
        known_rows.extend(known.to_dict("records"))

        top = analog.top_candidates(scored, n=N_CANDIDATES)
        top = placenames.annotate(top)
        stable, _ = sensitivity.stability(scored, percentiles, profile, top)
        for rank, (i, row) in enumerate(top.iterrows(), start=1):
            strong, weak = analog.characterize(row, profile, top_k=2)
            candidate_rows.append({
                "profile": key, "rank": rank, "place_name": row["place_name"],
                "latitude": round(float(row["latitude"]), 3),
                "longitude": round(float(row["longitude"]), 3),
                "analog_score": round(float(row["analog_score"]), 3),
                "top_percent": round(analog.top_percent(scores, row["analog_score"]), 3),
                "stability": round(float(stable[i]), 2),
                "elevation_m": round(float(row["elevation_m"])),
                "annual_precip_mm": round(float(row["annual_precip_mm"])),
                "lst_mean_c": round(float(row["lst_mean_c"]), 1),
                "lst_diurnal_range_c": round(float(row["lst_diurnal_range_c"]), 1),
                "lst_night_min_c": round(float(row["lst_night_min_c"]), 1),
                "ndvi_max": round(float(row["ndvi_max"]), 2),
                "strongest": "; ".join("%s %.2f" % s for s in strong),
                "weakest": "; ".join("%s %.2f" % s for s in weak),
                "known_site_distance_km": round(float(row["known_site_distance_km"])),
            })

    pd.DataFrame(validation_rows).to_csv(OUT_DIR / "validation.csv", index=False)
    pd.DataFrame(dim_rows).to_csv(OUT_DIR / "dimension_auc.csv", index=False)
    pd.DataFrame(ablation_rows).to_csv(OUT_DIR / "ablation.csv", index=False)
    pd.DataFrame(candidate_rows).to_csv(OUT_DIR / "candidates.csv", index=False)
    pd.DataFrame(known_rows).to_csv(OUT_DIR / "known_sites.csv", index=False)

    summary = {
        "land_cells": int(len(grid)),
        "modis_years": "2020-2024",
        "known_sites": len({r["site"] for r in known_rows}),
        "stability_runs": sensitivity.RUNS,
        "weight_spread": sensitivity.WEIGHT_SPREAD,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote", OUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
