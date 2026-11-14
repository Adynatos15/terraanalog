"""Turkiye case study: tables and a map.

Outputs:
  docs/results/turkiye_top.csv     best cells inside Turkiye, per profile
  docs/results/turkiye_places.csv  named Turkish places, per profile
  docs/figures/turkiye_moon_mare.png  lunar mare score across Turkiye
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as path_effects  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import analog  # noqa: E402
import countries  # noqa: E402
import placenames  # noqa: E402
import turkiye  # noqa: E402
from planet_profiles import PROFILE_ORDER, get_profile  # noqa: E402

GRID_PATH = ROOT / "data" / "processed" / "earth_grid.parquet"
RESULTS = ROOT / "docs" / "results"
FIGURES = ROOT / "docs" / "figures"
MAP_PROFILE = "moon_mare"

BACKGROUND = "#0E1420"
CMAP = LinearSegmentedColormap.from_list(
    "analog", [(0.0, "#18233A"), (0.45, "#2F4A70"), (0.72, "#B4532F"),
               (0.88, "#E8703F"), (1.0, "#FFD9A0")])
CMAP.set_bad(BACKGROUND)


def render_map(grid: pd.DataFrame, scores: np.ndarray, inside: np.ndarray,
               best: pd.Series) -> Path:
    lat0, lat1, lon0, lon1 = 35.5, 42.5, 25.5, 45.0
    rows = grid["grid_row"].to_numpy()
    cols = grid["grid_col"].to_numpy()
    raster = np.full((1080, 2160), np.nan, dtype="float32")
    raster[rows[inside], cols[inside]] = scores[inside]
    r0, r1 = int((90 - lat1) * 6), int((90 - lat0) * 6)
    c0, c1 = int((lon0 + 180) * 6), int((lon1 + 180) * 6)

    fig = plt.figure(figsize=(16, 6.2), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BACKGROUND)
    ax.imshow(raster[r0:r1, c0:c1], extent=[lon0, lon1, lat0, lat1], cmap=CMAP,
              vmin=0.30, vmax=0.90, interpolation="nearest", aspect="auto")
    for ring in countries.outline("TUR"):
        ax.plot(ring[:, 0], ring[:, 1], color="#8793A8", linewidth=1.2)

    outline = [path_effects.withStroke(linewidth=6, foreground=BACKGROUND)]
    for place in turkiye.PLACES:
        ax.scatter([place.longitude], [place.latitude], s=160, color="#F2EEE6",
                   edgecolors=BACKGROUND, linewidths=2, zorder=3)
        label = place.name.split(" (")[0].split(" and ")[0]
        # East-edge labels and Tuz Golu (which would collide with Cappadocia)
        # sit to the left of their marker.
        on_left = place.longitude > 41.0 or label.startswith("Tuz")
        offset = (-10, 8) if on_left else (14, 8)
        if label.startswith("Meke"):
            offset = (26, 6)
        ax.annotate(label, (place.longitude, place.latitude),
                    xytext=offset,
                    textcoords="offset points", color="#F2EEE6",
                    ha="right" if on_left else "left",
                    fontsize=24, fontweight="bold", path_effects=outline, zorder=4)
    ax.scatter([best["longitude"]], [best["latitude"]], s=1600, facecolors="none",
               edgecolors="#E8703F", linewidths=4, zorder=5)
    ax.set_xlim(lon0, lon1)
    ax.set_ylim(lat0, lat1)
    ax.axis("off")
    out = FIGURES / "turkiye_moon_mare.png"
    fig.savefig(out, facecolor=BACKGROUND)
    plt.close(fig)
    return out


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    grid = pd.read_parquet(GRID_PATH)
    features = sorted({d.feature for k in PROFILE_ORDER for d in get_profile(k).dimensions})
    percentiles = analog.to_percentiles(grid, features)
    known_distance = analog.known_site_distance_km(grid)
    inside = turkiye.turkey_mask(grid)
    print("Turkiye land cells:", int(inside.sum()))

    top_rows, place_rows = [], []
    map_scores, map_best = None, None
    for key in PROFILE_ORDER:
        profile = get_profile(key)
        scored = analog.build_scored_frame(grid, percentiles, profile, known_distance)
        scores = scored["analog_score"].to_numpy()
        top = placenames.annotate(turkiye.local_ranking(scored, inside, n=5))
        for rank, (_, row) in enumerate(top.iterrows(), start=1):
            strong, weak = analog.characterize(row, profile, top_k=2)
            top_rows.append({
                "profile": key, "rank": rank, "place_name": row["place_name"],
                "latitude": round(float(row["latitude"]), 3),
                "longitude": round(float(row["longitude"]), 3),
                "analog_score": round(float(row["analog_score"]), 3),
                "global_top_percent": round(analog.top_percent(scores, row["analog_score"]), 1),
                "elevation_m": round(float(row["elevation_m"])),
                "annual_precip_mm": round(float(row["annual_precip_mm"])),
                "lst_diurnal_range_c": round(float(row["lst_diurnal_range_c"]), 1),
                "ndvi_max": round(float(row["ndvi_max"]), 2),
                "strongest": "; ".join("%s %.2f" % s for s in strong),
                "weakest": "; ".join("%s %.2f" % s for s in weak),
            })
        place_rows.extend(turkiye.score_places(grid, scored, profile, inside).to_dict("records"))
        if key == MAP_PROFILE:
            map_scores, map_best = scores, top.iloc[0]

    pd.DataFrame(top_rows).to_csv(RESULTS / "turkiye_top.csv", index=False)
    pd.DataFrame(place_rows).to_csv(RESULTS / "turkiye_places.csv", index=False)
    print("wrote", render_map(grid, map_scores, inside, map_best))
    return 0


if __name__ == "__main__":
    sys.exit(main())
