"""Figures for the README, the demo deck and the project page.

Reads docs/results/ (run scripts/run_analysis.py first) and the grid.
Outputs to docs/figures/:
  map_mars_midlat.png    global score, Mars desert profile (physics)
  map_moon_mare.png      global score, lunar mare profile
  gap_candidates.png     physics vs practice candidates, plus Karapinar
  dimension_auc_mars.png what established Mars desert analogs share
"""
from __future__ import annotations

import re
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
from planet_profiles import get_profile  # noqa: E402

GRID = ROOT / "data" / "processed" / "earth_grid.parquet"
RESULTS = ROOT / "docs" / "results"
OUT = ROOT / "docs" / "figures"

BG = "#0E1420"
LAND = "#26344F"
PAPER = "#F2EEE6"
MUTED = "#8793A8"
COLD = "#7CC4E4"   # physics: cold-desert candidates
HOT = "#E8703F"    # practice: hot-desert candidates
TURKIYE = "#FFD9A0"
LAT_TOP, LAT_BOTTOM = 84.0, -58.0
HOT_DESERT_C = 25.0

CMAP = LinearSegmentedColormap.from_list(
    "analog", [(0.0, "#18233A"), (0.45, "#2F4A70"), (0.72, "#B4532F"),
               (0.88, "#E8703F"), (1.0, "#FFD9A0")])
CMAP.set_bad(BG)

_DROP = re.compile(
    r"\b(Prefecture|Region|Provincia( de| del)?|Province of|City|District( of)?|"
    r"Autonomous|County|League|Federal|Mongol and Tibetan)\b", re.I)


def short_name(place: str) -> str:
    """'Ngari Prefecture, Gerze County, China' -> 'Ngari, China'."""
    parts = [p.strip() for p in place.split(",")]
    head = re.sub(r"\s+", " ", _DROP.sub("", parts[0])).strip(" -")
    return head + ", " + parts[-1] if len(parts) > 1 else head


def raster(grid: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    r = np.full((1080, 2160), np.nan, dtype="float32")
    r[grid["grid_row"].to_numpy(), grid["grid_col"].to_numpy()] = values
    return r[int((90 - LAT_TOP) * 6): int((90 - LAT_BOTTOM) * 6), :]


def world_axes():
    fig = plt.figure(figsize=(21.6, 8.52), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG)
    ax.set_xlim(-180, 180)
    ax.set_ylim(LAT_BOTTOM, LAT_TOP)
    ax.axis("off")
    return fig, ax


def score_map(grid, percentiles, key) -> Path:
    scores = analog.score(grid, percentiles, get_profile(key))["analog_score"].to_numpy()
    fig, ax = world_axes()
    ax.imshow(raster(grid, scores), extent=[-180, 180, LAT_BOTTOM, LAT_TOP],
              cmap=CMAP, vmin=0.30, vmax=0.90, interpolation="nearest", aspect="auto")
    out = OUT / ("map_%s.png" % key)
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    return out


def gap_map(grid, candidates, turkiye_top) -> Path:
    land = np.where(np.isfinite(raster(grid, np.ones(len(grid)))), 1.0, np.nan)
    fig, ax = world_axes()
    ax.imshow(land, extent=[-180, 180, LAT_BOTTOM, LAT_TOP],
              cmap=LinearSegmentedColormap.from_list("l", [LAND, LAND]),
              interpolation="nearest", aspect="auto")
    stroke = [path_effects.withStroke(linewidth=8, foreground=BG)]

    def plot(rows, colour, size, labelled):
        seen = []
        for _, r in rows.iterrows():
            ax.scatter([r["longitude"]], [r["latitude"]], s=size * 3.5, color=colour,
                       alpha=0.22, linewidths=0, zorder=2)
            ax.scatter([r["longitude"]], [r["latitude"]], s=size, color=colour,
                       edgecolors=PAPER, linewidths=2, zorder=3)
            name = short_name(r["place_name"])
            if labelled and name not in seen:
                seen.append(name)
                ax.annotate(name, (r["longitude"], r["latitude"]), xytext=(16, -6),
                            textcoords="offset points", color=PAPER, fontsize=26,
                            fontweight="bold", path_effects=stroke, zorder=4)

    # The two lists share their Tibetan leaders; plot what separates them.
    # Physics: the top six. Practice: its hot-desert entries (mean surface
    # temperature of 25 C or more), none of which the physics list contains.
    practice = candidates[(candidates["profile"] == "mars_midlat_practice")
                          & (candidates["lst_mean_c"] >= HOT_DESERT_C)]
    physics = candidates[(candidates["profile"] == "mars_midlat") & (candidates["rank"] <= 6)]
    plot(practice, HOT, 260, False)
    plot(physics, COLD, 260, False)

    best_tr = turkiye_top[(turkiye_top["profile"] == "moon_mare")
                          & (turkiye_top["rank"] == 1)].iloc[0]
    ax.scatter([best_tr["longitude"]], [best_tr["latitude"]], s=700, facecolors="none",
               edgecolors=TURKIYE, linewidths=4, zorder=5)
    ax.annotate("Karapınar", (best_tr["longitude"], best_tr["latitude"]), xytext=(-40, 30),
                textcoords="offset points", ha="right", color=TURKIYE, fontsize=30,
                fontweight="bold", path_effects=stroke, zorder=6)
    ax.annotate("Tibet & Xinjiang", (84.0, 36.5), xytext=(0, 48), textcoords="offset points",
                ha="center", color=COLD, fontsize=30, fontweight="bold", path_effects=stroke,
                zorder=6)
    ax.annotate("Atacama", (-69.5, -20.0), xytext=(24, -8), textcoords="offset points",
                color=HOT, fontsize=30, fontweight="bold", path_effects=stroke, zorder=6)
    ax.annotate("Sahara", (20.0, 23.0), xytext=(-30, -70), textcoords="offset points",
                color=HOT, fontsize=30, fontweight="bold", path_effects=stroke, zorder=6)
    out = OUT / "gap_candidates.png"
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    return out


def dimension_chart(dims) -> Path:
    sub = dims[dims["profile"] == "mars_midlat"].sort_values("auc")
    fig = plt.figure(figsize=(12, 6.2), dpi=150)
    ax = fig.add_axes([0.30, 0.12, 0.64, 0.84])
    fig.patch.set_facecolor("#172033")
    ax.set_facecolor("#172033")
    colours = [HOT if a < 0.5 else COLD for a in sub["auc"]]
    ax.barh(sub["dimension"], sub["auc"], color=colours, height=0.62)
    ax.axvline(0.5, color=MUTED, linestyle="--", linewidth=2)
    for y, a in enumerate(sub["auc"]):
        ax.text(a + 0.015, y, "%.2f" % a, va="center", color=PAPER, fontsize=17,
                fontweight="bold")
    ax.set_xlim(0, 1.0)
    ax.tick_params(colors=PAPER, labelsize=17)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.text(0.505, len(sub) - 0.45, "chance", color=MUTED, fontsize=15, ha="left")
    out = OUT / "dimension_auc_mars.png"
    fig.savefig(out, facecolor="#172033")
    plt.close(fig)
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    grid = pd.read_parquet(GRID)
    feats = sorted({d.feature for k in ("mars_midlat", "moon_mare")
                    for d in get_profile(k).dimensions})
    percentiles = analog.to_percentiles(grid, feats)
    candidates = pd.read_csv(RESULTS / "candidates.csv")
    turkiye_top = pd.read_csv(RESULTS / "turkiye_top.csv")
    dims = pd.read_csv(RESULTS / "dimension_auc.csv")
    for path in (score_map(grid, percentiles, "mars_midlat"),
                 score_map(grid, percentiles, "moon_mare"),
                 gap_map(grid, candidates, turkiye_top),
                 dimension_chart(dims)):
        print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
