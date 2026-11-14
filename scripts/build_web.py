"""Build the static web map data (docs/web) from the grid and results.

Writes one Web Mercator PNG score layer per profile, a 1-degree score
lookup for click queries, and a JSON bundle with candidates, known sites
and validation numbers. The page itself is docs/index.html.
"""
from __future__ import annotations

import base64
import warnings
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import analog  # noqa: E402
from planet_profiles import PROFILE_ORDER, get_profile  # noqa: E402
from render_figures import CMAP  # noqa: E402

GRID = ROOT / "data" / "processed" / "earth_grid.parquet"
RESULTS = ROOT / "docs" / "results"
OUT = ROOT / "docs" / "web"

ROWS, COLS, CELLS_PER_DEG = 1080, 2160, 6
MERC_LAT = 85.0511287798
VMIN, VMAX = 0.30, 0.90
LOOKUP_NODATA = 255


def score_image(frame: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    """Scores on the full 10-arc-minute lat/lon grid, NaN where no land."""
    img = np.full((ROWS, COLS), np.nan)
    img[frame["grid_row"].to_numpy(), frame["grid_col"].to_numpy()] = scores
    return img


def to_mercator(img: np.ndarray) -> np.ndarray:
    """Resample an equirectangular grid onto Web Mercator rows."""
    height = COLS  # square tile-world
    y = (np.arange(height) + 0.5) / height
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * y))))
    src = np.clip(((90 - lat) * CELLS_PER_DEG).astype(int), 0, ROWS - 1)
    return img[src]


def write_png(img: np.ndarray, path: Path) -> None:
    norm = np.clip((img - VMIN) / (VMAX - VMIN), 0, 1)
    rgba = (CMAP(np.nan_to_num(norm)) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(np.isnan(img), 0, 235)
    Image.fromarray(rgba).save(path, optimize=True)


def lookup_grid(img: np.ndarray) -> str:
    """1-degree mean score as base64 uint8 (0-250), 255 = no data."""
    blocks = img.reshape(180, CELLS_PER_DEG, 360, CELLS_PER_DEG)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(blocks, axis=(1, 3))
    out = np.where(np.isnan(mean), LOOKUP_NODATA,
                   np.round(np.clip(mean, 0, 1) * 250)).astype(np.uint8)
    return base64.b64encode(out.tobytes()).decode("ascii")


def records(path: Path) -> list:
    frame = pd.read_csv(path)
    return json.loads(frame.to_json(orient="records"))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    grid = pd.read_parquet(GRID)
    features = sorted({d.feature for k in PROFILE_ORDER for d in get_profile(k).dimensions})
    percentiles = analog.to_percentiles(grid, features)
    ice = analog.ice_sheet_mask(grid)

    profiles = []
    for key in PROFILE_ORDER:
        profile = get_profile(key)
        scores = analog.score(grid, percentiles, profile)["analog_score"].to_numpy()
        scores = np.where(ice, np.nan, scores)
        img = score_image(grid, scores)
        write_png(to_mercator(img), OUT / f"score_{key}.png")
        profiles.append({
            "key": key,
            "name": profile.name,
            "provenance": profile.provenance,
            "lookup": lookup_grid(img),
            "dimensions": [{"label": d.label, "weight": d.weight} for d in profile.dimensions],
        })
        print("layer:", key, flush=True)

    bundle = {
        "profiles": profiles,
        "candidates": records(RESULTS / "candidates.csv"),
        "known_sites": records(RESULTS / "known_sites.csv"),
        "validation": records(RESULTS / "validation.csv"),
        "turkiye_top": records(RESULTS / "turkiye_top.csv"),
        "scale": {"vmin": VMIN, "vmax": VMAX, "merc_lat": MERC_LAT},
    }
    (OUT / "data.json").write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
