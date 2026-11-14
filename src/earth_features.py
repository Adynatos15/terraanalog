"""Build the global Earth feature grid the analog search runs on.

Source: WorldClim 2.1 bioclimatic variables and elevation at 10 arc-minutes
(https://worldclim.org/data/worldclim21.html), a widely used open climate
climatology. Every variable below is measured, not modeled by us.

The grid is land-only: ocean cells carry no analog meaning and dropping
them cuts the search space by roughly two thirds.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# WorldClim bioclimatic variables actually used, and why each one earns a
# place in a planetary-analog search.
BIO_VARIABLES: Dict[str, str] = {
    "bio_1": "annual_mean_temp_c",      # baseline thermal regime
    "bio_2": "diurnal_range_c",         # day-night swing: core Moon/Mars trait
    "bio_4": "temp_seasonality",        # how variable the year is
    "bio_5": "max_temp_warmest_c",      # upper thermal extreme
    "bio_6": "min_temp_coldest_c",      # lower thermal extreme
    "bio_7": "annual_temp_range_c",     # total thermal amplitude
    "bio_12": "annual_precip_mm",       # aridity: the single strongest filter
    "bio_14": "driest_month_precip_mm",  # hyperarid persistence
    "bio_15": "precip_seasonality",     # whether dryness is year-round
}

# Standard atmosphere constants, for converting elevation to surface
# pressure. Mars' mean surface pressure is about 6 hPa; Earth cannot come
# close, but high plateaus are the closest available low-pressure
# environments and that ordering is what the search uses.
SEA_LEVEL_PRESSURE_HPA = 1013.25
TEMPERATURE_LAPSE_K_PER_M = 0.0065
SEA_LEVEL_TEMP_K = 288.15
BAROMETRIC_EXPONENT = 5.255

# Window over which terrain roughness is measured, in grid cells. At 10
# arc-minutes one cell is roughly 18 km at the equator, so a 3-cell window
# describes regional ruggedness rather than local slope.
ROUGHNESS_WINDOW = 3


def _read_raster(path: Path) -> tuple:
    with rasterio.open(path) as src:
        band = src.read(1).astype("float32")
        nodata = src.nodata
        transform = src.transform
        shape = band.shape
    if nodata is not None:
        band[band == nodata] = np.nan
    band[band < -1e30] = np.nan
    return band, transform, shape


def _pressure_hpa(elevation_m: np.ndarray) -> np.ndarray:
    """Barometric formula on the international standard atmosphere."""
    ratio = 1.0 - (TEMPERATURE_LAPSE_K_PER_M * elevation_m) / SEA_LEVEL_TEMP_K
    ratio = np.clip(ratio, 1e-6, None)
    return SEA_LEVEL_PRESSURE_HPA * ratio**BAROMETRIC_EXPONENT


def _roughness(elevation_m: np.ndarray, window: int = ROUGHNESS_WINDOW) -> np.ndarray:
    """Local standard deviation of elevation, a standard ruggedness proxy.

    NaN cells (ocean) are filled with the local mean before filtering so
    coastlines do not manufacture false ruggedness.
    """
    filled = np.where(np.isnan(elevation_m), 0.0, elevation_m)
    valid = (~np.isnan(elevation_m)).astype("float32")

    sum_x = ndimage.uniform_filter(filled, size=window, mode="nearest")
    sum_x2 = ndimage.uniform_filter(filled**2, size=window, mode="nearest")
    weight = ndimage.uniform_filter(valid, size=window, mode="nearest")
    weight = np.where(weight < 1e-6, np.nan, weight)

    mean = sum_x / weight
    variance = np.clip(sum_x2 / weight - mean**2, 0.0, None)
    rough = np.sqrt(variance)
    rough[np.isnan(elevation_m)] = np.nan
    return rough


MODIS_SUMMARY = OUT_DIR / "modis_summary.npz"

# Ice sheet interiors have no vegetation index in the MODIS NDVI product,
# because there is nothing to measure. They are bare, so they are given the
# index of bare ground rather than dropped. Bounds match analog.ice_sheet_mask.
BARE_GROUND_NDVI = 0.0
ICE_LAT_MAX_SOUTH = -60.0
GREENLAND_BOX = (59.0, 84.0, -73.0, -11.0)


def _merge_modis(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach NASA MODIS surface temperature and NDVI, cell by cell.

    The MODIS layers were requested on the same 10 arc-minute grid as
    WorldClim (see modis_gibs.py), so the merge is an exact index lookup.
    """
    if not MODIS_SUMMARY.exists():
        print("MODIS summary not found; run scripts/build_modis.py for surface temperature.")
        return frame

    summary = np.load(MODIS_SUMMARY)
    rows = frame["grid_row"].to_numpy()
    cols = frame["grid_col"].to_numpy()
    out = frame.copy()
    for name in summary.files:
        out[name] = summary[name][rows, cols].astype("float32")

    lat = out["latitude"].to_numpy()
    lon = out["longitude"].to_numpy()
    lat_lo, lat_hi, lon_lo, lon_hi = GREENLAND_BOX
    on_ice = (lat < ICE_LAT_MAX_SOUTH) | (
        (lat >= lat_lo) & (lat <= lat_hi) & (lon >= lon_lo) & (lon <= lon_hi)
    )
    for name in ("ndvi_max", "ndvi_mean"):
        if name in out:
            missing_on_ice = on_ice & out[name].isna().to_numpy()
            out.loc[missing_on_ice, name] = BARE_GROUND_NDVI
    return out


def build_grid() -> pd.DataFrame:
    """Assemble every variable into one land-only table of grid cells."""
    elev_path = RAW_DIR / "wc2.1_10m_elev.tif"
    if not elev_path.exists():
        raise FileNotFoundError(
            "WorldClim rasters not found in " + str(RAW_DIR) +
            ". Run scripts/download_data.py first."
        )

    elevation, transform, shape = _read_raster(elev_path)
    n_rows, n_cols = shape

    # Cell-center coordinates from the raster's affine transform.
    cols = np.arange(n_cols)
    rows = np.arange(n_rows)
    lon = transform.c + (cols + 0.5) * transform.a
    lat = transform.f + (rows + 0.5) * transform.e
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    data = {
        "latitude": lat_grid,
        "longitude": lon_grid,
        "elevation_m": elevation,
        "surface_pressure_hpa": _pressure_hpa(elevation),
        "terrain_roughness_m": _roughness(elevation),
    }

    for bio_key, column in BIO_VARIABLES.items():
        number = bio_key.split("_")[1]
        band, _, band_shape = _read_raster(RAW_DIR / ("wc2.1_10m_" + bio_key + ".tif"))
        if band_shape != shape:
            raise ValueError("Raster shape mismatch for " + bio_key)
        data[column] = band

    frame = pd.DataFrame({k: v.ravel() for k, v in data.items()})
    frame["grid_row"] = np.repeat(rows, n_cols).astype("int32")
    frame["grid_col"] = np.tile(cols, n_rows).astype("int32")

    # Land only: elevation is the authoritative land mask here, and any
    # cell missing a climate variable cannot be scored anyway.
    frame = frame.dropna().reset_index(drop=True)
    frame = _merge_modis(frame)

    # Aridity index used throughout: lower means drier. Kept explicit
    # rather than derived in the UI so every consumer sees one definition.
    frame["aridity_mm_per_year"] = frame["annual_precip_mm"]

    return frame


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = build_grid()

    out_path = OUT_DIR / "earth_grid.parquet"
    try:
        frame.to_parquet(out_path, index=False)
        written = out_path
    except Exception:
        written = OUT_DIR / "earth_grid.pkl"
        frame.to_pickle(written)

    metadata = {
        "source": "WorldClim 2.1, 10 arc-minute resolution",
        "url": "https://worldclim.org/data/worldclim21.html",
        "n_land_cells": int(len(frame)),
        "columns": list(frame.columns),
        "cell_size_degrees": 1 / 6,
    }
    (OUT_DIR / "earth_grid_meta.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("land cells:", len(frame))
    print("written:", written)
    print(frame[[
        "elevation_m", "surface_pressure_hpa", "terrain_roughness_m",
        "annual_precip_mm", "diurnal_range_c", "annual_temp_range_c",
        "annual_mean_temp_c",
    ]].describe().round(1).to_string())


if __name__ == "__main__":
    main()
