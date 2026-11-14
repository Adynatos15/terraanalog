"""NASA MODIS land surface temperature and NDVI, read from NASA GIBS.

Why GIBS
--------
NASA Earth Observations (NEO), the usual no-login source for global MODIS
grids, was decommissioned in September 2026. The science archives (LP DAAC,
AppEEARS, Giovanni) all need an Earthdata login. NASA's Global Imagery Browse
Services (GIBS) needs none, but serves pictures rather than numbers.

GIBS publishes, for every layer, the exact colour table it used to draw the
picture: each colour stands for one value interval of the underlying data
(0.6 K wide for surface temperature, about 0.01 for NDVI), and no two
intervals share a colour. Reading the picture back through that table
recovers the data to within one interval. Pixels whose colour is not in the
table are treated as missing, never guessed.

Layers used
-----------
MODIS_Aqua_L3_Land_Surface_Temp_Monthly_Day / _Night
    Aqua crosses the equator at about 13:30 and 01:30 local time, close to
    the daily maximum and minimum of surface temperature. Terra (10:30 and
    22:30) would understate the day-night swing.
MODIS_Terra_L3_NDVI_Monthly
    Monthly vegetation index.

Each layer is requested on exactly the WorldClim 10 arc-minute grid
(2160 x 1080 over the globe), so pixel (row, col) is the same cell as in
earth_features.py.
"""
from __future__ import annotations

import io
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

WMS_URL = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
COLORMAP_URL = "https://gibs.earthdata.nasa.gov/colormaps/v1.3/{name}.xml"
USER_AGENT = "TerraAnalog/1.0 (NASA Space Apps 2026 project)"

GRID_WIDTH = 2160
GRID_HEIGHT = 1080
REQUEST_TIMEOUT_SECONDS = 180
RETRIES = 3
MAX_WORKERS = 4

LAYERS = {
    "lst_day": ("MODIS_Aqua_L3_Land_Surface_Temp_Monthly_Day", "MODIS_Land_Surface_Temp"),
    "lst_night": ("MODIS_Aqua_L3_Land_Surface_Temp_Monthly_Night", "MODIS_Land_Surface_Temp"),
    "ndvi": ("MODIS_Terra_L3_NDVI_Monthly", "MODIS_L3_NDVI"),
}

# Five full years: enough to average out a single anomalous season without
# reaching back to a different climate.
DEFAULT_YEARS = (2020, 2021, 2022, 2023, 2024)

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "gibs"

_INTERVAL = re.compile(r"\[\s*([-0-9.]+)\s*,\s*([-0-9.]+)\s*\)")


def _get(url: str) -> bytes:
    last_error: Optional[Exception] = None
    for attempt in range(RETRIES):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                return response.read()
        except Exception as exc:  # network errors are retried, then raised
            last_error = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError("GIBS request failed: " + url + " (" + str(last_error) + ")")


def load_colormap(name: str) -> Dict[Tuple[int, int, int], float]:
    """Map each RGB colour to the midpoint of the value interval it encodes.

    The open-ended bins at either end ("everything below 200 K") are mapped
    to their inner edge, so an extreme value is reported as the extreme the
    table can express rather than an invented midpoint.
    """
    cache = CACHE_DIR / (name + ".xml")
    if cache.exists():
        text = cache.read_text(encoding="utf-8")
    else:
        text = _get(COLORMAP_URL.format(name=name)).decode("utf-8")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(text, encoding="utf-8")

    entries = re.findall(r"<ColorMapEntry ([^>]*)/>", text)
    table: Dict[Tuple[int, int, int], float] = {}
    valued: List[Tuple[Tuple[int, int, int], float, float]] = []
    for entry in entries:
        if 'nodata="true"' in entry:
            continue
        rgb = re.search(r'rgb="([^"]+)"', entry)
        value = re.search(r'value="([^"]+)"', entry)
        if not rgb or not value:
            continue
        match = _INTERVAL.search(value.group(1))
        if not match:
            continue
        colour = tuple(int(c) for c in rgb.group(1).split(","))
        valued.append((colour, float(match.group(1)), float(match.group(2))))

    if not valued:
        raise ValueError("No usable entries in colormap " + name)

    widths = sorted(hi - lo for _, lo, hi in valued)
    typical = widths[len(widths) // 2]
    for i, (colour, lo, hi) in enumerate(valued):
        if hi - lo > 3 * typical:
            # Open-ended bin at the bottom or top of the scale.
            table[colour] = hi if i == 0 else lo
        else:
            table[colour] = (lo + hi) / 2.0
    return table


def _decode(png: bytes, table: Dict[Tuple[int, int, int], float]) -> Tuple[np.ndarray, float]:
    """Turn a rendered PNG back into values. Returns (values, match_rate)."""
    image = Image.open(io.BytesIO(png)).convert("RGBA")
    pixels = np.asarray(image)
    rgb = pixels[..., :3].astype(np.int32)
    alpha = pixels[..., 3]

    key = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
    lookup_keys = np.array([(r << 16) | (g << 8) | b for (r, g, b) in table], dtype=np.int64)
    lookup_vals = np.array(list(table.values()), dtype=np.float32)
    order = np.argsort(lookup_keys)
    lookup_keys = lookup_keys[order]
    lookup_vals = lookup_vals[order]

    flat = key.ravel().astype(np.int64)
    pos = np.searchsorted(lookup_keys, flat)
    pos = np.clip(pos, 0, len(lookup_keys) - 1)
    hit = lookup_keys[pos] == flat
    hit &= alpha.ravel() > 0

    values = np.full(flat.shape, np.nan, dtype=np.float32)
    values[hit] = lookup_vals[pos[hit]]

    painted = int((alpha > 0).sum())
    match_rate = float(hit.sum()) / painted if painted else 0.0
    return values.reshape(key.shape), match_rate


def fetch_month(key: str, year: int, month: int) -> Tuple[np.ndarray, float]:
    """One month of one layer on the 10 arc-minute global grid."""
    layer, cmap = LAYERS[key]
    cache = CACHE_DIR / key / ("%d-%02d.png" % (year, month))
    if cache.exists():
        png = cache.read_bytes()
    else:
        params = {
            "SERVICE": "WMS",
            "REQUEST": "GetMap",
            "VERSION": "1.3.0",
            "LAYERS": layer,
            "STYLES": "",
            "CRS": "EPSG:4326",
            # WMS 1.3.0 with EPSG:4326 takes lat,lon axis order.
            "BBOX": "-90,-180,90,180",
            "WIDTH": str(GRID_WIDTH),
            "HEIGHT": str(GRID_HEIGHT),
            "FORMAT": "image/png",
            "TRANSPARENT": "TRUE",
            "TIME": "%d-%02d-01" % (year, month),
        }
        png = _get(WMS_URL + "?" + urllib.parse.urlencode(params))
        if not png.startswith(b"\x89PNG"):
            raise RuntimeError("GIBS did not return a PNG for %s %d-%02d: %r"
                               % (layer, year, month, png[:200]))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(png)
    return _decode(png, load_colormap(cmap))


def fetch_stack(key: str, years: Iterable[int] = DEFAULT_YEARS) -> Tuple[np.ndarray, List[float]]:
    """All months of all years for one layer: array of shape (12, H, W).

    Each month is averaged across the years first, so the result is a
    monthly climatology rather than a single year's weather.
    """
    load_colormap(LAYERS[key][1])  # fetch once before the threads start
    jobs = [(y, m) for y in years for m in range(1, 13)]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results = list(pool.map(lambda ym: fetch_month(key, ym[0], ym[1]), jobs))

    rates = [rate for _, rate in results]
    by_month: Dict[int, List[np.ndarray]] = {m: [] for m in range(1, 13)}
    for (year, month), (values, _) in zip(jobs, results):
        by_month[month].append(values)

    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            climatology = np.stack(
                [np.nanmean(np.stack(by_month[m]), axis=0) for m in range(1, 13)]
            )
    return climatology, rates


def summarize(stacks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Annual surface-temperature and vegetation statistics per cell (deg C)."""
    import warnings

    day = stacks["lst_day"] - 273.15
    night = stacks["lst_night"] - 273.15
    ndvi = stacks["ndvi"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        swing = day - night
        return {
            "lst_day_mean_c": np.nanmean(day, axis=0),
            "lst_night_mean_c": np.nanmean(night, axis=0),
            "lst_mean_c": np.nanmean((day + night) / 2.0, axis=0),
            "lst_day_max_c": np.nanmax(day, axis=0),
            "lst_night_min_c": np.nanmin(night, axis=0),
            "lst_diurnal_range_c": np.nanmean(swing, axis=0),
            "lst_annual_range_c": np.nanmax(day, axis=0) - np.nanmin(night, axis=0),
            "ndvi_max": np.nanmax(ndvi, axis=0),
            "ndvi_mean": np.nanmean(ndvi, axis=0),
        }
