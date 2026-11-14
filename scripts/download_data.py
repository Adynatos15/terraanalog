"""Fetch the source rasters TerraAnalog is built on.

WorldClim 2.1 at 10 arc-minutes: bioclimatic variables (~50 MB) and
elevation (~1 MB). Both are open data under the WorldClim licence.
https://worldclim.org/data/worldclim21.html

Already-downloaded archives are skipped, so rerunning is cheap.
"""
from __future__ import annotations

import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Tuple

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
BASE_URL = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/"
CHUNK_BYTES = 1 << 20
TIMEOUT_SECONDS = 600

ARCHIVES: List[Tuple[str, str]] = [
    ("wc2.1_10m_elev.zip", "elevation"),
    ("wc2.1_10m_bio.zip", "bioclimatic variables"),
]


def _download(name: str, description: str) -> Path:
    destination = RAW_DIR / name
    if destination.exists():
        print("already present, skipping:", name)
        return destination

    url = BASE_URL + name
    print("downloading", description, "from", url)
    started = time.time()

    request = urllib.request.Request(url, headers={"User-Agent": "TerraAnalog/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            with open(destination, "wb") as handle:
                while True:
                    chunk = response.read(CHUNK_BYTES)
                    if not chunk:
                        break
                    handle.write(chunk)
    except Exception as exc:
        # A truncated file would fail confusingly later, so remove it.
        if destination.exists():
            destination.unlink()
        raise RuntimeError("download failed for " + name + ": " + str(exc)) from exc

    size_mb = destination.stat().st_size / 1e6
    print("  %.1f MB in %.1fs" % (size_mb, time.time() - started))
    return destination


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for name, description in ARCHIVES:
        archive = _download(name, description)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(RAW_DIR)
        print("  extracted", name)

    boundaries = RAW_DIR / "naturalearth" / "ne_50m_admin_0_countries.geojson"
    if not boundaries.exists():
        sys.path.insert(0, str(RAW_DIR.parent.parent / "src"))
        import countries

        print("downloading Natural Earth country boundaries")
        request = urllib.request.Request(countries.SOURCE_URL,
                                         headers={"User-Agent": "TerraAnalog/1.0"})
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            payload = response.read()
        boundaries.parent.mkdir(parents=True, exist_ok=True)
        boundaries.write_bytes(payload)

    print()
    print("Source data is in", RAW_DIR)
    print("Next: python scripts/build_modis.py, then python src/earth_features.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
