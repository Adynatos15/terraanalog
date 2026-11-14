"""Download five years of NASA MODIS surface temperature and NDVI from GIBS
and reduce them to a per-cell climatology (data/processed/modis_summary.npz).

About 180 requests, roughly two minutes. Cached; reruns are instant.
Then run: python src/earth_features.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import modis_gibs  # noqa: E402

OUT = ROOT / "data" / "processed" / "modis_summary.npz"


def main() -> int:
    stacks = {}
    for key in modis_gibs.LAYERS:
        stack, rates = modis_gibs.fetch_stack(key)
        stacks[key] = stack
        print("%-9s months decoded, worst colour match %.4f" % (key, min(rates)))
    summary = modis_gibs.summarize(stacks)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **summary)
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
