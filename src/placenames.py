"""Turn candidate coordinates into human-readable place names.

Candidates are far more persuasive as "Altiplano, Potosi, Bolivia" than as
"-20.8, -67.4". Names come from OpenStreetMap's Nominatim service, which is
rate limited to one request per second and requires an identifying
User-Agent. Results are cached on disk so a rerun costs nothing.

Naming is cosmetic: if the service is unreachable the search still works and
the UI falls back to coordinates.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Optional

CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "placenames.json"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
USER_AGENT = "TerraAnalog/1.0 (NASA Space Apps 2026 project)"
REQUEST_INTERVAL_SECONDS = 1.1
REQUEST_TIMEOUT_SECONDS = 20

_last_request_at = 0.0


def _load_cache() -> Dict[str, str]:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return {}
    return {}


def _save_cache(cache: Dict[str, str]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def _key(latitude: float, longitude: float) -> str:
    return "%.3f,%.3f" % (latitude, longitude)


def _compose(address: dict) -> str:
    """Build a short label from the most useful address parts available.

    Candidate sites are remote, so the usual city/town fields are usually
    absent and the natural-feature and region fields carry the meaning.
    """
    parts = []
    for field in ("natural", "mountain_range", "region", "state_district",
                  "county", "state", "province"):
        value = address.get(field)
        if value and value not in parts:
            parts.append(value)
        if len(parts) >= 2:
            break

    country = address.get("country")
    if country and country not in parts:
        parts.append(country)

    return ", ".join(parts) if parts else ""


def _reverse_one(latitude: float, longitude: float) -> Optional[str]:
    global _last_request_at

    elapsed = time.time() - _last_request_at
    if elapsed < REQUEST_INTERVAL_SECONDS:
        time.sleep(REQUEST_INTERVAL_SECONDS - elapsed)

    query = urllib.parse.urlencode(
        {
            "lat": "%.4f" % latitude,
            "lon": "%.4f" % longitude,
            "format": "jsonv2",
            "zoom": "8",
            "addressdetails": "1",
            # Without this, remote regions come back in the local script
            # (Tibetan, Uyghur, Cyrillic), which most judges cannot read.
            "accept-language": "en",
        }
    )
    request = urllib.request.Request(
        NOMINATIM_URL + "?" + query, headers={"User-Agent": USER_AGENT}
    )

    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None
    finally:
        _last_request_at = time.time()

    if not isinstance(payload, dict):
        return None
    label = _compose(payload.get("address", {}) or {})
    return label or payload.get("display_name", "") or None


def lookup(latitude: float, longitude: float) -> str:
    """Return a place label, falling back to coordinates."""
    cache = _load_cache()
    key = _key(latitude, longitude)
    if key in cache:
        return cache[key]

    label = _reverse_one(latitude, longitude)
    if label:
        cache[key] = label
        _save_cache(cache)
        return label

    return "%.2f, %.2f" % (latitude, longitude)


def annotate(frame, lat_col: str = "latitude", lon_col: str = "longitude"):
    """Add a place_name column to a small candidate frame."""
    out = frame.copy()
    out["place_name"] = [
        lookup(float(lat), float(lon))
        for lat, lon in zip(out[lat_col], out[lon_col])
    ]
    return out
