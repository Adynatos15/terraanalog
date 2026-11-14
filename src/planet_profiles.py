"""Target environments on the Moon and Mars, and how to look for them on Earth.

Earth cannot reproduce lunar or Martian absolute values. Lunar noon reaches
about +127 C and lunar night about -173 C; Martian mean surface temperature
is roughly -63 C at about 6 hPa of pressure. No terrestrial site comes near
any of those numbers, and a search that pretended otherwise would be
dishonest.

So the search does not match absolute values. For each dimension it asks a
different question: *in which direction does this planetary environment lie,
and how far along that direction can Earth actually go?* Each target below is
therefore expressed as a position in Earth's own distribution, from 0.0
(Earth's minimum) to 1.0 (Earth's maximum), with a weight saying how much
that dimension defines the environment.

This keeps every claim defensible: the engine ranks Earth against Earth, and
the planetary science decides the direction and the weighting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class Dimension:
    """One axis of analogy."""

    feature: str
    target: float   # position in Earth's distribution, 0.0 - 1.0
    weight: float   # relative importance for this environment
    label: str
    rationale: str


@dataclass(frozen=True)
class PlanetProfile:
    key: str
    name: str
    body: str
    summary: str
    reality: str          # the actual planetary conditions, stated plainly
    earth_limits: str     # what Earth categorically cannot reproduce
    dimensions: Tuple[Dimension, ...]
    # Which known-site list validates this profile. Empty means its own key.
    validated_by: str = ""
    # How the weights were set. Every profile is "physics" (set from
    # planetary conditions before validation) unless stated otherwise.
    provenance: str = "physics"

    def site_key(self) -> str:
        return self.validated_by or self.key

    def weights(self) -> Dict[str, float]:
        return {d.feature: d.weight for d in self.dimensions}

    def targets(self) -> Dict[str, float]:
        return {d.feature: d.target for d in self.dimensions}


MOON_EQUATORIAL = PlanetProfile(
    key="moon_mare",
    name="Moon - equatorial mare",
    body="Moon",
    summary=(
        "Airless basaltic plains with the most extreme day-night temperature "
        "swing in the inner solar system and no water in any form."
    ),
    reality=(
        "Surface temperatures swing from about +127 C at lunar noon to about "
        "-173 C at night, a range near 300 K. Pressure is effectively vacuum "
        "(~3e-15 bar). There is no precipitation, no liquid water and no life."
    ),
    earth_limits=(
        "Earth has an atmosphere, so it cannot reproduce vacuum, unfiltered "
        "solar and cosmic radiation, electrostatic dust levitation, or 1/6 g. "
        "Earth's largest diurnal ranges are roughly 20 C against the Moon's "
        "300 C. What transfers is the *regime*: bare rock, no moisture, "
        "thermal cycling as the dominant weathering process."
    ),
    dimensions=(
        Dimension(
            "annual_precip_mm", 0.0, 3.0, "Aridity",
            "The Moon has no water in any form. Any rain at a candidate site "
            "is the sharpest break in the analogy, so this carries the "
            "heaviest weight.",
        ),
        Dimension(
            "lst_diurnal_range_c", 1.0, 2.5, "Day-night surface swing",
            "Thermal cycling of the surface is what breaks down lunar rock. "
            "Measured as MODIS surface temperature at 13:30 minus 01:30, "
            "because air temperature hides most of the surface swing.",
        ),
        Dimension(
            "ndvi_max", 0.0, 2.0, "Bare ground",
            "The lunar surface has no plants. Uses the greenest month of the "
            "year, so a desert that blooms after rain is not counted as bare.",
        ),
        Dimension(
            "terrain_roughness_m", 0.65, 1.0, "Terrain ruggedness",
            "Mare surfaces are cratered and blocky at human scale. "
            "Moderate-to-high ruggedness fits better than a salt flat or an "
            "alpine ridge.",
        ),
        Dimension(
            "driest_month_precip_mm", 0.0, 1.0, "Year-round dryness",
            "A site that is dry only part of the year is a weaker analog.",
        ),
        Dimension(
            "lst_annual_range_c", 0.75, 0.8, "Thermal amplitude",
            "Hottest day minus coldest night across the year: how hard the "
            "surface materials are cycled.",
        ),
        Dimension(
            "surface_pressure_hpa", 0.15, 0.8, "Thin atmosphere",
            "Vacuum is unreachable, but high ground has the thinnest air, the "
            "least weather and the harshest UV on Earth.",
        ),
    ),
)

MOON_POLAR = PlanetProfile(
    key="moon_polar",
    name="Moon - south polar / permanently shadowed",
    body="Moon",
    summary=(
        "The Artemis target region: permanent shadow, water ice at the "
        "coldest temperatures measured anywhere in the solar system."
    ),
    reality=(
        "Permanently shadowed regions near the lunar south pole fall below "
        "-230 C and trap water ice over geological time. Illumination comes "
        "only at grazing angles, and crater walls control both sunlight and "
        "line of sight to Earth."
    ),
    earth_limits=(
        "No terrestrial location is permanently shadowed for billions of "
        "years, and Earth's coldest recorded surface is near -98 C against "
        "the Moon's -230 C. What transfers is cold-trapped ice in shadowed "
        "topography, and operations in permanent low-angle light."
    ),
    dimensions=(
        Dimension(
            "lst_night_min_c", 0.0, 3.0, "Extreme cold",
            "Cold trapping is the defining process of a shadowed polar "
            "region. Measured as the coldest monthly night-time surface "
            "temperature from MODIS.",
        ),
        Dimension(
            "annual_precip_mm", 0.05, 2.0, "Aridity",
            "Lunar polar ice is ancient and cold-trapped, not resupplied by "
            "snowfall. A polar desert fits better than a snowy coast.",
        ),
        Dimension(
            "lst_mean_c", 0.0, 2.0, "Mean thermal regime",
            "Sustained cold, not just a cold snap, is what preserves ice.",
        ),
        Dimension(
            "ndvi_max", 0.0, 1.5, "Bare ground",
            "The lunar poles have no plants; tundra and taiga break the analogy.",
        ),
        Dimension(
            "terrain_roughness_m", 0.7, 1.2, "Shadowing topography",
            "Permanent shadow exists because of crater relief. Flat terrain "
            "cannot produce it.",
        ),
        Dimension(
            "temp_seasonality", 0.8, 0.8, "Illumination extremes",
            "High seasonality stands in for the extreme and prolonged "
            "lighting cycles of the lunar poles.",
        ),
    ),
)

MARS_MIDLATITUDE = PlanetProfile(
    key="mars_midlat",
    name="Mars - mid-latitude desert",
    body="Mars",
    summary=(
        "Cold hyperarid basaltic desert under a thin atmosphere, shaped by "
        "wind and dust: the terrain most Mars missions actually land on."
    ),
    reality=(
        "Mean surface temperature is about -63 C, with diurnal swings of "
        "60-80 K. Mean surface pressure is roughly 6 hPa, under 1% of "
        "Earth's. No liquid water is stable at the surface; dust storms and "
        "aeolian transport dominate surface change."
    ),
    earth_limits=(
        "Earth's atmosphere is about 160 times denser, it has an oxygen-rich "
        "chemistry, a magnetic field that blocks the radiation Mars receives, "
        "and 2.6 times the gravity. Earth's soils are also biologically "
        "active almost everywhere, while Martian regolith is not."
    ),
    dimensions=(
        Dimension(
            "annual_precip_mm", 0.0, 3.0, "Hyperaridity",
            "Liquid water is unstable on the Martian surface. Hyperarid "
            "terrain is the single most important filter.",
        ),
        Dimension(
            "ndvi_max", 0.0, 2.0, "Bare ground",
            "Martian regolith carries no plants. Uses the greenest month of "
            "the year, so seasonal steppe does not pass as desert.",
        ),
        Dimension(
            "surface_pressure_hpa", 0.05, 2.0, "Low pressure",
            "6 hPa is unreachable, but high plateaus give the lowest pressure, "
            "strongest UV and thinnest air on Earth, which matters for testing "
            "hardware and spacesuit systems.",
        ),
        Dimension(
            "lst_mean_c", 0.1, 2.0, "Cold surface",
            "Mars is cold as well as dry. Hot deserts fail this dimension, "
            "which is why the high Atacama outranks the Sahara for Mars work.",
        ),
        Dimension(
            "lst_diurnal_range_c", 1.0, 1.5, "Day-night surface swing",
            "Thin air couples surface temperature tightly to sunlight, "
            "producing large daily swings on both planets. MODIS surface "
            "temperature, 13:30 minus 01:30.",
        ),
        Dimension(
            "driest_month_precip_mm", 0.0, 1.0, "Year-round dryness",
            "Mars has no wet season.",
        ),
        Dimension(
            "terrain_roughness_m", 0.5, 0.8, "Terrain",
            "Landing sites favour moderate relief: trafficable, but with "
            "geological structure worth exploring.",
        ),
    ),
)

MARS_POLAR = PlanetProfile(
    key="mars_polar",
    name="Mars - polar / ground ice",
    body="Mars",
    summary=(
        "Ice-bearing high-latitude terrain with seasonal frost cycling: the "
        "environment behind the search for accessible Martian water."
    ),
    reality=(
        "Martian polar caps cycle carbon dioxide and water ice seasonally, "
        "with temperatures near -125 C at the winter pole. Ground ice "
        "persists under shallow regolith at high latitudes."
    ),
    earth_limits=(
        "Earth's polar regions are far warmer, have liquid water at depth, "
        "an active hydrological cycle, and biology throughout. No seasonal "
        "carbon dioxide frost cycle exists on Earth at all."
    ),
    dimensions=(
        Dimension(
            "lst_night_min_c", 0.02, 2.5, "Winter extreme",
            "Frost cycling and ground ice depend on deep winter cold. "
            "Coldest monthly night-time surface temperature from MODIS.",
        ),
        Dimension(
            "lst_mean_c", 0.05, 2.0, "Permafrost regime",
            "A mean surface temperature below freezing keeps ground ice stable.",
        ),
        Dimension(
            "annual_precip_mm", 0.1, 2.0, "Polar desert",
            "Martian polar terrain is a desert. Snowy maritime polar sites are "
            "poor analogs despite being cold.",
        ),
        Dimension(
            "ndvi_max", 0.0, 1.5, "Bare ground",
            "Martian polar terrain has no plants. Forested cold regions such "
            "as the taiga fail here, however cold they are.",
        ),
        Dimension(
            "lst_annual_range_c", 0.95, 1.5, "Seasonal cycling",
            "The seasonal advance and retreat of frost is the defining "
            "process of Martian polar terrain. Hottest monthly day minus "
            "coldest monthly night at the surface.",
        ),
        Dimension(
            "surface_pressure_hpa", 0.2, 0.8, "Thin atmosphere",
            "Elevation again stands in for Mars' low surface pressure.",
        ),
    ),
)

# The Mars desert profile as analog scientists actually use it.
#
# Validation against established sites showed one dimension working against
# current practice: established Mars desert analogs are *warmer* than typical
# land (AUC 0.24 for "Cold surface"), because field teams go to hot deserts
# such as Death Valley and the Danakil. Dropping that one dimension lifts the
# profile from AUC 0.66 to 0.77 while every other dimension keeps or adds
# value.
#
# This variant was derived AFTER seeing that result, so its own validation
# score is optimistic and is labelled as such everywhere. It exists to make
# the gap visible: the physics profile asks for a cold desert because Mars is
# cold; current practice does not. Sites that rank high under physics but not
# under practice are where cold-desert analogs are being missed.
MARS_MIDLATITUDE_PRACTICE = PlanetProfile(
    key="mars_midlat_practice",
    name="Mars - mid-latitude desert (as currently practised)",
    body="Mars",
    summary=(
        "The same Mars desert profile with the temperature dimension removed: "
        "what established Mars desert analogs actually have in common."
    ),
    reality=MARS_MIDLATITUDE.reality,
    earth_limits=MARS_MIDLATITUDE.earth_limits,
    dimensions=tuple(d for d in MARS_MIDLATITUDE.dimensions if d.feature != "lst_mean_c"),
    validated_by=MARS_MIDLATITUDE.key,
    provenance="calibrated after validation (post hoc)",
)

PROFILES: Dict[str, PlanetProfile] = {
    p.key: p
    for p in (
        MOON_EQUATORIAL,
        MOON_POLAR,
        MARS_MIDLATITUDE,
        MARS_MIDLATITUDE_PRACTICE,
        MARS_POLAR,
    )
}

PROFILE_ORDER: List[str] = [
    MARS_MIDLATITUDE.key,
    MARS_MIDLATITUDE_PRACTICE.key,
    MOON_EQUATORIAL.key,
    MOON_POLAR.key,
    MARS_POLAR.key,
]


def get_profile(key: str) -> PlanetProfile:
    if key not in PROFILES:
        raise KeyError("Unknown profile: " + key)
    return PROFILES[key]
