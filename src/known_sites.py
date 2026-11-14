"""Established terrestrial analog sites, with why each one was chosen.

Used two ways:

* **Validation.** If the engine is sound, sites chosen for their
  *environment* (cold, dry, high, barren) must score well for the profiles
  they are used for. Sites chosen for their *geology* (an impact structure,
  an anorthosite body, a mineral assemblage) or for *operations* (a habitat,
  a training ground) need not, because climate and terrain cannot see
  geology. The validation reports these groups separately rather than
  pooling them into one flattering number.

* **Exclusion.** The challenge asks for analogs that have had less attention,
  so these sites and their surroundings are masked when searching for new
  candidates.

Coordinates are approximate centroids of the area used in the cited work,
not facility positions. References are the first place to check a claim;
several sites have a much larger literature.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

ENVIRONMENT = "environment"
GEOLOGY = "geology"
OPERATIONS = "operations"


@dataclass(frozen=True)
class KnownSite:
    name: str
    country: str
    latitude: float
    longitude: float
    profiles: Tuple[str, ...]   # profile keys this site is an analog for
    basis: str                  # why it was chosen: environment | geology | operations
    note: str
    reference: str

    @property
    def used_for(self) -> str:
        bodies = sorted({"Mars" if p.startswith("mars") else "Moon" for p in self.profiles})
        return " / ".join(bodies)


KNOWN_SITES: List[KnownSite] = [
    # ---------------------------------------------------------------- Mars
    KnownSite(
        "Atacama Desert (Yungay)", "Chile", -24.1, -70.0, ("mars_midlat",), ENVIRONMENT,
        "Hyperarid core; Mars-like soils that defeated the Viking life-detection approach.",
        "Navarro-Gonzalez et al. (2003) Science 302:1018",
    ),
    KnownSite(
        "Licancabur and the High Andes lakes", "Bolivia / Chile", -22.83, -67.88,
        ("mars_midlat",), ENVIRONMENT,
        "High-altitude lakes under extreme UV and low pressure.",
        "Cabrol et al. (2009) J. Geophys. Res. 114, G00D06 (High-Lakes Project)",
    ),
    KnownSite(
        "Qaidam Basin", "China", 37.8, 93.0, ("mars_midlat",), ENVIRONMENT,
        "High, cold, hyperarid basin with evaporites and yardangs.",
        "Xiao et al. (2017) Earth-Science Reviews 164:84",
    ),
    KnownSite(
        "Death Valley", "United States", 36.5, -117.0, ("mars_midlat",), ENVIRONMENT,
        "Hot hyperarid basin with evaporite mineralogy.",
        "Preston & Dartnell (2014) Int. J. Astrobiology 13:81 (review)",
    ),
    KnownSite(
        "Dallol / Danakil Depression", "Ethiopia", 14.24, 40.30, ("mars_midlat",), ENVIRONMENT,
        "Hot, acid, hypersaline hydrothermal field; limits of habitability.",
        "Cavalazzi et al. (2019) Astrobiology 19:553",
    ),
    KnownSite(
        "Erfoud - Merzouga (Ibn Battuta Centre sites)", "Morocco", 31.1, -4.0,
        ("mars_midlat",), ENVIRONMENT,
        "Saharan field sites used for rover and instrument testing.",
        "Ori et al. (2013) Planetary and Space Science (Ibn Battuta Centre)",
    ),
    KnownSite(
        "Arkaroola", "Australia", -30.3, 139.35, ("mars_midlat",), ENVIRONMENT,
        "Arid ranges with hydrothermal deposits; proposed Mars analog station.",
        "Thomas & Walter (2002) Australian Mars Exploration Conference",
    ),
    KnownSite(
        "McMurdo Dry Valleys (Beacon Valley)", "Antarctica", -77.82, 160.65,
        ("mars_polar", "mars_midlat", "moon_polar"), ENVIRONMENT,
        "Cold hyperarid valleys with buried ground ice.",
        "Marchant & Head (2007) Icarus 192:187",
    ),
    KnownSite(
        "Haughton Crater, Devon Island", "Canada", 75.38, -89.68,
        ("mars_polar", "moon_polar", "moon_mare"), ENVIRONMENT,
        "Polar desert impact crater; long-running analog field campaigns.",
        "Lee & Osinski (2005) Meteoritics & Planetary Science 40:1755",
    ),
    KnownSite(
        "Axel Heiberg Island (Expedition Fiord)", "Canada", 79.4, -90.75,
        ("mars_polar",), ENVIRONMENT,
        "Perennial cold springs flowing through thick permafrost.",
        "Andersen, Pollard & McKay (2002) J. Geophys. Res. 107(E3)",
    ),
    KnownSite(
        "Svalbard (Bockfjorden volcanic complex)", "Norway", 79.43, 13.33,
        ("mars_polar",), GEOLOGY,
        "Volcanics with carbonate deposits like those in Martian meteorite ALH84001.",
        "Steele et al. (2007) Meteoritics & Planetary Science 42:1549 (AMASE)",
    ),
    KnownSite(
        "Rio Tinto", "Spain", 37.72, -6.56, ("mars_midlat",), GEOLOGY,
        "Acidic iron-rich river forming jarosite, as found by the Opportunity rover.",
        "Fernandez-Remolar et al. (2005) Earth Planet. Sci. Lett. 240:149",
    ),
    KnownSite(
        "Pilbara (Strelley Pool)", "Australia", -21.2, 119.7, ("mars_midlat",), GEOLOGY,
        "3.4-billion-year-old stromatolites; early-life biosignature analog.",
        "Allwood et al. (2006) Nature 441:714",
    ),
    KnownSite(
        "Salda Lake", "Turkiye", 37.55, 29.68, ("mars_midlat",), GEOLOGY,
        "Hydromagnesite carbonates compared with those in Jezero crater.",
        "Garczynski et al. (2020) Lunar and Planetary Science Conference LI",
    ),
    KnownSite(
        "Holuhraun", "Iceland", 64.87, -16.83, ("mars_midlat",), GEOLOGY,
        "Fresh basaltic lava and sand; sediment transport analog.",
        "Ewing et al. (2019) LPSC L (SAND-E project)",
    ),
    KnownSite(
        "Ka'u Desert, Kilauea", "United States", 19.35, -155.33,
        ("mars_midlat",), GEOLOGY,
        "Acid-altered basaltic tephra with opaline silica coatings.",
        "Seelos et al. (2010) Geophys. Res. Lett. 37, L01202",
    ),
    KnownSite(
        "Mauna Kea (Pu'u Nene)", "United States", 19.7, -155.45,
        ("mars_midlat", "moon_mare"), GEOLOGY,
        "Source of the JSC Mars-1 regolith simulant; lunar ISRU test site.",
        "Allen et al. (1998) Lunar and Planetary Science XXIX (JSC Mars-1)",
    ),
    KnownSite(
        "Tenerife (Teide)", "Spain", 28.27, -16.64, ("mars_midlat",), GEOLOGY,
        "Volcanic terrain used to test ExoMars Raman spectroscopy.",
        "Lalla et al. (2016) Geoscience Frontiers 7:673",
    ),
    KnownSite(
        "Lanzarote (La Corona lava tube)", "Spain", 29.15, -13.47,
        ("mars_midlat", "moon_mare"), GEOLOGY,
        "Lava tubes used in ESA PANGAEA astronaut geology training.",
        "Sauro et al. (2020) Earth-Science Reviews 209:103288",
    ),
    KnownSite(
        "Lonar Crater", "India", 19.98, 76.51, ("moon_mare", "mars_midlat"), GEOLOGY,
        "Impact crater in basalt, a rare terrestrial match to lunar mare targets.",
        "Fredriksson et al. (1973) Science 180:862",
    ),
    KnownSite(
        "Mars Desert Research Station, Utah", "United States", 38.41, -110.79,
        ("mars_midlat",), OPERATIONS,
        "Habitat for simulated crewed Mars missions.",
        "The Mars Society, MDRS (2001-present)",
    ),
    KnownSite(
        "Ramon Crater (D-MARS)", "Israel", 30.6, 34.8, ("mars_midlat",), OPERATIONS,
        "Desert habitat used for simulated Mars missions.",
        "D-MARS / Israel Space Agency analog missions (2018-present)",
    ),
    # ---------------------------------------------------------------- Moon
    KnownSite(
        "Nordlinger Ries Crater", "Germany", 48.88, 10.56, ("moon_mare",), GEOLOGY,
        "Impact structure used for Apollo 14 crew geology training.",
        "Phinney et al. (1978) Proc. 9th Lunar Planet. Sci. Conf. (Apollo training)",
    ),
    KnownSite(
        "Meteor Crater (Barringer)", "United States", 35.03, -111.02, ("moon_mare",), GEOLOGY,
        "Simple impact crater used for Apollo crew training.",
        "Kring (2017) Guidebook to the Geology of Barringer Meteorite Crater, LPI",
    ),
    KnownSite(
        "SP Crater, San Francisco Volcanic Field", "United States", 35.58, -111.63,
        ("moon_mare",), OPERATIONS,
        "Apollo training ground and NASA Desert RATS field tests.",
        "Ross et al. (2013) Acta Astronautica 90:182 (Desert RATS)",
    ),
    KnownSite(
        "Craters of the Moon", "United States", 43.42, -113.52, ("moon_mare",), GEOLOGY,
        "Basaltic lava field used for Apollo 14 training and NASA FINESSE.",
        "Hughes et al. (2019) Planetary and Space Science 169:15 (FINESSE)",
    ),
    KnownSite(
        "Kilbourne Hole", "United States", 31.97, -106.97, ("moon_mare",), GEOLOGY,
        "Maar crater used for Apollo crew training.",
        "Schmitt (1987) Lunar Bases and Space Activities of the 21st Century",
    ),
    KnownSite(
        "Sudbury impact structure", "Canada", 46.6, -81.18, ("moon_mare",), GEOLOGY,
        "Impact melt and breccias used for Apollo 16 and 17 training.",
        "Phinney et al. (1978) Proc. 9th Lunar Planet. Sci. Conf. (Apollo training)",
    ),
    KnownSite(
        "Mistastin Lake impact structure", "Canada", 55.88, -63.31, ("moon_mare",), GEOLOGY,
        "Anorthosite target rock, like the lunar highlands.",
        "Mader & Osinski (2018) Geochim. Cosmochim. Acta 236:214",
    ),
    KnownSite(
        "Pavilion Lake", "Canada", 50.86, -121.74, ("moon_mare", "mars_midlat"), OPERATIONS,
        "NASA research project testing exploration operations and science teams.",
        "Lim et al. (2011) GSA Special Paper 483:85 (PLRP)",
    ),
    KnownSite(
        "Concordia Station, Dome C", "Antarctica", -75.10, 123.33,
        ("moon_polar", "mars_polar"), ENVIRONMENT,
        "Extreme cold, isolation and darkness; ESA long-duration analog.",
        "ESA Concordia isolation studies (2005-present)",
    ),
]

# Radius around a known site treated as "already studied". Generous on
# purpose: proposing the next valley over is not a discovery.
EXCLUSION_RADIUS_KM = 300.0


def sites_for(profile_key: str, basis: str = "") -> List[KnownSite]:
    """Sites used as analogs for a profile, optionally only of one basis."""
    return [
        s for s in KNOWN_SITES
        if profile_key in s.profiles and (not basis or s.basis == basis)
    ]
