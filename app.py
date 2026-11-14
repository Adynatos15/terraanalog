"""TerraAnalog - a systematic search of Earth for Moon and Mars analogs.

NASA Space Apps Challenge 2026.
Challenge: Identify Earth Locations that Analog the Permanent Moon Base
Locations and Mars.

Run with:  streamlit run app.py
Every table outside the live map is read from docs/results/, which
scripts/run_analysis.py and scripts/turkiye_case.py write.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

import analog  # noqa: E402
from known_sites import KNOWN_SITES  # noqa: E402
from planet_profiles import PROFILE_ORDER, get_profile  # noqa: E402

DATA_DIR = ROOT / "data" / "processed"
RESULTS = ROOT / "docs" / "results"
FIGURES = ROOT / "docs" / "figures"
CARTO_DARK_STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"

# The global map is drawn from a coarsened copy of the grid: 808k points
# render, but slowly enough to ruin a live demo.
DISPLAY_CELL_DEGREES = 0.5

st.set_page_config(page_title="TerraAnalog", page_icon="🌋", layout="wide")


# ------------------------------------------------------------------ data
@st.cache_data(show_spinner=False)
def load_grid() -> pd.DataFrame:
    path = DATA_DIR / "earth_grid.parquet"
    if not path.exists():
        raise FileNotFoundError(
            "Earth grid not found. Run scripts/download_data.py, "
            "scripts/build_modis.py and src/earth_features.py first."
        )
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_result(name: str) -> pd.DataFrame:
    path = RESULTS / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def display_layer(profile_key: str, hide_ice: bool) -> pd.DataFrame:
    """Score one profile and coarsen it so the global map draws instantly."""
    grid = load_grid()
    profile = get_profile(profile_key)
    features = sorted({d.feature for d in profile.dimensions})
    percentiles = analog.to_percentiles(grid, features)
    scores = analog.score(grid, percentiles, profile)["analog_score"].to_numpy()
    keep = np.isfinite(scores)
    if hide_ice:
        keep &= ~analog.ice_sheet_mask(grid)
    cell = DISPLAY_CELL_DEGREES
    binned = pd.DataFrame({
        "lat": (grid["latitude"].to_numpy()[keep] / cell).round() * cell,
        "lon": (grid["longitude"].to_numpy()[keep] / cell).round() * cell,
        "analog_score": scores[keep],
    })
    return binned.groupby(["lat", "lon"], as_index=False)["analog_score"].max()


def score_color(score: float) -> list:
    """Dark blue (poor) through rust to pale gold (strong). Readable on dark."""
    s = float(np.clip((score - 0.3) / 0.6, 0.0, 1.0))
    if s < 0.5:
        t = s / 0.5
        return [int(24 + 23 * t), int(35 + 39 * t), int(58 + 54 * t), 150]
    t = (s - 0.5) / 0.5
    return [int(47 + 208 * t), int(74 + 143 * t), int(112 + 48 * t), 175 + int(60 * t)]


# ------------------------------------------------------------------ search tab
def render_profile(profile) -> None:
    st.subheader(profile.name)
    st.write(profile.summary)
    if profile.provenance != "physics":
        st.warning(
            "This variant was derived after seeing the validation result "
            "(see the Physics vs practice tab), so its validation score is "
            "optimistic."
        )
    left, right = st.columns(2)
    with left:
        st.markdown("**What this environment actually is**")
        st.info(profile.reality)
    with right:
        st.markdown("**What Earth cannot reproduce**")
        st.warning(profile.earth_limits)


def render_map(profile_key: str, hide_ice: bool) -> None:
    layer = display_layer(profile_key, hide_ice).copy()
    layer["color"] = layer["analog_score"].apply(score_color)
    known = pd.DataFrame([
        {"lat": s.latitude, "lon": s.longitude, "name": s.name, "used_for": s.used_for}
        for s in KNOWN_SITES
    ])
    st.pydeck_chart(pdk.Deck(
        layers=[
            pdk.Layer("ScatterplotLayer", data=layer, get_position=["lon", "lat"],
                      get_fill_color="color", get_radius=32000, pickable=False),
            pdk.Layer("ScatterplotLayer", data=known, get_position=["lon", "lat"],
                      get_fill_color=[242, 238, 230, 230], get_line_color=[14, 20, 32, 255],
                      stroked=True, line_width_min_pixels=1, get_radius=70000, pickable=True),
        ],
        initial_view_state=pdk.ViewState(latitude=20, longitude=20, zoom=0.9),
        map_style=CARTO_DARK_STYLE,
        tooltip={"html": "<b>{name}</b><br/>Established analog: {used_for}"},
    ))
    st.caption(
        "Analog score across every land cell: blue is a poor match, gold a strong "
        "one. White circles are the 31 established analog sites used for validation."
    )


def render_dimensions(profile) -> None:
    rows = []
    for d in profile.dimensions:
        if d.target <= 0.2:
            look = "Earth's lowest"
        elif d.target >= 0.8:
            look = "Earth's highest"
        else:
            look = "around the %d%% mark" % round(d.target * 100)
        rows.append({"Dimension": d.label, "Look for": look, "Weight": d.weight,
                     "Why": d.rationale})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_candidates(profile_key: str) -> None:
    table = load_result("candidates.csv")
    table = table[table["profile"] == profile_key]
    if table.empty:
        st.info("Run scripts/run_analysis.py to compute candidates.")
        return
    show = pd.DataFrame({
        "Rank": table["rank"],
        "Location": table["place_name"],
        "Score": table["analog_score"],
        "Stability": (table["stability"] * 100).round().astype(int).astype(str) + "%",
        "Rain (mm/yr)": table["annual_precip_mm"],
        "Surface mean (C)": table["lst_mean_c"],
        "Day-night swing (C)": table["lst_diurnal_range_c"],
        "Greenest NDVI": table["ndvi_max"],
        "Strongest": table["strongest"],
        "Weakest": table["weakest"],
        "Lat": table["latitude"],
        "Lon": table["longitude"],
    })
    st.dataframe(show, width="stretch", hide_index=True)
    st.caption(
        "Established sites, their 300 km surroundings and the Antarctic and "
        "Greenland ice sheets are excluded. Stability is the share of 300 reruns, "
        "each with every weight randomly scaled by 0.5-1.5, in which a top-ten "
        "pick lands within 250 km of this candidate."
    )
    st.download_button(
        "Download candidates (CSV)", table.to_csv(index=False).encode("utf-8"),
        file_name="terraanalog_%s.csv" % profile_key, mime="text/csv",
    )


def tab_search() -> None:
    left, right = st.columns([3, 1])
    with left:
        key = st.selectbox("Target environment", PROFILE_ORDER,
                           format_func=lambda k: get_profile(k).name)
    with right:
        hide_ice = st.checkbox("Hide ice sheets on map", value=True)
    profile = get_profile(key)
    render_profile(profile)
    render_map(key, hide_ice)
    st.markdown("#### How the search is defined")
    render_dimensions(profile)
    st.markdown("#### Overlooked candidates")
    render_candidates(key)


# ------------------------------------------------------------------ validation tab
def parse_interval(text) -> list:
    """'[0.53, 0.79]' or '[nan, nan]' as written by pandas -> two floats."""
    parts = str(text).strip("[] ").split(",")
    try:
        return [float(p) for p in parts[:2]]
    except ValueError:
        return [float("nan"), float("nan")]


def tab_validation() -> None:
    st.markdown(
        "Before proposing anywhere new, the method has to recognise the places "
        "analog scientists already use. **31 established sites**, each with a "
        "published reference, are scored at their nearest land cell. The "
        "measure is ROC AUC: the chance that a known site outscores a random "
        "piece of land (0.5 = chance, 1.0 = perfect). Sites are split by *why* "
        "they were chosen, because climate and terrain cannot see geology."
    )
    table = load_result("validation.csv")
    if table.empty:
        st.info("Run scripts/run_analysis.py.")
        return
    names = {k: get_profile(k).name for k in PROFILE_ORDER}
    rows = []
    for _, r in table.iterrows():
        ci = parse_interval(r["auc_ci"])
        ci_text = "" if any(np.isnan(x) for x in ci) else "%.2f - %.2f" % tuple(ci)
        rows.append({
            "Profile": names.get(r["profile"], r["profile"]),
            "Sites chosen for": r["sites"], "n": r["n"], "AUC": r["auc"],
            "95% interval": ci_text, "Rainfall alone": r["baseline_auc"],
            "Weights": r["provenance"],
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.markdown("#### What do established sites actually have in common?")
    dims = load_result("dimension_auc.csv")
    for key in ["mars_midlat", "moon_mare", "mars_polar", "moon_polar"]:
        sub = dims[dims["profile"] == key]
        if sub.empty:
            continue
        st.markdown("**%s** (%s sites, n = %d)" % (names[key], sub["sites"].iloc[0], sub["n"].iloc[0]))
        st.bar_chart(sub.set_index("dimension")["auc"], horizontal=True, height=260)
    st.caption("AUC of each dimension on its own. Above 0.5: established sites "
               "share the trait. Below 0.5: they systematically avoid it.")

    st.markdown("#### Every established site, with its source")
    known = load_result("known_sites.csv")
    if not known.empty:
        known = known[known["profile"].isin(["mars_midlat", "moon_mare", "moon_polar", "mars_polar"])]
        st.dataframe(known[["profile", "site", "country", "basis", "analog_score",
                            "top_percent", "reference"]],
                     width="stretch", hide_index=True)


# ------------------------------------------------------------------ physics vs practice
def tab_gap() -> None:
    st.markdown(
        "### The physics-practice gap\n"
        "For the Martian desert, every dimension agrees with where analog "
        "scientists actually work **except one: temperature**. Established "
        "Mars desert analogs are *warmer* than typical land (AUC 0.24 for "
        "*Cold surface*), because field teams go to hot deserts such as Death "
        "Valley and the Danakil. Mars averages about -63 C."
    )
    ablation = load_result("ablation.csv")
    sub = ablation[ablation["profile"] == "mars_midlat"]
    if not sub.empty:
        st.markdown("**Remove one dimension at a time (environment-chosen Mars sites)**")
        st.dataframe(sub[["removed", "auc", "change"]].rename(columns={
            "removed": "Dimension removed", "auc": "AUC without it", "change": "Change"}),
            width="stretch", hide_index=True)
    st.markdown(
        "Removing temperature lifts agreement with practice from 0.66 to 0.77; "
        "every other dimension holds or adds value. So the two rankings below "
        "differ on that one axis, deliberately. The left list is what Mars "
        "physics asks for: none of its top ten averages above 20 C at the "
        "surface. The right list is what current practice rewards: half of its "
        "top ten are hot deserts above 27 C. **The cold, dry, bare, high-swing "
        "deserts that lead on the left are the analogs practice is not looking "
        "for.**"
    )
    candidates = load_result("candidates.csv")
    left, right = st.columns(2)
    for column, key, title in [(left, "mars_midlat", "Physics: Mars as it is"),
                               (right, "mars_midlat_practice", "Practice: Mars as studied")]:
        with column:
            st.markdown("**%s**" % title)
            sub = candidates[candidates["profile"] == key]
            st.dataframe(pd.DataFrame({
                "Rank": sub["rank"], "Location": sub["place_name"],
                "Surface mean (C)": sub["lst_mean_c"],
                "Stability": (sub["stability"] * 100).round().astype(int).astype(str) + "%",
            }), width="stretch", hide_index=True)
    st.caption("The practice variant was derived after seeing the validation, "
               "so its own AUC is optimistic. It is shown to make the gap "
               "visible, not as an independent result.")


# ------------------------------------------------------------------ Turkiye
def tab_turkiye() -> None:
    st.markdown(
        "### Case study: Türkiye\n"
        "The same search, cut to one country. Across every Moon and Mars "
        "profile, Türkiye's best cell is the same place: the **Karapınar "
        "basin** in Konya, beside the Meke Maar. It is the barest ground in "
        "the country and swings about 25 C between day and night at the "
        "surface. For the lunar mare profile it ranks in the top 10% of all "
        "land on Earth."
    )
    image = FIGURES / "turkiye_moon_mare.png"
    if image.exists():
        st.image(str(image), caption="Lunar mare analog score across Türkiye. "
                 "Orange ring: the best cell in the country.")
    top = load_result("turkiye_top.csv")
    if not top.empty:
        st.markdown("**Best cells in Türkiye per profile**")
        st.dataframe(top[top["rank"] <= 3][["profile", "rank", "place_name", "latitude",
                                            "longitude", "analog_score", "global_top_percent",
                                            "ndvi_max", "lst_diurnal_range_c", "weakest"]],
                     width="stretch", hide_index=True)
    places = load_result("turkiye_places.csv")
    if not places.empty:
        st.markdown("**Named places**")
        st.dataframe(places[places["profile"].isin(["moon_mare", "mars_midlat"])][
            ["profile", "place", "analog_score", "global_top_percent", "turkiye_rank",
             "weakest", "why"]], width="stretch", hide_index=True)
    st.markdown(
        "**Why Karapınar is worth a field visit.** The climate signal comes from "
        "the data alone. Independently, the site is a Quaternary monogenetic "
        "volcanic field of cinder cones, maars and lava flows, basaltic to "
        "andesitic in composition (Gall et al., 2021, *Lithos*). We found no "
        "published study using it as a lunar or Martian analog.\n\n"
        "**Where it breaks.** About 390 mm of rain a year is far wetter than "
        "any lunar analog should be; its lavas are only partly basaltic, while "
        "the lunar maria are basalt; and it is farmland in places.\n\n"
        "**Salda Lake**, Türkiye's established Mars analog, ranks low here. "
        "That is expected: it was chosen for its carbonate mineralogy, which "
        "climate data cannot see. The two approaches are complementary."
    )


# ------------------------------------------------------------------ method
def tab_method() -> None:
    st.markdown(
        """
**Data**
- NASA MODIS Aqua land surface temperature, monthly day (13:30) and night
  (01:30), 2020-2024, read from NASA GIBS and decoded through its published
  colour tables (every pixel matched a table entry).
- NASA MODIS Terra NDVI, monthly, 2020-2024, same route.
- WorldClim 2.1 precipitation, seasonality and elevation (SRTM-derived), 10
  arc-minutes; surface pressure from elevation via the barometric formula.
- 31 established analog sites with published references; Natural Earth
  boundaries; OpenStreetMap Nominatim for place names.

**Method**
1. Every variable becomes its rank among all land cells (0 to 1).
2. Each profile sets a target rank and a weight per dimension, from planetary
   conditions, because Earth cannot reach lunar or Martian absolute values.
3. Closeness per dimension = 1 - |rank - target| / furthest reachable distance.
   Score = weighted mean over measured dimensions.
4. Candidates: greedy non-maximum suppression at 400 km; established sites
   (300 km) and ice sheets excluded.
5. Validation: ROC AUC against established sites split by why they were
   chosen, bootstrap 95% intervals, rainfall-only baseline, drop-one ablation.
6. Robustness: 300 reruns with every weight scaled by 0.5-1.5.

**Limits**
- About 18 km cells: a shortlist of regions, not landing sites.
- No geology or mineralogy layer; craters, lava tubes and mineral
  assemblages are invisible to this method.
- Gravity, vacuum, radiation and atmospheric chemistry cannot be scored on
  Earth; each profile states them instead.
"""
    )


def main() -> None:
    st.title("TerraAnalog")
    st.markdown(
        "**Every land cell on Earth, scored against the Moon and Mars, and checked "
        "against the 31 places analog scientists already use.**"
    )
    tabs = st.tabs(["Search", "Validation", "Physics vs practice", "Türkiye", "Method"])
    with tabs[0]:
        tab_search()
    with tabs[1]:
        tab_validation()
    with tabs[2]:
        tab_gap()
    with tabs[3]:
        tab_turkiye()
    with tabs[4]:
        tab_method()


if __name__ == "__main__":
    main()
