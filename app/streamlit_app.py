"""Forest fire simulator: set the forest, the weather and the fire management, and watch the fire spread.

Run with:  streamlit run app/streamlit_app.py   (from the project root, so .streamlit/config.toml applies)
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(Path(__file__).resolve().parent)]

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import player  # noqa: E402
from forest_fire import real_data as rd  # noqa: E402
from forest_fire.weather import dry_spread_probability, wind_factor  # noqa: E402

st.set_page_config(page_title="Forest Fire Simulator", page_icon="🔥", layout="wide")
st.html("""<style>
  @import url("https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap");
  [data-testid="stMetricValue"] { font-family: "JetBrains Mono", monospace !important; }
  [data-testid="stSidebar"] { border-right: 1px solid #3a2c22; }
  [data-testid="stSidebar"] h3 { font-size: 13px; letter-spacing: .08em; text-transform: uppercase; color: #a8968a; margin: 8px 0 0; }
  .verdict { padding: 10px 14px; border-radius: 8px; border: 1px solid #3a2c22; background: #1c1511; margin-bottom: 6px; }
</style>""")

STRATEGY_LABEL = {"none": "None", "firebreaks": "Firebreaks (cleared lines)", "thinning": "Random thinning",
                  "controlled_burns": "Controlled burns (patches)"}
DIRECTIONS = {"→ east": 0.0, "↗ north-east": 45.0, "↑ north": 90.0, "↖ north-west": 135.0, "← west": 180.0,
              "↙ south-west": 225.0, "↓ south": 270.0, "↘ south-east": 315.0}
PRESETS = {
    "Normal year": dict(density=0.65, q_normal=0.6, dryness=0.0, wind_speed=0.0, burn_steps=1, ignition="left_edge",
                        strategy="none", compare=False, spot_prob=0.0),
    "El Niño drought": dict(density=0.65, q_normal=0.6, dryness=0.8, wind_speed=0.0, burn_steps=1, ignition="left_edge",
                            strategy="none", compare=False, spot_prob=0.0),
    "Drought + wind": dict(density=0.8, q_normal=0.6, dryness=0.75, wind_speed=10.0, wind_dir="→ east", burn_steps=10,
                           ignition="centre", strategy="none", compare=False, spot_prob=0.0),
    "Drought + firebreaks": dict(density=0.7, q_normal=0.6, dryness=0.8, wind_speed=0.0, burn_steps=1, ignition="centre",
                                 strategy="firebreaks", budget=0.06, compare=True, spot_prob=0.0),
}
DEFAULTS = dict(size=150, density=0.65, neighbourhood="von_neumann", q_normal=0.6, dryness=0.5, wind_speed=0.0,
                wind_dir="→ east", burn_steps=1, spot_prob=0.0, spot_radius=5, ignition="left_edge", strategy="none",
                budget=0.06, compare=False, seed=42)
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
    st.session_state[k] = st.session_state[k]       # keep values of widgets hidden on this run


def load_preset(name: str):
    st.session_state.update(PRESETS[name])


@st.cache_data
def critical_density_table() -> pd.DataFrame | None:
    """p_c(q) from Phase 2, used to tell the user which side of the threshold their forest is on."""
    path = ROOT / "results" / "phase2_critical_density" / "settings.json"
    if not path.exists():
        return None
    return pd.DataFrame(json.loads(path.read_text())["summary"]["fits"])


def critical_density(q: float, neighbourhood: str) -> float | None:
    tab = critical_density_table()
    if tab is None:
        return None
    d = tab[(tab["neighbourhood"] == neighbourhood)].dropna(subset=["p50"]).sort_values("q")
    if q < d["q"].min():
        return np.inf                                    # too wet: no forest carries fire
    return float(np.interp(q, d["q"], d["p50"]))


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Scenarios")
    cols = st.columns(2)
    for i, name in enumerate(PRESETS):
        cols[i % 2].button(name, key=f"preset_{i}", on_click=load_preset, args=(name,), width="stretch")

    st.markdown("### Forest")
    density = st.slider("Tree density (fraction of ground covered)", 0.30, 1.00, step=0.01, key="density")
    size = st.select_slider("Forest size (cells per side)", [100, 150, 200, 300], key="size")
    neighbourhood = st.radio("Fire can jump to", ["von_neumann", "moore"], key="neighbourhood", horizontal=True,
                             format_func={"von_neumann": "4 side neighbours", "moore": "8 incl. diagonals"}.get)

    st.markdown("### Weather")
    q_normal = st.slider("Spread chance in a normal year (q)", 0.30, 1.00, step=0.01, key="q_normal",
                         help="Chance that a burning tree ignites a neighbouring tree, with normal fuel moisture")
    dryness = st.slider("Dryness (El Niño drought)", 0.0, 1.0, step=0.05, key="dryness",
                        help="0 = normal year, 1 = bone dry. Drought shrinks the chance that fuel fails to catch.")
    q = dry_spread_probability(q_normal, dryness)
    st.caption(f"Spread chance this year: **q = {q:.2f}**")
    wind_speed = st.slider("Wind speed (m/s)", 0.0, 20.0, step=0.5, key="wind_speed")
    wind_dir = st.selectbox("Wind blows towards", list(DIRECTIONS), key="wind_dir", disabled=wind_speed == 0)
    if wind_speed > 0:
        st.caption(f"Downwind spread ×{wind_factor(1, wind_speed):.1f}, upwind ×{wind_factor(-1, wind_speed):.2f}. "
                   + ("Tip: set burn duration ≥ 5 (Advanced) so wind changes the fire's speed." if st.session_state["burn_steps"] < 5 else ""))

    st.markdown("### Ignition")
    ignition = st.radio("Fire starts", list(player.IGNITIONS), key="ignition", format_func=player.IGNITIONS.get)

    st.markdown("### Fire management")
    strategy = st.selectbox("Before the fire season", list(STRATEGY_LABEL), key="strategy", format_func=STRATEGY_LABEL.get)
    budget = st.slider("Trees removed (fraction)", 0.01, 0.30, step=0.01, key="budget", disabled=strategy == "none")
    compare = st.checkbox("Compare side by side with no management", key="compare", disabled=strategy == "none")

    with st.expander("Advanced"):
        burn_steps = st.slider("Burn duration (steps a tree keeps burning)", 1, 15, key="burn_steps",
                               help="1 = the classic percolation model. Longer burns let wind change the fire's speed.")
        spot_prob = st.slider("Ember chance per burning tree per step", 0.0, 0.10, step=0.005, key="spot_prob",
                              help="Wind-blown embers can jump gaps such as firebreaks")
        spot_radius = st.slider("Ember reach (cells)", 1, 15, key="spot_radius", disabled=spot_prob == 0)
        seed = int(st.number_input("Random seed", step=1, key="seed"))

scenario = player.Scenario(size=size, density=density, neighbourhood=neighbourhood, q_normal=q_normal, dryness=dryness,
                           wind_speed=wind_speed, wind_direction=DIRECTIONS[wind_dir], burn_steps=burn_steps,
                           spot_prob=spot_prob, spot_radius=spot_radius if spot_prob > 0 else 0, ignition=ignition,
                           strategy=strategy, budget=budget if strategy != "none" else 0.0, seed=seed)


@st.cache_data(max_entries=24, show_spinner=False)
def run(s: player.Scenario, strategy: str) -> dict:
    return player.simulate(s, strategy)


# ---------------------------------------------------------------- page
st.markdown("## 🔥 Forest fire simulator")
tab_sim, tab_exp, tab_nino, tab_about = st.tabs(["SIMULATE", "EXPERIMENTS", "EL NIÑO DATA", "ABOUT"])

with tab_sim:
    pc = critical_density(q, neighbourhood) if burn_steps == 1 and wind_speed == 0 and spot_prob == 0 else None
    if pc is not None:
        if np.isinf(pc):
            verdict = f"At q = {q:.2f} fuel is so damp that <b>no forest, however dense, can carry a fire across</b>."
        elif density > pc + 0.01:
            verdict = (f"Tree density {density:.2f} is <b>above</b> the critical density {pc:.2f} for q = {q:.2f}: "
                       "fires can spread across the whole forest.")
        elif density < pc - 0.01:
            verdict = (f"Tree density {density:.2f} is <b>below</b> the critical density {pc:.2f} for q = {q:.2f}: "
                       "fires should stay local and burn out.")
        else:
            verdict = f"Tree density {density:.2f} is <b>right at</b> the critical density {pc:.2f}: fires are unpredictable and long."
        st.markdown(f'<div class="verdict">{verdict}</div>', unsafe_allow_html=True)

    with st.spinner("Burning…"):
        main = run(scenario, strategy)
        runs = [(main, STRATEGY_LABEL[strategy] if strategy != "none" else "Wildfire", f"q = {q:.2f}")]
        if compare and strategy != "none":
            runs.append((run(scenario, "none"), "No management", "same forest, same fire"))

    cols = st.columns(len(runs) * 3)
    for k, (r, label, _) in enumerate(runs):
        cols[3 * k].metric(f"{label}: trees burned", f"{r['fraction_burned']:.0%}")
        cols[3 * k + 1].metric(r["reach_label"].capitalize(), "yes" if r["reached_edge"] else "no")
        cols[3 * k + 2].metric("Fire lasted", f"{r['duration']} steps")
    html = player.build_html(runs, scenario)
    components.html(html, height=player.player_height(len(runs)))
    with st.popover("⬇ Export"):
        st.download_button("Replay (standalone HTML)", html, "fire_replay.html", mime="text/html")
        st.download_button("Scenario settings (JSON)", json.dumps(player.scenario_dict(scenario), indent=2),
                           "fire_scenario.json")

with tab_exp:
    st.markdown("Results of the project's experiments (`scripts/run_phase*.py`). Each is an average over many random "
                "forests; settings and seeds are saved next to every figure in `results/`.")
    for folder, title, caption in [
        ("phase1_percolation", "1 · The critical density",
         "Fire crosses the forest only above p ≈ 0.593, and the switch sharpens with forest size."),
        ("phase2_critical_density", "2 · Wetter fuel needs a denser forest",
         "Critical density against spread probability q, for 4 and 8 neighbours."),
        ("phase3_drought", "3 · Drought flips forests from safe to fire-prone",
         "Fraction burned against dryness, for four tree densities."),
        ("phase3_wind_shapes", "3 · Wind drives the head and stalls the back",
         "Burnt area 120 steps after one spark, under different winds."),
        ("phase4_management", "4 · Which fuel treatment saves the most forest?",
         "Under drought: firebreaks save far more trees per tree removed than thinning or random controlled burns."),
        ("phase4_embers", "4 · Do firebreaks survive embers?",
         "Strong ember spotting breaches widely spaced firebreaks; closely spaced ones still hold."),
        ("phase5_enso_fire", "5 · Real fire seasons follow El Niño",
         "NASA FIRMS fire detections vs NOAA Niño 3.4, Indonesia and Australia, 2001-2025."),
    ]:
        fig = ROOT / "results" / folder / "figure.png"
        if fig.exists():
            with st.expander(title, expanded=folder == "phase3_drought"):
                st.image(str(fig), caption=caption, width="stretch")

with tab_nino:
    fires_path, nino_path = rd.PROCESSED / "firms_monthly.csv", rd.PROCESSED / "nino34_monthly.csv"
    if not fires_path.exists():
        st.info("Run `python scripts/download_real_data.py` once to fetch the NASA FIRMS and NOAA data.")
    else:
        fires, nino = pd.read_csv(fires_path), pd.read_csv(nino_path)
        country = st.radio("Region", list(rd.SEASONS), horizontal=True, key="country")
        latest = nino.iloc[-1]
        st.markdown(f"Latest Niño 3.4 anomaly in the data: **{latest['nino34']:+.2f} °C** "
                    f"({int(latest['year'])}-{int(latest['month']):02d}). Above +0.5 °C is El Niño.")
        f = fires[fires["country"] == country].assign(date=lambda d: pd.to_datetime(dict(year=d.year, month=d.month, day=15)))
        n = nino[nino["year"] >= 2001].assign(date=lambda d: pd.to_datetime(dict(year=d.year, month=d.month, day=15)))
        bars = alt.Chart(f).mark_bar(color="#ff7a1a").encode(
            x=alt.X("date:T", title=None), y=alt.Y("detections:Q", title="fire detections / month"),
            tooltip=["year", "month", "detections"])
        line = alt.Chart(n).mark_line(color="#f3e9df").encode(
            x="date:T", y=alt.Y("nino34:Q", title="Niño 3.4 anomaly (°C)"), tooltip=["year", "month", "nino34"])
        st.altair_chart(alt.layer(bars, line).resolve_scale(y="independent").properties(height=260,
                        title=f"{country}: monthly fire detections (bars) and Niño 3.4 (line)"), width="stretch")
        table = rd.seasonal_table(fires, nino, country)
        scatter = alt.Chart(table).mark_circle(size=90, opacity=0.9).encode(
            x=alt.X("nino34:Q", title=f"Niño 3.4, {rd.SEASONS[country]['label']} mean (°C)"),
            y=alt.Y("detections:Q", scale=alt.Scale(type="log"), title=f"fire detections, {rd.SEASONS[country]['label']}"),
            color=alt.Color("phase:N", scale=alt.Scale(domain=["La Niña", "neutral", "El Niño"],
                                                       range=["#4f8fd6", "#a8968a", "#ff5a36"])),
            tooltip=["year", "detections", alt.Tooltip("nino34:Q", format=".2f"), "phase"])
        a, b = st.columns([3, 2])
        a.altair_chart(scatter.properties(height=320, title="Each dot is one fire season"), width="stretch")
        cmp = rd.phase_comparison(table).round(1)
        rho = spearmanr(table["nino34"], table["detections"]).statistic
        b.markdown(f"**Fire season {rd.SEASONS[country]['label']}, 2001–2025**")
        b.dataframe(cmp, width="stretch")
        b.markdown(f"Rank correlation with Niño 3.4: **ρ = {rho:.2f}**. "
                   f"El Niño seasons average **{cmp.loc['El Niño', 'mean_vs_neutral']:.1f}×** the detections of neutral ones.")
        st.caption("Source: NASA FIRMS MODIS active fires (confidence ≥ 30, vegetation fires) and NOAA PSL Niño 3.4. "
                   "Detections are a proxy for fire activity, not burned area. FIRMS has not yet published 2026 yearly files.")

with tab_about:
    st.markdown("""
**What you are watching.** Each cell is bare ground or a tree. Fire starts where you choose. Every step, each burning
tree tries to ignite its neighbouring trees, succeeding with probability *q*, then burns out. With the default settings
that is all. The colours: green = tree, orange = burning, dark = burnt, brown = bare ground, blue = cleared by
management, grey = burnt on purpose beforehand.

**The big idea: a critical density.** Below a certain tree density, gaps break every fire path and fires stay small.
Above it, connected paths span the forest and one spark can burn almost everything. For dry fuel with 4 neighbours
the threshold is 0.593. Wetter fuel (lower *q*) raises it; drought lowers it. That is how an El Niño drought can turn a
normally safe forest into a fire-prone one without adding a single tree.

**Settings, briefly**
* *Dryness* turns a normal-year spread chance into this year's: q = 1 − (1 − q_normal)(1 − dryness).
* *Wind* multiplies the spread chance by e^(0.131·V·cos angle): up downwind, down upwind (Alexandridis et al. 2008).
* *Burn duration* lets trees smoulder and keep trying; use 5+ with wind.
* *Embers* let fire jump gaps, including firebreaks.
* *Management* removes trees beforehand with the chosen budget; *compare* reruns the same forest and fire untreated.

**Limits.** This is a teaching model on a grid of cells, not an operational fire-behaviour model: no terrain, fuel
types or fire suppression, and wind effects are idealised. The real-data tab shows correlation with El Niño, not a forecast.
""")
