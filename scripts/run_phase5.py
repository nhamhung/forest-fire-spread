"""Phase 5: real fire activity vs El Niño (NASA FIRMS MODIS detections, NOAA Niño 3.4).

Needs data/processed/*.csv from scripts/download_real_data.py. Writes results/phase5_enso_fire/.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from forest_fire import real_data as rd  # noqa: E402
from forest_fire.results import save_result  # noqa: E402

PHASE_COLOR = {"La Niña": "#2166ac", "neutral": "#999999", "El Niño": "#d6604d"}


def main():
    fires = pd.read_csv(rd.PROCESSED / "firms_monthly.csv")
    nino = pd.read_csv(rd.PROCESSED / "nino34_monthly.csv")
    tables = {c: rd.seasonal_table(fires, nino, c) for c in rd.SEASONS}

    stats = {}
    for c, t in tables.items():
        rho = spearmanr(t["nino34"], t["detections"])
        stats[c] = {"season": rd.SEASONS[c]["label"], "years": f"{t['year'].min()}-{t['year'].max()}",
                    "spearman_rho": float(rho.statistic), "spearman_p": float(rho.pvalue),
                    "permutation_p_el_nino_higher": float(rd.permutation_test(t)),
                    "by_phase": rd.phase_comparison(t).round(1).reset_index().to_dict(orient="records"),
                    "top_years": t.nlargest(5, "detections")[["year", "detections", "nino34", "phase"]]
                    .round(2).to_dict(orient="records")}

    latest = nino.iloc[-1]
    strong = tables["Indonesia"][tables["Indonesia"]["nino34"] >= 1.5]
    context = {"latest_nino34": {"year": int(latest["year"]), "month": int(latest["month"]), "value": float(latest["nino34"])},
               "indonesia_seasons_with_nino34_ge_1.5": strong[["year", "detections", "nino34"]].round(2).to_dict(orient="records"),
               "note": "FIRMS yearly country files for 2026 were not yet published, so the 2026 fire season is not scored."}

    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])
    ax = fig.add_subplot(gs[0, :])
    indo = fires[fires["country"] == "Indonesia"].assign(date=lambda d: pd.to_datetime(dict(year=d.year, month=d.month, day=15)))
    ax.bar(indo["date"], indo["detections"], width=25, color="#e66101", label="Indonesia: fire detections per month")
    ax.set(ylabel="MODIS fire detections / month", title="Indonesia's big fire years line up with El Niño")
    ax2 = ax.twinx()
    n = nino[(nino["year"] >= 2001) & (nino["year"] <= 2026)].assign(date=lambda d: pd.to_datetime(dict(year=d.year, month=d.month, day=15)))
    ax2.plot(n["date"], n["nino34"], color="k", lw=1.2, label="Niño 3.4 anomaly (°C)")
    ax2.axhline(rd.EL_NINO, color="#d6604d", ls="--", lw=0.8)
    ax2.axhline(rd.LA_NINA, color="#2166ac", ls="--", lw=0.8)
    ax2.set_ylabel("Niño 3.4 anomaly (°C)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
    ax.annotate(f"Sep 2026: +{latest['nino34']:.2f} °C", xy=(n["date"].iloc[-1], latest["nino34"]), xycoords=("data", ax2.transData),
                xytext=(-120, -10), textcoords="offset points", arrowprops=dict(arrowstyle="->"), fontsize=9)

    for k, (c, t) in enumerate(tables.items()):
        a = fig.add_subplot(gs[1, k])
        for phase, d in t.groupby("phase"):
            a.scatter(d["nino34"], d["detections"], s=45, color=PHASE_COLOR[phase], label=phase, edgecolor="k", lw=0.4)
        for _, row in t.nlargest(3, "detections").iterrows():
            a.annotate(int(row["year"]), (row["nino34"], row["detections"]), xytext=(4, 3), textcoords="offset points", fontsize=8)
        a.set(yscale="log", xlabel=f"Niño 3.4 anomaly, {rd.SEASONS[c]['label']} mean (°C)",
              ylabel=f"fire detections, {rd.SEASONS[c]['label']}",
              title=f"{c}: El Niño seasons {stats[c]['by_phase'][-1]['mean_vs_neutral']:.1f}× neutral, "
                    f"Spearman ρ = {stats[c]['spearman_rho']:.2f}")
        a.grid(True, which="both", alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    table = pd.concat(tables.values(), ignore_index=True)
    out = save_result("phase5_enso_fire", {"sources": {"fires": rd.FIRMS_URL, "nino34": rd.NINO34_URL},
                                           "filters": "MODIS confidence >= 30, type 0 (vegetation fire)",
                                           "seasons": rd.SEASONS, "thresholds_degC": [rd.LA_NINA, rd.EL_NINO]},
                      table, {"by_country": stats, "context_2026": context}, fig)
    plt.close(fig)
    for c, s in stats.items():
        print(c, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items() if k not in ("by_phase", "top_years")})
    print(context, f"-> {out}")


if __name__ == "__main__":
    main()
