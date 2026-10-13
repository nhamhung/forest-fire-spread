"""Phase 2: probabilistic spread (q) and the 8-neighbour (Moore) option.

Writes results/phase2_critical_density (p_c for each q and neighbourhood),
results/phase2_full_forest (q threshold in a completely full forest) and
results/phase2_examples (fire shapes on one forest under four spread rules).
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from forest_fire import experiments as ex  # noqa: E402
from forest_fire.grid import ignite_left_edge, make_forest  # noqa: E402
from forest_fire.plotting import NB_LABEL, plot_arrival, plot_phase_diagram  # noqa: E402
from forest_fire.results import save_result  # noqa: E402
from forest_fire.spread import run_fire  # noqa: E402

FULL_FOREST_QS = [round(x, 3) for x in np.arange(0.15, 0.701, 0.01)]
EXAMPLE_RULES = [("von_neumann", 1.0), ("von_neumann", 0.8), ("moore", 1.0), ("moore", 0.5)]


def main():
    s = ex.Phase2Settings()

    # 1. full forest (p = 1): only q matters -> bond percolation threshold
    full = pd.concat([ex.spread_probability_sweep(s.size, FULL_FOREST_QS, 100, s.seed, nb)
                      for nb in ("von_neumann", "moore")], ignore_index=True)
    bond = pd.DataFrame([{"neighbourhood": nb, "q50": ex.fit_transition(d, x="q")["p50"],
                          "q50_err": ex.fit_transition(d, x="q")["p50_err"], "known": ex.BOND_THRESHOLD[nb]}
                         for nb, d in full.groupby("neighbourhood")])
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for nb, d in full.groupby("neighbourhood"):
        ax.errorbar(d["q"], d["spanning_prob"], d["spanning_sem"], fmt="o", ms=3, capsize=2, label=NB_LABEL[nb])
        ax.axvline(ex.BOND_THRESHOLD[nb], color="k", ls="--", lw=0.8)
    ax.set(xlabel="spread probability q", ylabel="P(fire crosses)", title="A completely full forest (p = 1)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    save_result("phase2_full_forest", {**s.to_dict(), "density": 1.0, "qs": FULL_FOREST_QS, "n_runs": 100},
                full, {"thresholds": bond.to_dict(orient="records")}, fig)
    plt.close(fig)
    print(bond.round(4).to_string(index=False))

    # 2. critical density for every q, both neighbourhoods
    fits, sweeps = [], []
    for nb, qs in (("von_neumann", s.q_von_neumann), ("moore", s.q_moore)):
        for q in qs:
            fit, sweep = ex.critical_density(s.size, q, nb, s)
            fits.append(fit)
            sweeps.append(sweep)
    fits, sweeps = pd.DataFrame(fits), pd.concat(sweeps, ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    plot_phase_diagram(fits, bond, ax=axes[0])
    axes[0].set_title("Wetter fuel (lower q) needs a denser forest to carry fire")
    vn = sweeps[(sweeps["neighbourhood"] == "von_neumann") & (sweeps["sweep"] == "coarse")]
    for q in [1.0, 0.85, 0.7, 0.6]:
        d = vn[vn["q"] == q]
        axes[1].plot(d["density"], d["spanning_prob"], "o-", ms=3, label=f"q = {q}")
    axes[1].set(xlabel="tree density p", ylabel="P(fire crosses)", title="4 neighbours: the switch moves right as q drops")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    save_result("phase2_critical_density", s.to_dict(), sweeps, {"fits": fits.to_dict(orient="records"),
                                                                 "bond_thresholds": bond.to_dict(orient="records")}, fig)
    plt.close(fig)
    print(fits[["neighbourhood", "q", "p50", "p50_err", "width"]].round(4).to_string(index=False))

    # 3. one forest, four spread rules
    forest = make_forest(s.size, 0.7, np.random.default_rng(11))
    fig, axes = plt.subplots(1, 4, figsize=(21, 5))
    rows = []
    for ax, (nb, q) in zip(axes, EXAMPLE_RULES):
        _, arrival, duration = run_fire(ignite_left_edge(forest), q, nb, seed=3)
        m = ex.fire_metrics(forest, q, nb, seed=3)
        rows.append({"neighbourhood": nb, "q": q, **m})
        plot_arrival(arrival, ax=ax, title=f"{NB_LABEL[nb]}, q = {q}\n{100 * m['fraction_burned']:.0f}% burned, "
                                         f"{duration} steps, {'crossed' if m['spanned'] else 'did not cross'}")
    fig.tight_layout()
    save_result("phase2_examples", {"size": s.size, "density": 0.7, "forest_seed": 11, "fire_seed": 3,
                                    "rules": EXAMPLE_RULES}, pd.DataFrame(rows), figure=fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
