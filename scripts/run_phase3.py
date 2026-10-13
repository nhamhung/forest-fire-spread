"""Phase 3: drought (dryness) and wind.

Writes results/phase3_drought (fraction burned vs dryness at several densities),
results/phase3_wind (fire speed and shape vs wind speed) and
results/phase3_wind_shapes (one forest under different winds).
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from forest_fire import experiments as ex  # noqa: E402
from forest_fire.grid import ignite_point, make_forest  # noqa: E402
from forest_fire.plotting import plot_arrival  # noqa: E402
from forest_fire.results import save_result  # noqa: E402
from forest_fire.spread import run_fire  # noqa: E402

SHAPE_SPEEDS = [0.0, 5.0, 10.0, 15.0]
SHAPE_DIRECTIONS = [0.0, 90.0, 225.0]
SNAPSHOT_STEP = 120   # every panel shows the burnt area at this step, so sizes and shapes compare directly


def drought(s: ex.Phase3Settings):
    sweeps = pd.concat([ex.dryness_sweep(s.size, p, s.dryness, s.q_normal, s.drought_runs, s.seed, s.neighbourhood)
                        for p in s.densities], ignore_index=True)
    crit = pd.DataFrame([{"density": p, **ex.critical_dryness(d)} for p, d in sweeps.groupby("density")])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for p, d in sweeps.groupby("density"):
        axes[0].errorbar(d["dryness"], d["fraction_burned"], d["fraction_burned_sem"], fmt="o-", ms=3, capsize=2,
                         label=f"tree density {p:.2f}")
        axes[1].plot(d["dryness"], d["spanning_prob"], "o-", ms=3, label=f"tree density {p:.2f}")
    axes[0].set(xlabel="dryness d (0 = normal year, 1 = bone dry)", ylabel="fraction of trees burned",
                title=f"Drought turns small fires into forest-wide fires (q in a normal year = {s.q_normal})")
    axes[1].set(xlabel="dryness d", ylabel="P(fire crosses the forest)", title="The switch happens at a critical dryness")
    q_axis = axes[0].secondary_xaxis("top", functions=(lambda d: 1 - (1 - s.q_normal) * (1 - d),
                                                       lambda q: 1 - (1 - q) / (1 - s.q_normal)))
    q_axis.set_xlabel("spread probability q")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = save_result("phase3_drought", {**s.to_dict(), "experiment": "drought"}, sweeps,
                      {"critical_dryness": crit.to_dict(orient="records")}, fig)
    plt.close(fig)
    print(crit.round(3).to_string(index=False), f"-> {out}")


def wind(s: ex.Phase3Settings):
    runs = ex.wind_sweep(s)
    summary = ex.summarise_wind(runs)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for col, lab in [("head_speed", "head (downwind)"), ("back_speed", "back (upwind)")]:
        axes[0].errorbar(summary["wind_speed"], summary[col], summary[f"{col}_sem"], fmt="o-", capsize=3, label=lab)
    axes[0].set(xlabel="wind speed (m/s)", ylabel="spread speed (cells per step)", title="Wind speeds up the head, stalls the back")
    axes[1].errorbar(summary["wind_speed"], summary["length_to_width"], summary["length_to_width_sem"], fmt="o-", capsize=3,
                     color="C2")
    axes[1].axhline(1, color="k", lw=0.8, ls=":")
    axes[1].set(xlabel="wind speed (m/s)", ylabel="length / width", title="…and stretches the burnt area downwind")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    out = save_result("phase3_wind", {**s.to_dict(), "experiment": "wind", "shape_measured_at": "head 80% to edge"},
                      runs, {"by_wind_speed": summary.round(4).to_dict(orient="records")}, fig)
    plt.close(fig)
    print(summary[["wind_speed", "head_speed", "back_speed", "length_to_width", "fraction_burned", "n_spread"]]
          .round(3).to_string(index=False), f"-> {out}")


def shapes(s: ex.Phase3Settings, forest_seed: int = 21, fire_seed: int = 5):
    forest = make_forest(s.size, s.wind_density, np.random.default_rng(forest_seed))
    panels = [(v, s.wind_direction) for v in SHAPE_SPEEDS] + [(10.0, d) for d in SHAPE_DIRECTIONS]
    fig, axes = plt.subplots(2, 4, figsize=(21, 10))
    axes = axes.ravel()
    for ax, (v, direction) in zip(axes, panels):
        _, arrival, duration = run_fire(ignite_point(forest), s.wind_q, s.wind_neighbourhood, fire_seed, direction, v,
                                        s.burn_steps)
        snapshot = np.where(arrival <= SNAPSHOT_STEP, arrival, -1)
        plot_arrival(snapshot, ax=ax, title=f"wind {v:g} m/s towards {direction:g}°: step {SNAPSHOT_STEP}")
        if v > 0:   # arrow showing where the wind blows to
            a = np.deg2rad(direction)
            ax.annotate("", xy=(s.size * (0.5 + 0.18 * np.cos(a)), s.size * (0.5 - 0.18 * np.sin(a))),
                        xytext=(s.size * 0.5, s.size * 0.5), arrowprops=dict(color="cyan", width=3, headwidth=10))
    axes[-1].axis("off")
    axes[-1].text(0.05, 0.5, f"Burnt area {SNAPSHOT_STEP} steps after a single\nspark in the centre.\n\n"
                  "Top row: stronger wind towards the right.\nBottom row: 10 m/s towards three directions.\n\n"
                  "Colour = step when each cell caught fire;\nwhite = not (yet) burnt. Arrow = wind.",
                  fontsize=12, va="center")
    fig.tight_layout()
    save_result("phase3_wind_shapes", {**s.to_dict(), "forest_seed": forest_seed, "fire_seed": fire_seed,
                                       "panels": panels, "snapshot_step": SNAPSHOT_STEP}, figure=fig)
    plt.close(fig)


if __name__ == "__main__":
    settings = ex.Phase3Settings()
    drought(settings)
    wind(settings)
    shapes(settings)
