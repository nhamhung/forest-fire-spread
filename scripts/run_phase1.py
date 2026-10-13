"""Phase 1: deterministic von Neumann fire. Critical density, finite-size sharpening, example fires.

Writes results/phase1_percolation/ (sweeps + fits) and results/phase1_examples/ (snapshots + GIF).
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.animation import PillowWriter  # noqa: E402

from forest_fire import experiments as ex  # noqa: E402
from forest_fire.grid import ignite_left_edge, make_forest  # noqa: E402
from forest_fire.plotting import animate_fire, plot_arrival, plot_forest, plot_metric, plot_spanning  # noqa: E402
from forest_fire.results import RESULTS_DIR, save_result  # noqa: E402
from forest_fire.spread import run_fire  # noqa: E402

EXAMPLE_DENSITIES = [0.55, 0.593, 0.65]


def percolation():
    s = ex.Phase1Settings()
    coarse = pd.concat([ex.density_sweep(n, s.densities, s.n_runs, s.seed) for n in s.sizes], ignore_index=True)
    fine = pd.concat([ex.density_sweep(n, s.fine_densities, s.fine_runs, s.seed + 1) for n in s.sizes],
                     ignore_index=True)
    fits = pd.DataFrame([{"size": n, **ex.fit_transition(fine[fine["size"] == n])} for n in s.sizes])
    scaling = ex.finite_size_scaling(fits)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    plot_spanning(coarse, ax=axes[0, 0])
    axes[0, 0].set_title("Fire reaches the far edge: overview (100 runs per point)")
    plot_spanning(fine, fits, ax=axes[0, 1])
    axes[0, 1].set_title("Zoom near p_c (200 runs per point) with logistic fits")
    plot_metric(coarse, "fraction_burned", "fraction of trees burned", ax=axes[1, 0])
    axes[1, 0].set_title("Fraction burned")
    plot_metric(coarse, "duration", "fire duration (steps)", ax=axes[1, 1], logy=True)
    axes[1, 1].set_title("Fire duration peaks at p_c")
    fig.tight_layout()

    summary = {"fits": fits.to_dict(orient="records"), **scaling, "p_c_exact": ex.P_C_SQUARE}
    table = pd.concat([coarse.assign(sweep="overview"), fine.assign(sweep="fine")], ignore_index=True)
    out = save_result("phase1_percolation", s.to_dict(), table, summary, fig)
    plt.close(fig)
    print(fits.round(4).to_string(index=False))
    print(f"width exponent {scaling['width_exponent']:.3f} (theory -0.75) -> {out}")

    # sharpening on its own: width vs size, log-log
    fig, ax = plt.subplots(figsize=(5, 3.8))
    ax.errorbar(fits["size"], fits["width"], fits["width_err"], fmt="o", capsize=3)
    L = np.array([fits["size"].min(), fits["size"].max()])
    ax.plot(L, fits["width"].iloc[0] * (L / L[0]) ** -0.75, "k--", lw=1, label="theory slope -3/4")
    ax.set(xscale="log", yscale="log", xlabel="forest size L", ylabel="transition width",
           title=f"Transition sharpens: slope {scaling['width_exponent']:.2f}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(out / "width_scaling.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def examples(size: int = 200, seed: int = 7):
    """Snapshots of one forest per density: the burnt area and when each cell caught fire."""
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, len(EXAMPLE_DENSITIES), figsize=(5 * len(EXAMPLE_DENSITIES), 9.5))
    for k, p in enumerate(EXAMPLE_DENSITIES):
        forest = make_forest(size, p, rng)
        final, arrival, duration = run_fire(ignite_left_edge(forest))
        m = ex.fire_metrics(forest)
        plot_forest(final, ax=axes[0, k], title=f"p = {p}: {100 * m['fraction_burned']:.0f}% burned, "
                                                f"{'crossed' if m['spanned'] else 'did not cross'}")
        plot_arrival(arrival, ax=axes[1, k], title=f"ignition time ({duration} steps)")
        if p == 0.593:
            anim = animate_fire(forest, arrival)
            anim.save(RESULTS_DIR / "phase1_examples_anim.gif", writer=PillowWriter(fps=12))
    fig.tight_layout()
    out = save_result("phase1_examples", {"size": size, "densities": EXAMPLE_DENSITIES, "seed": seed}, figure=fig)
    (RESULTS_DIR / "phase1_examples_anim.gif").replace(out / "fire_p0.593.gif")
    plt.close("all")
    print(f"examples -> {out}")


if __name__ == "__main__":
    percolation()
    examples()
