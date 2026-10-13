"""Phase 4: fire management under drought. Firebreaks vs random thinning vs controlled burns.

Writes results/phase4_management (loss and efficiency vs budget), results/phase4_embers
(robustness when embers can jump gaps) and results/phase4_examples (maps of each strategy).
"""

from dataclasses import replace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from forest_fire import experiments as ex  # noqa: E402
from forest_fire.grid import TREE, ignite_point, make_forest  # noqa: E402
from forest_fire.management import treat  # noqa: E402
from forest_fire.plotting import plot_managed  # noqa: E402
from forest_fire.results import save_result  # noqa: E402
from forest_fire.spread import run_fire  # noqa: E402

LABEL = {"none": "no treatment", "firebreaks": "firebreaks", "thinning": "random thinning",
         "controlled_burns": "controlled burns"}
EMBER_SCENARIOS = [("no embers", 0.0, 0), ("embers: 1% per step, up to 5 cells", 0.01, 5),
                   ("embers: 5% per step, up to 10 cells", 0.05, 10)]
EMBER_BUDGETS = [0.04, 0.10, 0.20]
EXAMPLE_BUDGET = 0.08


def main_sweep(s: ex.Phase4Settings):
    runs = ex.management_sweep(s)
    summary = ex.summarise_management(runs)
    baseline = summary.loc[summary["strategy"] == "none", "loss"].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for strategy in s.strategies:
        d = summary[summary["strategy"] == strategy]
        axes[0].errorbar(d["removed"], d["loss"], d["loss_sem"], fmt="o-", capsize=2, label=LABEL[strategy])
        axes[1].plot(d["removed"], d["trees_saved_per_tree_removed"], "o-", label=LABEL[strategy])
    axes[0].axhline(baseline, color="k", ls="--", lw=1, label=f"no treatment ({baseline:.0%})")
    axes[0].set(xlabel="trees removed beforehand (fraction of forest)", ylabel="trees burned by the wildfire (fraction)",
                title=f"Drought year: density {s.density}, q = {s.q:.2f}, one lightning strike")
    axes[1].axhline(1, color="k", ls=":", lw=1)
    axes[1].set(xlabel="trees removed beforehand (fraction of forest)", ylabel="trees saved per tree removed",
                yscale="log", title="Efficiency: firebreaks win at every budget")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = save_result("phase4_management", s.to_dict(), runs, {"summary": summary.round(4).to_dict(orient="records")}, fig)
    plt.close(fig)
    print(summary[["strategy", "budget", "removed", "loss", "total_loss", "trees_saved_per_tree_removed"]]
          .round(3).to_string(index=False), f"-> {out}")


def embers(s: ex.Phase4Settings):
    frames = []
    for name, prob, radius in EMBER_SCENARIOS:
        si = replace(s, spot_prob=prob, spot_radius=radius, budgets=EMBER_BUDGETS, n_runs=60)
        frames.append(ex.summarise_management(ex.management_sweep(si)).assign(scenario=name))
    df = pd.concat(frames, ignore_index=True)
    fig, axes = plt.subplots(1, len(EMBER_BUDGETS), figsize=(15, 4.2), sharey=True)
    for ax, budget in zip(axes, EMBER_BUDGETS):
        d = df[(df["budget"] == budget) | (df["strategy"] == "none")]
        table = d.pivot_table(index="strategy", columns="scenario", values="loss").reindex(["none"] + s.strategies)
        table = table[[n for n, _, _ in EMBER_SCENARIOS]]
        table.index = [LABEL[i] for i in table.index]
        table.plot.bar(ax=ax, rot=15, width=0.8, legend=ax is axes[0])
        ax.set(title=f"budget: {budget:.0%} of trees removed", ylabel="trees burned (fraction)", xlabel="")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    out = save_result("phase4_embers", {**s.to_dict(), "scenarios": EMBER_SCENARIOS, "budgets": EMBER_BUDGETS, "n_runs": 60},
                      df, figure=fig)
    plt.close(fig)
    print(df.pivot_table(index=["strategy", "budget"], columns="scenario", values="loss").round(3).to_string(), f"-> {out}")


def strike_point(forest: np.ndarray, min_burn: float = 0.5) -> tuple[int, int]:
    """Tree nearest the centre that belongs to the forest's big connected cluster.

    Many trees sit in small clusters cut off by empty ground, where no fire can
    spread whatever the weather. The check uses a q = 1 fire, which burns exactly
    the strike's cluster.
    """
    n = (forest == TREE).sum()
    trees = np.argwhere(forest == TREE)
    for r, c in trees[np.argsort(((trees - forest.shape[0] // 2) ** 2).sum(axis=1))]:
        final, _, _ = run_fire(ignite_point(forest, r, c), 1.0)
        if (final == 3).sum() / n >= min_burn:
            return int(r), int(c)
    raise RuntimeError("no large cluster found")


def first_spreading_seed(forest: np.ndarray, r: int, c: int, s: ex.Phase4Settings, min_burn: float = 0.5) -> int:
    """First fire seed for which the untreated forest burns at least `min_burn` of its trees.

    Even inside the big cluster a single spark can fizzle out by chance; the example
    should show a fire that takes hold, so the treatments have something to stop.
    """
    n = (forest == TREE).sum()
    for seed in range(1000):
        final, _, _ = run_fire(ignite_point(forest, r, c), s.q, s.neighbourhood, seed)
        if (final == 3).sum() / n >= min_burn:
            return seed
    raise RuntimeError("no spreading fire found")


def examples(s: ex.Phase4Settings, forest_seed: int = 31):
    forest = make_forest(s.size, s.density, np.random.default_rng(forest_seed))
    # the same lightning strike for every strategy (kept as a tree by every treatment below)
    r, c = strike_point(forest)
    fire_seed = first_spreading_seed(forest, r, c, s)
    fig, axes = plt.subplots(1, 4, figsize=(21, 5.6))
    rows = []
    for ax, strategy in zip(axes, ["none"] + s.strategies):
        rng = np.random.default_rng([forest_seed, 1])          # treatment randomness, independent of the forest
        treated, removed = treat(forest, strategy, EXAMPLE_BUDGET, rng, s.burn_patch, s.break_width)
        treated[r, c] = TREE if treated[r, c] != TREE else treated[r, c]   # make sure the strike lands on a tree
        final, _, _ = run_fire(ignite_point(treated, r, c), s.q, s.neighbourhood, fire_seed)
        n = (forest == TREE).sum()
        loss = ((final == 3) & (treated == TREE)).sum() / n
        rows.append({"strategy": strategy, "removed": removed / n, "loss": loss})
        plot_managed(forest, treated, final, ax=ax,
                     title=f"{LABEL[strategy]}: {removed / n:.0%} removed, {loss:.0%} burned")
        ax.plot(c, r, "*", color="#ff6a00", ms=16, mec="k")
    fig.suptitle("Blue = removed beforehand, dark grey = burned by the wildfire, green = survived, star = lightning strike")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_result("phase4_examples", {**s.to_dict(), "budget": EXAMPLE_BUDGET, "forest_seed": forest_seed,
                                    "fire_seed": fire_seed, "strike": [int(r), int(c)]},
                pd.DataFrame(rows), figure=fig)
    plt.close(fig)


if __name__ == "__main__":
    settings = ex.Phase4Settings()
    main_sweep(settings)
    embers(settings)
    examples(settings)
