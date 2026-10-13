"""Shared plots and animations."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.colors import ListedColormap

from .experiments import P_C_SQUARE, logistic
from .spread import frame_at

# empty ground, tree, burning, burnt
FIRE_CMAP = ListedColormap(["#efe6d2", "#2f7d32", "#ff6a00", "#3b3b3b"])


def plot_forest(grid: np.ndarray, ax=None, title: str = ""):
    """Draw a grid with ground beige, trees green, fire orange and ash dark grey."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(grid, cmap=FIRE_CMAP, vmin=0, vmax=3, interpolation="nearest")
    ax.set(title=title, xticks=[], yticks=[])
    return ax


def plot_arrival(arrival: np.ndarray, ax=None, title: str = ""):
    """Colour each burnt cell by the step at which it caught fire; unburnt cells are white."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    shown = np.ma.masked_less(arrival, 0)
    im = ax.imshow(shown, cmap="inferno_r", interpolation="nearest")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, label="step when ignited")
    ax.set(title=title, xticks=[], yticks=[])
    return ax


def plot_spanning(df, fits=None, ax=None):
    """Spanning probability vs density, one curve per forest size, with fitted logistic steps."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4.2))
    for size, d in df.groupby("size"):
        line = ax.errorbar(d["density"], d["spanning_prob"], d["spanning_sem"], fmt="o", ms=3, capsize=2,
                           label=f"{size} × {size}")
        if fits is not None:
            f = fits.set_index("size").loc[size]
            p = np.linspace(d["density"].min(), d["density"].max(), 300)
            ax.plot(p, logistic(p, f["p50"], f["width"]), color=line[0].get_color(), lw=1)
    ax.axvline(P_C_SQUARE, color="k", ls="--", lw=1, label=f"p_c = {P_C_SQUARE:.4f}")
    ax.set(xlabel="tree density p", ylabel="P(fire reaches the far edge)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    return ax


def plot_metric(df, column: str, ylabel: str, ax=None, logy: bool = False):
    """Any per-density metric (fraction burned, duration) vs density, one curve per size."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4.2))
    for size, d in df.groupby("size"):
        ax.errorbar(d["density"], d[column], d.get(f"{column}_sem"), fmt="o-", ms=3, capsize=2, label=f"{size} × {size}")
    ax.axvline(P_C_SQUARE, color="k", ls="--", lw=1)
    ax.set(xlabel="tree density p", ylabel=ylabel, yscale="log" if logy else "linear")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    return ax


def animate_fire(forest: np.ndarray, arrival: np.ndarray, n_frames: int = 60, interval: int = 80):
    """Animate a recorded fire. Frames are evenly spaced over its duration (FuncAnimation)."""
    duration = int(arrival.max()) + 1
    steps = np.unique(np.linspace(0, duration, min(n_frames, duration + 1)).round().astype(int))
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(frame_at(arrival, forest, 0), cmap=FIRE_CMAP, vmin=0, vmax=3, interpolation="nearest")
    ax.set(xticks=[], yticks=[])

    def update(k):
        im.set_data(frame_at(arrival, forest, steps[k]))
        ax.set_title(f"step {steps[k]} / {duration}")
        return (im,)

    return animation.FuncAnimation(fig, update, frames=len(steps), interval=interval)


def gif_bytes(anim, fps: int = 12) -> bytes:
    """Render an animation to GIF bytes, for compact embedding in notebooks."""
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "anim.gif")
        anim.save(path, writer=animation.PillowWriter(fps=fps))
        with open(path, "rb") as f:
            return f.read()


NB_LABEL = {"von_neumann": "4 neighbours (von Neumann)", "moore": "8 neighbours (Moore)"}


def plot_phase_diagram(fits, bond=None, ax=None):
    """Critical density p_c against spread probability q, one line per neighbourhood.

    Above a line (denser forest or easier spread) fire usually crosses the forest;
    below it, fires burn out. The line ends at q = 1 on the site-percolation
    threshold and runs up to p = 1 at the bond-percolation threshold.
    """
    from .experiments import BOND_THRESHOLD, SITE_THRESHOLD

    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4.5))
    for nb, d in fits.groupby("neighbourhood"):
        d = d.dropna(subset=["p50"]).sort_values("q")
        qs, ps = list(d["q"]), list(d["p50"])
        if bond is not None and nb in set(bond["neighbourhood"]):
            qs, ps = [float(bond.set_index("neighbourhood").loc[nb, "q50"])] + qs, [1.0] + ps
        line = ax.plot(qs, ps, "o-", label=NB_LABEL.get(nb, nb))[0]
        ax.plot(1.0, SITE_THRESHOLD[nb], "*", ms=14, color=line.get_color(), mec="k")
        ax.plot(BOND_THRESHOLD[nb], 1.0, "*", ms=14, color=line.get_color(), mec="k")
    ax.plot([], [], "*", color="grey", mec="k", ms=12, label="exact / known thresholds")
    ax.set(xlabel="spread probability q (higher = drier fuel)", ylabel="critical tree density p_c",
           xlim=(0.2, 1.02), ylim=(0.35, 1.02))
    ax.text(0.97, 0.97, "fire crosses", ha="right", va="top", transform=ax.transAxes, fontsize=9, color="#b2182b")
    ax.text(0.03, 0.03, "fire burns out", ha="left", va="bottom", transform=ax.transAxes, fontsize=9, color="#2166ac")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)
    return ax


# empty ground, surviving tree, wildfire-burnt, removed by treatment
MANAGED_CMAP = ListedColormap(["#efe6d2", "#2f7d32", "#3b3b3b", "#4fa3e0"])


def plot_managed(forest: np.ndarray, treated: np.ndarray, final: np.ndarray, ax=None, title: str = ""):
    """Forest after treatment and wildfire: trees removed by the treatment in blue, wildfire damage dark grey."""
    from .grid import BURNT, TREE

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    show = np.zeros(forest.shape, np.uint8)
    show[final == TREE] = 1
    show[(final == BURNT) & (treated == TREE)] = 2          # burnt by the wildfire
    show[(forest == TREE) & (treated != TREE)] = 3           # removed beforehand (cleared or pre-burnt)
    ax.imshow(show, cmap=MANAGED_CMAP, vmin=0, vmax=3, interpolation="nearest")
    ax.set(title=title, xticks=[], yticks=[])
    return ax
