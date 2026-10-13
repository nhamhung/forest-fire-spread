"""Experiments: run many random forests and measure how fires behave.

One forest is a single random draw, so its fire could be lucky or unlucky.
Every experiment therefore averages many forests, each grown from its own random
stream spawned from one master seed. The whole experiment is reproducible from
that single number.
"""

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from .grid import BURNT, TREE, ignite_left_edge, ignite_point, make_forest
from .spread import run_fire
from .weather import dry_spread_probability, wind_unit_vector

P_C_SQUARE = 0.592746  # site percolation threshold of the square lattice (known to 6 digits)


@dataclass
class Phase1Settings:
    """Everything needed to reproduce a Phase 1 result. Saved next to every result."""

    sizes: list = field(default_factory=lambda: [50, 100, 200, 400])
    # overview sweep across the whole requested range
    densities: list = field(default_factory=lambda: [round(x, 3) for x in np.arange(0.40, 0.8001, 0.01)])
    n_runs: int = 100
    # fine sweep around p_c for fitting: on a 400 x 400 forest the whole switch is only ~0.003 wide
    fine_densities: list = field(default_factory=lambda: [round(x, 4) for x in np.arange(0.560, 0.6301, 0.002)])
    fine_runs: int = 200
    seed: int = 2026

    def to_dict(self) -> dict:
        return {"model": "deterministic_von_neumann", "ignition": "left edge", **asdict(self)}


def fire_metrics(forest: np.ndarray, q: float = 1.0, neighbourhood: str = "von_neumann", seed: int = 0) -> dict:
    """Light the left edge of `forest`, burn it out and measure the fire.

    fraction_burned : burnt trees / all trees. 0 = the fire died at once, 1 = every tree burned
    duration        : steps until nothing was burning (how long the fire lasted)
    spanned         : True if the fire reached the right edge
    """
    final, _, duration = run_fire(ignite_left_edge(forest), q, neighbourhood, seed)
    n_trees = int((forest == TREE).sum())
    burnt = final == BURNT
    return {
        "fraction_burned": burnt.sum() / n_trees if n_trees else 0.0,
        "duration": duration,
        "spanned": bool(burnt[:, -1].any()),
    }


def density_sweep(size: int, densities, n_runs: int, seed: int, q: float = 1.0,
                  neighbourhood: str = "von_neumann") -> pd.DataFrame:
    """For each density, burn `n_runs` random forests of side `size` and average the metrics.

    Returns one row per density: spanning probability (fraction of forests in which
    the fire crossed) with its binomial standard error, and the mean and standard
    error of the fraction burned and the duration.

    The forests depend only on (seed, size, density), not on q or the neighbourhood.
    Comparing two spread rules therefore burns the very same forests, so the
    difference comes from the rule and not from luck.
    """
    rows = []
    # one independent stream per density, so adding densities never changes existing results
    streams = np.random.SeedSequence([seed, size]).spawn(len(densities))
    for p, stream in zip(densities, streams):
        rng = np.random.default_rng(stream)
        forests = [make_forest(size, p, rng) for _ in range(n_runs)]
        # the random spread draws get their own stream; none are needed when q = 1
        fire_seeds = (np.random.default_rng([seed, size, int(round(p * 1e6))]).integers(2**31 - 1, size=n_runs)
                      if q < 1 else np.zeros(n_runs, np.int64))
        runs = pd.DataFrame([fire_metrics(f, q, neighbourhood, int(s)) for f, s in zip(forests, fire_seeds)])
        span = runs["spanned"].mean()
        rows.append({
            "size": size, "density": p, "q": q, "neighbourhood": neighbourhood, "n_runs": n_runs,
            "spanning_prob": span, "spanning_sem": np.sqrt(span * (1 - span) / n_runs),
            "fraction_burned": runs["fraction_burned"].mean(),
            "fraction_burned_sem": runs["fraction_burned"].std(ddof=1) / np.sqrt(n_runs),
            "duration": runs["duration"].mean(),
            "duration_sem": runs["duration"].std(ddof=1) / np.sqrt(n_runs),
        })
    return pd.DataFrame(rows)


def logistic(p, p50, width):
    """Smooth step from 0 to 1: equals 1/2 at p = p50; `width` sets how gradual the step is."""
    return 1.0 / (1.0 + np.exp(-(p - p50) / width))


def fit_transition(df: pd.DataFrame, x: str = "density") -> dict:
    """Fit a logistic step to spanning probability vs `x` (density, or q).

    p50 is the value at which half the fires cross. It estimates the critical
    point. `width` measures how sharp the switch is: a big forest switches over a
    narrow range, and an infinite one would jump straight from 0 to 1.
    """
    xs, ys = df[x].to_numpy(float), df["spanning_prob"].to_numpy(float)
    guess = float(xs[np.argmin(np.abs(ys - 0.5))])
    # starting width: the distance between the 20% and 80% crossings, about 2.8 widths for a logistic
    above20, above80 = xs[ys >= 0.2], xs[ys >= 0.8]
    w0 = (above80.min() - above20.min()) / 2.8 if len(above20) and len(above80) else 0.01
    w0 = float(np.clip(w0, 1e-3, 0.1))
    # bounds keep the optimiser away from overflow when the step is nearly vertical
    (p50, width), cov = curve_fit(logistic, xs, ys, p0=[guess, w0],
                                  bounds=([xs.min(), 1e-5], [xs.max(), 0.5]))
    return {"p50": float(p50), "p50_err": float(np.sqrt(cov[0, 0])),
            "width": float(abs(width)), "width_err": float(np.sqrt(cov[1, 1]))}


def finite_size_scaling(fits: pd.DataFrame) -> dict:
    """How the transition sharpens with forest size.

    Percolation theory predicts width ∝ L^(-1/ν) with ν = 4/3, i.e. slope -0.75
    on a log-log plot of width against size L. A fitted slope near -0.75 confirms
    the simulation sits in the percolation universality class.
    """
    slope, intercept = np.polyfit(np.log(fits["size"]), np.log(fits["width"]), 1)
    return {"width_exponent": float(slope), "width_exponent_expected": -0.75}


@dataclass
class Phase2Settings:
    """Everything needed to reproduce a Phase 2 result. Saved next to every result."""

    size: int = 200
    q_von_neumann: list = field(default_factory=lambda: [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6])
    q_moore: list = field(default_factory=lambda: [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3])
    coarse_step: float = 0.02   # first pass: locate the transition roughly
    coarse_runs: int = 40
    fine_halfwidth: float = 0.04  # second pass: dense sweep around it, used for the fit
    fine_step: float = 0.004
    fine_runs: int = 150
    seed: int = 2027

    def to_dict(self) -> dict:
        return {"model": "probabilistic_spread", "ignition": "left edge", **asdict(self)}


def critical_density(size: int, q: float, neighbourhood: str, s: Phase2Settings) -> tuple[dict, pd.DataFrame]:
    """Find the density at which half the fires cross, for spread probability q.

    Two passes: a coarse sweep over 0.20-1.00 locates the switch, then a fine sweep
    around it is fitted with a logistic step. If even a completely full forest
    (p = 1) stops most fires, the fire cannot spread at this q at any density, and
    p50 is reported as NaN.
    """
    coarse_p = [round(x, 4) for x in np.arange(0.20, 1.0001, s.coarse_step)]
    coarse = density_sweep(size, coarse_p, s.coarse_runs, s.seed, q, neighbourhood)
    if coarse["spanning_prob"].iloc[-1] < 0.5:
        return {"q": q, "neighbourhood": neighbourhood, "p50": np.nan, "p50_err": np.nan,
                "width": np.nan, "width_err": np.nan}, coarse.assign(sweep="coarse")
    centre = float(coarse.loc[(coarse["spanning_prob"] - 0.5).abs().idxmin(), "density"])
    fine_p = [round(x, 4) for x in np.arange(centre - s.fine_halfwidth, centre + s.fine_halfwidth + 1e-9, s.fine_step)
              if 0 < x <= 1.0]
    fine = density_sweep(size, fine_p, s.fine_runs, s.seed + 1, q, neighbourhood)
    fit = {"q": q, "neighbourhood": neighbourhood, **fit_transition(fine)}
    return fit, pd.concat([coarse.assign(sweep="coarse"), fine.assign(sweep="fine")], ignore_index=True)


def spread_probability_sweep(size: int, qs, n_runs: int, seed: int, neighbourhood: str,
                             density: float = 1.0) -> pd.DataFrame:
    """Spanning probability vs q at a fixed density (default: a completely full forest).

    In a full forest only q matters, and the question becomes "bond percolation":
    each link between neighbouring trees passes fire with probability q. Known
    thresholds are q_c = 1/2 exactly for 4 neighbours and ≈ 0.25 for 8.
    """
    return pd.concat([density_sweep(size, [density], n_runs, seed, q, neighbourhood) for q in qs], ignore_index=True)


BOND_THRESHOLD = {"von_neumann": 0.5, "moore": 0.2504}      # square lattice; with diagonals (approx.)
SITE_THRESHOLD = {"von_neumann": P_C_SQUARE, "moore": 0.407254}


# ---------------------------------------------------------------- Phase 3: drought and wind

@dataclass
class Phase3Settings:
    """Everything needed to reproduce a Phase 3 result. Saved next to every result."""

    size: int = 200
    neighbourhood: str = "von_neumann"
    # drought experiment: fraction burned vs dryness at fixed density (left-edge ignition)
    q_normal: float = 0.6                    # spread probability in a normal (non-drought) year
    dryness: list = field(default_factory=lambda: [round(x, 3) for x in np.arange(0.0, 1.0001, 0.025)])
    densities: list = field(default_factory=lambda: [0.60, 0.65, 0.70, 0.80])
    drought_runs: int = 100
    # wind experiment: one fire lit at the centre
    wind_density: float = 0.8
    wind_q: float = 0.9                      # total chance of igniting a neighbour over a cell's burn
    wind_speeds: list = field(default_factory=lambda: [0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0])  # m/s
    wind_direction: float = 0.0              # towards the right edge
    # 4 neighbours: with 8, a diagonal step moves a fire forward and sideways at once, so it widens
    # as fast as it lengthens and wind cannot stretch it
    wind_neighbourhood: str = "von_neumann"
    burn_steps: int = 10                     # cells smoulder for several steps, so speed depends on wind (spread.py)
    wind_runs: int = 60
    seed: int = 2028

    def to_dict(self) -> dict:
        return {"model": "drought_and_wind", **asdict(self)}


def dryness_sweep(size: int, density: float, dryness_values, q_normal: float, n_runs: int, seed: int,
                  neighbourhood: str = "von_neumann") -> pd.DataFrame:
    """Fraction burned (and spanning, duration) as drought increases, at a fixed tree density.

    Each dryness level is turned into a spread probability (weather.py) and the
    same forests are burned at every level. A rise in fire size therefore comes
    from the drought alone, as if the same landscape lived through wetter and drier years.
    """
    rows = [density_sweep(size, [density], n_runs, seed, dry_spread_probability(q_normal, d), neighbourhood)
            .assign(dryness=d, q_normal=q_normal) for d in dryness_values]
    return pd.concat(rows, ignore_index=True)


def fire_shape(arrival: np.ndarray, origin: tuple[int, int], wind_direction: float, reach: float) -> dict:
    """Shape and speed of a fire lit at `origin`, measured when its head first gets `reach` cells downwind.

    Every burnt cell's position is split into a downwind distance (along the wind)
    and a sideways distance (across it). At the moment the head reaches `reach`
    cells downwind (before the fire is cut off by the grid edges), we measure:

    head_speed      : reach / time taken, in cells per step (how fast the front runs downwind)
    back_speed      : how far the fire has crept upwind by then / the same time
    length_to_width : (downwind + upwind extent) / sideways extent. 1 = round, >1 = stretched by the wind
    Returns NaNs if the fire never gets that far (it died out first).
    """
    rows, cols = np.nonzero(arrival >= 0)
    times = arrival[rows, cols]
    wind = wind_unit_vector(wind_direction)
    rel = np.stack([rows - origin[0], cols - origin[1]], axis=1).astype(float)
    along = rel @ wind                                     # + downwind, - upwind
    across = rel @ np.array([-wind[1], wind[0]])
    reached = times[along >= reach]
    if len(reached) == 0:
        return {"head_speed": np.nan, "back_speed": np.nan, "length_to_width": np.nan, "t_reach": np.nan}
    t_reach = int(reached.min())
    now = times <= t_reach
    width = across[now].max() - across[now].min() + 1
    length = along[now].max() - along[now].min() + 1
    return {"head_speed": reach / t_reach, "back_speed": max(0.0, -along[now].min()) / t_reach,
            "length_to_width": length / width, "t_reach": t_reach}


def wind_sweep(s: Phase3Settings) -> pd.DataFrame:
    """Fire shape and speed for each wind speed, averaged over random forests.

    One fire per forest, lit at the centre. Forests and fire seeds are the same at
    every wind speed, so differences come from the wind alone. Shapes are
    measured when the head is 80% of the way to the downwind edge.
    """
    origin = (s.size // 2, s.size // 2)
    reach = 0.8 * (s.size // 2)
    rng = np.random.default_rng([s.seed, s.size])
    forests = [make_forest(s.size, s.wind_density, rng) for _ in range(s.wind_runs)]
    fire_seeds = rng.integers(2**31 - 1, size=s.wind_runs)
    rows = []
    for w in s.wind_speeds:
        for k, (forest, fseed) in enumerate(zip(forests, fire_seeds)):
            final, arrival, duration = run_fire(ignite_point(forest), s.wind_q, s.wind_neighbourhood, int(fseed),
                                                s.wind_direction, w, s.burn_steps)
            n_trees = int((forest == TREE).sum())
            rows.append({"wind_speed": w, "run": k, "duration": duration,
                         "fraction_burned": (final == BURNT).sum() / n_trees,
                         **fire_shape(arrival, origin, s.wind_direction, reach)})
    return pd.DataFrame(rows)


def summarise_wind(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard error of each wind metric per wind speed (fires that died early are left out)."""
    cols = ["head_speed", "back_speed", "length_to_width", "fraction_burned", "duration"]
    g = runs.groupby("wind_speed")
    out = g[cols].mean()
    for c in cols:
        out[f"{c}_sem"] = g[c].std(ddof=1) / np.sqrt(g[c].count())
    out["n_spread"] = g["head_speed"].count()
    return out.reset_index()


def critical_dryness(df: pd.DataFrame) -> dict:
    """Dryness at which half the fires cross, from a dryness sweep (NaN if the switch is not inside the sweep)."""
    span = df["spanning_prob"]
    if span.max() < 0.5 or span.min() > 0.5:
        return {"d50": np.nan, "d50_err": np.nan}
    fit = fit_transition(df, x="dryness")
    return {"d50": fit["p50"], "d50_err": fit["p50_err"]}


# ---------------------------------------------------------------- Phase 4: fire management

@dataclass
class Phase4Settings:
    """Everything needed to reproduce a Phase 4 result. Saved next to every result."""

    size: int = 200
    density: float = 0.7
    q_normal: float = 0.6
    dryness: float = 0.8                     # a drought year: q = 0.92
    neighbourhood: str = "von_neumann"
    budgets: list = field(default_factory=lambda: [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30])
    strategies: list = field(default_factory=lambda: ["firebreaks", "thinning", "controlled_burns"])
    burn_patch: int = 10                     # side of a controlled-burn patch, cells
    break_width: int = 1
    spot_prob: float = 0.0                   # ember spotting (spread.py); 0 = embers off
    spot_radius: int = 0
    n_runs: int = 100
    seed: int = 2029

    @property
    def q(self) -> float:
        return dry_spread_probability(self.q_normal, self.dryness)

    def to_dict(self) -> dict:
        return {"model": "fire_management", "ignition": "one random lightning strike on a tree",
                "q": self.q, **asdict(self)}


def managed_fire(forest: np.ndarray, strategy: str, budget: float, s: Phase4Settings,
                 rng: np.random.Generator, fire_seed: int) -> dict:
    """Treat the forest, then start one wildfire on a random surviving tree and burn it out.

    loss    : wildfire-burnt trees / trees in the untreated forest
    removed : trees removed by the treatment / trees in the untreated forest
    """
    from .management import treat

    n_trees = int((forest == TREE).sum())
    treated, removed = treat(forest, strategy, budget, rng, s.burn_patch, s.break_width)
    trees = np.flatnonzero(treated == TREE)
    r, c = divmod(int(trees[rng.integers(len(trees))]), forest.shape[1])
    final, _, duration = run_fire(ignite_point(treated, r, c), s.q, s.neighbourhood, fire_seed,
                                  spot_prob=s.spot_prob, spot_radius=s.spot_radius)
    burnt = int((final == BURNT).sum() - (treated == BURNT).sum())   # do not count pre-burnt patches
    return {"loss": burnt / n_trees, "removed": removed / n_trees, "duration": duration}


def management_sweep(s: Phase4Settings) -> pd.DataFrame:
    """Every strategy at every budget, on the same forests and the same lightning seeds.

    Forest k is identical across strategies and budgets, so comparisons are paired.
    """
    rows = []
    base = np.random.SeedSequence(s.seed)
    forest_seq, fire_seq = base.spawn(2)
    forests = [make_forest(s.size, s.density, np.random.default_rng(fs)) for fs in forest_seq.spawn(s.n_runs)]
    fire_seeds = np.random.default_rng(fire_seq).integers(2**31 - 1, size=s.n_runs)
    for strategy in ["none"] + list(s.strategies):
        budgets = [0.0] if strategy == "none" else [b for b in s.budgets if b > 0]
        for budget in budgets:
            for k, (forest, fseed) in enumerate(zip(forests, fire_seeds)):
                rng = np.random.default_rng([s.seed, k, int(budget * 1000)])   # same lightning cell draw per k
                rows.append({"strategy": strategy, "budget": budget, "run": k,
                             **managed_fire(forest, strategy, budget, s, rng, int(fseed))})
    return pd.DataFrame(rows)


def summarise_management(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean loss and removal per strategy and budget, plus efficiency against no treatment.

    trees_saved_per_tree_removed = (baseline loss - loss) / removed. Above 1, a
    treatment saves more trees than it cuts.
    total_loss = loss + removed: all trees gone, whether to fire or to the treatment.
    """
    baseline = runs.loc[runs["strategy"] == "none", "loss"].mean()
    g = runs.groupby(["strategy", "budget"])
    out = g[["loss", "removed", "duration"]].mean()
    out["loss_sem"] = g["loss"].std(ddof=1) / np.sqrt(g["loss"].count())
    out["total_loss"] = out["loss"] + out["removed"]
    out["trees_saved_per_tree_removed"] = np.where(out["removed"] > 0, (baseline - out["loss"]) / out["removed"], np.nan)
    return out.reset_index()
