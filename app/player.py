"""Run fires for the app and package them for the in-browser player (app/fire_player.html).

The simulation runs once in Python and records when every cell caught fire; the
browser then replays it with play / pause / scrub, so playback is smooth at any speed.
"""

import base64
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from forest_fire.grid import BURNT, EMPTY, TREE, ignite_left_edge, ignite_point, make_forest
from forest_fire.management import treat
from forest_fire.spread import run_fire
from forest_fire.weather import dry_spread_probability

TEMPLATE = Path(__file__).with_name("fire_player.html")
IGNITIONS = {"left_edge": "Whole left edge", "centre": "Spark in the centre", "lightning": "Random lightning strike"}


@dataclass(frozen=True)
class Scenario:
    """Every input of one app run (saved with the export so it can be reproduced)."""

    size: int = 150
    density: float = 0.65
    neighbourhood: str = "von_neumann"
    q_normal: float = 0.6
    dryness: float = 0.5
    wind_speed: float = 0.0
    wind_direction: float = 0.0
    burn_steps: int = 1
    spot_prob: float = 0.0
    spot_radius: int = 0
    ignition: str = "left_edge"
    strategy: str = "none"
    budget: float = 0.0
    seed: int = 42

    @property
    def q(self) -> float:
        return dry_spread_probability(self.q_normal, self.dryness)


def simulate(s: Scenario, strategy: str | None = None) -> dict:
    """Grow the forest, apply management, ignite and burn. Same forest and fire seed for any strategy."""
    strategy = s.strategy if strategy is None else strategy
    rng = np.random.default_rng(s.seed)
    forest = make_forest(s.size, s.density, rng)
    treated, removed = treat(forest, strategy, s.budget if strategy != "none" else 0.0,
                             np.random.default_rng([s.seed, 1]))
    strike = None
    if s.ignition == "left_edge":
        grid = ignite_left_edge(treated)
    else:
        if s.ignition == "centre":
            strike = (s.size // 2, s.size // 2)
        else:                                                   # a random tree that survived the treatment
            trees = np.argwhere(forest == TREE)
            strike = tuple(int(v) for v in trees[np.random.default_rng([s.seed, 2]).integers(len(trees))])
        if treated[strike] != TREE:
            treated[strike] = TREE                              # the strike always hits fuel
        grid = ignite_point(treated, *strike)
    final, arrival, duration = run_fire(grid, s.q, s.neighbourhood, s.seed, s.wind_direction, s.wind_speed,
                                        s.burn_steps, s.spot_prob, s.spot_radius)
    n_trees = int((forest == TREE).sum())
    lit = np.sort(arrival[arrival >= 0])
    burned_curve = np.searchsorted(lit, np.arange(duration + 1), side="right")   # trees alight by each step
    # "reached": the far (right) edge for an edge fire, any edge of the map for a single spark
    reach = np.zeros(forest.shape, bool)
    if s.ignition == "left_edge":
        reach[:, -1] = True
        reach_label = "reached far edge"
    else:
        reach[[0, -1], :] = True
        reach[:, [0, -1]] = True
        reach_label = "reached forest edge"
    cells = np.where(treated == TREE, 1, np.where(treated == BURNT, 3, EMPTY)).astype(np.uint8)
    cells[(forest == TREE) & (treated == EMPTY)] = 4            # cleared by firebreaks or thinning
    return {
        "strategy": strategy, "forest": forest, "cells": cells, "arrival": arrival, "duration": int(duration),
        "n_trees": n_trees, "strike": strike, "removed": removed / n_trees,
        "fraction_burned": len(lit) / n_trees,               # wildfire-burnt trees (not pre-burnt patches) / trees
        "reached_edge": bool((reach & (arrival >= 0)).any()),
        "burned_curve": (burned_curve / n_trees).round(4).tolist(), "reach_label": reach_label,
    }


def payload_run(run: dict, label: str, tag: str) -> dict:
    return {
        "label": label, "tag": tag, "size": int(run["arrival"].shape[0]), "duration": run["duration"],
        "n_trees": run["n_trees"], "removed": run["removed"], "reached_edge": run["reached_edge"],
        "strike": list(run["strike"]) if run["strike"] else None, "reach_label": run["reach_label"],
        "cells_b64": base64.b64encode(run["cells"].tobytes()).decode(),
        "arrival_b64": base64.b64encode(np.ascontiguousarray(run["arrival"], "<i4").tobytes()).decode(),
        "burned_curve": run["burned_curve"],
    }


def build_html(runs: list[tuple[dict, str, str]], s: Scenario, autoplay: bool = True) -> str:
    """Player page for one or more runs (side by side). The JSON payload is embedded safely."""
    items = [payload_run(r, label, tag) for r, label, tag in runs]
    payload = {"runs": items, "t_max": max(r["duration"] for r, _, _ in runs) + s.burn_steps,
               "burn_steps": s.burn_steps, "wind_speed": s.wind_speed, "wind_direction": s.wind_direction,
               "autoplay": autoplay}
    # "</" is escaped so the JSON can never close the <script> tag early
    return TEMPLATE.read_text().replace("__PAYLOAD__", json.dumps(payload).replace("</", "<\\/"))


def player_height(n_runs: int) -> int:
    """Height for the embedded player: square map(s) + stats + trace + controls."""
    map_px = 640 if n_runs == 1 else 480          # single map is capped at 640 px wide (fire_player.html)
    return map_px + 40 + 62 + 30 + 116 + 56 + 30


def scenario_dict(s: Scenario) -> dict:
    return {**asdict(s), "q": s.q}
