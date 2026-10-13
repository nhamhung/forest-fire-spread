"""How fire moves: the spread rule, run step by step until the fire goes out.

At every time step, each burning cell tries to set fire to each of its
neighbouring trees, then burns out. The run ends when nothing is burning.

Two choices define the rule:

* Neighbourhood: which cells count as neighbours.
  - von Neumann (4): up, down, left, right. Fire cannot jump diagonally.
  - Moore (8): the 4 side neighbours plus the 4 diagonal ones. Fire jumps more
    easily, so it needs fewer trees to cross a forest.
* Spread probability q: the chance that a burning cell ignites a given neighbouring
  tree. q = 1 (Phase 1) means fire always spreads; lower q stands for wetter
  fuel, which often fails to catch. Each burning cell gets one try per
  neighbour, but a tree that survives one try can still be lit by another
  burning neighbour later. Trees with several burning neighbours are therefore
  more likely to catch, as in a real fire.

With q = 1 and 4 neighbours the fire burns exactly the trees connected to the
left edge, which is classic site percolation with threshold p_c ≈ 0.5927. With
q < 1 it becomes "site-bond" percolation, and a denser forest is needed for
fire to cross.

Phase 3 adds weather (see weather.py): drought raises q, and wind gives each
neighbour direction its own probability, higher downwind and lower upwind. The
kernel therefore takes one probability per neighbour direction.

Burn duration: by default a cell burns for exactly one step and gets a single
try at each neighbour. A fire that survives then always advances at about the
grid's speed limit (one cell per step), so wind could change *where* fire goes
but not how *fast*. Real fuel smoulders and keeps trying, so `burn_steps` lets a
cell stay alight for several steps, trying its neighbours each step. q stays the
total chance of igniting a neighbour over the whole burn, so the per-step chance
is 1 - (1 - q)^(1 / burn_steps). Spread speed then depends on that per-step
chance, and wind, which scales it by direction, makes heads fast, backs slow
and shapes long. burn_steps = 1 is exactly the Phase 1-2 model.

Ember spotting (Phase 4): burning cells can throw embers. Each step, each
burning cell has a chance `spot_prob` of launching one ember to a random cell
up to `spot_radius` cells away in each direction; if that cell is a tree it
catches fire. Embers are how real fires jump firebreaks, roads and rivers.
spot_prob = 0 (the default) draws no random numbers, so it is the Phase 1-3 model.

Speed: the loop visits every burning cell individually, which is slow in plain
Python, so it is compiled with Numba (@njit). It keeps a list of the cells
currently burning (the "fire front"), so each cell is handled once.
"""

import numpy as np
from numba import njit

from .grid import BURNING, BURNT, TREE
from .weather import direction_probabilities

NOT_BURNT = -1

NEIGHBOURHOODS = {
    "von_neumann": np.array([[-1, 0], [1, 0], [0, -1], [0, 1]]),
    "moore": np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [-1, 1], [1, -1], [1, 1]]),
}


@njit(cache=True)
def burn(grid, arrival, offsets, probs, burn_steps, seed, spot_prob=0.0, spot_radius=0):
    """Run a fire to completion, in place. Returns the number of steps it burned for.

    grid    : cell states (EMPTY / TREE / BURNING / BURNT); burning cells are the
              starting fire. On return every reached tree is BURNT.
    arrival : int32 array, same shape, filled with the step at which each cell
              caught fire (0 = initially burning) or NOT_BURNT. This one array is
              a complete record of the fire: frame t of an animation is
              "arrival < t is burnt, arrival == t is burning".
    offsets : neighbour offsets, one (row, column) pair per neighbour
    probs   : per-step probability that fire spreads towards each neighbour (same
              order as offsets); all equal when there is no wind
    burn_steps : how many steps a cell stays alight, trying its neighbours each step
    spot_prob, spot_radius : ember spotting (see module docstring); 0 = off
    seed    : seeds Numba's random generator. No random numbers are drawn for
              directions with probability 1, so a q = 1 fire is fully
              determined by the forest.
    """
    np.random.seed(seed)
    n_rows, n_cols = grid.shape
    n_nb = offsets.shape[0]
    front = np.empty(n_rows * n_cols, np.int64)  # flat indices of cells burning now
    new_front = np.empty(n_rows * n_cols, np.int64)
    steps_left = np.zeros(n_rows * n_cols, np.int32)  # remaining burn time of each burning cell
    arrival[:, :] = NOT_BURNT
    n_front = 0
    for r in range(n_rows):
        for c in range(n_cols):
            if grid[r, c] == BURNING:
                front[n_front] = r * n_cols + c
                steps_left[r * n_cols + c] = burn_steps
                arrival[r, c] = 0
                n_front += 1

    step = 0
    while n_front > 0:
        step += 1
        n_new = 0
        for k in range(n_front):
            r, c = front[k] // n_cols, front[k] % n_cols
            for d in range(n_nb):
                rr, cc = r + offsets[d, 0], c + offsets[d, 1]
                if 0 <= rr < n_rows and 0 <= cc < n_cols and grid[rr, cc] == TREE:
                    if probs[d] >= 1.0 or np.random.random() < probs[d]:
                        grid[rr, cc] = BURNING       # marked now so it is not added twice
                        arrival[rr, cc] = step
                        steps_left[rr * n_cols + cc] = burn_steps
                        new_front[n_new] = rr * n_cols + cc
                        n_new += 1
            if spot_prob > 0.0 and np.random.random() < spot_prob:   # an ember flies
                rr = r + np.random.randint(-spot_radius, spot_radius + 1)
                cc = c + np.random.randint(-spot_radius, spot_radius + 1)
                if 0 <= rr < n_rows and 0 <= cc < n_cols and grid[rr, cc] == TREE:
                    grid[rr, cc] = BURNING
                    arrival[rr, cc] = step
                    steps_left[rr * n_cols + cc] = burn_steps
                    new_front[n_new] = rr * n_cols + cc
                    n_new += 1
            steps_left[front[k]] -= 1
            if steps_left[front[k]] > 0:             # still alight: keep it on the front
                new_front[n_new] = front[k]
                n_new += 1
            else:
                grid[r, c] = BURNT                   # this cell has used up its fuel
        front, new_front = new_front, front
        n_front = n_new
    return step


def per_step_probability(q: float, burn_steps: int) -> float:
    """Per-step ignition chance that adds up to a total of q over `burn_steps` tries."""
    return 1.0 - (1.0 - q) ** (1.0 / burn_steps)


def run_fire(grid: np.ndarray, q: float = 1.0, neighbourhood: str = "von_neumann", seed: int = 0,
             wind_direction: float = 0.0, wind_speed: float = 0.0,
             burn_steps: int = 1, spot_prob: float = 0.0, spot_radius: int = 0) -> tuple[np.ndarray, np.ndarray, int]:
    """Burn a copy of `grid`; returns (final grid, arrival times, duration in steps).

    The duration counts steps in which something was burning. A fire that only
    burns the left column (one-step burning) lasts 1 step.
    """
    final = grid.copy()
    arrival = np.empty(grid.shape, np.int32)
    offsets = NEIGHBOURHOODS[neighbourhood]
    probs = direction_probabilities(offsets, per_step_probability(q, burn_steps), wind_direction, wind_speed)
    duration = burn(final, arrival, offsets, probs, int(burn_steps), int(seed), float(spot_prob), int(spot_radius))
    return final, arrival, duration


def frame_at(arrival: np.ndarray, forest: np.ndarray, t: int, burn_steps: int = 1) -> np.ndarray:
    """Rebuild the grid as it looked at step t, from the arrival-time record.

    A cell that caught fire at step a burns during steps a .. a + burn_steps - 1
    and is burnt afterwards; every other cell keeps its original state.
    """
    frame = forest.copy()
    lit = arrival >= 0
    frame[lit & (arrival + burn_steps <= t)] = BURNT
    frame[lit & (arrival <= t) & (t < arrival + burn_steps)] = BURNING
    return frame
