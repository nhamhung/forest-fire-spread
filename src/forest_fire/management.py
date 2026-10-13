"""Fire management: ways of removing fuel before the fire season, and how to compare them.

Each strategy removes some trees in advance. The question is which one prevents
the most burning per tree removed. All three return a new grid plus the number
of trees removed, and never touch the original forest.

* Firebreaks: straight cleared lines in a square grid pattern, splitting the
  forest into compartments. In this model a cleared line of width 1 cannot be
  crossed (there is no ember spotting), so a fire is trapped in the compartment
  where it starts.
* Random thinning: remove a random fraction of trees everywhere. This lowers
  the density p, so it only helps much once p falls below the critical density.
* Controlled (prescribed) burns: burn square patches on purpose in the cool,
  wet season. Burnt ground has no fuel, so the patches act as scattered barriers.
"""

import numpy as np

from .grid import BURNT, EMPTY, TREE

STRATEGIES = ("none", "firebreaks", "thinning", "controlled_burns")


def firebreaks(forest: np.ndarray, spacing: int, width: int = 1) -> tuple[np.ndarray, int]:
    """Clear lines `width` cells wide every `spacing` cells, horizontally and vertically.

    Leaves square compartments of about (spacing - width)^2 cells. Cleared cells
    become empty ground.
    """
    grid = forest.copy()
    lines = np.zeros(grid.shape[0], bool)
    for start in range(spacing - width, grid.shape[0], spacing):
        lines[start:start + width] = True
    mask = lines[:, None] | lines[None, :]
    removed = int((grid[mask] == TREE).sum())
    grid[mask] = EMPTY
    return grid, removed


def random_thinning(forest: np.ndarray, fraction: float, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """Remove each tree independently with probability `fraction`."""
    grid = forest.copy()
    cut = (grid == TREE) & (rng.random(grid.shape) < fraction)
    grid[cut] = EMPTY
    return grid, int(cut.sum())


def controlled_burns(forest: np.ndarray, fraction: float, patch: int,
                     rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """Burn square `patch` x `patch` blocks at random places until `fraction` of the trees are gone.

    Patches may overlap. Burnt cells are marked BURNT, so they cannot carry a later fire.
    """
    grid = forest.copy()
    n_trees = int((forest == TREE).sum())
    target, removed = int(round(fraction * n_trees)), 0
    n = grid.shape[0]
    while removed < target:
        r, c = rng.integers(0, n - patch + 1, size=2)
        block = grid[r:r + patch, c:c + patch]
        hit = block == TREE
        removed += int(hit.sum())
        block[hit] = BURNT
    return grid, removed


def spacing_for_budget(size: int, budget: float, width: int = 1) -> int:
    """Firebreak spacing that clears about `budget` of the area.

    A square grid of lines every s cells clears 1 - (1 - w/s)^2 of the cells.
    Solving for s gives s = w / (1 - sqrt(1 - budget)).
    """
    if budget <= 0:
        return size + 1
    return max(width + 1, int(round(width / (1 - np.sqrt(1 - budget)))))


def treat(forest: np.ndarray, strategy: str, budget: float, rng: np.random.Generator,
          patch: int = 10, width: int = 1) -> tuple[np.ndarray, int]:
    """Apply `strategy` with roughly `budget` (fraction of trees) to remove; returns (grid, trees removed)."""
    if strategy == "none" or budget <= 0:
        return forest.copy(), 0
    if strategy == "firebreaks":
        return firebreaks(forest, spacing_for_budget(forest.shape[0], budget, width), width)
    if strategy == "thinning":
        return random_thinning(forest, budget, rng)
    if strategy == "controlled_burns":
        return controlled_burns(forest, budget, patch, rng)
    raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")
