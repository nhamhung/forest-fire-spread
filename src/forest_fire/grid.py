"""Building the forest: a square grid of cells, each either empty ground or a tree.

The grid is a 2D NumPy array of small integers. Row 0 is the top, column 0 the
left edge. Fire is lit along the left edge and we ask whether it can cross to the
right edge.
"""

import numpy as np

# Cell states. uint8 keeps a 400 x 400 forest at 160 kB.
EMPTY = 0    # bare ground: cannot burn, so it blocks the fire
TREE = 1     # fuel
BURNING = 2  # on fire right now; spreads to neighbouring trees on the next step
BURNT = 3    # burnt out: no fuel left, cannot burn again


def make_forest(size: int, density: float, rng: np.random.Generator) -> np.ndarray:
    """A size x size grid where each cell is independently a tree with probability `density`.

    `density` (often called p) is the fraction of ground covered by trees. Each
    cell is decided by its own coin flip, so trees form random clumps and gaps,
    with no pattern. Mathematically this is "site percolation" on a square lattice.
    """
    return (rng.random((size, size)) < density).astype(np.uint8)


def ignite_left_edge(forest: np.ndarray) -> np.ndarray:
    """Copy of the forest with every tree in the left column set on fire.

    Lighting the whole edge (instead of one tree) asks a clean question: can a
    fire that starts anywhere along one side find a connected path of trees to
    the opposite side? Empty cells in that column stay empty.
    """
    grid = forest.copy()
    edge = grid[:, 0]
    edge[edge == TREE] = BURNING
    return grid


def ignite_point(forest: np.ndarray, row: int | None = None, col: int | None = None) -> np.ndarray:
    """Copy of the forest with a single fire at (row, col), by default the centre.

    Used to watch the shape of a fire spreading from one spot, e.g. under wind.
    The cell is set burning even if it was empty, as if a spark landed there.
    """
    grid = forest.copy()
    n_rows, n_cols = grid.shape
    grid[n_rows // 2 if row is None else row, n_cols // 2 if col is None else col] = BURNING
    return grid
