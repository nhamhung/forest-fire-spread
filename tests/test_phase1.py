"""Validation checks for Phase 1: deterministic 4-neighbour fire = site percolation."""

import numpy as np
import pandas as pd
import pytest

from forest_fire import experiments as ex
from forest_fire.grid import BURNING, BURNT, EMPTY, TREE, ignite_left_edge, make_forest
from forest_fire.spread import frame_at, run_fire


def grid_from(rows: list[str]) -> np.ndarray:
    """'T' = tree, '.' = empty, so small test forests are readable."""
    return np.array([[TREE if ch == "T" else EMPTY for ch in row] for row in rows], dtype=np.uint8)


def test_density_matches_request():
    forest = make_forest(400, 0.6, np.random.default_rng(0))
    assert (forest == TREE).mean() == pytest.approx(0.6, abs=0.005)


def test_only_trees_in_left_column_ignite():
    g = ignite_left_edge(grid_from(["T.", ".T", "TT"]))
    assert g[:, 0].tolist() == [BURNING, EMPTY, BURNING] and g[1, 1] == TREE


def test_straight_row_of_trees_spans():
    m = ex.fire_metrics(grid_from(["TTTTT", ".....", "....."]))
    assert m["spanned"] and m["fraction_burned"] == 1.0 and m["duration"] == 5


def test_gap_stops_the_fire():
    m = ex.fire_metrics(grid_from(["TT.TT", ".....", "....."]))
    assert not m["spanned"] and m["fraction_burned"] == pytest.approx(0.5)


def test_fire_does_not_jump_diagonally():
    m = ex.fire_metrics(grid_from(["T.", ".T"]))
    assert not m["spanned"] and m["fraction_burned"] == pytest.approx(0.5)


def test_fire_follows_a_winding_path():
    final, arrival, duration = run_fire(ignite_left_edge(grid_from(["TTT.", "..T.", "..TT"])))
    # path (0,0) (0,1) (0,2) (1,2) (2,2) (2,3): the last tree ignites at step 5, and it burns out in step 6
    assert final[2, 3] == BURNT and arrival[2, 3] == 5 and duration == 6


def test_frames_rebuild_from_arrival_times():
    forest = make_forest(30, 0.65, np.random.default_rng(3))
    final, arrival, duration = run_fire(ignite_left_edge(forest))
    assert np.array_equal(frame_at(arrival, forest, duration + 1), final)       # after the end = final state
    assert (frame_at(arrival, forest, 0)[:, 0] == np.where(forest[:, 0] == TREE, BURNING, EMPTY)).all()


def test_spanning_far_from_threshold():
    low = ex.density_sweep(200, [0.45], 50, seed=1)["spanning_prob"].iloc[0]
    high = ex.density_sweep(200, [0.75], 50, seed=1)["spanning_prob"].iloc[0]
    assert low == 0.0 and high == 1.0


def test_spanning_probability_jumps_near_0593():
    ps = [round(x, 3) for x in np.arange(0.56, 0.631, 0.005)]
    fit = ex.fit_transition(ex.density_sweep(200, ps, 100, seed=5))
    assert fit["p50"] == pytest.approx(ex.P_C_SQUARE, abs=0.01)
    assert fit["width"] < 0.012


def test_transition_sharpens_with_size():
    ps = [round(x, 3) for x in np.arange(0.54, 0.651, 0.005)]
    fits = pd.DataFrame([{"size": n, **ex.fit_transition(ex.density_sweep(n, ps, 100, seed=9))} for n in (50, 200)])
    assert fits["width"].iloc[1] < 0.6 * fits["width"].iloc[0]


def test_same_seed_is_reproducible():
    a = ex.density_sweep(40, [0.6], 10, seed=4)
    b = ex.density_sweep(40, [0.6], 10, seed=4)
    assert a.equals(b)
