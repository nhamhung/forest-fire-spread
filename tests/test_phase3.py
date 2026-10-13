"""Validation checks for Phase 3: drought, wind and multi-step burning."""

import numpy as np
import pytest

from forest_fire import experiments as ex
from forest_fire.grid import BURNING, TREE, ignite_point, make_forest
from forest_fire.spread import NEIGHBOURHOODS, frame_at, per_step_probability, run_fire
from forest_fire.weather import direction_probabilities, dry_spread_probability, wind_unit_vector


def test_dryness_mapping():
    assert dry_spread_probability(0.6, 0.0) == pytest.approx(0.6)
    assert dry_spread_probability(0.6, 1.0) == pytest.approx(1.0)
    assert dry_spread_probability(0.6, 0.5) == pytest.approx(0.8)
    with pytest.raises(ValueError):
        dry_spread_probability(0.6, 1.5)


def test_wind_directions():
    assert np.allclose(wind_unit_vector(0), [0, 1])     # towards the right edge
    assert np.allclose(wind_unit_vector(90), [-1, 0])   # towards the top (row index decreases)


def test_wind_boosts_downwind_and_cuts_upwind():
    p = direction_probabilities(NEIGHBOURHOODS["von_neumann"], 0.2, wind_direction=0, wind_speed=10)
    up, down, left, right = p
    assert right == pytest.approx(0.2 * np.exp(1.31)) and left == pytest.approx(0.2 * np.exp(-1.31))
    assert up == pytest.approx(0.2) and down == pytest.approx(0.2)
    assert np.allclose(direction_probabilities(NEIGHBOURHOODS["moore"], 0.4), 0.4)   # no wind: all equal


def test_probabilities_are_capped_at_one():
    assert direction_probabilities(NEIGHBOURHOODS["von_neumann"], 0.9, 0, 20).max() == 1.0


def test_one_step_burning_is_the_phase2_model():
    forest = make_forest(60, 0.7, np.random.default_rng(2))
    a = run_fire(ignite_point(forest), 0.8, seed=5)
    b = run_fire(ignite_point(forest), 0.8, seed=5, burn_steps=1)
    assert np.array_equal(a[1], b[1]) and a[2] == b[2]


def test_per_step_probability_adds_up_to_q():
    p = per_step_probability(0.9, 10)
    assert 1 - (1 - p) ** 10 == pytest.approx(0.9)


def test_multi_step_cells_stay_alight():
    forest = np.zeros((3, 3), np.uint8)
    forest[1, 1] = TREE
    _, arrival, duration = run_fire(ignite_point(forest), burn_steps=4)
    assert duration == 4 and arrival[1, 1] == 0
    assert frame_at(arrival, forest, 3, burn_steps=4)[1, 1] == BURNING


def test_drought_makes_fires_bigger():
    df = ex.dryness_sweep(100, 0.7, [0.0, 0.5, 1.0], 0.6, 30, seed=3)
    burned = df["fraction_burned"].to_numpy()
    assert burned[0] < 0.1 and burned[2] > 0.9 and burned[0] < burned[1] < burned[2]


def test_wind_speeds_up_the_head_and_slows_the_back():
    s = ex.Phase3Settings(size=120, wind_runs=12, wind_speeds=[0.0, 10.0])
    w = ex.summarise_wind(ex.wind_sweep(s)).set_index("wind_speed")
    assert w.loc[10.0, "head_speed"] > 1.8 * w.loc[0.0, "head_speed"]
    assert w.loc[10.0, "back_speed"] < 0.5 * w.loc[0.0, "back_speed"]
    assert abs(w.loc[0.0, "length_to_width"] - 1) < 0.1          # no wind: round on average


@pytest.mark.parametrize("direction, axis, sign", [(0, 1, +1), (90, 0, -1), (180, 1, -1), (270, 0, +1)])
def test_fire_runs_downwind(direction, axis, sign):
    forest = make_forest(120, 0.8, np.random.default_rng(8))
    _, arrival, _ = run_fire(ignite_point(forest), 0.9, seed=1, wind_direction=direction, wind_speed=10, burn_steps=10)
    rows, cols = np.nonzero((arrival >= 0) & (arrival <= 100))
    centre_of_burn = (rows.mean() - 60, cols.mean() - 60)
    assert sign * centre_of_burn[axis] > 10      # the burnt area has moved well downwind of the spark
