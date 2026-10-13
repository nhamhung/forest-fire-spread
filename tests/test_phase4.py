"""Validation checks for Phase 4: fire management strategies and ember spotting."""

import numpy as np
import pytest

from forest_fire import experiments as ex
from forest_fire import management as mg
from forest_fire.grid import BURNT, EMPTY, TREE, ignite_point, make_forest
from forest_fire.spread import run_fire


@pytest.fixture
def forest():
    return make_forest(100, 0.7, np.random.default_rng(0))


def test_firebreak_lines_are_cleared(forest):
    treated, removed = mg.firebreaks(forest, spacing=20, width=2)
    lines = [18, 19, 38, 39, 58, 59, 78, 79, 98, 99]
    assert (treated[lines, :] == EMPTY).all() and (treated[:, lines] == EMPTY).all()
    assert removed == (forest == TREE).sum() - (treated == TREE).sum()
    assert (forest == TREE).sum() > 0 and (treated[:5, :5] == forest[:5, :5]).all()   # interior untouched


def test_firebreak_spacing_matches_budget():
    for budget in (0.04, 0.1, 0.2):
        s = mg.spacing_for_budget(200, budget)
        cleared = 1 - (1 - 1 / s) ** 2
        assert cleared == pytest.approx(budget, abs=0.02)


def test_thinning_removes_the_requested_fraction(forest):
    treated, removed = mg.random_thinning(forest, 0.2, np.random.default_rng(1))
    assert removed / (forest == TREE).sum() == pytest.approx(0.2, abs=0.02)
    assert ((treated == TREE) <= (forest == TREE)).all()     # nothing planted


def test_controlled_burns_reach_the_budget(forest):
    treated, removed = mg.controlled_burns(forest, 0.1, 10, np.random.default_rng(2))
    n = (forest == TREE).sum()
    assert removed >= 0.1 * n and removed == (treated == BURNT).sum()


def test_unknown_strategy_is_rejected(forest):
    with pytest.raises(ValueError):
        mg.treat(forest, "goats", 0.1, np.random.default_rng(0))


def test_fire_is_trapped_in_its_compartment():
    full = np.full((60, 60), TREE, np.uint8)
    treated, _ = mg.firebreaks(full, spacing=20)
    final, _, _ = run_fire(ignite_point(treated, 5, 5))           # q = 1: burns everything it can reach
    assert (final == BURNT).sum() == 19 * 19                      # only the first 19 x 19 compartment


def test_embers_jump_firebreaks():
    full = np.full((60, 60), TREE, np.uint8)
    treated, _ = mg.firebreaks(full, spacing=20)
    jumped = [(run_fire(ignite_point(treated, 5, 5), seed=s, spot_prob=0.2, spot_radius=4)[0] == BURNT).sum() > 19 * 19
              for s in range(20)]
    assert sum(jumped) >= 15          # embers are random: most fires escape the compartment, not every one


def test_no_embers_draws_nothing_extra(forest):
    a = run_fire(ignite_point(forest), 0.8, seed=4)
    b = run_fire(ignite_point(forest), 0.8, seed=4, spot_prob=0.0, spot_radius=5)
    assert np.array_equal(a[1], b[1])


def test_firebreaks_beat_thinning_and_burns_at_low_budget():
    s = ex.Phase4Settings(size=100, n_runs=30, budgets=[0.06])
    sm = ex.summarise_management(ex.management_sweep(s)).set_index("strategy")
    assert sm.loc["firebreaks", "loss"] < 0.5 * sm.loc["thinning", "loss"]
    assert sm.loc["firebreaks", "loss"] < 0.5 * sm.loc["controlled_burns", "loss"]
    assert sm.loc["firebreaks", "trees_saved_per_tree_removed"] > 3
