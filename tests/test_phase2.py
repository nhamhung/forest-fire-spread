"""Validation checks for Phase 2: probabilistic spread and the 8-neighbour option."""

import numpy as np
import pytest

from forest_fire import experiments as ex
from forest_fire.grid import EMPTY, TREE, make_forest


def grid_from(rows: list[str]) -> np.ndarray:
    return np.array([[TREE if ch == "T" else EMPTY for ch in row] for row in rows], dtype=np.uint8)


def test_q_one_is_the_deterministic_fire():
    forest = make_forest(60, 0.6, np.random.default_rng(1))
    assert ex.fire_metrics(forest) == ex.fire_metrics(forest, q=1.0, seed=123)


def test_q_zero_never_spreads():
    m = ex.fire_metrics(grid_from(["TTTT", "TTTT"]), q=0.0, seed=1)
    assert m["fraction_burned"] == pytest.approx(0.25) and m["duration"] == 1 and not m["spanned"]


def test_moore_jumps_diagonally_von_neumann_does_not():
    diagonal = grid_from(["T..", ".T.", "..T"])
    assert ex.fire_metrics(diagonal, neighbourhood="moore")["spanned"]
    assert not ex.fire_metrics(diagonal, neighbourhood="von_neumann")["spanned"]


def test_same_seed_same_random_fire():
    a = ex.density_sweep(60, [0.75], 10, seed=3, q=0.7)
    b = ex.density_sweep(60, [0.75], 10, seed=3, q=0.7)
    assert a.equals(b)


def test_q_only_changes_the_fire_not_the_forests():
    # same seed -> same forests, so the drop in average burn comes from q alone
    hi = ex.density_sweep(80, [0.7], 30, seed=4, q=1.0)["fraction_burned"].iloc[0]
    lo = ex.density_sweep(80, [0.7], 30, seed=4, q=0.6)["fraction_burned"].iloc[0]
    assert lo < hi


@pytest.fixture(scope="module")
def settings():
    return ex.Phase2Settings(coarse_runs=30, fine_runs=80)


def test_moore_site_threshold(settings):
    fit, _ = ex.critical_density(150, 1.0, "moore", settings)
    assert fit["p50"] == pytest.approx(ex.SITE_THRESHOLD["moore"], abs=0.01)


def test_wetter_fuel_raises_the_critical_density(settings):
    dry, _ = ex.critical_density(150, 1.0, "von_neumann", settings)
    wet, _ = ex.critical_density(150, 0.8, "von_neumann", settings)
    assert wet["p50"] - dry["p50"] == pytest.approx(0.105, abs=0.025)


def test_too_wet_means_no_fire_at_any_density(settings):
    fit, _ = ex.critical_density(100, 0.4, "von_neumann", settings)   # below the bond threshold 0.5
    assert np.isnan(fit["p50"])


@pytest.mark.parametrize("neighbourhood", ["von_neumann", "moore"])
def test_full_forest_bond_thresholds(neighbourhood):
    qs = [round(x, 3) for x in np.arange(0.15, 0.7, 0.01)]
    fit = ex.fit_transition(ex.spread_probability_sweep(150, qs, 60, 2, neighbourhood), x="q")
    assert fit["p50"] == pytest.approx(ex.BOND_THRESHOLD[neighbourhood], abs=0.015)
