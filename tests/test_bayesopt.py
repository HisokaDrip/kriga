import numpy as np
import pytest

from kriga import BayesOpt, expected_improvement, random_search
from kriga.bayesopt import branin, BRANIN_BOUNDS, BRANIN_MIN, hartmann3, HARTMANN3_MIN


def test_ei_basic_properties():
    mu = np.array([0.0, 0.0, 1.0])
    sd = np.array([1.0, 2.0, 1.0])
    ei = expected_improvement(mu, sd, best=0.5, xi=0.0)
    assert np.all(ei >= 0)
    assert ei[1] > ei[0] > ei[2]  # more uncertainty / lower mean -> more EI
    # zero variance collapses to plain improvement
    assert expected_improvement(np.array([0.2]), np.array([0.0]), 0.5, 0.0)[0] == pytest.approx(0.3)


def test_known_minima():
    assert branin(np.array([np.pi, 2.275])) == pytest.approx(BRANIN_MIN, abs=1e-5)
    assert hartmann3([0.114614, 0.555649, 0.852547]) == pytest.approx(HARTMANN3_MIN, abs=1e-4)


@pytest.mark.parametrize("seed", [0, 1])
def test_bo_beats_random_on_branin(seed):
    bo = BayesOpt(branin, BRANIN_BOUNDS, seed=seed)
    _, best = bo.run(25)
    _, ry = random_search(branin, BRANIN_BOUNDS, 25, seed=seed)
    assert best - BRANIN_MIN < 0.05
    assert best < ry.min()


def test_lcb_works_too():
    bo = BayesOpt(lambda x: (x[0] - 0.3) ** 2, [(-2, 2)], acquisition="lcb", seed=0)
    x, y = bo.run(12)
    assert abs(x[0] - 0.3) < 0.05


def test_suggestions_stay_in_bounds():
    bo = BayesOpt(branin, BRANIN_BOUNDS, seed=3)
    bo.run(10)
    X = np.array(bo.X)
    lo, hi = np.array(BRANIN_BOUNDS).T
    assert np.all(X >= lo) and np.all(X <= hi)
