import numpy as np
import pytest

from kriga import RBF, Matern52, Periodic

rng = np.random.default_rng(0)


def kernels():
    return [
        RBF(1.3, 2.0),
        RBF([0.5, 2.0, 1.1], 0.7),
        Matern52(0.8, 1.5),
        Matern52([0.4, 1.7, 3.0], 2.2),
        Periodic(0.9, 1.7, 1.3),
        RBF(2.0) + Periodic(1.0, 0.5),
        RBF(3.0) * Periodic(0.7, 1.2) + Matern52(0.5, 0.3),
        Matern52([1.0, 0.5, 2.0]) * RBF([2.0, 2.0, 0.3]),
    ]


@pytest.mark.parametrize("k", kernels(), ids=lambda k: k.describe()[:40])
def test_gradient_matches_finite_differences(k):
    X = rng.normal(size=(9, 3))
    K, G = k(X, grad=True)
    assert G.shape == (k.n_params, 9, 9)
    assert np.allclose(K, k(X))
    theta = k.theta
    eps = 1e-6
    for j in range(k.n_params):
        t = theta.copy(); t[j] += eps; k.theta = t; Kp = k(X)
        t = theta.copy(); t[j] -= eps; k.theta = t; Km = k(X)
        k.theta = theta
        assert np.allclose((Kp - Km) / (2 * eps), G[j], atol=1e-6), k.param_names()[j]


@pytest.mark.parametrize("k", kernels(), ids=lambda k: k.describe()[:40])
def test_psd_and_diag(k):
    X = rng.normal(size=(30, 3))
    K = k(X)
    assert np.allclose(K, K.T)
    assert np.linalg.eigvalsh(K).min() > -1e-8
    assert np.allclose(np.diag(K), k.diag(X))


def test_periodic_really_is_periodic():
    k = Periodic(0.8, 1.7)
    X = rng.normal(size=(6, 2))
    shift = np.array([1.7, -3.4])  # whole periods in every dimension
    assert np.allclose(k(X, X + shift), k(X, X))


def test_cross_covariance_shape():
    k = RBF(1.0) * Periodic()
    assert k(rng.normal(size=(4, 2)), rng.normal(size=(7, 2))).shape == (4, 7)


@pytest.mark.parametrize("ours,theirs", [
    (lambda: RBF(0.7, 1.9), lambda K: K.ConstantKernel(1.9) * K.RBF(0.7)),
    (lambda: Matern52(1.3, 0.6), lambda K: K.ConstantKernel(0.6) * K.Matern(1.3, nu=2.5)),
    (lambda: Periodic(0.8, 2.1, 1.4), lambda K: K.ConstantKernel(1.4) * K.ExpSineSquared(0.8, 2.1)),
    (lambda: RBF([0.5, 2.0]), lambda K: K.RBF([0.5, 2.0])),
])
def test_matches_sklearn_kernels(ours, theirs):
    from sklearn.gaussian_process import kernels as K
    # 1d for the periodic one - sklearn uses euclidean distance there, which
    # only agrees with the per-dimension form in one dimension
    d = 1 if "Periodic" in ours().describe() else 2
    X = rng.normal(size=(12, d))
    Y = rng.normal(size=(5, d))
    assert np.allclose(ours()(X, Y), theirs(K)(X, Y))
