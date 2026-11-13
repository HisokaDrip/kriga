import numpy as np
import pytest

from kriga import GaussianProcess, RBF, Matern52, Periodic


def toy(n=40, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 10, size=(n, 1))
    y = np.sin(X[:, 0]) + 0.5 * np.cos(2.3 * X[:, 0]) + 0.1 * rng.standard_normal(n)
    return X, y


def test_lml_gradient_finite_difference():
    X, y = toy()
    gp = GaussianProcess(RBF(1.0) * Periodic(1.0, 3.0) + Matern52(0.5, 0.2), noise=0.05)
    gp.fit(X, y, optimize=False)
    theta = gp._get_theta()
    lml, g = gp.log_marginal_likelihood(theta, grad=True)
    eps = 1e-6
    for j in range(len(theta)):
        tp, tm = theta.copy(), theta.copy()
        tp[j] += eps; tm[j] -= eps
        num = (gp.log_marginal_likelihood(tp) - gp.log_marginal_likelihood(tm)) / (2 * eps)
        assert num == pytest.approx(g[j], rel=1e-4, abs=1e-6)


def test_matches_sklearn_fixed_hyperparams():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF as SkRBF, ConstantKernel, WhiteKernel
    X, y = toy()
    ours = GaussianProcess(RBF(1.2, 0.8), noise=0.03, normalize_y=False, jitter=0.0)
    ours.fit(X, y, optimize=False)
    sk = GaussianProcessRegressor(ConstantKernel(0.8) * SkRBF(1.2) + WhiteKernel(0.03),
                                  optimizer=None, alpha=0.0).fit(X, y)
    Xs = np.linspace(-1, 11, 50)[:, None]
    m1, s1 = ours.predict(Xs, return_std=True)
    m2, s2 = sk.predict(Xs, return_std=True)
    assert ours.lml_ == pytest.approx(sk.log_marginal_likelihood_value_, rel=1e-9)
    assert np.allclose(m1, m2, atol=1e-8)
    # sklearn's std includes the white noise, ours is the latent function
    assert np.allclose(s1 ** 2 + 0.03, s2 ** 2, atol=1e-8)


def test_optimised_lml_at_least_as_good_as_sklearn():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF as SkRBF, ConstantKernel, WhiteKernel
    X, y = toy(60, seed=3)
    ours = GaussianProcess(RBF(1.0), noise=0.1, normalize_y=False).fit(X, y, restarts=3)
    sk = GaussianProcessRegressor(ConstantKernel() * SkRBF() + WhiteKernel(0.1),
                                  n_restarts_optimizer=3, random_state=0).fit(X, y)
    assert ours.lml_ >= sk.log_marginal_likelihood_value_ - 1e-3


def test_interpolates_noise_free_data():
    X = np.linspace(0, 5, 12)
    y = np.sin(X)
    gp = GaussianProcess(Matern52(1.0), noise=1e-8, fix_noise=True).fit(X, y)
    m, s = gp.predict(X, return_std=True)
    assert np.allclose(m, y, atol=1e-4)
    assert s.max() < 1e-3


def test_uncertainty_grows_away_from_data():
    X, y = toy()
    gp = GaussianProcess(RBF(1.0)).fit(X, y)
    _, s_in = gp.predict([[5.0]], return_std=True)
    _, s_out = gp.predict([[25.0]], return_std=True)
    assert s_out[0] > 5 * s_in[0]


def test_recovers_period():
    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0, 12, 80))
    y = np.sin(2 * np.pi * t / 2.5) + 0.05 * rng.standard_normal(80)
    gp = GaussianProcess(Periodic(1.0, 1.9, period_bounds=(1.0, 4.0)), noise=0.01)
    # the period likelihood is badly multimodal (one bad basin just calls
    # everything noise), so this genuinely needs a handful of restarts
    gp.fit(t, y, restarts=8)
    assert np.exp(gp.kernel.theta[2]) == pytest.approx(2.5, rel=0.01)


def test_samples_have_posterior_moments():
    X, y = toy(20)
    gp = GaussianProcess(RBF(1.0)).fit(X, y)
    Xs = np.linspace(0, 10, 15)
    S = gp.sample(Xs, n=4000, seed=0)
    m, s = gp.predict(Xs, return_std=True)
    assert np.allclose(S.mean(0), m, atol=4 * s.max() / np.sqrt(4000) + 1e-3)
    assert np.allclose(S.std(0), s, rtol=0.1, atol=1e-3)
