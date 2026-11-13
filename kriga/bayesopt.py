"""Bayesian optimisation on top of the GP.

Minimisation, box-constrained. The loop is the textbook one:
fit GP -> maximise acquisition -> evaluate -> repeat. The acquisition is
maximised with a big random sweep followed by L-BFGS-B polishing of the
best few candidates, which is crude but hard to beat at this size.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from .gp import GaussianProcess
from .kernels import Matern52


def expected_improvement(mu, sigma, best, xi=0.01):
    sigma = np.maximum(sigma, 1e-12)
    imp = best - mu - xi
    z = imp / sigma
    return imp * norm.cdf(z) + sigma * norm.pdf(z)


def lower_confidence_bound(mu, sigma, kappa=2.0):
    # negated so that "bigger is better" like EI
    return -(mu - kappa * sigma)


class BayesOpt:
    def __init__(self, f, bounds, acquisition="ei", n_init=5, seed=0, xi=0.01, kappa=2.0):
        self.f = f
        self.bounds = np.asarray(bounds, dtype=float)
        self.dim = len(self.bounds)
        self.acquisition = acquisition
        self.n_init = n_init
        self.rng = np.random.default_rng(seed)
        self.xi, self.kappa = xi, kappa
        self.X, self.y = [], []

    # work in the unit cube so one set of lengthscale bounds fits every problem
    def _to_unit(self, x):
        lo, hi = self.bounds.T
        return (x - lo) / (hi - lo)

    def _from_unit(self, u):
        lo, hi = self.bounds.T
        return lo + u * (hi - lo)

    def _new_gp(self):
        kernel = Matern52(lengthscale=np.full(self.dim, 0.3), ls_bounds=(1e-2, 1e1))
        return GaussianProcess(kernel, noise=1e-6, noise_bounds=(1e-9, 1e-1))

    def _acq(self, gp, U, best):
        mu, sd = gp.predict(U, return_std=True)
        if self.acquisition == "ei":
            return expected_improvement(mu, sd, best, self.xi)
        return lower_confidence_bound(mu, sd, self.kappa)

    def suggest(self):
        if len(self.X) < self.n_init:
            return self._from_unit(self.rng.uniform(size=self.dim))
        U = self._to_unit(np.array(self.X))
        y = np.array(self.y)
        gp = self._new_gp().fit(U, y, restarts=2, seed=int(self.rng.integers(1 << 30)))
        self.gp_ = gp
        best = y.min()

        cand = self.rng.uniform(size=(4000, self.dim))
        scores = self._acq(gp, cand, best)
        top = cand[np.argsort(scores)[-5:]]
        best_u, best_s = top[-1], scores.max()
        for u0 in top:
            res = minimize(lambda u: -self._acq(gp, u[None], best)[0], u0,
                           method="L-BFGS-B", bounds=[(0, 1)] * self.dim)
            if -res.fun > best_s:
                best_u, best_s = res.x, -res.fun
        return self._from_unit(np.clip(best_u, 0, 1))

    def step(self):
        x = self.suggest()
        y = float(self.f(x))
        self.X.append(x)
        self.y.append(y)
        return x, y

    def run(self, n_iter, callback=None):
        for i in range(n_iter):
            x, y = self.step()
            if callback:
                callback(i, x, y, min(self.y))
        i = int(np.argmin(self.y))
        return self.X[i], self.y[i]


def random_search(f, bounds, n_iter, seed=0):
    bounds = np.asarray(bounds, dtype=float)
    rng = np.random.default_rng(seed)
    xs = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_iter, len(bounds)))
    ys = np.array([f(x) for x in xs])
    return xs, ys


# a couple of standard test functions, handy for the demo and tests
def branin(x):
    x1, x2 = x[0], x[1]
    a, b, c = 1.0, 5.1 / (4 * np.pi ** 2), 5 / np.pi
    r, s, t = 6.0, 10.0, 1 / (8 * np.pi)
    return a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s

BRANIN_BOUNDS = [(-5, 10), (0, 15)]
BRANIN_MIN = 0.397887


def hartmann3(x):
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]])
    P = 1e-4 * np.array([[3689, 1170, 2673], [4699, 4387, 7470],
                         [1091, 8732, 5547], [381, 5743, 8828]])
    x = np.asarray(x)
    return -np.sum(alpha * np.exp(-np.sum(A * (x - P) ** 2, axis=1)))

HARTMANN3_BOUNDS = [(0, 1)] * 3
HARTMANN3_MIN = -3.86278
