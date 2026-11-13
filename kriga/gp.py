import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.optimize import minimize

from .kernels import _as2d

LOG_2PI = np.log(2 * np.pi)


class GaussianProcess:
    """Exact GP regression with a Gaussian likelihood.

    noise is the observation noise *variance*; it's learned along with the
    kernel hyperparameters unless fix_noise=True.
    """

    def __init__(self, kernel, noise=1e-2, noise_bounds=(1e-8, 1e1),
                 normalize_y=True, fix_noise=False, jitter=1e-10):
        self.kernel = kernel
        self.log_noise = np.log(noise)
        self.noise_bounds = np.log(noise_bounds)
        self.normalize_y = normalize_y
        self.fix_noise = fix_noise
        self.jitter = jitter

    @property
    def noise(self):
        return float(np.exp(self.log_noise))

    # -- the one function everything else leans on ------------------------
    def _factor(self, X, y, grad=False):
        n = X.shape[0]
        if grad:
            K, dK = self.kernel(X, grad=True)
        else:
            K, dK = self.kernel(X), None
        K = K + (self.noise + self.jitter) * np.eye(n)
        # a cholesky failure here almost always means the optimizer wandered
        # into a corner where noise ~ 0 and two inputs coincide
        L = np.linalg.cholesky(K)
        alpha = cho_solve((L, True), y)
        lml = -0.5 * y @ alpha - np.log(np.diag(L)).sum() - 0.5 * n * LOG_2PI
        if not grad:
            return L, alpha, lml, None
        # d lml / d theta_j = 0.5 tr((aa^T - K^-1) dK_j)
        inner = np.outer(alpha, alpha) - cho_solve((L, True), np.eye(n))
        g = 0.5 * np.einsum("ij,kji->k", inner, dK)
        if not self.fix_noise:
            g_noise = 0.5 * self.noise * np.trace(inner)
            g = np.append(g, g_noise)
        return L, alpha, lml, g

    def _prep_y(self, y):
        y = np.asarray(y, dtype=float).ravel()
        if self.normalize_y:
            self._y_mean = y.mean()
            self._y_std = y.std() or 1.0
        else:
            self._y_mean, self._y_std = 0.0, 1.0
        return (y - self._y_mean) / self._y_std

    def _get_theta(self):
        t = self.kernel.theta
        return t if self.fix_noise else np.append(t, self.log_noise)

    def _set_theta(self, theta):
        k = self.kernel.n_params
        self.kernel.theta = theta[:k]
        if not self.fix_noise:
            self.log_noise = theta[k]

    def _bounds(self):
        b = self.kernel.bounds
        return b if self.fix_noise else np.vstack([b, self.noise_bounds])

    def log_marginal_likelihood(self, theta=None, grad=False, X=None, y=None):
        X = self.X_ if X is None else _as2d(X)
        y = self.y_ if y is None else y
        if theta is not None:
            saved = self._get_theta()
            self._set_theta(theta)
        try:
            _, _, lml, g = self._factor(X, y, grad=grad)
        finally:
            if theta is not None:
                self._set_theta(saved)
        return (lml, g) if grad else lml

    # -- fitting ------------------------------------------------------------
    def fit(self, X, y, optimize=True, restarts=3, seed=0):
        X = _as2d(X)
        y = self._prep_y(y)
        self.X_, self.y_ = X, y
        if optimize:
            self._optimize(restarts, np.random.default_rng(seed))
        self.L_, self.alpha_, self.lml_, _ = self._factor(X, y)
        return self

    def _optimize(self, restarts, rng):
        bounds = self._bounds()

        def nll(theta):
            self._set_theta(theta)
            try:
                _, _, lml, g = self._factor(self.X_, self.y_, grad=True)
            except np.linalg.LinAlgError:
                return 1e25, np.zeros_like(theta)
            return -lml, -g

        starts = [self._get_theta()]
        for _ in range(restarts):
            starts.append(rng.uniform(bounds[:, 0], bounds[:, 1]))

        best = None
        for x0 in starts:
            res = minimize(nll, x0, jac=True, method="L-BFGS-B", bounds=bounds)
            if best is None or res.fun < best.fun:
                best = res
        self._set_theta(best.x)
        self.opt_result_ = best

    # -- prediction ---------------------------------------------------------
    def predict(self, Xs, return_std=False, return_cov=False):
        Xs = _as2d(Xs)
        Ks = self.kernel(self.X_, Xs)
        mean = Ks.T @ self.alpha_
        mean = mean * self._y_std + self._y_mean
        if not (return_std or return_cov):
            return mean
        v = solve_triangular(self.L_, Ks, lower=True)
        if return_cov:
            cov = self.kernel(Xs) - v.T @ v
            return mean, cov * self._y_std ** 2
        var = self.kernel.diag(Xs) - (v * v).sum(0)
        # tiny negatives from round-off at the training points
        var = np.maximum(var, 0.0)
        return mean, np.sqrt(var) * self._y_std

    def sample(self, Xs, n=1, seed=None):
        mean, cov = self.predict(Xs, return_cov=True)
        rng = np.random.default_rng(seed)
        L = np.linalg.cholesky(cov + 1e-9 * np.eye(len(mean)) * cov.diagonal().mean())
        return mean[None] + (L @ rng.standard_normal((len(mean), n))).T

    def summary(self):
        lines = [f"kernel : {self.kernel.describe()}",
                 f"noise  : {_fmtvar(self.noise * self._y_std ** 2)} (variance, data units)",
                 f"log ML : {self.lml_:.3f}"]
        return "\n".join(lines)


def _fmtvar(v):
    return f"{v:.4g}"
