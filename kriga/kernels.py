"""Covariance functions.

Every hyperparameter lives in log space (theta). That keeps them positive
for free and makes the optimizer's life a lot easier, since lengthscales
can sensibly range over several orders of magnitude.

Each kernel knows how to give you K, its diagonal, and dK/dtheta as a
stack of (n_params, n, n) matrices. The gradients are what make
marginal-likelihood fitting fast - see test_kernels.py for the
finite-difference checks.
"""
import numpy as np


def _as2d(X):
    X = np.asarray(X, dtype=float)
    return X[:, None] if X.ndim == 1 else X


class Kernel:
    # subclasses set self._names / self._log / self._bounds

    @property
    def theta(self):
        return np.array(self._log, dtype=float)

    @theta.setter
    def theta(self, value):
        self._log = list(np.asarray(value, dtype=float))

    @property
    def bounds(self):
        return np.array(self._bounds, dtype=float).reshape(-1, 2)

    @property
    def n_params(self):
        return len(self._log)

    def __add__(self, other):
        return Sum(self, other)

    def __mul__(self, other):
        return Product(self, other)

    def __call__(self, X1, X2=None, grad=False):
        raise NotImplementedError

    def diag(self, X):
        raise NotImplementedError

    def describe(self):
        vals = np.exp(self.theta)
        inner = ", ".join(f"{n}={_fmt(v)}" for n, v in zip(self.param_names(), vals))
        return f"{type(self).__name__}({inner})"

    def __repr__(self):
        return self.describe()


def _fmt(v):
    return f"{v:.4g}"


class _Stationary(Kernel):
    """Shared plumbing for kernels with a variance and (maybe ARD) lengthscales."""

    def __init__(self, lengthscale=1.0, variance=1.0,
                 ls_bounds=(1e-3, 1e3), var_bounds=(1e-3, 1e3)):
        ls = np.atleast_1d(np.asarray(lengthscale, dtype=float))
        self.ard = ls.size > 1
        self._log = [np.log(variance)] + list(np.log(ls))
        self._bounds = [np.log(var_bounds)] + [np.log(ls_bounds)] * ls.size

    def param_names(self):
        n_ls = self.n_params - 1
        if n_ls == 1:
            return ["variance", "lengthscale"]
        return ["variance"] + [f"lengthscale[{i}]" for i in range(n_ls)]

    def _split(self):
        variance = np.exp(self._log[0])
        ls = np.exp(np.array(self._log[1:]))
        return variance, ls

    def _scaled_sqdiff(self, X1, X2):
        # (n1, n2, d) of ((x - x') / l)^2 - only materialised when needed
        _, ls = self._split()
        diff = (X1[:, None, :] - X2[None, :, :]) / ls
        return diff * diff

    def _sqdist(self, X1, X2):
        _, ls = self._split()
        A, B = X1 / ls, X2 / ls
        d2 = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T
        return np.maximum(d2, 0.0)

    def diag(self, X):
        return np.full(_as2d(X).shape[0], np.exp(self._log[0]))


class RBF(_Stationary):
    def __call__(self, X1, X2=None, grad=False):
        X1 = _as2d(X1)
        X2 = X1 if X2 is None else _as2d(X2)
        variance, _ = self._split()
        if not grad:
            return variance * np.exp(-0.5 * self._sqdist(X1, X2))
        sq = self._scaled_sqdiff(X1, X2)
        K = variance * np.exp(-0.5 * sq.sum(-1))
        if self.ard:
            dls = K[..., None] * sq                      # one slab per dimension
            dls = np.moveaxis(dls, -1, 0)
        else:
            dls = (K * sq.sum(-1))[None]
        return K, np.concatenate([K[None], dls])


class Matern52(_Stationary):
    def __call__(self, X1, X2=None, grad=False):
        X1 = _as2d(X1)
        X2 = X1 if X2 is None else _as2d(X2)
        variance, _ = self._split()
        s5 = np.sqrt(5.0)
        if not grad:
            r = np.sqrt(self._sqdist(X1, X2))
            return variance * (1 + s5 * r + 5 * r * r / 3) * np.exp(-s5 * r)
        sq = self._scaled_sqdiff(X1, X2)
        r = np.sqrt(sq.sum(-1))
        e = np.exp(-s5 * r)
        K = variance * (1 + s5 * r + 5 * r * r / 3) * e
        # dk/dlog(l_d) = var * 5/3 * (1 + sqrt5 r) e^{-sqrt5 r} * (dx_d/l_d)^2
        # nice property: no 1/r, so the diagonal (r = 0) is perfectly safe
        common = variance * (5.0 / 3.0) * (1 + s5 * r) * e
        if self.ard:
            dls = np.moveaxis(common[..., None] * sq, -1, 0)
        else:
            dls = (common * sq.sum(-1))[None]
        return K, np.concatenate([K[None], dls])


class Periodic(Kernel):
    """exp-sine-squared: k = var * exp(-2 sum_d sin^2(pi (x_d - x'_d) / p) / l^2).

    Note the sum over dimensions *inside* the exp. The more common form
    plugs the euclidean distance |x - x'| into the sine, and that is not
    positive definite once d > 1 (the psd test caught this - eigenvalues
    of -1.8 on random 3d inputs). Summing per-dimension sines is the
    MacKay "warp to a circle" construction, so it's psd by design. In 1d
    the two are identical.
    """

    def __init__(self, lengthscale=1.0, period=1.0, variance=1.0,
                 ls_bounds=(1e-2, 1e2), period_bounds=(1e-2, 1e2), var_bounds=(1e-3, 1e3)):
        self._log = [np.log(variance), np.log(lengthscale), np.log(period)]
        self._bounds = [np.log(var_bounds), np.log(ls_bounds), np.log(period_bounds)]

    def param_names(self):
        return ["variance", "lengthscale", "period"]

    def __call__(self, X1, X2=None, grad=False):
        X1 = _as2d(X1)
        X2 = X1 if X2 is None else _as2d(X2)
        variance, ls, period = np.exp(self._log)
        arg = np.pi * (X1[:, None, :] - X2[None, :, :]) / period
        s = np.sin(arg)
        s2 = (s * s).sum(-1)
        K = variance * np.exp(-2 * s2 / ls ** 2)
        if not grad:
            return K
        dls = K * 4 * s2 / ls ** 2
        dper = K * (4 / ls ** 2) * (s * np.cos(arg) * arg).sum(-1)
        return K, np.stack([K, dls, dper])

    def diag(self, X):
        return np.full(_as2d(X).shape[0], np.exp(self._log[0]))


class _Combo(Kernel):
    def __init__(self, a, b):
        self.a, self.b = a, b

    @property
    def theta(self):
        return np.concatenate([self.a.theta, self.b.theta])

    @theta.setter
    def theta(self, value):
        value = np.asarray(value, dtype=float)
        self.a.theta = value[: self.a.n_params]
        self.b.theta = value[self.a.n_params:]

    @property
    def bounds(self):
        return np.vstack([self.a.bounds, self.b.bounds])

    @property
    def n_params(self):
        return self.a.n_params + self.b.n_params

    def param_names(self):
        return [f"a.{n}" for n in self.a.param_names()] + [f"b.{n}" for n in self.b.param_names()]


class Sum(_Combo):
    def __call__(self, X1, X2=None, grad=False):
        if not grad:
            return self.a(X1, X2) + self.b(X1, X2)
        Ka, Ga = self.a(X1, X2, grad=True)
        Kb, Gb = self.b(X1, X2, grad=True)
        return Ka + Kb, np.concatenate([Ga, Gb])

    def diag(self, X):
        return self.a.diag(X) + self.b.diag(X)

    def describe(self):
        return f"{self.a.describe()} + {self.b.describe()}"


class Product(_Combo):
    def __call__(self, X1, X2=None, grad=False):
        if not grad:
            return self.a(X1, X2) * self.b(X1, X2)
        Ka, Ga = self.a(X1, X2, grad=True)
        Kb, Gb = self.b(X1, X2, grad=True)
        return Ka * Kb, np.concatenate([Ga * Kb, Ka * Gb])

    def diag(self, X):
        return self.a.diag(X) * self.b.diag(X)

    def describe(self):
        def wrap(k):
            s = k.describe()
            return f"({s})" if isinstance(k, Sum) else s
        return f"{wrap(self.a)} * {wrap(self.b)}"
