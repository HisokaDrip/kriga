"""python -m kriga [fit|bo|all]"""
import sys
import time

import numpy as np

from . import GaussianProcess, RBF, Periodic, Matern52, BayesOpt, random_search
from .bayesopt import branin, BRANIN_BOUNDS, BRANIN_MIN, hartmann3, HARTMANN3_BOUNDS, HARTMANN3_MIN
from .plot import band_plot, convergence_plot


def co2ish(t, rng=None):
    # a toy version of the mauna loa curve: slow quadratic trend,
    # a yearly cycle that isn't a pure sine, and some measurement noise
    y = 0.05 * t ** 2 + 1.2 * t + 2.0 * np.sin(2 * np.pi * t) + 0.6 * np.cos(4 * np.pi * t + 0.3)
    if rng is not None:
        y = y + 0.25 * rng.standard_normal(t.shape)
    return y


def demo_fit():
    print("== structure discovery on a seasonal series ==\n")
    rng = np.random.default_rng(7)
    t = np.sort(rng.uniform(0, 8, 90))
    y = co2ish(t, rng)

    # long-range trend + locally-periodic seasonality. The periodic part
    # starts deliberately off (period 0.7) so you can see it get learned.
    kernel = (RBF(lengthscale=5.0, variance=10.0, ls_bounds=(1, 1e3))
              + RBF(lengthscale=4.0, ls_bounds=(1, 1e2)) * Periodic(lengthscale=1.0, period=0.7,
                                                                   period_bounds=(0.3, 3)))
    gp = GaussianProcess(kernel, noise=0.1)
    tic = time.perf_counter()
    gp.fit(t, y, restarts=4, seed=1)
    print(gp.summary())
    period = np.exp(gp.kernel.b.b.theta[2])
    print(f"learned period: {period:.4f}  (true 1.0)    fit took {time.perf_counter() - tic:.2f}s\n")

    ts = np.linspace(0, 10, 400)
    mean, std = gp.predict(ts, return_std=True)
    print(band_plot(ts, mean, std, t, y, truth=co2ish(ts), vline=8.0))
    print("  o data   - posterior mean   . +/-2 sd   : truth   | end of data\n")

    future = ts > 8
    rmse = np.sqrt(np.mean((mean[future] - co2ish(ts[future])) ** 2))
    cover = np.mean(np.abs(mean[future] - co2ish(ts[future])) < 2 * std[future])
    print(f"extrapolation over the 2 unseen years: rmse {rmse:.3f}, 2sd band covers {cover:.0%} of truth\n")


def demo_bo(seeds=5):
    print("== bayesian optimisation vs random search ==\n")
    problems = [("branin", branin, BRANIN_BOUNDS, BRANIN_MIN, 30),
                ("hartmann3", hartmann3, HARTMANN3_BOUNDS, HARTMANN3_MIN, 30)]
    for name, f, bounds, fmin, n in problems:
        bo_curves, rs_curves = [], []
        tic = time.perf_counter()
        for s in range(seeds):
            bo = BayesOpt(f, bounds, seed=s)
            bo.run(n)
            bo_curves.append(np.minimum.accumulate(bo.y) - fmin)
            _, ry = random_search(f, bounds, n, seed=s)
            rs_curves.append(np.minimum.accumulate(ry) - fmin)
        bo_med = np.median(bo_curves, axis=0)
        rs_med = np.median(rs_curves, axis=0)
        # how many random evals it takes to match BO's final answer
        _, big = random_search(f, bounds, 20000, seed=99)
        hit = np.flatnonzero(np.minimum.accumulate(big) - fmin <= bo_med[-1])
        need = f"~{hit[0] + 1}" if hit.size else f"more than {len(big)}"
        print(f"{name}: {n} evals, median over {seeds} seeds "
              f"({time.perf_counter() - tic:.1f}s)")
        print(f"  final regret   bayes-opt {bo_med[-1]:.2e}   random {rs_med[-1]:.2e}")
        print(f"  random search needed {need} evals to match bayes-opt's {n}\n")
        print(convergence_plot([bo_med, rs_med], ["bayes-opt (EI)", "random"]))
        print()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    what = argv[0] if argv else "all"
    if what in ("fit", "all"):
        demo_fit()
    if what in ("bo", "all"):
        demo_bo()


if __name__ == "__main__":
    main()
