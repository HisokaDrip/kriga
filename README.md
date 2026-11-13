# kriga

Gaussian processes and Bayesian optimisation from scratch, in NumPy (plus
scipy for L-BFGS-B and the normal cdf). I kept using sklearn's
`GaussianProcessRegressor` as a black box and wanted to actually understand
where the uncertainty band comes from, so I wrote the whole thing myself.

What's in it:

- **kernels** with analytic gradients: RBF, Matérn 5/2 (both with optional ARD
  lengthscales), periodic, and arbitrary sums/products of them. Hyperparameters
  live in log space.
- **exact GP regression**: Cholesky fit, posterior mean/std/covariance,
  sampling, and the log marginal likelihood *with its gradient*, which is what
  lets L-BFGS-B tune every hyperparameter (noise included) in well under a
  second.
- **Bayesian optimisation**: expected improvement or lower confidence bound,
  ARD Matérn surrogate, acquisition maximised with a random sweep + L-BFGS-B
  polish.
- **terminal plots**, because I wanted to see the posterior without opening
  matplotlib.

It matches sklearn exactly (same log-likelihood to 1e-9, same predictions) for
fixed hyperparameters, and every kernel gradient is checked against finite
differences in the tests.

## run it

```
pip install -r requirements.txt
python -m kriga          # both demos
python -m kriga fit      # just the seasonal-series one
python -m kriga bo       # just bayes-opt vs random search
pytest -q
```

## demo 1: let the kernel find the structure

90 noisy points from a toy Mauna-Loa-ish curve (quadratic trend + a yearly
cycle that isn't a pure sine). The kernel is `RBF + RBF * Periodic`, and the
period is deliberately initialised wrong at 0.7. Then it predicts two years
past the end of the data:

```
== structure discovery on a seasonal series ==

kernel : RBF(variance=27.47, lengthscale=17.23) + RBF(variance=0.384, lengthscale=100) * Periodic(variance=2.272, lengthscale=2.225, period=0.9998)
noise  : 0.05198 (variance, data units)
log ML : 97.846
learned period: 0.9998  (true 1.0)    fit took 0.40s

   18.73 |                                                              |               
         |                                                              |         :    -
         |                                                              |       ----   .
         |                                                              | --   :.   : - 
   14.19 |                                                              --..   -    -:. 
         |                                                        oo   o.   - :      -  
         |                                                 o    oo  -  o|    --         
         |                                               -oo-   o   o o |    .          
    9.66 |                                         oo   -.  oo -     -o |               
         |                                       oo .o  o    o          |               
         |                               oooo   oo   - -      o         |               
         |                         o-    o   o  o     -.                |               
    5.12 |                  oo   ooo o  .     .o                        |               
         |           o    oo.o   o   o  -     oo                        |               
         |   o    ooo-   o    o o     -o                                |               
         | oo-o   -   o  o    ..-                                       |               
    0.58 |o   -  -    o o      -                                        |               
         |     -       o                                                |               
         |     :o                                                       |               
         |                                                              |               
         +------------------------------------------------------------------------------
          0.00                                                                     10.00
  o data   - posterior mean   . +/-2 sd   : truth   | end of data

extrapolation over the 2 unseen years: rmse 0.272, 2sd band covers 100% of truth
```

It found the period (0.9998), and the long lengthscale on the RBF multiplying
the periodic part says "the seasonal shape barely changes over time", which is
true for this data. The 2-sd band contains the true curve over the whole
extrapolated stretch.

## demo 2: bayes-opt vs random search

Same evaluation budget (30), median best-so-far regret over 5 seeds:

```
branin: 30 evals, median over 5 seeds (6.0s)
  final regret   bayes-opt 2.64e-03   random 1.37e+00
  random search needed ~14519 evals to match bayes-opt's 30

  1e+1.5 |ooo                                                         
         |   oooo                                                     
         |       oooo**                                               
  1e+0.4 |           oooooooooooooooooooooo                           
         |                 **********      ooooooooooooooooooooooooooo
         |                                                            
  1e-0.7 |                           ********                         
         |                                                            
         |                                   ****                     
  1e-1.8 |                                                            
         |                                       **************       
         |                                                     *******
         +------------------------------------------------------------
          1                                                         30  evals
          * bayes-opt (EI)   o random

hartmann3: 30 evals, median over 5 seeds (6.9s)
  final regret   bayes-opt 1.73e-03   random 2.48e-01
  random search needed more than 20000 evals to match bayes-opt's 30
```

(full output has a convergence plot for hartmann3 too.)

## usage

```python
import numpy as np
from kriga import GaussianProcess, RBF, Periodic, BayesOpt

gp = GaussianProcess(RBF(5.0) + RBF(4.0) * Periodic(period=0.7), noise=0.1)
gp.fit(t, y, restarts=4)
mean, std = gp.predict(t_new, return_std=True)
print(gp.summary())

bo = BayesOpt(my_expensive_fn, bounds=[(-5, 10), (0, 15)], seed=0)
x_best, y_best = bo.run(30)
```

## things I learned the hard way

- The "textbook" periodic kernel that plugs `|x - x'|` into the sine is
  **not** positive definite in more than one dimension. My PSD test found
  eigenvalues around -1.8 on random 3-d points. kriga sums per-dimension sines
  instead (MacKay's construction), which is the same thing in 1-d and actually
  valid in n-d.
- The marginal likelihood over the period is badly multimodal. There's a wide,
  comfortable basin where the model just calls everything noise. Restarts
  aren't optional.
- Doing BO in the unit cube means one set of lengthscale bounds works for
  every problem, which got rid of most of my tuning.

## what I'd do next

- sparse GPs (inducing points / SVGP) so it scales past a few thousand points
- a kernel search à la the "automatic statistician": greedily grow
  sums/products and pick by BIC
- batch BO (q-EI or just Kriging-believer) for parallel evaluations
- Matérn 3/2 and rational quadratic kernels

MIT licensed.
