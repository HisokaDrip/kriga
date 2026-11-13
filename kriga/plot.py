"""Terminal plotting. No matplotlib on purpose - I wanted the demo to
show the uncertainty band right in the terminal."""
import numpy as np


def band_plot(xs, mean, std, X_obs=None, y_obs=None, width=78, height=20,
              truth=None, vline=None):
    lo, hi = mean - 2 * std, mean + 2 * std
    ys = [lo, hi]
    if y_obs is not None:
        ys.append(np.asarray(y_obs))
    if truth is not None:
        ys.append(truth)
    ymin = min(a.min() for a in ys)
    ymax = max(a.max() for a in ys)
    pad = 0.05 * (ymax - ymin or 1)
    ymin, ymax = ymin - pad, ymax + pad

    grid = [[" "] * width for _ in range(height)]
    x0, x1 = xs.min(), xs.max()

    def col(x):
        return int(round((x - x0) / (x1 - x0) * (width - 1)))

    def row(y):
        return int(round((ymax - y) / (ymax - ymin) * (height - 1)))

    # resample onto columns
    cx = np.linspace(x0, x1, width)
    m = np.interp(cx, xs, mean)
    l = np.interp(cx, xs, lo)
    h = np.interp(cx, xs, hi)
    for c in range(width):
        for r in range(row(h[c]), row(l[c]) + 1):
            if 0 <= r < height:
                grid[r][c] = "."
    if truth is not None:
        t = np.interp(cx, xs, truth)
        for c in range(width):
            r = row(t[c])
            if 0 <= r < height:
                grid[r][c] = ":"
    for c in range(width):
        r = row(m[c])
        if 0 <= r < height:
            grid[r][c] = "-"
    if vline is not None and x0 <= vline <= x1:
        c = col(vline)
        for r in range(height):
            if grid[r][c] == " ":
                grid[r][c] = "|"
    if X_obs is not None:
        for x, y in zip(np.ravel(X_obs), np.ravel(y_obs)):
            if x0 <= x <= x1:
                r, c = row(y), col(x)
                if 0 <= r < height:
                    grid[r][c] = "o"

    out = []
    for i, line in enumerate(grid):
        yv = ymax - i * (ymax - ymin) / (height - 1)
        label = f"{yv:8.2f} |" if i % 4 == 0 else "         |"
        out.append(label + "".join(line))
    out.append("         +" + "-" * width)
    out.append(f"          {x0:<10.2f}{'':{width - 20}}{x1:>10.2f}")
    return "\n".join(out)


def convergence_plot(curves, labels, width=60, height=12, log=True):
    """curves: list of 1d arrays of best-so-far regret."""
    curves = [np.asarray(c, dtype=float) for c in curves]
    vals = np.concatenate(curves)
    if log:
        curves = [np.log10(np.maximum(c, 1e-6)) for c in curves]
        vals = np.concatenate(curves)
    ymin, ymax = vals.min(), vals.max()
    if ymax - ymin < 1e-9:
        ymax = ymin + 1
    n = max(len(c) for c in curves)
    grid = [[" "] * width for _ in range(height)]
    marks = "*o+x"
    for k, c in enumerate(curves):
        for col in range(width):
            i = min(int(col / (width - 1) * (n - 1)), len(c) - 1)
            r = int(round((ymax - c[i]) / (ymax - ymin) * (height - 1)))
            grid[r][col] = marks[k % len(marks)]
    out = []
    for i, line in enumerate(grid):
        yv = ymax - i * (ymax - ymin) / (height - 1)
        txt = f"1e{yv:+.1f}" if log else f"{yv:.2f}"
        label = f"{txt:>8} |" if i % 3 == 0 else "         |"
        out.append(label + "".join(line))
    out.append("         +" + "-" * width)
    out.append(f"          1{'':{width - 6}}{n:>5}  evals")
    out.append("          " + "   ".join(f"{marks[k]} {lab}" for k, lab in enumerate(labels)))
    return "\n".join(out)
