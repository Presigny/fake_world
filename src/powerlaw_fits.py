"""
powerlaw_fits.py
================
Four ways to measure the exponent gamma of xi(r) = (r / r0)^(-gamma), all on the
same pair counts as the calibration 2PCF (notebook 11):

  fit_wls          weighted least squares on xi, the method of the paper
                   (same as sp_helpers.fit_2pcf)
  fit_ols_log      unweighted least squares on log xi against log r
  fit_poisson_mle  maximum likelihood on the data pair counts DD
  fit_powerlaw_pkg `powerlaw` package (Alstott et al. 2014) on the pair
                   distances, i.e. the Takens estimator of the correlation
                   dimension D2, with gamma = 2 - D2

The pair counts are computed with cKDTree, with the binning convention of
methods_two_point_correlation (np.digitize(right=True)) and the split random
catalogues of Keihanen et al. (2019).
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit, minimize
import warnings

# --------------------------------------------------------------------------
# Pair counts, same binning convention as methods_two_point_correlation
# (np.digitize(right=True): index i holds edges[i-1] < d <= edges[i]).
# --------------------------------------------------------------------------

def mean_nn(xy):
    """compute_rmin: mean distance to the nearest distinct neighbour."""
    d, _ = cKDTree(xy).query(xy, k=3)
    nn = d[:, 1]
    zero = nn == 0
    return np.mean(np.concatenate([nn, d[zero, 2]]))


def _cum(tA, tB, edges):
    return tA.count_neighbors(tB, edges).astype(np.float64)


def hist_auto(xy, edges):
    t = cKDTree(xy)
    c = (_cum(t, t, edges) - len(xy)) / 2.0           # unordered pairs, no self
    h = np.empty(len(edges) + 1)
    h[0] = c[0]; h[1:-1] = np.diff(c); h[-1] = np.nan
    return h


def hist_cross(xyA, xyB, edges):
    c = _cum(cKDTree(xyA), cKDTree(xyB), edges)
    h = np.empty(len(edges) + 1)
    h[0] = c[0]; h[1:-1] = np.diff(c); h[-1] = np.nan
    return h


def random_in(gdf_edge, size, rng):
    return gdf_edge.sample_points(size, rng=rng).get_coordinates().to_numpy()


def counts_one(xy, gdf_edge, edges, N_run, size, rng):
    """DD, DR, RR (summed over N_run split catalogues) for one repetition."""
    DD = hist_auto(xy, edges)
    DR = np.zeros(len(edges) + 1); RR = np.zeros(len(edges) + 1)
    for _ in range(N_run):
        R = random_in(gdf_edge, size, rng)
        DR += hist_cross(xy, R, edges)
        RR += hist_auto(R, edges)
    return dict(DD=DD, DR=DR, RR=RR, Nd=len(xy), Nrp=size, Nr=size * N_run,
                N_run=N_run)


def ls_xi(c):
    """Keihanen et al. (2019) split-random Landy-Szalay estimator."""
    Nd, Nrp, Nr = c["Nd"], c["Nrp"], c["Nr"]
    with np.errstate(divide="ignore", invalid="ignore"):
        return (Nr * (Nrp - 1) * c["DD"]) / (Nd * (Nd - 1) * c["RR"]) \
            - ((Nrp - 1) * c["DR"]) / (Nd * c["RR"]) + 1


def calib_counts(xy, gdf_edge, rmax, rng, nbins=20, N_run=5, size=7000, k=3,
                 boot_size=10_000, boot_n=10):
    """Repetitions of the calibration 2PCF (PCF_with_variance or bootstrap).

    Returns edges, list of counts, and same_data (True when the repetitions
    share the data points and only the random catalogues change).
    """
    if len(xy) < boot_size:
        edges = np.logspace(np.log10(mean_nn(xy)), np.log10(rmax), nbins)
        reps = [counts_one(xy, gdf_edge, edges, N_run, size, rng) for _ in range(k)]
        return edges, reps, True
    reps = []
    edges = None
    for _ in range(boot_n):
        sub = xy[rng.choice(len(xy), boot_size, replace=False)]
        # The original code takes the mean NN distance of each sub-sample. Here
        # the edges of the first sub-sample are kept, so that the counts of
        # all sub-samples share the same bins (the difference is small).
        if edges is None:
            edges = np.logspace(np.log10(mean_nn(sub)), np.log10(rmax), nbins)
        reps.append(counts_one(sub, gdf_edge, edges, N_run, size, rng))
    return edges, reps, False


# --------------------------------------------------------------------------
# Fits. Bins 1..18 (FIT_RANGE = (1, 19)), position = upper edge, as in
# sp_helpers.fit_2pcf.
# --------------------------------------------------------------------------
A, B = 1, 19


def pl(r, g, r0):
    return (r / r0) ** (-g)


def fit_wls(edges, xis):
    """Paper method: curve_fit on the mean xi, sigma = std over repetitions."""
    xi = np.mean(xis, axis=0); er = np.std(xis, axis=0)
    r = edges[A:B]; y = xi[A:B]; s = er[A:B]
    ok = np.isfinite(y)
    s = np.maximum(s[ok], 0.01 * np.mean(np.abs(y[ok])) + 1e-9)
    try:
        p, cov = curve_fit(pl, r[ok], y[ok], p0=[1.0, np.median(r)], sigma=s,
                           absolute_sigma=False, maxfev=20000)
        return p[0], np.sqrt(cov[0, 0]), p[1]
    except Exception:
        return np.nan, np.nan, np.nan


def fit_ols_log(edges, xis):
    xi = np.mean(xis, axis=0)
    r = edges[A:B]; y = xi[A:B]
    ok = np.isfinite(y) & (y > 0)
    if ok.sum() < 3:
        return np.nan, np.nan, np.nan
    sl, ic = np.polyfit(np.log(r[ok]), np.log(y[ok]), 1)
    return -sl, np.nan, np.exp(ic / -sl)


def _bin_mean_powerlaw(lo, hi, g):
    """<r^-g> over an annulus [lo, hi] with pair density ~ r (2D)."""
    return 2.0 / (2.0 - g) * (hi ** (2 - g) - lo ** (2 - g)) / (hi ** 2 - lo ** 2)


def fit_poisson_mle(edges, reps, same_data):
    """Maximum likelihood on the data pair counts.

    In bin i, DD_i ~ Poisson(mu_i), mu_i = P_i * Npairs * (1 + xi_i), where
    P_i is the random-pair fraction (geometry of the territory) and xi_i the
    power law averaged over the bin. The repetitions with independent data
    (sub-samples) add their log-likelihoods. With shared data, DD is counted
    once and P_i is the mean of the random catalogues. The pairs are not
    independent, so the Fisher error is a lower bound.
    """
    lo, hi = edges[A - 1:B - 1], edges[A:B]
    blocks = []
    if same_data:
        c0 = reps[0]
        P = np.mean([c["RR"][A:B] / (c["N_run"] * c["Nrp"] * (c["Nrp"] - 1) / 2)
                     for c in reps], axis=0)
        blocks.append((c0["DD"][A:B], P, c0["Nd"] * (c0["Nd"] - 1) / 2))
    else:
        for c in reps:
            P = c["RR"][A:B] / (c["N_run"] * c["Nrp"] * (c["Nrp"] - 1) / 2)
            blocks.append((c["DD"][A:B], P, c["Nd"] * (c["Nd"] - 1) / 2))

    def nll(th):
        g, lr0 = th
        if not (0.01 < g < 1.99):
            return 1e30
        x = np.exp(lr0 * g) * _bin_mean_powerlaw(lo, hi, g)   # r0^g <r^-g>
        out = 0.0
        for DD, P, Np in blocks:
            mu = P * Np * (1 + x)
            ok = (P > 0)
            out += np.sum(mu[ok] - DD[ok] * np.log(mu[ok]))
        return out

    g0, _, r00 = fit_wls(edges, [ls_xi(c) for c in reps])
    if not np.isfinite(g0):
        g0, r00 = 0.8, np.median(hi)
    best = None
    for gs in (g0, 0.5, 1.0):
        res = minimize(nll, [gs, np.log(r00)], method="Nelder-Mead",
                       options=dict(xatol=1e-6, fatol=1e-6, maxiter=4000))
        if best is None or res.fun < best.fun:
            best = res
    g, lr0 = best.x
    # Fisher error from a numerical Hessian
    h = np.array([1e-3, 1e-3]); H = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            e_i = np.eye(2)[i] * h[i]; e_j = np.eye(2)[j] * h[j]
            H[i, j] = (nll(best.x + e_i + e_j) - nll(best.x + e_i - e_j)
                       - nll(best.x - e_i + e_j) + nll(best.x - e_i - e_j)) / (4 * h[i] * h[j])
    try:
        err = np.sqrt(np.linalg.inv(H)[0, 0])
    except Exception:
        err = np.nan
    return g, err, np.exp(lr0)


def fit_powerlaw_pkg(xy_list, r_hi, compare=True, max_pairs=2_000_000, seed=0):
    """`powerlaw` package on the pair distances d <= r_hi (Takens estimator).

    If the correlation integral is C(d) ~ d^D2 below r_hi, then x = r_hi / d
    follows a power law of exponent alpha = D2 + 1 on [1, inf), and
    gamma = 2 - D2 = 3 - alpha. This holds where xi >> 1 and far from the
    border, since the package has no geometric correction.
    The package (v1.5) ignores xmax in the continuous normalisation, so the
    distances are not cut from below. Above max_pairs, a random subset of the
    pairs is used (the estimate is unbiased, its error grows slightly).
    Returns gamma, its standard error, the normalised log-likelihood ratio R
    and p-value of power law against lognormal (R > 0 favours the power law),
    and the number of pairs.
    """
    import powerlaw
    ds = []
    for xy in xy_list:
        t = cKDTree(xy)
        pr = t.query_pairs(r_hi, output_type="ndarray")
        d = np.linalg.norm(xy[pr[:, 0]] - xy[pr[:, 1]], axis=1)
        ds.append(d[d > 0])
    d = np.concatenate(ds)
    n_pairs = len(d)
    if n_pairs > max_pairs:
        d = np.random.default_rng(seed).choice(d, max_pairs, replace=False)
    x = r_hi / d
    R = p = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = powerlaw.Fit(x, xmin=1.0, verbose=False)
        a, s = f.power_law.alpha, f.power_law.sigma
        if compare:
            try:
                R, p = f.distribution_compare("power_law", "lognormal",
                                              normalized_ratio=True)
            except Exception:
                pass
    return 3 - a, s, R, p, n_pairs
