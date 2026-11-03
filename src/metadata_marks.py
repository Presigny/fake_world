"""
metadata_marks.py
=================
Population marks (metadata) for the SP model and the observables used to
compare marked point patterns.

1. Marks in the generative model
   An SP mixture (fractal + uniform background, see
   synthesis_utils.build_sp_mixture) is generated, then each node receives a
   population from its local density (number of neighbours within a radius,
   or inverse squared distance to the k-th neighbour). This encodes "large
   cities are where nodes are dense" without using the SP tree, so it applies
   to the whole mixture, while population_from_level only applies to the
   fractal part. Two alternatives are provided: a power law of the density
   (pop_powerlaw) and rank matching with the empirical populations
   (pop_rank_matching), plus a split between SP and background nodes
   (pop_split_sp_background).

2. Observables that include the marks
   - cumulative population profile M(r): for each point, the sum of the
     populations of the points within a radius r, and its normalised version
     M_norm(r) = <M(r)> / (<pop> <N(r)>), which isolates the coupling between
     marks and positions;
   - "people 2PCF": each point is replicated a number of times proportional
     to its population and the 2PCF of the extended cloud is measured. It is
     implemented literally (superpose_people) and with a mark-weighted
     Landy-Szalay estimator that avoids the quadratic cost of the extended
     cloud (compute_2pcf_people).

All coordinates are assumed to be normalised to [0, 1]^2
(see synthesis_utils.normalize_coords).
"""

from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import synthesis_utils as su


# ===========================================================================
# 1. Local density and population assignment
# ===========================================================================

def local_density_knn(xy, k=10):
    """Local density from the distance to the k-th neighbour.

    rho_k(i) = k / (pi d_k(i)^2), where d_k(i) is the distance from point i
    to its k-th nearest neighbour. This classical estimator does not need an
    absolute radius.
    """
    d, _ = cKDTree(xy).query(xy, k=k + 1)          # the first neighbour is the point itself
    dk = d[:, -1]
    return k / (np.pi * np.maximum(dk, 1e-12) ** 2)


def local_density_radius(xy, h):
    """Local density from the number of neighbours within a radius h (self excluded)."""
    tree = cKDTree(xy)
    counts = np.asarray(tree.query_ball_point(xy, r=h, return_length=True)) - 1
    return counts / (np.pi * h ** 2)


def default_radius(xy, k=10):
    """Default counting radius: median k-NN distance of the cloud."""
    d, _ = cKDTree(xy).query(xy, k=k + 1)
    return float(np.median(d[:, -1]))


def pop_powerlaw(rho, beta, pop_total):
    """Power-law mapping, population proportional to rho^beta, summing to pop_total."""
    w = np.maximum(np.asarray(rho, dtype=float), 1e-12) ** beta
    return pop_total * w / w.sum()


def pop_rank_matching(rho, pop_empirical):
    """Rank matching between local density and empirical populations.

    The densest node receives the largest empirical population, and so on.
    The population marginal is therefore exactly the one of the data, and
    only the placement comes from the model. When the sizes differ, the
    empirical quantiles are interpolated.
    """
    rho = np.asarray(rho, dtype=float)
    p = np.sort(np.asarray(pop_empirical, dtype=float))[::-1]
    n = len(rho)
    if len(p) != n:
        q = np.linspace(0, 1, n)
        p = np.quantile(np.asarray(pop_empirical, dtype=float),
                        1 - q, method="linear")
    out = np.empty(n)
    out[np.argsort(rho)[::-1]] = p
    return out


def pop_split_sp_background(is_sp, pop_empirical, seed=0):
    """Population assignment from the generative process itself (SP versus background).

    The SP mixture already separates the nodes into structured nodes (from
    the SP fractal) and uniform background nodes. The |SP| largest empirical
    populations go to the SP nodes and the others to the background. Inside
    each group the assignment is uniformly random (no local density is used),
    so results should be averaged over several seeds.

    This tests the hypothesis that being a structured node is enough: the
    coupling can only come from SP nodes being more clustered than the
    background, not from a node-by-node link between density and size as in
    pop_powerlaw and pop_rank_matching.

    is_sp comes from build_sp_mixture(..., return_labels=True). When the
    sizes differ, the empirical quantiles are interpolated.
    Returns the population vector.
    """
    is_sp = np.asarray(is_sp, dtype=bool)
    n = len(is_sp)
    p = np.sort(np.asarray(pop_empirical, dtype=float))[::-1]
    if len(p) != n:
        q = np.linspace(0, 1, n)
        p = np.quantile(np.asarray(pop_empirical, dtype=float),
                        1 - q, method="linear")
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    n_sp = int(is_sp.sum())
    out[is_sp] = rng.permutation(p[:n_sp])          # the n_sp largest go to SP nodes
    out[~is_sp] = rng.permutation(p[n_sp:])         # the rest goes to the background
    return out


def calibrate_beta_on_zipf(rho, target_zipf, pop_total,
                           betas=np.linspace(0.25, 4.0, 16)):
    """Pick beta so that the rank-size exponent of rho^beta matches the data.

    Returns (beta*, zipf(beta*)).
    """
    best = None
    for b in betas:
        z = su.fit_zipf_exponent(pop_powerlaw(rho, b, pop_total))
        err = abs(z - target_zipf) if np.isfinite(z) else np.inf
        if best is None or err < best[1]:
            best = (float(b), err, z)
    return best[0], best[2]


def delta_gamma(xy, pop, geom_norm, frac_high=0.2):
    """Coupling between marks and structure, Delta gamma = gamma_hi - gamma_lo.

    Wraps synthesis_utils.marked_clustering_2pcf (split by rank). A positive
    value means that high-population nodes are more clustered than the
    others, as in the real data. Returns (full result dict, Delta gamma).
    """
    res = su.marked_clustering_2pcf(xy, pop, geom_norm, frac_high=frac_high)
    g_hi, g_lo = res["gamma_hi"], res["gamma_lo"]
    dg = (g_hi - g_lo) if (np.isfinite(g_hi) and np.isfinite(g_lo)) else np.nan
    return res, dg


# ===========================================================================
# 2a. Cumulative population profile M(r)
# ===========================================================================

def cumulative_population_profile(xy, pop, r_grid, include_self=False):
    """M_i(r) = sum_{j: d_ij <= r} pop_j for each point i, on the grid r_grid.

    Returns an array (n_points, n_r). Its mean over axis 0 is the mean
    profile M(r). The log-log slope of M(r) is a mass dimension, the
    population-weighted analogue of the correlation dimension D_2.
    Cost O(N^2 log N), for clouds of at most about 10,000 points.
    """
    d_sorted, order = _neighbour_sort(xy, include_self)
    return _profile_from_sort(d_sorted, order, np.asarray(pop, float), r_grid)


def _neighbour_sort(xy, include_self=False):
    """Sort the neighbours of each point by increasing distance.

    Returns (d_sorted, order). It only depends on the positions, so it can be
    reused when only the marks change (shuffled null), which avoids redoing
    the O(N^2) cdist at each draw.
    """
    xy = np.asarray(xy, dtype=float)
    d = cdist(xy, xy)
    if not include_self:
        np.fill_diagonal(d, np.inf)
    order = np.argsort(d, axis=1)
    return np.take_along_axis(d, order, axis=1), order


def _profile_from_sort(d_sorted, order, marks, r_grid):
    """Cumulative profile (n_points, n_r) from a precomputed neighbour sort.

    M_i(r) is a step function that jumps at each neighbour and is constant
    in between. It is evaluated exactly with a binary search. A linear
    interpolation between successive neighbours would anticipate the mark of
    the next neighbour, overestimate M(r) by up to about 30% at small r, and
    break the identity <M(r)> = <m><N(r)> + Cov(m_j, n_j(r)).
    """
    m_sorted = np.cumsum(marks[order], axis=1)
    out = np.empty((len(order), len(r_grid)))
    for i in range(len(order)):
        # number of neighbours at distance <= r (0 means no mark accumulated)
        idx = np.searchsorted(d_sorted[i], r_grid, side="right")
        out[i] = np.where(idx > 0, m_sorted[i][np.maximum(idx - 1, 0)], 0.0)
    return out


def mass_dimension(r_grid, M_mean, a=2, b=None):
    """Log-log slope D_M of M(r), fitted on bins a..b-1.

    D_M is dominated by the geometry of the points and hardly detects the
    coupling between marks and positions: shuffling the marks leaves it
    unchanged (within about 0.5 sigma), while changing the positions moves
    it clearly. Use normalized_population_profile for the coupling.
    """
    if b is None:
        b = len(r_grid) - 2
    valid = (M_mean[a:b] > 0) & (r_grid[a:b] > 0)
    if valid.sum() < 3:
        return np.nan
    return float(np.polyfit(np.log(r_grid[a:b][valid]),
                            np.log(M_mean[a:b][valid]), 1)[0])


def normalized_population_profile(xy, pop, r_grid, include_self=False):
    """M_norm(r) = <M(r)> / (<pop> <N(r)>), the coupling between marks and positions.

    <M(r)> = <pop><N(r)> + Cov(pop_j, n_j(r)). The first term only depends on
    the positions and the coupling enters through the covariance, so the
    division removes the geometry:
      - M_norm(r) = 1: the population within r is the one expected from the
        number of neighbours, no coupling (value of the shuffled null by
        construction);
      - M_norm(r) > 1: nodes are surrounded by more population than expected
        at scale r, large populations cluster together;
      - M_norm(r) < 1: large populations avoid each other.

    Contrary to the slope D_M, this observable separates coupled and
    uncoupled marks (for Belgium, peak of about 1.6 near r = 0.012, flat null
    at 1.00). Returns (M_norm, M_mean, N_mean).
    """
    pop = np.asarray(pop, dtype=float)
    d_sorted, order = _neighbour_sort(xy, include_self)
    M = _profile_from_sort(d_sorted, order, pop, r_grid).mean(axis=0)
    N = _profile_from_sort(d_sorted, order, np.ones(len(pop)), r_grid).mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        M_norm = np.where(N > 0, M / (pop.mean() * N), np.nan)
    return M_norm, M, N


def covariance_profile(M_mean, N_mean, pop):
    """Raw coupling term Cov(pop_j, n_j(r)), deduced from the existing profiles.

    It follows from <M(r)> = <pop><N(r)> + Cov(pop_j, n_j(r)) without any
    extra computation. M_norm - 1 is the same quantity rescaled by
    <pop><N(r)>, so both carry the same information with complementary
    drawbacks:
      - Cov goes smoothly to 0 at both ends (n_j = 0 everywhere when r -> 0,
        n_j = N - 1 everywhere when r -> infinity), but it has units
        (inhabitants x neighbours), so it is not comparable between
        datasets. It scales with the mean population <pop>, so two
        countries with different mean city sizes have different peaks for
        this reason alone.
      - M_norm has no units and reads directly ("+74% of population
        around"), like xi(r) = DD/RR - 1 or k_mm = <m_i m_j>/<m>^2, but it is
        0/0 at small r, where the division amplifies the noise.
    """
    return np.asarray(M_mean, float) - np.mean(pop) * np.asarray(N_mean, float)


def shuffled_null_profile(xy, pop, r_grid, n_shuffle=30, seed=0,
                          include_self=False):
    """Null model of M_norm(r): shuffled marks, averaged over n_shuffle draws.

    A single permutation is very noisy when the mark distribution is heavy
    tailed (Brussels holds about 15% of the Belgian population), hence the
    average. Positions are fixed, so the neighbour sort is computed once.
    Returns (norm_mean, norm_std, DM_mean, DM_std).
    """
    pop = np.asarray(pop, dtype=float)
    rng = np.random.default_rng(seed)
    d_sorted, order = _neighbour_sort(xy, include_self)
    N = _profile_from_sort(d_sorted, order, np.ones(len(pop)), r_grid).mean(axis=0)
    denom = pop.mean() * np.maximum(N, 1e-12)

    norms, dms = [], []
    for _ in range(n_shuffle):
        M = _profile_from_sort(d_sorted, order, rng.permutation(pop),
                               r_grid).mean(axis=0)
        norms.append(np.where(N > 0, M / denom, np.nan))
        dms.append(mass_dimension(r_grid, M, a=1, b=16))
    norms = np.array(norms)
    return (np.nanmean(norms, axis=0), np.nanstd(norms, axis=0),
            float(np.nanmean(dms)), float(np.nanstd(dms)))


# ===========================================================================
# 2b. People 2PCF
# ===========================================================================

def superpose_people(xy, pop, n_total=6000, jitter=0.0, seed=0):
    """Replace each point by n_i nodes, n_i proportional to pop_i (multinomial draw).

    Literal implementation of the idea: the returned cloud has n_total
    points. jitter > 0 adds Gaussian noise, only for plotting (keep 0 for
    the 2PCF).
    """
    rng = np.random.default_rng(seed)
    p = np.asarray(pop, dtype=float)
    p = p / p.sum()
    counts = rng.multinomial(n_total, p)
    reps = np.repeat(np.arange(len(xy)), counts)
    out = np.asarray(xy, dtype=float)[reps]
    if jitter > 0:
        out = out + rng.normal(0, jitter, out.shape)
    return out


def compute_2pcf_people(xy, pop, geom_norm, n_random=su.N_RANDOM,
                        n_runs=su.N_RUNS_2PCF, n_bins=su.N_BINS):
    """People 2PCF, a Landy-Szalay estimator weighted by the marks.

    In expectation, it is the 2PCF of the cloud where each point is replaced
    by n_i nodes with n_i proportional to pop_i, without the quadratic cost
    of that extended cloud:

        DD_w(r) = sum_{i<j, d_ij in bin} w_i w_j / sum_{i<j} w_i w_j
        DR_w(r) = sum_{i,k, d_ik in bin} w_i     / (sum_i w_i N_R)
        RR(r)   = unchanged (unmarked random catalogue)

        xi_people(r) = (DD_w - 2 DR_w + RR) / RR

    with w_i = pop_i / <pop>. Pairs of clones of the same point (d = 0) are
    excluded, as in the standard estimator. Returns (r_centers, xi).
    """
    xy = np.asarray(xy, dtype=float)
    w = np.asarray(pop, dtype=float)
    w = w / w.mean()

    iu = np.triu_indices(len(xy), k=1)
    d_pairs = cdist(xy, xy)[iu]
    w_pairs = w[iu[0]] * w[iu[1]]
    pos = d_pairs > 0
    d_pairs, w_pairs = d_pairs[pos], w_pairs[pos]

    # Same radial grid as compute_2pcf_norm (rmin = half the mean NN distance)
    nn_dist, _ = cKDTree(xy).query(xy, k=2)
    rmin = max(np.mean(nn_dist[:, 1]) * 0.5, 1e-5)
    rmax = np.max(d_pairs) * 0.55
    r_edges = np.logspace(np.log10(rmin), np.log10(rmax), n_bins + 1)
    r_centers = np.sqrt(r_edges[:-1] * r_edges[1:])

    DD_hist, _ = np.histogram(d_pairs, bins=r_edges, weights=w_pairs)
    DD_norm = DD_hist / max(w_pairs.sum(), 1e-12)

    rng_r = np.random.default_rng(99)
    DR_list, RR_list = [], []
    for _ in range(n_runs):
        rand_xy = su.rejection_sample_in_boundary(n_random, geom_norm, rng_r)
        dr = cdist(xy, rand_xy)
        w_dr = np.repeat(w, dr.shape[1])
        DR_hist, _ = np.histogram(dr.ravel(), bins=r_edges, weights=w_dr)
        DR_list.append(DR_hist / max(w.sum() * len(rand_xy), 1e-12))
        rr = cdist(rand_xy, rand_xy)
        rr = np.triu(rr, k=1).ravel()
        rr = rr[rr > 0]
        RR_hist, _ = np.histogram(rr, bins=r_edges)
        RR_list.append(RR_hist / max(len(rr), 1))

    DR_norm = np.mean(DR_list, axis=0)
    RR_norm = np.mean(RR_list, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = np.where(RR_norm > 0,
                      (DD_norm - 2 * DR_norm + RR_norm) / RR_norm, np.nan)
    return r_centers, xi


if __name__ == "__main__":
    # Smoke test on Belgium: local density and both observables.
    name = "Belgium Cities"
    ds = su.DATASETS[name]
    print(f"=== metadata_marks smoke test: {name} ===")
    ld = su.load_dataset(ds)
    xy, pop = ld["xy_norm"], ld["pop_data"]

    rho = local_density_knn(xy, k=10)
    zipf_data = su.fit_zipf_exponent(pop)
    beta, zipf_model = calibrate_beta_on_zipf(rho, zipf_data, pop.sum())
    print(f"Zipf data={zipf_data:.2f}  beta*={beta:.2f}  Zipf model={zipf_model:.2f}")

    r_grid = np.logspace(-2.5, -0.3, 12)
    M_norm, M, N = normalized_population_profile(xy, pop, r_grid)
    d_m = mass_dimension(r_grid, M)
    null_m, null_s, dm_null, dm_null_s = shuffled_null_profile(
        xy, pop, r_grid, n_shuffle=10)
    i_pk = int(np.nanargmax(M_norm[1:-1]) + 1)
    print(f"D_M = {d_m:.2f}  vs null {dm_null:.2f}+-{dm_null_s:.2f} "
          f"({abs(d_m - dm_null) / max(dm_null_s, 1e-9):.1f} sigma)")
    print(f"M_norm peak = {M_norm[i_pk]:.2f} at r={r_grid[i_pk]:.4f} "
          f"(shuffled null at {np.nanmean(null_m):.2f})")

    r_p, xi_p = compute_2pcf_people(xy, pop, ld["geom_norm"])
    g, _ = su.fit_powerlaw_norm(r_p, xi_p)
    print(f"gamma_people = {g:.3f}  (gamma_nodes = {ld['gamma_fit']:.3f})")
