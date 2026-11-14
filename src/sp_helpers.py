"""
sp_helpers.py
=============
Soneira-Peebles (SP) model and 2PCF helpers used by the calibration pipeline.

Calibration overview (see `calibration.py`)
------------------------------------------
1. Power-law fit of the data 2PCF, xi(r) = (r/r0)^(-gamma).
2. Branching factor eta from a k-means silhouette scan.
3. m-scan on the data sorted by a node attribute:
       m* = argmin_m  integral |xi_AB(r)| dr
4. Complete SP model: smallest L such that eta^L >= m*, generated with the
   N-dimensional generator `soneira_peebles_nd` (dim = 2 for maps) with
   lambda = eta^(1/(dim-gamma)). A child falling outside the territory is
   redrawn (child-by-child rejection), so every leaf is inside.
5. Uniform background points to match the 2PCF amplitude. The model says
   xi_data ~ a xi_SP, so log a is estimated as the mean of
   log xi_data - log xi_SP over the bins of the power-law fit window
   (`log_amplitude_ratio`), then
       n_rand = m_SP * (a^(-1/2) - 1)
6. Uniform random removal of nodes from the mixture to reach exactly N.

Power-law fit window (`fit_window`): bins [a, b) of the 20 logarithmic
bins, minus the small-r bins that hold fewer than MIN_PAIRS data pairs
(DD). The same window is used for gamma and for log a.

The functions called inside joblib workers re-import their dependencies, so
that they are self-contained in sub-processes.
"""

from pathlib import Path
import sys

# Make src/ importable from anywhere, including joblib worker sub-processes.
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import geopandas as gpd
from scipy.optimize import curve_fit
import methods_two_point_correlation as mtpc


# ===========================================================================
# Soneira-Peebles model (2D, inside a territory)
# ===========================================================================

def soneira_peebles_border(lamb, eta, L, R, gdf_edge, crs):
    """
    Generate a Soneira-Peebles fractal point set.

    Starting from the centroid of gdf_edge, each point spawns eta children
    placed at a random radius r in [0, R/lambda^i] and angle theta in
    [0, 2 pi] at level i. After L levels the cloud contains eta^L points.

    Parameters
    ----------
    lamb : float
        Scale ratio lambda = eta^(1/(2-gamma)). Controls how quickly the
        cluster radius shrinks from one level to the next.
    eta : int
        Branching factor: each node spawns eta children.
    L : int
        Number of levels (depth of the fractal tree).
    R : float
        Radius of the level-0 disc, in metres.
    gdf_edge : GeoDataFrame
        Territory boundary used for the seed centroid and the inside test.
    crs : str
        Projected CRS (metres).

    Returns
    -------
    position : ndarray, shape (eta^L, 2)
        All generated point coordinates.
    point_outside : GeoDataFrame
        Subset of `position` that falls outside the territory.
    """
    position = gdf_edge.centroid.get_coordinates().to_numpy()
    for i in range(L):
        l_pos = []
        for x, y in position:
            r     = (R / (lamb ** i)) * np.random.uniform(0, 1, size=eta)
            theta = np.random.uniform(0, 2 * np.pi, size=eta)
            for k in range(eta):
                l_pos.append((x - r[k] * np.cos(theta[k]),
                              y - r[k] * np.sin(theta[k])))
        position = np.array(l_pos)

    s    = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(position[:, 0], position[:, 1]), crs=crs)
    join = gpd.sjoin(s, gdf_edge, how="left", predicate="within")
    join["is_inside"] = join["index_right"].notna()
    return position, join[~join["is_inside"]]


def get_inside_points(position, point_outside):
    """
    Return only the points that fall inside the territory.

    Parameters
    ----------
    position : ndarray, shape (n, 2)
    point_outside : GeoDataFrame
        Rows whose index matches the rows of `position` that are outside.

    Returns
    -------
    ndarray, shape (n_inside, 2)
    """
    mask = np.ones(len(position), dtype=bool)
    mask[point_outside.index.to_numpy()] = False
    return position[mask]


def build_sp_model_truncated(m_star, eta, gamma, R_ref, gdf_edge, crs):
    """
    Historical 2D builder (truncation at the border), kept for reference.
    The calibration now uses `build_sp_model` (child-by-child rejection).

    Build a complete Soneira-Peebles model with at least m_star nodes inside
    the territory.

    The number of levels L starts at the smallest value with eta^L >= m_star
    and is incremented until enough points fall inside the territory. Points
    outside the territory are discarded, and the full fractal is returned.

    Parameters
    ----------
    m_star : int
        Minimum number of nodes inside the territory.
    eta : int
        Branching factor.
    gamma : float
        Power-law exponent of the data 2PCF. Sets lambda = eta^(1/(2-gamma)).
    R_ref : float
        Radius of the level-0 disc, in metres.
    gdf_edge : GeoDataFrame
        Territory boundary.
    crs : str
        Projected CRS.

    Returns
    -------
    pos_sp : ndarray, shape (m_sp, 2), with m_sp >= m_star
    L      : int, number of levels used
    lamb   : float, scale ratio lambda
    """
    lamb = eta ** (1.0 / (2.0 - gamma))
    L = max(int(np.ceil(np.log(max(m_star, eta)) / np.log(eta))), 4)

    for _attempt in range(10):
        pos_all, pt_out = soneira_peebles_border(lamb, eta, L, R_ref, gdf_edge, crs)
        pos_in = get_inside_points(pos_all, pt_out)
        if len(pos_in) >= m_star:
            return pos_in, L, lamb
        L += 1
        print(f"  L={L - 1}: {len(pos_in):,} inside < {m_star:,}, trying L={L}")

    print(f"  Warning: could only generate {len(pos_in):,} nodes (wanted {m_star:,})")
    return pos_in, L, lamb


def build_sp_model(m_star, eta, gamma, R_ref, domain, crs=None, dim=2,
                   center=None, rng=None, radial="uniform", max_tries=64,
                   verbose=True):
    """
    Build a complete Soneira-Peebles model with at least m_star nodes inside
    the domain, with the N-dimensional generator (`soneira_peebles_nd`).

    Why this generator even in 2D (discussion with Charles, Nov. 2026):
    the border is handled child by child. A child drawn outside the domain
    is redrawn around the same parent, so the tree keeps its eta^L leaves
    (up to the rare children still outside after max_tries draws), the
    generation converges whatever the size of the tree, and the tree is not
    biased towards the centre of the domain. The two historical variants
    either redraw the whole tree while one point is outside
    (check_inside, cost 1/p with p = 2-16 % of accepted trees) or truncate
    the tree at the border (`build_sp_model_truncated`, which loses the
    outer branches and needs more levels than eta^L >= m*).

    The number of levels is the smallest L with eta^L >= m_star. If, after
    the rare drops, fewer than m_star leaves remain, L is incremented.

    Parameters
    ----------
    m_star : int
        Minimum number of nodes.
    eta : int
        Branching factor.
    gamma : float
        Target exponent of the 2PCF. Sets lambda = eta^(1/(dim-gamma)).
    R_ref : float
        Radius of the level-0 ball, in the units of the domain (metres).
    domain : GeoDataFrame / shapely geometry (2D), scipy ConvexHull, or
        callable P -> bool mask. See `inside_domain`.
    crs : str, unused (kept for the historical signature).
    dim : int
        Dimension of the space (2 for maps, specified explicitly).
    center : array (dim,), seed of the tree. Default: centroid of the
        territory (2D) or mean of the hull vertices.
    rng : np.random.Generator (default: fresh generator).
    radial : "uniform" (SP definition, r = R u^(1/dim)) or "linear"
        (r = R u, historical 2D code). See `sample_in_ball`.
    max_tries : int, redraws of a child before it is dropped.

    Returns
    -------
    pos_sp : ndarray, shape (m_sp, dim), with m_sp >= m_star
    L      : int, number of levels used
    lamb   : float, scale ratio lambda
    """
    rng = np.random.default_rng() if rng is None else rng
    lamb = lambda_of_gamma(eta, gamma, dim)
    L = max(int(np.ceil(np.log(max(m_star, eta)) / np.log(eta))), 1)
    if center is None:
        center = domain_center(domain, dim)
    center = np.asarray(center, dtype=float).reshape(-1)
    if center.shape[0] != dim:
        raise ValueError(f"center has {center.shape[0]} coordinates, dim={dim}")

    for _attempt in range(10):
        pos = soneira_peebles_nd(eta, lamb, L, R_ref, center, domain=domain,
                                 rng=rng, radial=radial, max_tries=max_tries)
        if len(pos) >= m_star:
            return pos, L, lamb
        if verbose:
            print(f"  L={L}: {len(pos):,} nodes < {m_star:,}, trying L={L + 1}")
        L += 1

    print(f"  Warning: could only generate {len(pos):,} nodes (wanted {m_star:,})")
    return pos, L, lamb


def domain_center(domain, dim=2):
    """Default seed of the tree: centroid of the territory (GeoDataFrame,
    GeoSeries or shapely geometry) or mean of the hull vertices."""
    if hasattr(domain, "geometry"):
        domain = domain.geometry.union_all()
    if hasattr(domain, "centroid"):
        c = domain.centroid
        return np.array([c.x, c.y], dtype=float)
    if hasattr(domain, "points"):
        return domain.points[domain.vertices].mean(axis=0)
    raise ValueError("center must be given for a callable domain")


# ===========================================================================
# 2PCF helpers
# ===========================================================================

def power_law(r, gamma, r0):
    """Power-law model xi(r) = (r / r0)^(-gamma)."""
    return (r / r0) ** (-gamma)


MIN_PAIRS = 100   # minimum number of data pairs (DD) for a bin to enter the fit
FIT_RANGE = (1, 19)


def fit_window(r_edges, xi, dd=None, a=FIT_RANGE[0], b=FIT_RANGE[1],
               min_pairs=MIN_PAIRS):
    """
    Boolean mask of the bins used for the power-law fit (and for log a).

    Three rules, applied in this order:
    1. index range [a, b): bin 0 (d <= mean nearest-neighbour distance) and
       the last bins (near rmax, where the random catalogue is cut by the
       border) are left out;
    2. minimum-pairs criterion: the small-r bins are dropped, starting from
       the first one, as long as they hold fewer than `min_pairs` data pairs
       (DD). With few pairs, xi is dominated by the Poisson noise of DD and
       the Landy-Szalay estimator can even be negative; keeping these bins
       biases gamma. The first bin with DD >= min_pairs opens the window and
       every bin above it (up to b) stays, so the criterion only moves the
       lower bound of the window. When `dd` is None the rule is skipped;
    3. the bins where xi is NaN or infinite (RR = 0) are dropped.

    Parameters
    ----------
    r_edges : ndarray, bin positions.
    xi : ndarray, mean 2PCF (one value per bin).
    dd : ndarray or None, number of data pairs per bin (counts["DD"] of
        PCF_with_variance(..., return_counts=True)).
    a, b : int, index range [a, b).
    min_pairs : int, threshold of rule 2.

    Returns
    -------
    mask : ndarray of bool, length min(len(r_edges), len(xi)).
    info : dict with the lower bin `first`, the bins dropped by the
        criterion `dropped_min_pairs`, and the window bounds in metres
        `r_lo`, `r_hi`.
    """
    n = min(len(r_edges), len(xi))
    xi = np.asarray(xi)[:n]; r_edges = np.asarray(r_edges)[:n]
    b = min(b, n - 1)
    idx = np.arange(n)
    mask = (idx >= a) & (idx < b)

    dropped = []
    if dd is not None:
        dd = np.asarray(dd)[:n]
        ok = np.where((idx >= a) & (dd >= min_pairs))[0]
        first = int(ok[0]) if len(ok) else b
        dropped = [int(i) for i in idx[(idx >= a) & (idx < first)]]
        mask &= idx >= first
    mask &= np.isfinite(xi) & np.isfinite(r_edges)

    used = np.where(mask)[0]
    info = dict(first=int(used[0]) if len(used) else None,
                last=int(used[-1]) if len(used) else None,
                dropped_min_pairs=dropped, min_pairs=min_pairs,
                r_lo=float(r_edges[used[0]]) if len(used) else np.nan,
                r_hi=float(r_edges[used[-1]]) if len(used) else np.nan,
                n_bins=int(mask.sum()))
    return mask, info


def fit_2pcf(r_edges, l_xi, a=FIT_RANGE[0], b=FIT_RANGE[1], dd=None,
             min_pairs=MIN_PAIRS, return_window=False):
    """
    Fit a power law xi(r) = (r/r0)^(-gamma) to a 2PCF estimate, weighted by
    the spread of the realisations (CP convention: curve_fit with
    sigma = max(std over the k realisations, 1 % of mean |xi|)).

    Parameters
    ----------
    r_edges : ndarray
        Bin positions in metres.
    l_xi : list of ndarray
        Individual 2PCF realisations (from PCF_with_variance or bootstrap).
    a, b : int
        Bin range [a, b) used for the fit (avoids r = 0 and the noisy
        large-r bins).
    dd : ndarray or None
        Data pairs per bin; enables the minimum-pairs criterion of
        `fit_window` (small-r bins with DD < min_pairs are left out).
    min_pairs : int
    return_window : bool
        Also return the mask and the info dict of `fit_window`.

    Returns
    -------
    gamma, r0, gamma_err, r0_err : floats, or four None on failure.
    (mask, info) appended when return_window is True.
    """
    xi    = np.mean(l_xi, axis=0)
    error = np.std(l_xi,  axis=0)

    # Align lengths (bincount can return arrays of different lengths).
    n = min(len(xi), len(r_edges))
    xi = xi[:n]; error = error[:n]; r_edges = r_edges[:n]

    mask, info = fit_window(r_edges, xi, dd=dd, a=a, b=b, min_pairs=min_pairs)
    fail = (None, None, None, None)
    if mask.sum() < 3:
        print("Power-law fit failed: fewer than 3 usable bins.")
        return fail + ((mask, info),) if return_window else fail

    r_fit  = r_edges[mask]
    xi_fit = xi[mask]
    sigma  = np.maximum(error[mask], 0.01 * np.mean(np.abs(xi_fit)) + 1e-9)
    try:
        popt, pcov = curve_fit(
            power_law, r_fit, xi_fit,
            sigma=sigma, absolute_sigma=False,
            p0=[1.0, 1e4],
            bounds=([0.05, 1.0], [1.95, 1e8]),
            maxfev=5000)
        errs = np.sqrt(np.diag(pcov))
        res = (popt[0], popt[1], errs[0], errs[1])
    except RuntimeError:
        print("Power-law fit failed (curve_fit did not converge).")
        res = fail
    return res + ((mask, info),) if return_window else res


def log_amplitude_ratio(xi_data, xi_sp, mask=None):
    """
    Amplitude ratio a of the data 2PCF to the SP 2PCF, estimated in log.

    The mixture model says xi_data(r) ~ a xi_SP(r) on the fit window, so
    log xi_data - log xi_SP is constant and equal to log a:
        log a = < log xi_data - log xi_SP >_{window, xi_data>0, xi_SP>0}.
    Why a log estimate rather than a ratio (discussion with Charles):
    symmetric (a factor 2 too high and a factor 2 too low weigh the same,
    which a ratio does not do, hence the gap between the former estimators
    alpha = 1/<|xi_SP/xi_data|> and rho = <xi_data/xi_SP>), robust (a bin
    where one xi is almost zero no longer blows up the mean), and
    consistent (gamma and a are estimated on the same bins).

    Parameters
    ----------
    xi_data, xi_sp : ndarray, 2PCF on the same bins (use `_align` first
        when the SP grid differs from the data grid).
    mask : ndarray of bool or None, fit window (`fit_window`). None = all
        bins.

    Returns
    -------
    a : float (exp of log a), np.nan when no bin is usable.
    log_a : float
    n_used : int, number of bins entering the mean.
    """
    n = min(len(xi_data), len(xi_sp))
    xd = np.asarray(xi_data, dtype=float)[:n]
    xs = np.asarray(xi_sp, dtype=float)[:n]
    m = np.ones(n, dtype=bool) if mask is None else np.asarray(mask, bool)[:n]
    m &= np.isfinite(xd) & np.isfinite(xs) & (xd > 0) & (xs > 0)
    if m.sum() == 0:
        return np.nan, np.nan, 0
    log_a = float(np.mean(np.log(xd[m]) - np.log(xs[m])))
    return float(np.exp(log_a)), log_a, int(m.sum())


def dilution_from_a(a, m_sp):
    """
    Number of uniform random points to add to m_sp SP nodes so that the
    mixture has the amplitude of the data.

    In a mixture of m_SP clustered nodes and n_rand uniform ones, the pairs
    of the clustered part are a fraction f^2 of all pairs with
    f = m_SP/(m_SP+n_rand), so xi_mix ~ f^2 xi_SP = a xi_SP gives
        n_rand = m_SP * (a^(-1/2) - 1).
    a >= 1 (data at least as clustered as the SP tree) gives n_rand = 0.
    """
    if not np.isfinite(a) or a >= 1.0:
        return 0
    return int(round(m_sp * (a ** -0.5 - 1.0)))


def _xi_mean(r_e, l_xi_loc):
    """
    Mean 2PCF across realisations, cut to the length of r_e.

    NaN and +-Inf (bins with RR = 0) are replaced by 0.
    """
    xi = np.mean(l_xi_loc, axis=0)
    n  = min(len(xi), len(r_e))
    xi = xi[:n]
    return np.nan_to_num(xi, nan=0.0, posinf=0.0, neginf=0.0)


def _align(r_src, xi_src, r_edges_ref):
    """
    Interpolate xi_src (defined on r_src) onto the reference grid
    r_edges_ref, linearly in log(r).

    Bins outside the range of r_src are set to 0 (no extrapolation).
    """
    n = min(len(r_src), len(xi_src))
    r_src = r_src[:n]; xi_src = xi_src[:n]
    return np.nan_to_num(
        np.interp(
            np.log10(np.clip(r_edges_ref, 1e-9, None)),
            np.log10(np.clip(r_src,       1e-9, None)),
            xi_src,
            left=np.nan, right=np.nan),
        nan=0.0)


def _pcf_safe(gdf, gdf_edge, crs, N_run, size, k, rmax, scale, nbins,
              bootstrap_size, bootstrap_N_real, rmin=None, return_counts=False):
    """
    Compute the 2PCF, switching to bootstrap sub-sampling for large datasets.

    Uses PCF_with_variance when len(gdf) < bootstrap_size, and
    compute_two_point_correlation_bootstrap otherwise.

    Returns
    -------
    r_edges : ndarray
    l_xi    : list of ndarray, one per realisation
    counts  : dict (DD, DR, RR, ...) when return_counts is True
    """
    if len(gdf) < bootstrap_size:
        kwargs = dict(nbins=nbins, return_counts=return_counts)
        if rmin is not None:
            kwargs["rmin"] = rmin
        return mtpc.PCF_with_variance(
            gdf, gdf_edge, crs, N_run, size, k, rmax, scale, **kwargs)
    return mtpc.compute_two_point_correlation_bootstrap(
        gdf, gdf_edge, crs, N_run, size,
        rmin if rmin is not None else False,
        rmax, scale, nbins, bootstrap_size, bootstrap_N_real,
        return_counts=return_counts)


# ===========================================================================
# Territory sampling
# ===========================================================================

def _random_in_territory(n, rng, gdf_edge, crs, terr_geom=None, terr_bounds=None):
    """
    Draw n uniform random points inside the territory by rejection sampling.

    Parameters
    ----------
    n           : int
    rng         : numpy Generator (np.random.default_rng)
    gdf_edge    : GeoDataFrame, territory boundary
    crs         : str
    terr_geom   : shapely geometry (optional, computed if None)
    terr_bounds : tuple (minx, miny, maxx, maxy) (optional, computed if None)

    Returns
    -------
    ndarray, shape (n, 2)
    """
    if n <= 0:
        return np.zeros((0, 2))
    if terr_geom is None:
        terr_geom   = gdf_edge.geometry.union_all()
    if terr_bounds is None:
        terr_bounds = terr_geom.bounds

    minx, miny, maxx, maxy = terr_bounds
    result, total = [], 0
    while total < n:
        batch = max(4 * (n - total), 200)
        xs = rng.uniform(minx, maxx, batch)
        ys = rng.uniform(miny, maxy, batch)
        gdf_cand = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(xs, ys), crs=crs)
        joined = gpd.sjoin(gdf_cand, gdf_edge[["geometry"]],
                           how="inner", predicate="within")
        idx = joined.index.unique().to_numpy()
        if len(idx) > 0:
            result.append(np.column_stack([xs[idx], ys[idx]]))
            total += len(idx)
    return np.vstack(result)[:n]


# ===========================================================================
# m-scan worker
# ===========================================================================

def _run_cross_pcf(m, xy_data_sorted, xi_data, r_edges, gdf_edge, crs,
                   N_run, size_r, k, rmax, scale, nbins,
                   bootstrap_size=10_000, bootstrap_N_real=5):
    """
    Integral of the cross-2PCF for a given split m of the sorted data.

    The data is sorted by decreasing node attribute before calling this
    function. Sub-population A contains the m first nodes and B the N - m
    remaining ones. The 2PCF of their union (the full dataset) is passed as
    xi_data, so that it is not recomputed for every m.

    Cross-2PCF (Landy-Szalay decomposition):

        xi_AB(r) = [N^2 xi_data(r) - m^2 xi_AA(r) - (N-m)^2 xi_BB(r)] / (2 m (N-m))

    Designed to be called inside a joblib Parallel loop.

    Parameters
    ----------
    m                : int, split index
    xy_data_sorted   : ndarray (N, 2), data sorted by decreasing attribute
    xi_data          : ndarray, 2PCF of the full dataset
    r_edges          : ndarray, bins of the data 2PCF
    gdf_edge         : GeoDataFrame, territory boundary
    crs              : str
    N_run, size_r, k : 2PCF parameters (see PCF_with_variance)
    rmax, scale, nbins : 2PCF range and binning
    bootstrap_size   : int, size above which bootstrap sub-sampling is used
    bootstrap_N_real : int, number of bootstrap realisations

    Returns
    -------
    dict with keys 'm', 'r_edges', 'xi', 'integral', or None on failure.
    """
    # Re-import inside the worker for joblib sub-process compatibility.
    import numpy as _np
    import geopandas as _gpd
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
    import methods_two_point_correlation as _mtpc

    N_pool = len(xy_data_sorted)
    N_A, N_B = m, N_pool - m
    if N_B <= 0:
        return None

    n_ref = len(xi_data)

    def _xi(xy_sub):
        """Mean 2PCF of a sub-population, cut to n_ref values."""
        gdf_sub = _gpd.GeoDataFrame(
            geometry=_gpd.points_from_xy(xy_sub[:, 0], xy_sub[:, 1]), crs=crs)
        try:
            with _np.errstate(divide='ignore', invalid='ignore'):
                if len(gdf_sub) >= bootstrap_size:
                    _, l_xi = _mtpc.compute_two_point_correlation_bootstrap(
                        gdf_sub, gdf_edge, crs, N_run, size_r, False,
                        rmax, scale, nbins, bootstrap_size, bootstrap_N_real)
                else:
                    _, l_xi = _mtpc.PCF_with_variance(
                        gdf_sub, gdf_edge, crs, N_run, size_r, k,
                        rmax, scale, nbins=nbins)
        except Exception:
            return _np.zeros(n_ref)
        xi = _np.mean(l_xi, axis=0)
        n  = min(len(xi), n_ref)
        return _np.nan_to_num(xi[:n], nan=0.0, posinf=0.0, neginf=0.0)

    xi_AA = _xi(xy_data_sorted[:m])
    xi_BB = _xi(xy_data_sorted[m:])

    n = min(n_ref, len(xi_AA), len(xi_BB))
    N_tot = N_A + N_B   # = N, the union is always the full dataset
    xi_cross = (
        N_tot**2 * xi_data[:n]
        - N_A**2  * xi_AA[:n]
        - N_B**2  * xi_BB[:n]
    ) / (2.0 * N_A * N_B)
    xi_cross = _np.nan_to_num(xi_cross, nan=0.0, posinf=0.0, neginf=0.0)

    r_e      = r_edges[:n]
    integral = float(_np.trapezoid(_np.abs(xi_cross[1:]), r_e[1:]))
    if not _np.isfinite(integral):
        return None
    return {"m": int(m), "r_edges": r_e, "xi": xi_cross, "integral": integral}


# ===========================================================================
# Soneira-Peebles in arbitrary dimension (notebook 05bis, fruit fly, 3D)
# ===========================================================================
# The Soneira & Peebles (1978) model was introduced in 3D, for galaxies, and
# the version above is its adaptation to maps. Three of its ingredients only
# hold in the plane:
#   - lamb = eta**(1/(2-gamma)): the 2 is the dimension of the space;
#   - the children are drawn in a disc (r, theta);
#   - check_inside (main branch) redraws the whole tree as long as one point
#     falls outside the domain, which never converges at eta**L ~ 2e5 points.
# The functions below keep the same definition of the model without these
# three assumptions. The domain is a scipy.spatial.ConvexHull, a shapely
# geometry / GeoDataFrame (2D maps) or any callable P -> bool mask, so that
# they work in 2D as well as in 3D. Since the discussion with Charles
# (Nov. 2026) this generator is the one used for the maps too, with dim = 2
# (`build_sp_model`).

def lambda_of_gamma(eta, gamma, dim):
    """Scale ratio lambda giving the target exponent gamma in dimension dim.

    The fractal has dimension D = ln(eta)/ln(lambda) and its 2PCF follows
    xi ~ r^(-gamma) with gamma = dim - D, so lambda = eta**(1/(dim-gamma)).
    In 2D this is the eta**(1/(2-gamma)) of notebooks 01-04."""
    return float(eta) ** (1.0 / (dim - gamma))


def sample_in_ball(size, dim, rng, radial="uniform"):
    """Points in the unit ball of dimension dim.

    radial="uniform": uniform in volume (r = u**(1/dim)), the definition of
    Soneira & Peebles. radial="linear": r = u, the law of the historical 2D
    code, which concentrates the children towards the centre (density in
    r**(1-dim)). It is kept to measure what this detail changes
    (see notebook 05bis)."""
    v = rng.normal(size=(size, dim))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    u = rng.random(size)
    r = u ** (1.0 / dim) if radial == "uniform" else u
    return v * r[:, None]


def inside_hull(P, hull, chunk=None):
    """Boolean mask "P is inside the convex hull" (A x + b <= 0 for every
    facet). Computed by chunks: the full matrix would hold len(P) x n_facets
    floats, i.e. 0.5 GB at 1.3e5 points and 439 facets."""
    A, b = hull.equations[:, :-1], hull.equations[:, -1]
    chunk = max(1, 2_000_000 // len(b)) if chunk is None else chunk
    out = np.empty(len(P), dtype=bool)
    for s in range(0, len(P), chunk):
        sl = slice(s, s + chunk)
        out[sl] = (P[sl] @ A.T + b <= 0).all(axis=1)
    return out


def inside_domain(P, domain):
    """Boolean mask "P is inside domain", for the three kinds of domain:
    - scipy.spatial.ConvexHull (any dimension): `inside_hull`;
    - shapely geometry, GeoSeries or GeoDataFrame (2D maps):
      shapely.contains_xy on the union of the geometries (the full
      multipolygon, islands included);
    - callable P -> bool mask (anything else, e.g. a raster mask).
    None means an unbounded domain."""
    return _domain_test(domain)(P)


def _domain_test(domain):
    """Turn a domain into a callable P -> bool mask (prepared once)."""
    if domain is None:
        return lambda P: np.ones(len(P), dtype=bool)
    if callable(domain):
        return domain
    if hasattr(domain, "equations"):
        return lambda P: inside_hull(P, domain)
    import shapely
    geom = domain.geometry.union_all() if hasattr(domain, "geometry") else domain
    shapely.prepare(geom)
    return lambda P: shapely.contains_xy(geom, P[:, 0], P[:, 1])


def soneira_peebles_nd(eta, lamb, L, R, center, domain=None, rng=None,
                       radial="uniform", max_tries=64, return_levels=False,
                       hull=None):
    """Soneira-Peebles fractal in arbitrary dimension.

    Each point of level i spawns eta children in the ball of radius
    R/lamb**i around it. After L levels the eta**L leaves are returned.

    The domain is handled child by child: a child falling outside the
    domain is redrawn around the same parent (up to max_tries times),
    instead of redrawing the whole tree like check_inside or truncating it
    like build_sp_model_truncated. Children still outside after max_tries
    attempts are dropped, so the fractal can have slightly fewer than
    eta**L leaves (notebook 05bis quantifies it).

    Input:
        eta: number of children per point
        lamb: scale ratio between two levels (see lambda_of_gamma)
        L: number of levels
        R: radius of level 0
        center: seed, shape (dim,) or (1, dim)
        domain: scipy.spatial.ConvexHull, shapely geometry / GeoDataFrame
            (2D) or callable P -> bool mask (None = unbounded), see
            `inside_domain`
        rng: np.random.Generator
        radial: "uniform" (Soneira-Peebles) or "linear" (historical 2D code)
        return_levels: also return, level by level, the pair
            (positions, index of the parent in the previous level)
        hull: alias of domain (historical name)
    Output:
        position: array (n_leaves, dim), with n_leaves <= eta**L
    """
    rng = np.random.default_rng() if rng is None else rng
    if domain is None:
        domain = hull
    inside = _domain_test(domain)
    pos = np.atleast_2d(np.asarray(center, dtype=float))
    dim = pos.shape[1]
    levels = []
    for i in range(L):
        radius = R / lamb ** i
        parent_of = np.repeat(np.arange(len(pos)), eta)
        parents = pos[parent_of]
        child = parents + radius * sample_in_ball(len(parents), dim, rng, radial)
        if domain is not None:
            bad = ~inside(child)
            for _ in range(max_tries):
                if not bad.any():
                    break
                child[bad] = (parents[bad]
                              + radius * sample_in_ball(int(bad.sum()), dim,
                                                        rng, radial))
                bad = ~inside(child)
            child, parent_of = child[~bad], parent_of[~bad]
        pos = child
        levels.append((pos, parent_of))
    return (pos, levels) if return_levels else pos
