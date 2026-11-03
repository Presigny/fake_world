"""
sp_helpers.py
=============
Soneira-Peebles (SP) model and 2PCF helpers used by the calibration pipeline.

Calibration overview (see `calibration.py` and notebook 00)
-----------------------------------------------------------
1. Power-law fit of the data 2PCF, xi(r) = (r/r0)^(-gamma).
2. Branching factor eta from a k-means silhouette scan.
3. m-scan on the data sorted by a node attribute:
       m* = argmin_m  integral |xi_AB(r)| dr
4. Complete SP model: smallest L such that at least m* points fall inside
   the territory, with lambda = eta^(1/(2-gamma)).
5. Uniform background points to match the 2PCF amplitude:
       rho    = <xi_data / xi_SP>_r
       n_rand = m_SP * (1/sqrt(rho) - 1)
6. Uniform random removal of nodes from the mixture to reach exactly N.

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


def build_sp_model(m_star, eta, gamma, R_ref, gdf_edge, crs):
    """
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


# ===========================================================================
# 2PCF helpers
# ===========================================================================

def power_law(r, gamma, r0):
    """Power-law model xi(r) = (r / r0)^(-gamma)."""
    return (r / r0) ** (-gamma)


def fit_2pcf(r_edges, l_xi, a=1, b=19):
    """
    Fit a power law xi(r) = (r/r0)^(-gamma) to a 2PCF estimate, weighted by
    the spread of the realisations.

    Parameters
    ----------
    r_edges : ndarray
        Bin positions in metres.
    l_xi : list of ndarray
        Individual 2PCF realisations (from PCF_with_variance or bootstrap).
    a, b : int
        Bin range [a, b) used for the fit (avoids r = 0 and the noisy
        large-r bins).

    Returns
    -------
    gamma, r0, gamma_err, r0_err : floats, or four None on failure.
    """
    xi    = np.mean(l_xi, axis=0)
    error = np.std(l_xi,  axis=0)

    # Align lengths (bincount can return arrays of different lengths).
    n = min(len(xi), len(r_edges))
    xi = xi[:n]; error = error[:n]; r_edges = r_edges[:n]
    b = min(b, n - 1)

    # Drop the NaN / Inf produced when RR = 0 in the Landy-Szalay estimator.
    valid = np.isfinite(xi[a:b]) & np.isfinite(r_edges[a:b])
    if valid.sum() < 3:
        print("Power-law fit failed: fewer than 3 finite points.")
        return None, None, None, None

    r_fit  = r_edges[a:b][valid]
    xi_fit = xi[a:b][valid]
    sigma  = np.maximum(error[a:b][valid],
                        0.01 * np.mean(np.abs(xi_fit)) + 1e-9)
    try:
        popt, pcov = curve_fit(
            power_law, r_fit, xi_fit,
            sigma=sigma, absolute_sigma=False,
            p0=[1.0, 1e4],
            bounds=([0.05, 1.0], [1.95, 1e8]),
            maxfev=5000)
        errs = np.sqrt(np.diag(pcov))
        return popt[0], popt[1], errs[0], errs[1]
    except RuntimeError:
        print("Power-law fit failed (curve_fit did not converge).")
        return None, None, None, None


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
              bootstrap_size, bootstrap_N_real, rmin=None):
    """
    Compute the 2PCF, switching to bootstrap sub-sampling for large datasets.

    Uses PCF_with_variance when len(gdf) < bootstrap_size, and
    compute_two_point_correlation_bootstrap otherwise.

    Returns
    -------
    r_edges : ndarray
    l_xi    : list of ndarray, one per realisation
    """
    if len(gdf) < bootstrap_size:
        kwargs = dict(nbins=nbins)
        if rmin is not None:
            kwargs["rmin"] = rmin
        return mtpc.PCF_with_variance(
            gdf, gdf_edge, crs, N_run, size, k, rmax, scale, **kwargs)
    return mtpc.compute_two_point_correlation_bootstrap(
        gdf, gdf_edge, crs, N_run, size,
        rmin if rmin is not None else False,
        rmax, scale, nbins, bootstrap_size, bootstrap_N_real)


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
# three assumptions. The domain is a scipy.spatial.ConvexHull (the role of
# gdf_edge in 2D), so that they work in 2D as well as in 3D.

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


def soneira_peebles_nd(eta, lamb, L, R, center, hull=None, rng=None,
                       radial="uniform", max_tries=64, return_levels=False):
    """Soneira-Peebles fractal in arbitrary dimension.

    Each point of level i spawns eta children in the ball of radius
    R/lamb**i around it. After L levels the eta**L leaves are returned.

    The domain is handled child by child: a child falling outside the hull
    is redrawn (up to max_tries times), instead of redrawing the whole tree
    like check_inside. Children still outside after max_tries attempts are
    dropped, so the fractal can have slightly fewer than eta**L leaves
    (notebook 05bis quantifies it).

    Input:
        eta: number of children per point
        lamb: scale ratio between two levels (see lambda_of_gamma)
        L: number of levels
        R: radius of level 0
        center: seed, shape (dim,) or (1, dim)
        hull: scipy.spatial.ConvexHull bounding the domain (None = unbounded)
        rng: np.random.Generator
        radial: "uniform" (Soneira-Peebles) or "linear" (historical 2D code)
        return_levels: also return, level by level, the pair
            (positions, index of the parent in the previous level)
    Output:
        position: array (n_leaves, dim), with n_leaves <= eta**L
    """
    rng = np.random.default_rng() if rng is None else rng
    pos = np.atleast_2d(np.asarray(center, dtype=float))
    dim = pos.shape[1]
    levels = []
    for i in range(L):
        radius = R / lamb ** i
        parent_of = np.repeat(np.arange(len(pos)), eta)
        parents = pos[parent_of]
        child = parents + radius * sample_in_ball(len(parents), dim, rng, radial)
        if hull is not None:
            bad = ~inside_hull(child, hull)
            for _ in range(max_tries):
                if not bad.any():
                    break
                child[bad] = (parents[bad]
                              + radius * sample_in_ball(int(bad.sum()), dim,
                                                        rng, radial))
                bad = ~inside_hull(child, hull)
            child, parent_of = child[~bad], parent_of[~bad]
        pos = child
        levels.append((pos, parent_of))
    return (pos, levels) if return_levels else pos
