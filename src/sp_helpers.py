"""
sp_helpers.py
=============
Helper functions for the Single-SP Cross-2PCF calibration notebooks.

Pipeline overview
-----------------
1. m-scan on data sorted by metadata → m* = argmin_m ∫|ξ_cross(r)|dr
2. Build a *complete* SP model: minimum L such that inside-territory count ≥ m*
3. Verify the slope γ of the SP model (should match data, it's a model parameter)
4. Add random background points to match the 2PCF amplitude:
       ρ = <ξ_data / ξ_SP>_r
       n_rand = m_SP * (1/√ρ - 1)
5. Randomly delete nodes from the SP+random mixture to reach exactly N

All functions that are called inside joblib parallel workers re-import their
dependencies explicitly so they are self-contained in sub-processes.
"""

from pathlib import Path
import sys

# ---------------------------------------------------------------------------
# Path setup: ensure src/ is found when this module is imported from anywhere
# (including joblib worker sub-processes).
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent          # .../src/ (module co-localisé)
_SRC  = _HERE
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import geopandas as gpd
from scipy.optimize import curve_fit
import methods_two_point_correlation as mtpc


# ===========================================================================
# Soneira-Peebles model
# ===========================================================================

def soneira_peebles_border(lamb, eta, L, R, gdf_edge, crs):
    """
    Generate a Soneira-Peebles fractal point set.

    Starting from the centroid of gdf_edge, each point spawns η children
    placed at a random radius r ∈ [0, R/λ^i] and angle θ ∈ [0, 2π] at
    level i.  After L levels the cloud contains η^L candidate points.

    Parameters
    ----------
    lamb : float
        Scale ratio λ = η^(1/(2-γ)).  Controls how quickly the cluster
        radius shrinks from one level to the next.
    eta : int
        Branching factor: each node spawns η children.
    L : int
        Number of levels (depth of the fractal tree).
    R : float
        Reference radius in metres (e.g. half the territory perimeter / 2π).
    gdf_edge : GeoDataFrame
        Territory boundary used for the seed centroid and inside-test.
    crs : str
        Projected CRS string (metres).

    Returns
    -------
    position : ndarray, shape (η^L, 2)
        All generated point coordinates.
    point_outside : GeoDataFrame
        Subset of *position* that falls outside the territory.
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
    return position, join[join["is_inside"] == False]


def get_inside_points(position, point_outside):
    """
    Return only the points that fall *inside* the territory.

    Parameters
    ----------
    position : ndarray, shape (n, 2)
    point_outside : GeoDataFrame
        Rows whose index matches the rows in *position* that are outside.

    Returns
    -------
    ndarray, shape (n_inside, 2)
    """
    outside_idx = set(point_outside.index.tolist())
    mask = np.array([i not in outside_idx for i in range(len(position))])
    return position[mask]


def build_sp_model(m_star, eta, gamma, R_ref, gdf_edge, crs):
    """
    Build a *complete* Soneira-Peebles model with at least m_star nodes
    inside the territory.

    The minimum number of levels L is chosen so that the expected total
    count η^L ≥ m_star, then incremented until the inside-territory count
    is sufficient.  The full fractal (all levels) is returned without any
    slicing — contrary to the pool-based approach.

    Parameters
    ----------
    m_star : int
        Target minimum number of inside-territory nodes.
    eta : int
        Branching factor.
    gamma : float
        Power-law slope of the data 2PCF.  Sets λ = η^(1/(2-γ)).
    R_ref : float
        Reference radius in metres.
    gdf_edge : GeoDataFrame
        Territory boundary.
    crs : str
        Projected CRS.

    Returns
    -------
    pos_sp : ndarray, shape (m_sp, 2)   — m_sp ≥ m_star
    L      : int    — number of levels used
    lamb   : float  — scale ratio λ
    """
    lamb = eta ** (1.0 / (2.0 - gamma))
    # Minimum L such that η^L ≥ m_star
    L = max(int(np.ceil(np.log(max(m_star, eta)) / np.log(eta))), 4)

    for _attempt in range(10):
        pos_all, pt_out = soneira_peebles_border(lamb, eta, L, R_ref, gdf_edge, crs)
        pos_in = get_inside_points(pos_all, pt_out)
        if len(pos_in) >= m_star:
            return pos_in, L, lamb
        L += 1
        print(f"  L={L - 1} → {len(pos_in):,} inside < {m_star:,} → trying L={L}")

    # Last resort: return whatever we have
    print(f"  Warning: could only generate {len(pos_in):,} nodes (wanted {m_star:,})")
    return pos_in, L, lamb


# ===========================================================================
# 2PCF helpers
# ===========================================================================

def power_law(r, gamma, r0):
    """Power-law model  ξ(r) = (r / r₀)^(−γ)."""
    return (r / r0) ** (-gamma)


def fit_2pcf(r_edges, l_xi, a=1, b=19):
    """
    Fit a power law ξ(r) = (r/r₀)^(−γ) to a 2PCF estimate.

    Parameters
    ----------
    r_edges : ndarray
        Bin centre positions in metres.
    l_xi : list of ndarray
        Individual 2PCF realisations (from PCF_with_variance or bootstrap).
    a, b : int
        Bin range [a, b) used for fitting (avoids r=0 and noisy large-r bins).

    Returns
    -------
    gamma, r0, gamma_err, r0_err : floats, or (None, None, None, None) on failure.
    """
    xi    = np.mean(l_xi, axis=0)
    error = np.std(l_xi,  axis=0)

    # Align lengths (bincount can return variable-length arrays)
    n = min(len(xi), len(r_edges))
    xi = xi[:n]; error = error[:n]; r_edges = r_edges[:n]
    b = min(b, n - 1)

    # Remove NaN / Inf produced when RR = 0 in the Landy-Szalay estimator
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
    Compute the mean 2PCF across realisations, aligned to r_e length.

    Replaces NaN / ±Inf (from RR = 0 bins) with 0.
    """
    xi = np.mean(l_xi_loc, axis=0)
    n  = min(len(xi), len(r_e))
    xi = xi[:n]
    return np.nan_to_num(xi, nan=0.0, posinf=0.0, neginf=0.0)


def _align(r_src, xi_src, r_edges_ref):
    """
    Interpolate xi_src (defined on r_src) onto the reference grid r_edges_ref
    using log-space linear interpolation.

    Bins outside the range of r_src are set to 0 (not extrapolated).
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
    Compute the 2PCF, automatically switching to bootstrap for large datasets.

    Uses PCF_with_variance when len(gdf) < bootstrap_size, otherwise uses
    compute_two_point_correlation_bootstrap.

    Returns
    -------
    r_edges : ndarray
    l_xi    : list of ndarray — individual realisations
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
    Draw n uniform random points inside the territory via rejection sampling.

    Parameters
    ----------
    n           : int
    rng         : numpy Generator (np.random.default_rng)
    gdf_edge    : GeoDataFrame  — territory boundary
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
    Compute the cross-2PCF integral for a given metadata split m.

    The data is sorted by descending metadata before calling this function.
    Sub-population A contains the m highest-metadata nodes; B contains the
    remaining N - m nodes.  The 2PCF of their union (full dataset) is passed
    in as xi_data to avoid recomputing it for every m.

    Cross-2PCF formula (Landy-Szalay decomposition):

        ξ_AB(r) = [N² ξ_data(r) − m² ξ_AA(r) − (N−m)² ξ_BB(r)] / (2·m·(N−m))

    The function is designed to be called inside a joblib Parallel loop.
    It re-imports its dependencies explicitly for sub-process compatibility.

    Parameters
    ----------
    m               : int   — split index
    xy_data_sorted  : ndarray (N, 2) — data sorted by metadata descending
    xi_data         : ndarray — pre-computed full-data 2PCF (ξ_union)
    r_edges         : ndarray — bin edges from the data 2PCF
    gdf_edge        : GeoDataFrame — territory boundary
    crs             : str
    N_run, size_r, k: PCF computation parameters
    rmax, scale, nbins: PCF range / binning parameters
    bootstrap_size  : int — threshold above which bootstrap is used
    bootstrap_N_real: int — number of bootstrap realisations

    Returns
    -------
    dict with keys 'm', 'r_edges', 'xi', 'integral', or None on failure.
    """
    # Re-import inside the worker for joblib sub-process compatibility
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
        """Compute mean 2PCF for a sub-population, returning n_ref values."""
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
    N_tot = N_A + N_B   # = N (constant — union is always the full dataset)
    xi_cross = (
        N_tot**2 * xi_data[:n]
        - N_A**2  * xi_AA[:n]
        - N_B**2  * xi_BB[:n]
    ) / (2.0 * N_A * N_B)
    xi_cross = _np.nan_to_num(xi_cross, nan=0.0, posinf=0.0, neginf=0.0)

    r_e      = r_edges[:n]
    integral = float(_np.trapz(_np.abs(xi_cross[1:]), r_e[1:]))
    if not _np.isfinite(integral):
        return None
    return {"m": int(m), "r_edges": r_e, "xi": xi_cross, "integral": integral}


# ═══════════════════════════════════════════════════════════════════════════
# Soneira-Peebles en dimension quelconque (notebook 05bis — mouche, 3D)
# ═══════════════════════════════════════════════════════════════════════════
# Le modèle de Soneira & Peebles (1978) est né en 3D, pour les galaxies ; c'est
# le passage aux cartes qui était l'adaptation. Les fonctions ci-dessus sont
# écrites pour d = 2 et trois choses y sont fausses hors du plan :
#   - lamb = eta**(1/(2-gamma)) : le 2 est la dimension de l'espace ;
#   - les enfants sont tirés dans un *disque* (r, theta) ;
#   - check_inside relance l'arbre entier tant qu'un point sort du domaine,
#     ce qui ne converge jamais à eta**L ~ 2e5 points.
# Les fonctions qui suivent reprennent la même définition du modèle sans ces
# trois hypothèses. Le domaine est un scipy.spatial.ConvexHull (le rôle tenu
# par gdf_edge en 2D), ce qui les rend utilisables en 2D comme en 3D.

def lambda_of_gamma(eta, gamma, dim):
    """Rapport d'échelle lambda donnant la pente gamma voulue en dimension dim.

    La fractale a pour dimension D = ln(eta)/ln(lambda) et sa 2PCF suit
    xi ~ r^(-gamma) avec gamma = dim - D, d'où lambda = eta**(1/(dim-gamma)).
    En 2D on retrouve le eta**(1/(2-gamma)) des notebooks 01-04."""
    return float(eta) ** (1.0 / (dim - gamma))


def sample_in_ball(size, dim, rng, radial="uniform"):
    """Points dans la boule unité de dimension dim.

    radial="uniform" : uniformes en volume (r = u**(1/dim)) — la définition de
    Soneira & Peebles. radial="linear" : r = u, la loi du code 2D historique,
    qui concentre les enfants vers le centre (densité en r**(1-dim)) ; gardée
    pour pouvoir chiffrer ce que ce détail change (cf. notebook 05bis)."""
    v = rng.normal(size=(size, dim))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    u = rng.random(size)
    r = u ** (1.0 / dim) if radial == "uniform" else u
    return v * r[:, None]


def inside_hull(P, hull, chunk=None):
    """Masque booléen « P est dans l'enveloppe convexe » (A·x + b <= 0 pour
    toutes les faces). Découpé en tranches : la matrice complète ferait
    len(P) x n_faces flottants, soit 0.5 Go à 1.3e5 points et 439 faces."""
    A, b = hull.equations[:, :-1], hull.equations[:, -1]
    chunk = max(1, 2_000_000 // len(b)) if chunk is None else chunk
    out = np.empty(len(P), dtype=bool)
    for s in range(0, len(P), chunk):
        sl = slice(s, s + chunk)
        out[sl] = (P[sl] @ A.T + b <= 0).all(axis=1)
    return out


def soneira_peebles_nd(eta, lamb, L, R, center, hull=None, rng=None,
                       radial="uniform", max_tries=64, return_levels=False):
    """Fractale de Soneira-Peebles en dimension quelconque.

    Chaque point du niveau i engendre eta enfants dans la boule de rayon
    R/lamb**i qui l'entoure ; après L niveaux on rend les eta**L feuilles.

    Le domaine est traité **enfant par enfant** : un enfant tombé hors de
    l'enveloppe est retiré (jusqu'à max_tries fois), au lieu de relancer
    l'arbre entier comme check_inside. Les enfants encore dehors après
    max_tries essais sont abandonnés — la fractale a alors un peu moins de
    eta**L feuilles, ce que le notebook chiffre.

    Input:
        eta: nombre d'enfants par point
        lamb: rapport d'échelle entre deux niveaux (cf. lambda_of_gamma)
        L: nombre de niveaux
        R: rayon du niveau 0
        center: graine, (dim,) ou (1, dim)
        hull: scipy.spatial.ConvexHull délimitant le domaine (None = illimité)
        rng: np.random.Generator
        radial: "uniform" (Soneira-Peebles) ou "linear" (code 2D historique)
        return_levels: rendre aussi, niveau par niveau, le couple
            (positions, indice du parent dans le niveau precedent)
    Output:
        position: array (n_feuilles, dim) — n_feuilles <= eta**L
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
