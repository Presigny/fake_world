"""
synthesis_utils.py
==================
Shared pipeline for the cross-dataset comparison of the paper (Table 2 and
the model-comparison figures).

Contents
--------
1. DATASETS: configuration and calibrated SP parameters of the 9 datasets
   (gamma, r0, eta, m*, R_ref). The calibration itself is done by
   `calibration.py` and the notebooks of `notebooks/emile_2026/calibration/`.
2. Normalisation to the unit square, Landy-Szalay 2PCF, power-law fit,
   Jensen-Shannon divergence between two 2PCFs.
3. Poisson reference process.
4. Soneira-Peebles mixture (SP fractal + uniform background), see sp_helpers.
5. Dataset loading (border, points, normalisation, data 2PCF).
6. Point-process competitors inferred by SVI with BSTPP (LGCP, Hawkes,
   Cox-Hawkes) and stochastic bands over independent realisations.
7. Comparison of all models at a constant number of nodes N.
8. Population marks: hierarchical SP, marked Hawkes, mark correlation,
   nearest-neighbour distances, N sweep, and the result cache written by
   `scripts/compute_all.py`.

All model coordinates live in the normalised square [0, 1]^2 (see
normalize_coords / normalize_geometry), so that 2PCFs and divergences are
comparable between territories of different physical sizes.
"""

from pathlib import Path
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import geopandas as gpd
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union
from shapely import affinity

# ---------------------------------------------------------------------------
# Paths: ROOT is the repository root, SRC holds the other modules. Data are
# expected under ROOT/data (see the README for where to get them).
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
SRC = Path(__file__).resolve().parent
# Only src/ is put on the path, first: the repository root holds an older
# methods_two_point_correlation.py that must not shadow the one of src/.
if str(SRC) in sys.path:
    sys.path.remove(str(SRC))
sys.path.insert(0, str(SRC))

import methods_two_point_correlation as mtpc           # noqa: E402
from sp_helpers import (                                # noqa: E402
    soneira_peebles_border, get_inside_points, build_sp_model,
    power_law, _random_in_territory,
)

SEED = 42

# 2PCF parameters (shared by every comparison notebook)
N_BINS = 18
N_RANDOM = 5000
N_RUNS_2PCF = 5


# ===========================================================================
# 1. Dataset configuration
# ===========================================================================
# gamma_data, r0_data_m : power-law fit of the data 2PCF
# eta                   : branching factor (KMeans silhouette)
# m_star                : number of "structured" points (cross-2PCF m-scan)
# R_ref_m               : SP reference radius, 0.5 * perimeter / (2 pi)
# rmax                  : maximum separation (m) of the calibration 2PCF
# These values come from the per-dataset calibration
# (out/00_dataset_calibration_reference.csv).

_FR = ROOT / "data" / "clean_transport_dataset" / "France"
_FR_BORDER = _FR / "france_nat_earth.geojson"

DATASETS = {
    # The Austrian city data are not distributed with this repository (see the
    # README). The entry is kept so that the results can be reproduced locally.
    "Austria Cities": dict(
        name="Austria", crs="EPSG:3416",
        path_border=ROOT / "data" / "map" / "Austria.geojson",
        path_csv=ROOT / "data" / "cities" / "Austria_cities.csv",
        threshold_variable="population", threshold=0,
        N=2254, area_km2=83895,
        gamma_data=0.789, r0_data_m=1945, rmax=300_000,
        eta=2, m_star=2142, R_ref_m=188_500,
    ),
    "Belgium Cities": dict(
        name="Belgium", crs="EPSG:3812",
        path_border=ROOT / "data" / "map" / "Belgium.geojson",
        path_csv=ROOT / "data" / "cities" / "Belgium_cities.csv",
        threshold_variable="population", threshold=0,
        N=815, area_km2=30659,
        gamma_data=0.808, r0_data_m=3083, rmax=200_000,
        eta=2, m_star=718, R_ref_m=102_700,
    ),
    "RTE France": dict(
        name="France", crs="EPSG:2154",
        path_border=_FR_BORDER,
        path_csv=_FR / "postes-electriques-rte.csv",
        threshold_variable="population_density", threshold=1,
        N=4932, area_km2=538401,
        gamma_data=0.931, r0_data_m=12427, rmax=160_000,
        eta=6, m_star=3661, R_ref_m=496_900,
    ),
    "Gares France": dict(
        name="France", crs="EPSG:2154",
        path_border=_FR_BORDER,
        path_csv=_FR / "noeuds_route500_gares.csv",
        threshold_variable="population_density", threshold=1,
        N=3470, area_km2=538401,
        gamma_data=0.643, r0_data_m=61622, rmax=160_000,
        eta=6, m_star=2896, R_ref_m=496_900,
    ),
    # Large datasets (N >> 10,000) are subsampled to N_data_cap points when
    # loaded: the O(N^2) cdist becomes impractical beyond that, and the
    # point-process competitors do not scale beyond about 10,000 nodes.
    "ENEDIS France": dict(
        name="France", crs="EPSG:2154",
        path_border=_FR_BORDER,
        path_csv=_FR / "postes-electriques-enedis.csv",
        threshold_variable="population_density", threshold=1,
        N=912_982, area_km2=538_401, N_data_cap=10_000,
        gamma_data=0.954, r0_data_m=7_995, rmax=160_000,
        eta=6, m_star=235_268, R_ref_m=496_900,
    ),
    "France Communes": dict(
        name="France", crs="EPSG:2154",
        path_border=_FR_BORDER,
        path_csv=_FR / "noeuds_route500_commune.csv",
        threshold_variable="population_density", threshold=1,
        N=34_269, area_km2=538_401, N_data_cap=10_000,
        gamma_data=0.254, r0_data_m=292, rmax=160_000,
        eta=2, m_star=24_647, R_ref_m=496_900,
    ),
    "France Routes": dict(
        name="France", crs="EPSG:2154",
        path_border=_FR_BORDER,
        path_csv=_FR / "noeuds_route500_routes.csv",
        threshold_variable="population_density", threshold=1,
        N=897_965, area_km2=538_401, N_data_cap=10_000,
        gamma_data=0.525, r0_data_m=1_744, rmax=160_000,
        eta=7, m_star=625_121, R_ref_m=496_900,
    ),
    "Germany Stops": dict(
        name="Germany", crs="EPSG:4839",
        path_border=ROOT / "data" / "clean_transport_dataset" / "Germany" / "germany_nat_earth.geojson",
        path_csv=ROOT / "data" / "clean_transport_dataset" / "Germany" / "localtransitgermany.csv",
        threshold_variable="population_density", threshold=1,
        N=660_298, area_km2=355_405, N_data_cap=10_000,
        gamma_data=0.612, r0_data_m=4_384, rmax=160_000,
        eta=4, m_star=413_956, R_ref_m=364_100,
    ),
    "Swiss Stops": dict(
        name="Switzerland", crs="EPSG:21781",
        path_border=ROOT / "data" / "clean_transport_dataset" / "Switzerland" / "switzerland_nat_earth.geojson",
        path_csv=ROOT / "data" / "clean_transport_dataset" / "Switzerland" / "swiss_stops.csv",
        threshold_variable="population_density", threshold=1,
        N=93_288, area_km2=41_439, N_data_cap=10_000,
        gamma_data=0.873, r0_data_m=6_690, rmax=160_000,
        eta=12, m_star=56_331, R_ref_m=117_700,
    ),
}

# Small datasets (N < 5000), processed in full.
SMALL_DATASETS = ["Austria Cities", "Belgium Cities", "RTE France", "Gares France"]

# Hard cap on the number of nodes for the comparison (O(N^2) cdist).
N_CAP_MODELS = 10_000


# ===========================================================================
# 2. Normalisation, border, 2PCF
# ===========================================================================

def normalize_coords(xy_metric, bounds_metric):
    """Map metric coordinates to [0, 1]^2 with an isotropic scale.

    Returns (xy_norm, scale) where scale is the metric length of the unit side.
    """
    xmin, ymin, xmax, ymax = bounds_metric
    scale = max(xmax - xmin, ymax - ymin)
    x_norm = (xy_metric[:, 0] - xmin) / scale
    y_norm = (xy_metric[:, 1] - ymin) / scale
    return np.column_stack([x_norm, y_norm]), scale


def normalize_geometry(gdf_edge, bounds_metric):
    """Apply the normalize_coords transformation to the territory border."""
    xmin, ymin, xmax, ymax = bounds_metric
    scale = max(xmax - xmin, ymax - ymin)
    geom = unary_union(gdf_edge.geometry)
    geom_shifted = affinity.translate(geom, xoff=-xmin, yoff=-ymin)
    return affinity.scale(geom_shifted, xfact=1.0 / scale, yfact=1.0 / scale,
                          origin=(0, 0))


def rejection_sample_in_boundary(n, geom_norm, rng, max_factor=8):
    """Draw n uniform points inside the normalised polygon."""
    from shapely.vectorized import contains
    if n <= 0:
        return np.zeros((0, 2))
    bx0, by0, bx1, by1 = geom_norm.bounds
    collected, total = [], 0
    while total < n:
        batch = max(max_factor * (n - total), 500)
        xs = rng.uniform(bx0, bx1, batch)
        ys = rng.uniform(by0, by1, batch)
        inside = contains(geom_norm, xs, ys)
        collected.append(np.column_stack([xs[inside], ys[inside]]))
        total += inside.sum()
    return np.vstack(collected)[:n]


def compute_2pcf_norm(xy, geom_norm, n_random=N_RANDOM, n_runs=N_RUNS_2PCF,
                      n_bins=N_BINS):
    """Landy-Szalay 2PCF in normalised coordinates, logarithmic bins.

    The random catalogue is drawn n_runs times (n_random points each) inside
    the border and the DR and RR counts are averaged.
    Returns (r_centers, xi) with xi = (DD - 2DR + RR) / RR.
    """
    DD_raw = cdist(xy, xy)
    DD_raw = np.triu(DD_raw, k=1).ravel()
    DD_raw = DD_raw[DD_raw > 0]

    tree = cKDTree(xy)
    nn_dist, _ = tree.query(xy, k=2)
    rmin = max(np.mean(nn_dist[:, 1]) * 0.5, 1e-5)
    rmax = np.max(DD_raw) * 0.55

    r_edges = np.logspace(np.log10(rmin), np.log10(rmax), n_bins + 1)
    r_centers = np.sqrt(r_edges[:-1] * r_edges[1:])
    DD_hist, _ = np.histogram(DD_raw, bins=r_edges)
    DD_norm = DD_hist / max(len(DD_raw), 1)

    rng_r = np.random.default_rng(99)
    DR_list, RR_list = [], []
    for _ in range(n_runs):
        rand_xy = rejection_sample_in_boundary(n_random, geom_norm, rng_r)
        DR_raw = cdist(xy, rand_xy).ravel()
        DR_hist, _ = np.histogram(DR_raw, bins=r_edges)
        DR_list.append(DR_hist / max(len(DR_raw), 1))
        RR_raw = cdist(rand_xy, rand_xy)
        RR_raw = np.triu(RR_raw, k=1).ravel()
        RR_raw = RR_raw[RR_raw > 0]
        RR_hist, _ = np.histogram(RR_raw, bins=r_edges)
        RR_list.append(RR_hist / max(len(RR_raw), 1))

    DR_norm = np.mean(DR_list, axis=0)
    RR_norm = np.mean(RR_list, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = np.where(RR_norm > 0,
                      (DD_norm - 2 * DR_norm + RR_norm) / RR_norm, np.nan)
    return r_centers, xi


def fit_powerlaw_norm(r, xi, a=2, b=None):
    """Fit xi(r) = (r / r0)^(-gamma) on bins a..b-1.

    Only finite positive xi are used. Returns (gamma, r0), or (None, None)
    when fewer than 3 bins are usable or the fit fails.
    """
    if b is None:
        b = len(r) - 2
    valid = np.isfinite(xi[a:b]) & (xi[a:b] > 0) & (r[a:b] > 0)
    if valid.sum() < 3:
        return None, None
    try:
        popt, _ = curve_fit(
            lambda r_, gamma, r0: (r_ / r0) ** (-gamma),
            r[a:b][valid], xi[a:b][valid],
            p0=[1.0, 0.01], bounds=([0.05, 1e-6], [1.95, 1.0]), maxfev=3000)
        return popt[0], popt[1]
    except Exception:
        return None, None


def js_divergence(r_ref, xi_ref, r_mod, xi_mod):
    """Jensen-Shannon divergence between a reference 2PCF and a model 2PCF.

    The model curve is interpolated (in log r) on the reference bins. Each
    curve is turned into a pair density p(r) proportional to
    max(xi(r) + 1, eps) * r (the factor r is the annulus weight of a
    logarithmic bin). JSD = 0 means the two 2PCFs coincide.
    """
    eps = 1e-9
    xi_mod_al = np.interp(
        np.log10(np.clip(r_ref, 1e-12, None)),
        np.log10(np.clip(r_mod, 1e-12, None)),
        np.nan_to_num(xi_mod, nan=0.0), left=np.nan, right=np.nan)
    mask = np.isfinite(xi_ref) & np.isfinite(xi_mod_al) & (r_ref > 0)
    if mask.sum() < 3:
        return np.nan
    r = r_ref[mask]
    p_raw = np.maximum(xi_ref[mask] + 1, eps) * r
    q_raw = np.maximum(xi_mod_al[mask] + 1, eps) * r
    P = p_raw / p_raw.sum()
    Q = q_raw / q_raw.sum()
    M = 0.5 * (P + Q)
    kl = lambda a, b: np.sum(a * np.log(a / b + eps))
    return float(0.5 * kl(P, M) + 0.5 * kl(Q, M))


# ===========================================================================
# 3. Poisson reference
# ===========================================================================

def simulate_poisson(N_target, geom_norm, seed=None):
    """Homogeneous Poisson process inside the border (reference, xi(r) ~ 0)."""
    rng = np.random.default_rng(seed)
    return rejection_sample_in_boundary(N_target, geom_norm, rng)


# ===========================================================================
# 4. Soneira-Peebles mixture (SP fractal + uniform background)
# ===========================================================================

def build_sp_mixture(ds, gdf_edge, bounds, geom_norm, r_data, xi_data,
                     N_target, seed=SEED, verbose=False, return_labels=False):
    """Build a calibrated SP fractal plus a uniform background.

    Steps (see sp_helpers):
      1. full SP with m_SP >= m* nodes inside the territory, where m* is
         rescaled by N_target / N when N_target < N
      2. n_rand uniform points are added so that the amplitude of the 2PCF
         matches the data (rho = <xi_data / xi_SP>, n_rand = m_SP (rho^-1/2 - 1))
      3. random deletion down to exactly N_target nodes

    Returns xy_norm (coordinates in [0, 1]^2). With return_labels=True,
    returns (xy_norm, is_sp) where is_sp[i] tells whether node i comes from
    the SP fractal (True) or from the background (False). The random stream
    is the same in both cases.
    """
    np.random.seed(seed)
    m_star_eff = ds["m_star"]
    if N_target < ds["N"]:
        m_star_eff = max(1, round(ds["m_star"] / ds["N"] * N_target))

    pos_sp, L_sp, lamb_sp = build_sp_model(
        m_star_eff, ds["eta"], ds["gamma_data"], ds["R_ref_m"], gdf_edge, ds["crs"])
    m_sp = len(pos_sp)
    if verbose:
        print(f"  SP: eta={ds['eta']} L={L_sp} lambda={lamb_sp:.3f} m_SP={m_sp:,}")

    sp_norm, _ = normalize_coords(pos_sp, bounds)
    r_tmp, xi_tmp = compute_2pcf_norm(sp_norm, geom_norm, N_RANDOM, 3, N_BINS)
    xi_data_al = np.interp(
        np.log10(np.clip(r_tmp, 1e-9, None)),
        np.log10(np.clip(r_data, 1e-9, None)),
        np.nan_to_num(xi_data), left=0.0, right=0.0)
    mask = np.isfinite(xi_tmp) & (xi_tmp > 1e-6) & np.isfinite(xi_data_al)
    rho = np.clip(np.mean(xi_data_al[mask] / xi_tmp[mask]) if mask.sum() else 1.0,
                  1e-4, 1.0)
    n_rand = int(np.round(m_sp * (1.0 / np.sqrt(rho) - 1.0)))

    rng_f = np.random.default_rng(seed + 1)
    terr_g = gdf_edge.geometry.union_all()
    rand_xy = _random_in_territory(n_rand, rng_f, gdf_edge, ds["crs"],
                                   terr_g, terr_g.bounds)
    pos_mix = np.vstack([pos_sp, rand_xy])
    is_sp = np.zeros(len(pos_mix), dtype=bool)
    is_sp[:len(pos_sp)] = True
    if len(pos_mix) > N_target:
        keep = rng_f.choice(len(pos_mix), size=N_target, replace=False)
        pos_mix, is_sp = pos_mix[keep], is_sp[keep]
    xy_norm, _ = normalize_coords(pos_mix, bounds)
    if return_labels:
        return xy_norm, is_sp
    return xy_norm


# ===========================================================================
# 5. Dataset loading
# ===========================================================================

def load_dataset(ds, compute_data_2pcf=True, verbose=True):
    """Load the border and the points, normalise them, compute the data 2PCF.

    Large datasets are subsampled to ds["N_data_cap"] points (seed 42). The
    function also stores in `ds` the normalised correlation length r0_norm and
    the offspring spread sigma = r0_norm / 2 used by the marked Hawkes process.

    Returns a dict `ld` shared by all the model builders.
    """
    crs = ds["crs"]
    gdf_edge_raw = gpd.read_file(ds["path_border"])
    geom0 = gdf_edge_raw.geometry.iloc[0]
    if isinstance(geom0, MultiPolygon):
        largest = max(geom0.geoms, key=lambda p: p.area)
        gdf_edge_raw = gpd.GeoDataFrame(geometry=[largest], crs=gdf_edge_raw.crs)
    gdf_edge = gdf_edge_raw.to_crs(crs)

    gdf_pts = mtpc.load_df_to_gdf(ds["path_csv"], ds["threshold"],
                                  ds["threshold_variable"])
    gdf_proj = gpd.clip(gdf_pts.to_crs(crs), gdf_edge)
    xy_metric_full = gdf_proj.get_coordinates().values
    N_full = len(xy_metric_full)
    # Mark (population or population density) aligned with the points.
    mvar = ds["threshold_variable"]
    pop_full = (gdf_proj[mvar].to_numpy(dtype=float)
                if mvar in gdf_proj.columns else None)

    N_cap = ds.get("N_data_cap")
    if N_cap is not None and N_full > N_cap:
        rng_sub = np.random.default_rng(42)
        sub_idx = rng_sub.choice(N_full, size=N_cap, replace=False)
        xy_metric = xy_metric_full[sub_idx]
        pop_data = pop_full[sub_idx] if pop_full is not None else None
        N_real = N_cap
        if verbose:
            print(f"  N_full={N_full:,} -> subsampled to {N_cap:,}")
    else:
        xy_metric = xy_metric_full
        pop_data = pop_full
        N_real = N_full

    bounds = gdf_edge.total_bounds
    xy_norm, scale_m = normalize_coords(xy_metric, bounds)
    geom_norm = normalize_geometry(gdf_edge, bounds)
    r0_norm = ds["r0_data_m"] / scale_m
    ds.update(r0_norm=r0_norm, sigma=max(r0_norm / 2.0, 1e-3))

    if verbose:
        print(f"  N={N_real:,}  scale={scale_m / 1e3:.1f} km")

    r_data = xi_data = gamma_fit = r0_fit = None
    if compute_data_2pcf:
        r_data, xi_data = compute_2pcf_norm(xy_norm, geom_norm)
        gamma_fit, r0_fit = fit_powerlaw_norm(r_data, xi_data)
        if verbose and gamma_fit is not None:
            print(f"  gamma_data={gamma_fit:.3f}  r0_norm={r0_fit:.5f}")

    return dict(
        gdf_edge=gdf_edge, xy_metric=xy_metric, xy_norm=xy_norm,
        pop_data=pop_data,
        N_real=N_real, N_full=N_full, scale_m=scale_m, bounds=bounds,
        geom_norm=geom_norm,
        r_data=r_data, xi_data=xi_data, gamma_fit=gamma_fit, r0_fit_norm=r0_fit,
    )


def data_2pcf_at_N(ld, N_target, seed=SEED):
    """Draw N_target data points and compute their 2PCF.

    Used to compare data and models at the same N. When N_target >= N_real,
    the full data and their precomputed 2PCF are returned.
    Returns (xy, r, xi).
    """
    if N_target >= ld["N_real"]:
        return ld["xy_norm"], ld["r_data"], ld["xi_data"]
    rng = np.random.default_rng(seed)
    idx = rng.choice(ld["N_real"], size=N_target, replace=False)
    xy = ld["xy_norm"][idx]
    r, xi = compute_2pcf_norm(xy, ld["geom_norm"])
    return xy, r, xi


# ===========================================================================
# 6. Point processes inferred by SVI (BSTPP)
# ===========================================================================
# LGCP, Hawkes and Cox-Hawkes models are fitted by stochastic variational
# inference with BSTPP (notebooks/emile_2026/09_bstpp_inference.ipynb). Their posterior
# samples are saved as .pkl files in BSTPP_RESULTS and reloaded here.

BSTPP_CAP = 2000                                  # N used for the inference
BSTPP_RESULTS = ROOT / "out" / "bstpp_results"
BSTPP_NAMES = ("LGCP", "Hawkes", "Cox Hawkes")


def _bstpp_xy(ld, cap=BSTPP_CAP, seed=123):
    """Rebuild the BSTPP training subsample (same seed as the inference)."""
    xy = ld["xy_norm"]
    if len(xy) > cap:
        rng = np.random.default_rng(seed)
        xy = xy[rng.choice(len(xy), size=cap, replace=False)]
    return xy


def load_bstpp_models(name, ld, T_max=365.0, seed=42):
    """Reload the three BSTPP models (LGCP, Hawkes, Cox Hawkes) of a dataset.

    Each model is rebuilt with the training data frame and the [0, 1]^2
    domain, then its posterior samples are read from the .pkl file.
    Returns {model name: model or None}. bstpp and jax are imported lazily.
    """
    import geopandas as _gpd
    from shapely.geometry import box as _box
    import numpyro.distributions as dist
    from bstpp.main import LGCP_Model, Hawkes_Model

    slug = name.replace(" ", "_")
    xy = _bstpp_xy(ld)
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"X": xy[:, 0], "Y": xy[:, 1],
                       "T": rng.uniform(0, T_max, len(xy))}
                      ).sort_values("T").reset_index(drop=True)
    A = _gpd.GeoDataFrame(geometry=[_box(0, 0, 1, 1)])

    out = {}
    for mname in BSTPP_NAMES:
        pkl = BSTPP_RESULTS / f"{slug}_{mname.replace(' ', '_')}.pkl"
        if not pkl.exists():
            out[mname] = None
            continue
        try:
            if mname == "LGCP":
                model = LGCP_Model(df, A, T_max, a_0=dist.Normal(0, 5))
            else:
                model = Hawkes_Model(
                    df, A, T_max, cox_background=(mname == "Cox Hawkes"),
                    a_0=dist.Normal(0, 5), alpha=dist.Beta(2, 5),
                    beta=dist.HalfNormal(2.0), sigmax_2=dist.HalfNormal(0.25))
            model.load_rslts(str(pkl))
            out[mname] = model
        except Exception as e:
            print(f"  [BSTPP {mname}] reload failed: {e}")
            out[mname] = None
    return out


def bstpp_simulate_pool(model, geom_norm, n_sim=6, seed=SEED,
                        cap_per_sim=8000):
    """Simulate n_sim patterns from the posterior and keep points inside the border.

    model.simulate() draws a new point pattern from the mean posterior
    intensity at each call. Returns the stacked normalised coordinates.
    """
    from shapely.vectorized import contains
    pool = []
    for k in range(n_sim):
        try:
            sim = model.simulate()
        except Exception:
            continue
        xy = sim[["X", "Y"]].values
        inside = contains(geom_norm, xy[:, 0], xy[:, 1])
        xin = xy[inside]
        if len(xin) < 10:
            continue
        if len(xin) > cap_per_sim:
            rng = np.random.default_rng(seed + k)
            xin = xin[rng.choice(len(xin), size=cap_per_sim, replace=False)]
        pool.append(xin)
    return np.vstack(pool) if pool else np.zeros((0, 2))


def _bstpp_realization(model, geom_norm, N, seed):
    """One independent realisation of a BSTPP model, resampled to N points."""
    from shapely.vectorized import contains
    try:
        sim = model.simulate()
    except Exception:
        return None
    xy = sim[["X", "Y"]].values
    inside = contains(geom_norm, xy[:, 0], xy[:, 1])
    xin = xy[inside]
    if len(xin) < 10:
        return None
    rng = np.random.default_rng(seed)
    return xin[rng.choice(len(xin), size=N, replace=len(xin) < N)]


def _align_xi(r_src, xi_src, r_ref):
    """Interpolate a curve (r_src, xi_src) on the grid r_ref (in log r)."""
    return np.interp(np.log10(np.clip(r_ref, 1e-12, None)),
                     np.log10(np.clip(r_src, 1e-12, None)),
                     np.nan_to_num(xi_src), left=np.nan, right=np.nan)


def stochastic_bands(ds, ld, N, n_instances=10, seed=SEED, bstpp_models=None,
                     models=("SP", "Poisson", "LGCP", "Hawkes", "Cox Hawkes")):
    """Variability of each model over n_instances independent realisations.

    For each model, returns the mean and standard deviation of the 2PCF, the
    mean and standard deviation of the JSD to the data, and the band width
    cv = <std(xi) / |mean(xi)|>. A narrow band means a more reproducible model.

    Returns {model: dict(r, xi_mean, xi_std, jsd_mean, jsd_std, cv)}.
    """
    bstpp_models = bstpp_models or {}
    geom_norm, gdf_edge, bounds = ld["geom_norm"], ld["gdf_edge"], ld["bounds"]
    data_xy, r_data, xi_data = data_2pcf_at_N(ld, N, seed=seed)
    out = {}
    for m in models:
        r_ref, xi_list, jsd_list = None, [], []
        for k in range(n_instances):
            if m == "SP":
                xy = build_sp_mixture(ds, gdf_edge, bounds, geom_norm,
                                      r_data, xi_data, N, seed=seed + 1000 + k)
            elif m == "Poisson":
                xy = simulate_poisson(N, geom_norm, seed=seed + 1000 + k)
            elif bstpp_models.get(m) is not None:
                xy = _bstpp_realization(bstpp_models[m], geom_norm, N,
                                        seed + 1000 + k)
            else:
                xy = None
            if xy is None or len(xy) < 10:
                continue
            r_k, xi_k = compute_2pcf_norm(xy, geom_norm)
            if r_ref is None:
                r_ref = r_k
            xi_list.append(_align_xi(r_k, xi_k, r_ref))
            jsd_list.append(js_divergence(r_data, xi_data, r_k, xi_k))
        if not xi_list:
            out[m] = dict(r=None, xi_mean=None, xi_std=None,
                          jsd_mean=np.nan, jsd_std=np.nan, cv=np.nan)
            continue
        xi_all = np.array(xi_list)
        xi_mean, xi_std = np.nanmean(xi_all, 0), np.nanstd(xi_all, 0)
        valid = np.isfinite(xi_mean) & (np.abs(xi_mean) > 1e-6)
        cv = float(np.nanmean(xi_std[valid] / np.abs(xi_mean[valid]))) if valid.any() else np.nan
        jsd_arr = np.array(jsd_list)
        out[m] = dict(r=r_ref, xi_mean=xi_mean, xi_std=xi_std,
                      jsd_mean=float(np.nanmean(jsd_arr)),
                      jsd_std=float(np.nanstd(jsd_arr)), cv=cv)
    return out


# ===========================================================================
# 7. Comparison of the models at constant N
# ===========================================================================

def compare_at_constant_N(ds, ld, N_target=None, models=("SP", "Poisson",
                          "LGCP", "Hawkes", "Cox Hawkes"), seed=SEED,
                          bstpp_pools=None):
    """Compare several models at the same number of nodes N_target.

    One sample of N_target data points is drawn and its 2PCF is the
    reference. Each model then generates N_target points inside the same
    border and its JSD to the reference is computed, so that data and models
    all have the same N. SP and Poisson are generated directly. LGCP, Hawkes
    and Cox Hawkes are resampled from their BSTPP pools ({name: pool_xy}).
    A model without a pool gets a NaN JSD.

    Returns {model: dict(xy, r, xi, gamma, jsd, n)} plus the key "data".
    """
    if N_target is None:
        N_target = ld["N_real"]
    N_target = min(N_target, ld["N_real"], N_CAP_MODELS)
    geom_norm, gdf_edge, bounds = ld["geom_norm"], ld["gdf_edge"], ld["bounds"]
    bstpp_pools = bstpp_pools or {}

    def _from_pool(pool):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pool), size=N_target, replace=len(pool) < N_target)
        return pool[idx]

    data_xy, r_data, xi_data = data_2pcf_at_N(ld, N_target, seed=seed)
    out = {"data": dict(xy=data_xy, r=r_data, xi=xi_data, n=len(data_xy))}
    for m in models:
        pool = bstpp_pools.get(m)
        if m == "SP":
            xy = build_sp_mixture(ds, gdf_edge, bounds, geom_norm,
                                  r_data, xi_data, N_target, seed=seed)
        elif m == "Poisson":
            xy = simulate_poisson(N_target, geom_norm, seed=seed)
        elif pool is not None and len(pool) >= 10:
            xy = _from_pool(pool)
        else:
            out[m] = dict(xy=np.zeros((0, 2)), r=None, xi=None,
                          gamma=np.nan, jsd=np.nan, n=0)
            continue
        r_m, xi_m = compute_2pcf_norm(xy, geom_norm)
        g_m, _ = fit_powerlaw_norm(r_m, xi_m)
        jsd = (js_divergence(r_data, xi_data, r_m, xi_m)
               if r_data is not None else np.nan)
        out[m] = dict(xy=xy, r=r_m, xi=xi_m, gamma=g_m, jsd=jsd, n=len(xy))
    return out


def sweep_constant_N(ds, ld, n_list, models=("SP", "Poisson", "LGCP",
                     "Hawkes", "Cox Hawkes"), seed=SEED, bstpp_pools=None):
    """JSD of each model for several values of N.

    The same BSTPP pools are reused for every N. Values of N larger than
    N_real are clipped to N_real. Returns dict(N=[...], jsd={model: [...]}).
    """
    ns = sorted({min(n, ld["N_real"]) for n in n_list})
    jsd = {m: [] for m in models}
    for N in ns:
        res = compare_at_constant_N(ds, ld, N_target=N, models=models,
                                    seed=seed, bstpp_pools=bstpp_pools)
        for m in models:
            jsd[m].append(res[m]["jsd"])
    return dict(N=np.array(ns), jsd={m: np.array(v) for m, v in jsd.items()})


# ===========================================================================
# 8. Population marks
# ===========================================================================
# The SP model is a hierarchy. If the whole tree is kept (not only the
# leaves), the nodes of the first levels are few and central, like large
# cities or hubs, and the leaves are many, like small units. A population
# that decreases with the level then gives a power-law rank-size
# distribution, and the marks are correlated with the spatial structure.

def soneira_peebles_tree(lamb, eta, L, R, center, seed=None):
    """Generate the full SP tree (all levels 0..L-1) with the level of each node.

    Contrary to soneira_peebles_border, which keeps only the leaves, the
    nodes of every level are kept. Level 0 holds the eta children of the root.
    Returns (positions (n, 2), levels (n,)).
    """
    rng = np.random.default_rng(seed)
    position = np.asarray(center, dtype=float).reshape(1, 2)
    all_pos, all_lvl = [], []
    for i in range(L):
        child = []
        for x, y in position:
            r = (R / (lamb ** i)) * rng.uniform(0, 1, size=eta)
            theta = rng.uniform(0, 2 * np.pi, size=eta)
            child.extend(np.column_stack([x - r * np.cos(theta),
                                          y - r * np.sin(theta)]))
        position = np.array(child)
        all_pos.append(position)
        all_lvl.append(np.full(len(position), i))
    return np.vstack(all_pos), np.concatenate(all_lvl)


def population_from_level(levels, eta, pop_max=1e6, slope=1.0):
    """Population pop_max / eta^(slope * level)."""
    return pop_max / (eta ** (slope * levels.astype(float)))


def build_sp_with_population(ds, gdf_edge, L=None, pop_max=1e6, slope=1.0,
                             max_nodes=12_000, seed=SEED):
    """Hierarchical SP tree marked by a population that decreases with the level.

    L is chosen to give about m* nodes, but capped so that the tree size
    sum_i eta^i stays below max_nodes (a large eta, e.g. eta = 12 for
    Switzerland, would otherwise give far too many nodes). Only nodes inside
    the territory are kept, then subsampled to max_nodes if needed.
    Returns dict(xy_norm, pop, levels).
    """
    crs = ds["crs"]
    eta, gamma = ds["eta"], ds["gamma_data"]
    lamb = eta ** (1.0 / (2.0 - gamma))
    if L is None:
        L = max(int(np.ceil(np.log(max(ds["m_star"], eta)) / np.log(eta))), 4)
    # Largest L such that sum_{i=1}^{L} eta^i <= max_nodes
    L_cap, cum = 1, eta
    while L_cap < L and cum + eta ** (L_cap + 1) <= max_nodes:
        L_cap += 1
        cum += eta ** L_cap
    L = L_cap
    center = gdf_edge.centroid.get_coordinates().to_numpy()[0]

    pos, lvl = soneira_peebles_tree(lamb, eta, L, ds["R_ref_m"], center, seed)
    pop = population_from_level(lvl, eta, pop_max, slope)

    s = gpd.GeoDataFrame(geometry=gpd.points_from_xy(pos[:, 0], pos[:, 1]),
                         crs=crs)
    inside = gpd.sjoin(s, gdf_edge, how="left",
                       predicate="within")["index_right"].notna().to_numpy()
    pos, pop, lvl = pos[inside], pop[inside], lvl[inside]

    if len(pos) > max_nodes:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(pos), size=max_nodes, replace=False)
        pos, pop, lvl = pos[keep], pop[keep], lvl[keep]

    bounds = gdf_edge.total_bounds
    xy_norm, _ = normalize_coords(pos, bounds)
    return dict(xy_norm=xy_norm, pop=pop, levels=lvl)


def fit_zipf_exponent(pop, a=1, b=None):
    """Rank-size exponent s, population ~ rank^(-s) (log-log fit on ranks a..b)."""
    p = np.sort(np.asarray(pop, dtype=float))[::-1]
    p = p[p > 0]
    if len(p) < 10:
        return np.nan
    if b is None:
        b = len(p)
    rank = np.arange(1, len(p) + 1)
    s = -np.polyfit(np.log(rank[a:b]), np.log(p[a:b]), 1)[0]
    return float(s)


def calibrate_sp_slope(ds, gdf_edge, target_zipf, seed=SEED,
                       slopes=np.linspace(0.3, 2.0, 12)):
    """Pick the slope of the hierarchical SP whose rank-size exponent is
    closest to the data. Returns (slope, zipf_sp)."""
    best = None
    for s in slopes:
        spp = build_sp_with_population(ds, gdf_edge, slope=s, seed=seed)
        z = fit_zipf_exponent(spp["pop"])
        err = abs(z - target_zipf) if np.isfinite(z) and np.isfinite(target_zipf) else np.inf
        if best is None or err < best[1]:
            best = (s, err, z)
    return best[0], best[2]


def simulate_marked_hawkes(N_target, geom_norm, marks_pool, sigma,
                           alpha0=0.6, decay=0.5, seed=None, max_iter=30):
    """Marked spatial Hawkes process, the point-process competitor for marks.

    Background points get a mark drawn from marks_pool (the empirical
    population distribution). A node of mark m has on average
    alpha0 * m / <m> children, spread with a Gaussian kernel of width sigma,
    and its children inherit the mark decay * m. Large marks are therefore
    surrounded by many nearby nodes, as in the data.
    Returns (xy (n, 2), marks (n,)).
    """
    from shapely.vectorized import contains
    rng = np.random.default_rng(seed)
    marks_pool = np.asarray(marks_pool, dtype=float)
    marks_pool = marks_pool[marks_pool > 0]
    mean_m = marks_pool.mean()
    n_bg = max(int(N_target / (1 + alpha0)), 10)

    all_xy, all_m = [], []
    for _ in range(max_iter):
        bg = rejection_sample_in_boundary(n_bg, geom_norm, rng)
        bg_m = rng.choice(marks_pool, size=len(bg))
        all_xy.append(bg)
        all_m.append(bg_m)
        for (px, py), pm in zip(bg, bg_m):
            n_off = rng.poisson(alpha0 * pm / mean_m)
            if n_off == 0:
                continue
            ox = px + rng.normal(0, sigma, n_off)
            oy = py + rng.normal(0, sigma, n_off)
            ins = contains(geom_norm, ox, oy)
            if ins.sum() > 0:
                all_xy.append(np.column_stack([ox[ins], oy[ins]]))
                all_m.append(np.full(ins.sum(), pm * decay))
        if sum(len(a) for a in all_xy) >= N_target:
            break
        n_bg = int(n_bg * 1.4)

    xy = np.vstack(all_xy)
    marks = np.concatenate(all_m)
    if len(xy) > N_target:
        idx = rng.choice(len(xy), size=N_target, replace=False)
        xy, marks = xy[idx], marks[idx]
    return xy, marks


def split_by_population(xy, pop, frac_high=0.2):
    """Split the nodes into (high, low) population groups by rank.

    Splitting by rank rather than by value keeps both groups non-empty when
    the mark has many ties (frequent for the discrete population_density).
    """
    n = len(pop)
    n_hi = max(1, min(n - 1, int(round(frac_high * n))))
    order = np.argsort(pop)[::-1]
    return xy[order[:n_hi]], xy[order[n_hi:]]


def _cap_pts(xy, max_pts, seed=SEED):
    """Subsample a point cloud to max_pts points (bounds the O(N^2) cdist)."""
    if len(xy) <= max_pts:
        return xy
    rng = np.random.default_rng(seed)
    return xy[rng.choice(len(xy), size=max_pts, replace=False)]


def marked_clustering_2pcf(xy, pop, geom_norm, frac_high=0.2, min_pts=20,
                           max_pts=6000):
    """2PCF of the high-population nodes versus the low-population nodes.

    If the mark is correlated with the structure, high-population nodes are
    more clustered (gamma_hi > gamma_lo). Gammas are NaN when a group has
    fewer than min_pts nodes. Each group is capped at max_pts nodes.
    Returns dict(r_hi, xi_hi, r_lo, xi_lo, gamma_hi, gamma_lo).
    """
    xy_hi, xy_lo = split_by_population(xy, pop, frac_high)
    if len(xy_hi) < min_pts or len(xy_lo) < min_pts:
        return dict(r_hi=None, xi_hi=None, r_lo=None, xi_lo=None,
                    gamma_hi=np.nan, gamma_lo=np.nan)
    xy_hi, xy_lo = _cap_pts(xy_hi, max_pts), _cap_pts(xy_lo, max_pts)
    r_hi, xi_hi = compute_2pcf_norm(xy_hi, geom_norm)
    r_lo, xi_lo = compute_2pcf_norm(xy_lo, geom_norm)
    g_hi, _ = fit_powerlaw_norm(r_hi, xi_hi)
    g_lo, _ = fit_powerlaw_norm(r_lo, xi_lo)
    return dict(r_hi=r_hi, xi_hi=xi_hi, r_lo=r_lo, xi_lo=xi_lo,
                gamma_hi=g_hi, gamma_lo=g_lo)


def mark_correlation_function(xy, marks, geom_norm=None, n_bins=18,
                              max_pts=6000, seed=SEED):
    """Mark correlation function k_mm(r) (Stoyan).

    Each pair is weighted by the product of its marks. For the pairs at
    distance about r,

        k_mm(r) = < m_i m_j >_{pairs at r} / <m>^2.

    k_mm = 1 means that marks do not depend on positions, k_mm > 1 that large
    marks are close to each other at scale r, k_mm < 1 that they repel. No
    random catalogue is needed. Returns (r_centers, k_mm).
    """
    marks = np.asarray(marks, dtype=float)
    if len(xy) > max_pts:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(xy), size=max_pts, replace=False)
        xy, marks = xy[keep], marks[keep]
    mean_m = marks.mean()
    if mean_m <= 0:
        return None, None

    iu = np.triu_indices(len(xy), k=1)
    d = cdist(xy, xy)[iu]
    prod = marks[iu[0]] * marks[iu[1]]
    pos = d > 0
    d, prod = d[pos], prod[pos]

    rmin, rmax = np.percentile(d, 1), np.max(d) * 0.55
    r_edges = np.logspace(np.log10(max(rmin, 1e-5)), np.log10(rmax), n_bins + 1)
    r_centers = np.sqrt(r_edges[:-1] * r_edges[1:])
    num, _ = np.histogram(d, bins=r_edges, weights=prod)
    cnt, _ = np.histogram(d, bins=r_edges)
    with np.errstate(divide="ignore", invalid="ignore"):
        k_mm = np.where(cnt > 0, (num / cnt) / mean_m ** 2, np.nan)
    return r_centers, k_mm


def nn_distances(xy):
    """Nearest-neighbour distance of each point."""
    if len(xy) < 2:
        return np.zeros(0)
    d, _ = cKDTree(xy).query(xy, k=2)
    return d[:, 1]


def _jsd_hist(p, q):
    """Jensen-Shannon divergence between two histograms."""
    eps = 1e-12
    p = np.asarray(p, float) + eps
    q = np.asarray(q, float) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(a * np.log(a / b))
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def nn_comparison(data_xy, model_xys, n_bins=25):
    """Compare nearest-neighbour distance distributions of data and models.

    The SP model is not calibrated on this observable, contrary to the 2PCF.
    All distributions share the same logarithmic bins, built from the data.
    Returns ({"r", "data", <model>: histogram}, {model: JSD}).
    """
    dd = nn_distances(data_xy)
    if len(dd) < 3:
        return None, {}
    rmin, rmax = np.percentile(dd, 0.5), np.percentile(dd, 99.5)
    edges = np.logspace(np.log10(max(rmin, 1e-9)), np.log10(rmax), n_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])

    def hist(x):
        h = np.histogram(x, bins=edges)[0]
        return h / max(h.sum(), 1)

    dists = {"r": centers, "data": hist(dd)}
    jsd = {}
    for name, xy in model_xys.items():
        if xy is None or len(xy) < 3:
            dists[name] = np.full(len(centers), np.nan)
            jsd[name] = np.nan
            continue
        dists[name] = hist(nn_distances(xy))
        jsd[name] = _jsd_hist(dists["data"], dists[name])
    return dists, jsd


# ---------------------------------------------------------------------------
# Result cache written by scripts/compute_all.py
# ---------------------------------------------------------------------------
CACHE_DIR = ROOT / "out" / "synthesis_cache"


def cache_slug(name):
    return name.lower().replace(" ", "_")


def load_cached(name):
    """Load the .npz written by compute_all.py for one dataset (None if absent)."""
    f = CACHE_DIR / f"{cache_slug(name)}.npz"
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    out = {}
    for k in d.files:
        v = d[k]
        out[k] = v.item() if v.ndim == 0 else v      # scalars and strings
    return out


def available_cached():
    """Datasets present in the cache, in the order of DATASETS."""
    return [n for n in DATASETS if (CACHE_DIR / f"{cache_slug(n)}.npz").exists()]


if __name__ == "__main__":
    # Smoke test on the smallest dataset (Belgium, N = 815).
    name = "Belgium Cities"
    ds = DATASETS[name]
    print(f"=== Smoke test: {name} ===")
    ld = load_dataset(ds)
    res = compare_at_constant_N(ds, ld, models=("SP", "Poisson"))
    print(f"\nData reference: sample of {res['data']['n']} points")
    print("Model        N      gamma    JSD")
    for m, r in res.items():
        if m == "data":
            continue
        g = f"{r['gamma']:.3f}" if r["gamma"] else "  -  "
        print(f"{m:<12} {r['n']:>5}   {g}   {r['jsd']:.4f}")
