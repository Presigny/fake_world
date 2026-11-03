"""
calibration.py
==============
Per-dataset calibration of the Soneira-Peebles (SP) mixture, in metric
coordinates. It produces the parameters of Table 1 of the paper, which are
stored in synthesis_utils.DATASETS.

Steps (one function each):
  1. territory and reference radius R_ref = 0.5 * perimeter / (2 pi)
  2. data 2PCF (Landy-Szalay, split random catalogues, bootstrap above
     `bootstrap_size` points) and power-law fit, giving gamma and r0
  3. branching factor eta, the number of KMeans clusters that maximises the
     silhouette score
  4. m-scan: the data are sorted by decreasing mark (population or
     population density), the m first nodes form A and the others B, and
     m* minimises the integral of |xi_AB(r)|
  5. SP fractal with at least m* nodes inside the territory,
     lambda = eta^(1/(2 - gamma)), and check of its slope
  6. dilution by uniform points, rho = <xi_data / xi_SP>,
     n_rand = m_SP (rho^-1/2 - 1), then random deletion down to N nodes

`calibrate(name)` runs the whole chain for one entry of DATASETS.
Notebook 00 shows each step with its figures, and scripts/calibrate_all.py
runs all datasets and writes out/00_dataset_calibration.csv.

The random catalogues of the 2PCF are drawn without a fixed seed (as in
methods_two_point_correlation), so two runs differ slightly. The values
obtained for the paper are kept in out/00_dataset_calibration_reference.csv.

The amplitude ratio rho depends strongly on the SP draw. On Belgium, with
m* fixed, the mean ratio of ten SP draws spreads over about 0 to 0.3 (0.12 +/-
0.09 in the stored run of notebook 00), and it can even be negative, in which
case rho is clipped to 1e-4. Section 6bis of notebook 00 shows this spread.
"""

from pathlib import Path
import sys
import multiprocessing as mp

import numpy as np
import geopandas as gpd
from shapely.geometry import MultiPolygon

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import methods_two_point_correlation as mtpc        # noqa: E402
from sp_helpers import (                             # noqa: E402
    build_sp_model, fit_2pcf, _xi_mean, _align, _pcf_safe,
    _random_in_territory, _run_cross_pcf,
)

# 2PCF parameters of the calibration (same for every dataset)
PCF_PARAMS = dict(N_run=5, size=7000, k=3, scale="log", nbins=20,
                  bootstrap_size=10_000, bootstrap_N_real=10)
FIT_RANGE = (1, 19)            # bins used for the power-law fit


def pcf_budget(n, N_run=5, size=7000, k=3):
    """Lighter 2PCF settings for large point sets.

    Returns (N_run, size, k): unchanged below 5,000 points, then fewer and
    smaller random catalogues.
    """
    if n < 5_000:
        return N_run, size, k
    if n < 30_000:
        return 2, 2_000, 1
    return 2, 1_500, 1


# ===========================================================================
# 1. Territory and data
# ===========================================================================

def load_territory(ds):
    """Border of the territory in its metric CRS, and the SP reference radius.

    For a MultiPolygon border only the largest polygon is kept (mainland).
    Returns (gdf_edge, R_ref) with R_ref = 0.5 * perimeter / (2 pi).
    """
    gdf_edge_raw = gpd.read_file(ds["path_border"])
    geom = gdf_edge_raw.geometry.iloc[0]
    if isinstance(geom, MultiPolygon):
        largest = max(geom.geoms, key=lambda p: p.area)
        gdf_edge_raw = gpd.GeoDataFrame(geometry=[largest], crs=gdf_edge_raw.crs)
    gdf_edge = gdf_edge_raw.to_crs(ds["crs"])
    R_ref = 0.5 * gdf_edge.length.sum() / (2 * np.pi)
    return gdf_edge, R_ref


def load_points(ds, gdf_edge):
    """Points of the dataset inside the territory, in the metric CRS.

    Returns (gdf_projected, coords (N, 2), marks (N,)) where marks is the
    column ds["threshold_variable"] (zeros if absent).
    """
    gdf = mtpc.load_df_to_gdf(ds["path_csv"], ds["threshold"],
                              ds["threshold_variable"])
    gdf_projected = gpd.clip(gdf.to_crs(ds["crs"]), gdf_edge)
    coords = gdf_projected.get_coordinates().values
    var = ds["threshold_variable"]
    if var in gdf_projected.columns:
        marks = gdf_projected[var].fillna(0).to_numpy(dtype=float)
    else:
        marks = np.zeros(len(coords))
    return gdf_projected, coords, marks


# ===========================================================================
# 2. Data 2PCF and power-law fit
# ===========================================================================

def data_2pcf(gdf_projected, gdf_edge, crs, rmax, params=PCF_PARAMS):
    """2PCF of the data in metres.

    PCF_with_variance (k realisations) below params["bootstrap_size"] points,
    bootstrap subsampling above. Returns (r_edges, l_xi).
    """
    p = params
    return _pcf_safe(gdf_projected, gdf_edge, crs, p["N_run"], p["size"],
                     p["k"], rmax, p["scale"], p["nbins"],
                     p["bootstrap_size"], p["bootstrap_N_real"])


def fit_gamma(r_edges, l_xi, fit_range=FIT_RANGE):
    """Power-law fit xi(r) = (r / r0)^(-gamma) weighted by the realisations.

    Returns (gamma, r0, gamma_err, r0_err), or four None on failure.
    """
    return fit_2pcf(r_edges, l_xi, a=fit_range[0], b=fit_range[1])


# ===========================================================================
# 3. Branching factor eta (silhouette)
# ===========================================================================

def silhouette_eta(coords, k_min=2, k_max=13, n_sub=1500, n_rep=3, seed=1):
    """Number of KMeans clusters that maximises the silhouette score.

    For each k in [k_min, k_max), n_rep random subsamples of n_sub points
    are clustered. n_sub is kept at 1,500 because silhouette_score needs all
    pairwise distances (O(n_sub^2) memory).
    Returns (eta, k_values, mean scores, std of scores).
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    rng = np.random.default_rng(seed)
    n_sub = min(n_sub, len(coords))
    k_values = list(range(k_min, k_max))
    means, stds = [], []
    for k in k_values:
        scores = []
        for _ in range(n_rep):
            c = coords[rng.choice(len(coords), size=n_sub, replace=False)]
            labels = KMeans(n_clusters=k, tol=1e-8, max_iter=200,
                            n_init=10, random_state=0).fit(c).labels_
            scores.append(silhouette_score(c, labels))
        means.append(np.mean(scores))
        stds.append(np.std(scores))
    means, stds = np.array(means), np.array(stds)
    eta = int(k_values[int(np.argmax(means))])
    return eta, np.array(k_values), means, stds


# ===========================================================================
# 4. m-scan on the cross-2PCF
# ===========================================================================

def m_scan(coords_sorted, xi_data, r_edges, gdf_edge, crs, rmax,
           params=PCF_PARAMS, n_scan=40, bootstrap_size_scan=3_000,
           n_jobs=None, progress=True):
    """Integral of |xi_AB| for n_scan splits of the sorted data.

    coords_sorted must be sorted by decreasing mark. The splits m are spread
    linearly between N/20 and N - N/20. Each split is computed by
    sp_helpers._run_cross_pcf, in parallel with joblib.
    Returns (m_values, integrals, list of result dicts).
    """
    from joblib import Parallel, delayed

    N = len(coords_sorted)
    m_min = max(10, N // 20)
    m_max = N - max(10, N // 20)
    m_values = np.unique(np.linspace(m_min, m_max, n_scan).astype(int))
    N_run, size, k = pcf_budget(N, params["N_run"], params["size"], params["k"])
    if n_jobs is None:
        n_jobs = min(max(1, mp.cpu_count() - 2), 8)

    iterator = m_values
    if progress:
        from tqdm import tqdm
        iterator = tqdm(m_values, desc="m-scan")
    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_cross_pcf)(
            int(m), coords_sorted, xi_data, r_edges, gdf_edge, crs,
            N_run, size, k, rmax, params["scale"], params["nbins"],
            bootstrap_size_scan, params["bootstrap_N_real"])
        for m in iterator)
    results = [r for r in results if r is not None]
    m_arr = np.array([r["m"] for r in results])
    int_arr = np.array([r["integral"] for r in results])
    return m_arr, int_arr, results


def find_m_star(m_arr, int_arr):
    """m* = argmin of the cross-2PCF integral. Returns (m*, index)."""
    i_star = int(np.argmin(int_arr))
    return int(m_arr[i_star]), i_star


# ===========================================================================
# 5-6. SP fractal, slope check, dilution
# ===========================================================================

def sp_2pcf(pos_sp, gdf_edge, crs, rmax, params=PCF_PARAMS):
    """2PCF of an SP point set (metric coordinates). Returns (r_edges, l_xi)."""
    gdf_sp = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(pos_sp[:, 0], pos_sp[:, 1]), crs=crs)
    N_run, size, k = pcf_budget(len(pos_sp), params["N_run"], params["size"],
                                params["k"])
    return _pcf_safe(gdf_sp, gdf_edge, crs, N_run, size, k, rmax,
                     params["scale"], params["nbins"],
                     params["bootstrap_size"], params["bootstrap_N_real"])


def dilution(xi_data, r_edges, xi_sp, r_e_sp, m_sp):
    """Amplitude ratio rho and number of uniform points to add.

    xi_mix ~ (m_SP / (m_SP + n_rand))^2 xi_SP, so matching xi_data gives
    rho = <xi_data / xi_SP> over the bins where xi_SP > 0 (first bin
    excluded), clipped to [1e-4, 1], and n_rand = m_SP (rho^-1/2 - 1).
    Returns (rho, n_rand).
    """
    xi_sp_al = _align(r_e_sp, xi_sp, r_edges)
    mask = xi_sp_al[1:] > 1e-6
    rho = float(np.mean(xi_data[1:][mask] / xi_sp_al[1:][mask])) if mask.sum() else 1.0
    if rho <= 1e-4:
        # A bin at large r where xi_SP is barely positive and xi_data is
        # negative can drive the mean below zero. The SP draw then gives no
        # usable amplitude ratio, and another draw (seed) should be used.
        print(f"warning: rho = {rho:.4f} <= 1e-4, clipped. The amplitude ratio "
              "is dominated by a large-r bin, try another SP seed.")
    rho = float(np.clip(rho, 1e-4, 1.0))
    n_rand = int(np.round(m_sp * (1.0 / np.sqrt(rho) - 1.0)))
    return rho, n_rand


def build_mixture(pos_sp, n_rand, gdf_edge, crs, seed=42):
    """SP points plus n_rand uniform points in the territory.

    Returns (pos_mix, labels) with label 0 for SP and 1 for random.
    """
    terr = gdf_edge.geometry.union_all()
    rng = np.random.default_rng(seed)
    pos_rand = _random_in_territory(n_rand, rng, gdf_edge, crs, terr, terr.bounds)
    if len(pos_rand) == 0:
        return pos_sp.copy(), np.zeros(len(pos_sp), dtype=np.int8)
    pos_mix = np.vstack([pos_sp, pos_rand])
    labels = np.r_[np.zeros(len(pos_sp)), np.ones(len(pos_rand))].astype(np.int8)
    return pos_mix, labels


def final_realisations(pos_mix, labels, N, gdf_edge, crs, r_edges, rmax,
                       params=PCF_PARAMS, n_real=5, seed0=100):
    """Random deletion (or padding) of the mixture down to N nodes, n_real times.

    Returns (xi_mean, xi_std, last positions, last labels), the 2PCFs being
    aligned on r_edges.
    """
    terr = gdf_edge.geometry.union_all()
    N_run, size, k = pcf_budget(N, params["N_run"], params["size"], params["k"])
    all_xi, pos_f, lab_f = [], None, None
    for s in range(n_real):
        rng = np.random.default_rng(seed0 + s)
        if len(pos_mix) > N:
            idx = rng.choice(len(pos_mix), size=N, replace=False)
            pos_f, lab_f = pos_mix[idx], labels[idx]
        elif len(pos_mix) < N:
            extra = _random_in_territory(N - len(pos_mix), rng, gdf_edge, crs,
                                         terr, terr.bounds)
            pos_f = np.vstack([pos_mix, extra])
            lab_f = np.r_[labels, np.ones(len(extra), dtype=np.int8)]
        else:
            pos_f, lab_f = pos_mix.copy(), labels.copy()
        gdf_f = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(pos_f[:, 0], pos_f[:, 1]), crs=crs)
        r_e_f, l_xi_f = _pcf_safe(gdf_f, gdf_edge, crs, N_run, size, k, rmax,
                                  params["scale"], params["nbins"],
                                  params["bootstrap_size"],
                                  params["bootstrap_N_real"])
        all_xi.append(_align(r_e_f, _xi_mean(r_e_f, l_xi_f), r_edges))
    all_xi = np.array(all_xi)
    return all_xi.mean(axis=0), all_xi.std(axis=0), pos_f, lab_f


# ===========================================================================
# Full chain
# ===========================================================================

def calibrate(name, datasets=None, n_scan=40, n_real=5, n_jobs=None,
              seed=42, verbose=True):
    """Run the full calibration of one dataset of synthesis_utils.DATASETS.

    Returns a dict holding the scalar results (N, R_ref, gamma, r0, eta, m*,
    lambda, L, m_SP, gamma_SP, rho, n_rand, final SP and random counts) and
    the arrays needed for the figures (2PCFs, silhouette and m-scan curves,
    final positions). `seed` fixes the SP draw. The random catalogues of the
    2PCF are not seeded.
    """
    if datasets is None:
        import synthesis_utils as su
        datasets = su.DATASETS
    ds = datasets[name]
    crs, rmax = ds["crs"], ds["rmax"]
    say = print if verbose else (lambda *a, **k: None)

    gdf_edge, R_ref = load_territory(ds)
    gdf_projected, coords, marks = load_points(ds, gdf_edge)
    N = len(coords)
    say(f"[{name}] N={N:,}  R_ref={R_ref / 1e3:.1f} km")

    r_edges, l_xi = data_2pcf(gdf_projected, gdf_edge, crs, rmax)
    gamma, r0, gamma_err, r0_err = fit_gamma(r_edges, l_xi)
    xi_data = _xi_mean(r_edges, l_xi)
    say(f"[{name}] gamma={gamma:.4f} +- {gamma_err:.4f}  r0={r0:.0f} m")

    eta, k_values, sil_means, sil_stds = silhouette_eta(coords)
    say(f"[{name}] eta={eta}  (silhouette {sil_means.max():.4f})")

    coords_sorted = coords[np.argsort(marks)[::-1]]
    m_arr, int_arr, scan = m_scan(coords_sorted, xi_data, r_edges, gdf_edge,
                                  crs, rmax, n_scan=n_scan, n_jobs=n_jobs,
                                  progress=verbose)
    m_star, i_star = find_m_star(m_arr, int_arr)
    say(f"[{name}] m*={m_star:,}  (m*/N={m_star / N:.3f})")

    np.random.seed(seed)                 # the SP draw uses the global stream
    pos_sp, L_sp, lamb_sp = build_sp_model(m_star, eta, gamma, R_ref,
                                           gdf_edge, crs)
    m_sp = len(pos_sp)
    r_e_sp, l_xi_sp = sp_2pcf(pos_sp, gdf_edge, crs, rmax)
    xi_sp = _xi_mean(r_e_sp, l_xi_sp)
    r_e_sp = r_e_sp[:len(xi_sp)]
    gamma_sp, r0_sp, gamma_sp_err, _ = fit_gamma(r_e_sp, l_xi_sp)
    say(f"[{name}] SP: lambda={lamb_sp:.3f} L={L_sp} m_SP={m_sp:,} "
        f"gamma_SP={gamma_sp:.4f}")

    rho, n_rand = dilution(xi_data, r_edges, xi_sp, r_e_sp, m_sp)
    pos_mix, labels = build_mixture(pos_sp, n_rand, gdf_edge, crs)
    xi_f_mean, xi_f_std, pos_f, lab_f = final_realisations(
        pos_mix, labels, N, gdf_edge, crs, r_edges, rmax, n_real=n_real)
    n_sp_final = int((lab_f == 0).sum())
    say(f"[{name}] rho={rho:.4f} n_rand={n_rand:,} -> final {N:,} "
        f"({n_sp_final:,} SP, {N - n_sp_final:,} random)")

    return dict(
        name=name, N=N, R_ref_m=R_ref, rmax=rmax,
        gamma=gamma, gamma_err=gamma_err, r0_m=r0, r0_err_m=r0_err,
        eta=eta, silhouette=float(sil_means.max()),
        m_min=int(m_arr.min()), m_max=int(m_arr.max()), m_star=m_star,
        lamb=lamb_sp, L=L_sp, m_SP=m_sp,
        gamma_SP=gamma_sp, gamma_SP_err=gamma_sp_err,
        rho=rho, n_rand=n_rand, final_SP=n_sp_final,
        final_random=N - n_sp_final,
        # arrays for the figures
        gdf_edge=gdf_edge, coords=coords, marks=marks,
        r_edges=r_edges, l_xi=l_xi, xi_data=xi_data,
        k_values=k_values, sil_means=sil_means, sil_stds=sil_stds,
        m_arr=m_arr, int_arr=int_arr, xi_cross_star=scan[i_star]["xi"],
        r_cross_star=scan[i_star]["r_edges"],
        r_e_sp=r_e_sp, xi_sp=xi_sp, r0_SP_m=r0_sp,
        xi_final_mean=xi_f_mean, xi_final_std=xi_f_std,
        pos_final=pos_f, labels_final=lab_f,
    )


#: scalar keys of the calibrate() result, in the order of the summary table
SUMMARY_KEYS = ["name", "N", "R_ref_m", "rmax", "gamma", "gamma_err", "r0_m",
                "r0_err_m", "eta", "silhouette", "m_min", "m_max", "m_star",
                "lamb", "L", "m_SP", "gamma_SP", "gamma_SP_err", "rho",
                "n_rand", "final_SP", "final_random"]
