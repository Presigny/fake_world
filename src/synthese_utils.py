"""
synthese_utils.py
=================
Module consolidé pour la synthèse du projet "Synthetic World".

Il rassemble en un seul endroit la pipeline éprouvée, dispersée jusqu'ici
dans plusieurs notebooks (`Manring/`, `BSTPP_comparison/`,
`notebooks/single_sp_cross_2pcf/`) :

  - configuration des datasets réels (DATASETS)
  - modèle Soneira-Peebles (réexporté depuis src/ et sp_helpers.py)
  - simulateurs "Manring" légers, pur numpy (LGCP, Hawkes, Neyman-Scott, Cox)
  - 2PCF de Landy-Szalay normalisée + fit loi de puissance + divergence JSD
  - chargement d'un dataset (frontière, normalisation, 2PCF des données)

et il ajoute les briques nécessaires aux prochaines étapes :

  - run_multi_instances() ......... TODO stochasticité (moyenne ± écart-type)
  - compare_at_constant_N() ....... TODO comparaison SP vs Manring à N constant

Toutes les coordonnées "modèle" sont manipulées dans le carré normalisé
[0,1]² (via normalize_coords / normalize_geometry) pour que la 2PCF et la JSD
soient comparables entre datasets de tailles physiques différentes.

Auteur : synthèse assemblée à partir du travail de Charley Presigny / Emile Emery.
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
# Chemins : on retrouve src/ (qui contient aussi sp_helpers.py) quel que soit
# le CWD. Les données sont atteintes via les symlinks de data/ (→ fake_world,
# fkwrld_emile) : rien n'est dupliqué.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent   # racine fake_world_paper_july_2026/
SRC = Path(__file__).resolve().parent           # src/ (contient sp_helpers.py)
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import methods_two_point_correlation as mtpc           # noqa: E402
from sp_helpers import (                                # noqa: E402
    soneira_peebles_border, get_inside_points, build_sp_model,
    power_law, _random_in_territory,
)

SEED = 42

# Paramètres de la 2PCF (cohérents avec les notebooks de comparaison)
N_BINS = 18
N_RANDOM = 5000
N_RUNS_2PCF = 5


# ===========================================================================
# 1. Configuration des datasets réels
# ===========================================================================
# Les 4 datasets "principaux" (N < 5000, calcul cdist direct possible).
# gamma_data, r0_data_m : fit loi de puissance de la 2PCF des données.
# eta                   : facteur de branchement (Silhouette KMeans).
# m_star                : nombre de points "structurés" (scan cross-2PCF).
# R_ref_m               : rayon de référence du SP (~ demi-diagonale du pays).
#
# Les paramètres "Manring dérivés" (ls_s, sigma, alpha, amp_s, a0, mu_hawkes)
# ne sont PAS stockés ici : ils dépendent de l'échelle métrique du territoire
# et sont calculés au chargement par finalize_manring_params().

DATASETS = {
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
        path_border=ROOT / "data" / "clean_transport_dataset" / "France" / "france_nat_earth.geojson",
        path_csv=ROOT / "data" / "clean_transport_dataset" / "France" / "postes-electriques-rte.csv",
        threshold_variable="population_density", threshold=1,
        N=4932, area_km2=538401,
        gamma_data=0.931, r0_data_m=12427, rmax=160_000,
        eta=6, m_star=3661, R_ref_m=496_900,
    ),
    "Gares France": dict(
        name="France", crs="EPSG:2154",
        path_border=ROOT / "data" / "clean_transport_dataset" / "France" / "france_nat_earth.geojson",
        path_csv=ROOT / "data" / "clean_transport_dataset" / "France" / "noeuds_route500_gares.csv",
        threshold_variable="population_density", threshold=1,
        N=3470, area_km2=538401,
        gamma_data=0.643, r0_data_m=61622, rmax=160_000,
        eta=6, m_star=2896, R_ref_m=496_900,
    ),
    # ── Gros datasets : N >> 10 000 → sous-échantillonnés à N_data_cap au
    #    chargement (cdist O(N²) impraticable au-delà ; les modèles hors-SP
    #    ne passent pas l'échelle au-delà de ~10 000 nœuds). ──────────────────
    "ENEDIS France": dict(
        name="France", crs="EPSG:2154",
        path_border=ROOT / "data" / "clean_transport_dataset" / "France" / "france_nat_earth.geojson",
        path_csv=ROOT / "data" / "clean_transport_dataset" / "France" / "postes-electriques-enedis.csv",
        threshold_variable="population_density", threshold=1,
        N=912_982, area_km2=538_401, N_data_cap=10_000,
        gamma_data=0.954, r0_data_m=7_995, rmax=160_000,
        eta=6, m_star=235_268, R_ref_m=496_900,
    ),
    "France Communes": dict(
        name="France", crs="EPSG:2154",
        path_border=ROOT / "data" / "clean_transport_dataset" / "France" / "france_nat_earth.geojson",
        path_csv=ROOT / "data" / "clean_transport_dataset" / "France" / "noeuds_route500_commune.csv",
        threshold_variable="population_density", threshold=1,
        N=34_269, area_km2=538_401, N_data_cap=10_000,
        gamma_data=0.254, r0_data_m=292, rmax=50_000,
        eta=2, m_star=24_647, R_ref_m=496_900,
    ),
    "France Routes": dict(
        name="France", crs="EPSG:2154",
        path_border=ROOT / "data" / "clean_transport_dataset" / "France" / "france_nat_earth.geojson",
        path_csv=ROOT / "data" / "clean_transport_dataset" / "France" / "noeuds_route500_routes.csv",
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
        gamma_data=0.873, r0_data_m=6_690, rmax=100_000,
        eta=12, m_star=56_331, R_ref_m=117_700,
    ),
}

# Datasets "petits" (N < 5000) : tournent en entier, rapides.
SMALL_DATASETS = ["Austria Cities", "Belgium Cities", "RTE France", "Gares France"]

# Cap dur sur le nombre de nœuds pour les modèles hors-SP (cdist O(N²)).
N_CAP_MODELS = 10_000


def finalize_manring_params(ds, scale_m, amp_s=1.5):
    """Dérive les paramètres des processus ponctuels à partir des params SP.

    Reproduit (en version "défauts documentés") la calibration des notebooks
    Manring : ls_s = r0_norm, sigma = r0_norm/2, alpha = m*/N, etc. La
    calibration fine de amp_s par matching de moments est optionnelle ; la
    valeur 1.5 reste un bon défaut.
    """
    r0_norm = ds["r0_data_m"] / scale_m
    alpha = float(np.clip(ds["m_star"] / ds["N"], 0.1, 0.9))
    ds.update(
        r0_norm=r0_norm,
        ls_s=max(r0_norm, 1e-3),
        sigma=max(r0_norm / 2.0, 1e-3),
        alpha=alpha,
        amp_s=amp_s,
        beta=80.0,
        a0=float(np.log(ds["N"]) - amp_s ** 2 / 2.0),
        a0_cox=float(np.log(ds["N"]) - amp_s ** 2 / 2.0 + np.log(max(1 - alpha, 1e-6))),
        mu_hawkes=float(ds["N"] * (1 - alpha)),
    )
    return ds


# ===========================================================================
# 2. Normalisation, frontière, 2PCF
# ===========================================================================

def normalize_coords(xy_metric, bounds_metric):
    """Mappe des coordonnées métriques dans [0,1]² (échelle isotrope)."""
    xmin, ymin, xmax, ymax = bounds_metric
    scale = max(xmax - xmin, ymax - ymin)
    x_norm = (xy_metric[:, 0] - xmin) / scale
    y_norm = (xy_metric[:, 1] - ymin) / scale
    return np.column_stack([x_norm, y_norm]), scale


def normalize_geometry(gdf_edge, bounds_metric):
    """Applique la même transformation à la frontière du territoire."""
    xmin, ymin, xmax, ymax = bounds_metric
    scale = max(xmax - xmin, ymax - ymin)
    geom = unary_union(gdf_edge.geometry)
    geom_shifted = affinity.translate(geom, xoff=-xmin, yoff=-ymin)
    return affinity.scale(geom_shifted, xfact=1.0 / scale, yfact=1.0 / scale,
                          origin=(0, 0))


def rejection_sample_in_boundary(n, geom_norm, rng, max_factor=8):
    """Tire n points uniformes à l'intérieur du polygone normalisé."""
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
    """2PCF de Landy-Szalay en coordonnées normalisées (bins log).

    Retourne (r_centers, xi) avec xi = (DD - 2DR + RR)/RR.
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
    """Fit ξ(r) = (r/r0)^(-gamma). Retourne (gamma, r0) ou (None, None)."""
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
    """Divergence de Jensen-Shannon entre 2PCF données et modèle.

    g(r) = max(xi(r)+1, eps) pondéré par r (volume annulaire en log) sert de
    densité de paires. JSD = 0 → reproduction parfaite de la 2PCF.
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
# 3. Simulateurs "Manring" légers (pur numpy, confinés à la frontière)
# ===========================================================================

def sample_gp_2d(grid_size, length_scale, amplitude, rng):
    """Échantillonne un champ gaussien 2D (kernel RBF) sur une grille."""
    x = np.linspace(0, 1, grid_size)
    xx, yy = np.meshgrid(x, x)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    sq_dist = cdist(pts, pts, "sqeuclidean")
    K = amplitude ** 2 * np.exp(-sq_dist / (2 * length_scale ** 2))
    K += 1e-6 * np.eye(len(pts))
    L = np.linalg.cholesky(K)
    z = L @ rng.randn(len(pts))
    return x, z.reshape(grid_size, grid_size)


def simulate_poisson(N_target, geom_norm, seed=None):
    """Processus de Poisson homogène (référence ξ(r) ≈ 0)."""
    rng = np.random.default_rng(seed)
    return rejection_sample_in_boundary(N_target, geom_norm, rng)


def simulate_lgcp(N_target, ls_s, amp_s, a0, geom_norm, grid_size_s=25, seed=None):
    """Log-Gaussian Cox Process confiné à la frontière."""
    from shapely.vectorized import contains
    rng_ = np.random.RandomState(seed)
    _, fs = sample_gp_2d(grid_size_s, ls_s, amp_s, rng_)
    lam_max = np.exp(np.max(fs) + a0)
    bx0, by0, bx1, by1 = geom_norm.bounds
    collected = []
    for _ in range(200):
        n_prop = max(int(lam_max * 5), 10 * N_target)
        xs = rng_.uniform(bx0, bx1, n_prop)
        ys = rng_.uniform(by0, by1, n_prop)
        xi_idx = np.clip((xs * grid_size_s).astype(int), 0, grid_size_s - 1)
        yi_idx = np.clip((ys * grid_size_s).astype(int), 0, grid_size_s - 1)
        lam_pts = np.exp(fs[yi_idx, xi_idx] + a0)
        keep = rng_.uniform(0, lam_max, n_prop) < lam_pts
        xs, ys = xs[keep], ys[keep]
        inside = contains(geom_norm, xs, ys)
        if inside.sum() > 0:
            collected.append(np.column_stack([xs[inside], ys[inside]]))
        if sum(len(c) for c in collected) >= N_target * 3:
            break
    pts = np.vstack(collected) if collected else np.zeros((0, 2))
    if len(pts) >= N_target:
        return pts[rng_.choice(len(pts), size=N_target, replace=False)]
    return pts


def simulate_hawkes_spatial(N_target, mu, alpha, beta, sigma, geom_norm,
                            seed=None, max_iter=50):
    """Processus de Hawkes spatial auto-excitant (cascade de Poisson clusters)."""
    from shapely.vectorized import contains
    rng_ = np.random.RandomState(seed)
    T = max(10.0, N_target / max(mu, 0.1) * 3)
    events = []
    for _it in range(max_iter):
        events = []
        rng_bg = np.random.default_rng((seed or 0) + 1000 + _it)
        n_bg_want = rng_.poisson(mu * T)
        bg_xy = rejection_sample_in_boundary(max(n_bg_want, 1), geom_norm, rng_bg)
        bg_t = np.sort(rng_.uniform(0, T, len(bg_xy)))
        queue = [(bg_xy[i, 0], bg_xy[i, 1], bg_t[i]) for i in range(len(bg_xy))]
        events.extend(queue)
        while queue:
            px, py, pt = queue.pop(0)
            n_off = rng_.poisson(alpha)
            if n_off == 0:
                continue
            child_t = pt + rng_.exponential(1.0 / beta, n_off)
            child_x = px + rng_.normal(0, sigma, n_off)
            child_y = py + rng_.normal(0, sigma, n_off)
            in_time = child_t < T
            if in_time.sum() == 0:
                continue
            cx, cy, ct = child_x[in_time], child_y[in_time], child_t[in_time]
            inside = contains(geom_norm, cx, cy)
            for i in np.where(inside)[0]:
                events.append((cx[i], cy[i], ct[i]))
                queue.append((cx[i], cy[i], ct[i]))
        if len(events) >= N_target:
            evs = np.array(events)
            return evs[rng_.choice(len(evs), size=N_target, replace=False), :2]
        mu = mu * N_target / max(len(events), 1) * 1.5
    evs = np.array(events) if events else np.zeros((0, 3))
    return evs[:N_target, :2]


def simulate_neyman_scott(N_target, alpha, sigma, geom_norm, seed=None):
    """Processus de Neyman-Scott / Thomas (clusters gaussiens), purement spatial."""
    from shapely.vectorized import contains
    rng = np.random.default_rng(seed)
    mu_c = max(N_target / max(alpha, 0.5), 1.0)
    all_pts = []
    for _ in range(20):
        n_centers = max(1, int(rng.poisson(mu_c)))
        centers = rejection_sample_in_boundary(n_centers, geom_norm, rng)
        for cx, cy in centers:
            n_off = int(rng.poisson(alpha))
            if n_off == 0:
                continue
            ox = cx + rng.normal(0, sigma, n_off)
            oy = cy + rng.normal(0, sigma, n_off)
            ins = contains(geom_norm, ox, oy)
            if ins.sum() > 0:
                all_pts.append(np.column_stack([ox[ins], oy[ins]]))
        if sum(len(b) for b in all_pts) >= N_target:
            break
        mu_c *= 1.5
    pts = np.vstack(all_pts) if all_pts else np.zeros((0, 2))
    if len(pts) >= N_target:
        return pts[rng.choice(len(pts), size=N_target, replace=False)]
    return pts


# ===========================================================================
# 4. Modèle Soneira-Peebles (mélange SP + fond aléatoire) — réexport + wrapper
# ===========================================================================

def build_sp_mixture(ds, gdf_edge, bounds, geom_norm, r_data, xi_data,
                     N_target, seed=SEED, verbose=False, return_labels=False):
    """Construit un nuage SP calibré + fond aléatoire et renvoie ses coords.

    Pipeline (cf sp_helpers) :
      1. SP complet avec m_SP ≥ m* nœuds dans le territoire
      2. ajout de n_rand points uniformes pour coller l'amplitude de la 2PCF
         (ρ = ⟨ξ_data/ξ_SP⟩, n_rand = m_SP·(1/√ρ − 1))
      3. suppression aléatoire pour atteindre exactement N_target

    Retourne xy_norm (coordonnées normalisées dans [0,1]²). Avec
    return_labels=True, retourne (xy_norm, is_sp) où is_sp[i] indique si le
    nœud i vient du fractal SP (True) ou du fond aléatoire (False) — le flux
    RNG est identique, donc le nuage est le même qu'avant à seed égal.
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
# 5. Chargement d'un dataset
# ===========================================================================

def load_dataset(ds, compute_data_2pcf=True, verbose=True):
    """Charge frontière + points, normalise, calcule la 2PCF des données.

    Remplit aussi les paramètres Manring dérivés (finalize_manring_params).
    Retourne un dict 'ld' réutilisable par tous les builders.
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
    # Métadonnée (population / densité) alignée sur les points, pour la TODO 3.
    mvar = ds["threshold_variable"]
    pop_full = (gdf_proj[mvar].to_numpy(dtype=float)
                if mvar in gdf_proj.columns else None)

    # Sous-échantillonnage des gros datasets (cdist O(N²) impraticable).
    N_cap = ds.get("N_data_cap")
    if N_cap is not None and N_full > N_cap:
        rng_sub = np.random.default_rng(42)
        sub_idx = rng_sub.choice(N_full, size=N_cap, replace=False)
        xy_metric = xy_metric_full[sub_idx]
        pop_data = pop_full[sub_idx] if pop_full is not None else None
        N_real = N_cap
        if verbose:
            print(f"  N_full={N_full:,} → sous-échantillonné à {N_cap:,}")
    else:
        xy_metric = xy_metric_full
        pop_data = pop_full
        N_real = N_full

    bounds = gdf_edge.total_bounds
    xy_norm, scale_m = normalize_coords(xy_metric, bounds)
    geom_norm = normalize_geometry(gdf_edge, bounds)
    finalize_manring_params(ds, scale_m)

    if verbose:
        print(f"  N={N_real:,}  échelle={scale_m / 1e3:.1f} km")

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
    """2PCF d'un extrait de N_target points de données (None si N_target ≥ N_real).

    Sert à comparer données et modèles **au même N** : on tire un sous-échantillon
    des données réelles puis on recalcule sa 2PCF.
    """
    if N_target >= ld["N_real"]:
        return ld["xy_norm"], ld["r_data"], ld["xi_data"]
    rng = np.random.default_rng(seed)
    idx = rng.choice(ld["N_real"], size=N_target, replace=False)
    xy = ld["xy_norm"][idx]
    r, xi = compute_2pcf_norm(xy, ld["geom_norm"])
    return xy, r, xi


# ===========================================================================
# 6. NOUVEAU — Stochasticité : plusieurs instances, moyenne ± écart-type
# ===========================================================================

def run_multi_instances(builder, n_instances=10, base_seed=0):
    """Exécute `builder(seed) -> xy_norm` plusieurs fois et agrège la 2PCF.

    `builder(seed)` doit retourner un tableau (n,2) de coordonnées normalisées
    (p.ex. une lambda fermant sur build_sp_mixture, simulate_lgcp, …).

    Retourne dict(r, xi_mean, xi_std, xi_all, gammas) où :
      - xi_mean/xi_std : moyenne et écart-type de ξ(r) sur les instances
      - xi_all         : (n_instances, n_bins)
      - gammas         : pente fittée pour chaque instance
    """
    # grille de référence : celle de la 1re instance
    xi_all, gammas, r_ref = [], [], None
    for k in range(n_instances):
        xy = builder(base_seed + k)
        # toutes les instances ont la même frontière donc ~même grille r ;
        # on aligne par interpolation sur r_ref pour être robuste.
        r_k, xi_k = run_multi_instances._last_2pcf
        if r_ref is None:
            r_ref = r_k
            xi_all.append(xi_k)
        else:
            xi_all.append(np.interp(
                np.log10(np.clip(r_ref, 1e-12, None)),
                np.log10(np.clip(r_k, 1e-12, None)),
                np.nan_to_num(xi_k), left=np.nan, right=np.nan))
        g, _ = fit_powerlaw_norm(r_k, xi_k)
        if g is not None:
            gammas.append(g)
    xi_all = np.array(xi_all)
    return dict(
        r=r_ref,
        xi_mean=np.nanmean(xi_all, axis=0),
        xi_std=np.nanstd(xi_all, axis=0),
        xi_all=xi_all,
        gammas=np.array(gammas),
    )


def make_2pcf_builder(model_fn, geom_norm):
    """Enrobe un simulateur pour qu'il renvoie xy_norm ET stocke sa 2PCF.

    Utilisé conjointement avec run_multi_instances : le builder renvoie les
    coordonnées, et la 2PCF calculée est mémorisée dans
    run_multi_instances._last_2pcf pour éviter de la recalculer.
    """
    def builder(seed):
        xy = model_fn(seed)
        run_multi_instances._last_2pcf = compute_2pcf_norm(xy, geom_norm)
        return xy
    return builder


# ===========================================================================
# 6bis. Modèles BSTPP inférés par SVI (étalon-or, params depuis les données)
# ===========================================================================
# On réutilise les modèles LGCP / Hawkes / Cox Hawkes déjà ajustés par
# inférence variationnelle dans BSTPP_comparison/ (fichiers .pkl). C'est la
# calibration *rigoureuse* (bayésienne, sans heuristique) — contrairement aux
# simulateurs "défauts" ci-dessus (amp_s=1.5 figé…) qui désavantageaient les PP.

BSTPP_CAP = 2000                       # N sur lequel BSTPP a été entraîné
BSTPP_RESULTS = ROOT / "data" / "bstpp_results"   # symlink → fake_world/BSTPP_comparison
BSTPP_NAMES = ("LGCP", "Hawkes", "Cox Hawkes")


def _bstpp_xy(ld, cap=BSTPP_CAP, seed=123):
    """Reconstruit le sous-échantillon d'entraînement BSTPP (même seed=123)."""
    xy = ld["xy_norm"]
    if len(xy) > cap:
        rng = np.random.default_rng(seed)
        xy = xy[rng.choice(len(xy), size=cap, replace=False)]
    return xy


def load_bstpp_models(name, ld, T_max=365.0, seed=42):
    """Recharge les 3 modèles BSTPP (LGCP/Hawkes/Cox Hawkes) inférés par SVI.

    Reconstruit chaque modèle avec le même data_df d'entraînement + le domaine
    [0,1]², puis charge les samples postérieurs depuis le .pkl. Retourne
    {nom: model | None}. Import de bstpp/jax fait ici (paresseux).
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
            print(f"  [BSTPP {mname}] échec reload : {e}")
            out[mname] = None
    return out


def bstpp_simulate_pool(model, geom_norm, n_sim=6, seed=SEED,
                        cap_per_sim=8000):
    """Simule n_sim patterns depuis la postérieure, filtre dans la frontière.

    Retourne un pool de coordonnées normalisées (empilé). simulate() tire un
    processus ponctuel aléatoire depuis l'intensité postérieure moyenne → les
    tirages diffèrent d'un appel à l'autre.
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


def build_bstpp_pools(name, ld, n_sim=6, seed=SEED):
    """Recharge + simule les 3 modèles BSTPP → {nom: pool_xy_norm}."""
    models = load_bstpp_models(name, ld)
    pools = {}
    for mname, model in models.items():
        pools[mname] = (bstpp_simulate_pool(model, ld["geom_norm"], n_sim, seed)
                        if model is not None else np.zeros((0, 2)))
    return pools


def _bstpp_realization(model, geom_norm, N, seed):
    """Un tirage indépendant d'un modèle BSTPP, sous-échantillonné à N."""
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
    """Aligne une courbe (r_src, xi_src) sur la grille r_ref (interp log)."""
    return np.interp(np.log10(np.clip(r_ref, 1e-12, None)),
                     np.log10(np.clip(r_src, 1e-12, None)),
                     np.nan_to_num(xi_src), left=np.nan, right=np.nan)


def stochastic_bands(ds, ld, N, n_instances=10, seed=SEED, bstpp_models=None,
                     models=("SP", "Poisson", "LGCP", "Hawkes", "Cox Hawkes")):
    """TODO 2 — reproductibilité : pour CHAQUE modèle, n_instances réalisations
    indépendantes → bande 2PCF (moyenne ± σ), JSD par tirage (incertitude sur la
    JSD) et un scalaire de largeur de bande (coefficient de variation).

    Un modèle à bande étroite est plus *reproductible* → critère de sélection
    complémentaire à la JSD moyenne. Retourne {model: dict(r, xi_mean, xi_std,
    jsd_mean, jsd_std, cv)}.
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
# 7. NOUVEAU — Comparaison SP vs processus ponctuels à N constant
# ===========================================================================

def compare_at_constant_N(ds, ld, N_target=None, models=("SP", "Poisson",
                          "LGCP", "Hawkes", "Cox Hawkes"), seed=SEED,
                          bstpp_pools=None):
    """Compare plusieurs modèles à un même N (nombre de nœuds fixé).

    Principe (comparaison équitable) : on prend **un seul extrait de N_target
    points des données**, on calcule SA 2PCF, puis chaque modèle génère
    N_target points dans la même frontière et on mesure la JSD à cette 2PCF de
    référence. Données et modèles sont donc tous au même N.

    Calibration des processus ponctuels (LGCP / Hawkes / Cox Hawkes) : si
    `bstpp_pools` est fourni ({nom: pool_xy}), on utilise les **modèles BSTPP
    inférés par SVI** (étalon-or) — on sous-échantillonne leur pool simulé à
    N_target. Sinon on retombe sur les simulateurs heuristiques (amp_s=1.5…),
    moins équitables. SP et Poisson sont toujours générés directement.

    Retourne dict {model: dict(xy, r, xi, gamma, jsd, n)} + clé "data".
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

    # Référence = un extrait de N_target points des données, sa propre 2PCF.
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
            xy = _from_pool(pool)                       # modèle BSTPP (SVI)
        elif m == "LGCP":
            xy = simulate_lgcp(N_target, ds["ls_s"], ds["amp_s"], ds["a0"],
                               geom_norm, seed=seed)
        elif m == "Neyman-Scott":
            xy = simulate_neyman_scott(N_target, ds["alpha"] * 5, ds["sigma"],
                                       geom_norm, seed=seed)
        elif m == "Hawkes":
            xy = simulate_hawkes_spatial(N_target, ds["mu_hawkes"], ds["alpha"],
                                         ds["beta"], ds["sigma"], geom_norm,
                                         seed=seed)
        else:
            # Modèle BSTPP demandé mais pool indisponible → JSD NaN.
            out[m] = dict(xy=np.zeros((0, 2)), r=None, xi=None,
                          gamma=np.nan, jsd=np.nan, n=0)
            continue
        r_m, xi_m = compute_2pcf_norm(xy, geom_norm)
        g_m, _ = fit_powerlaw_norm(r_m, xi_m)
        jsd = (js_divergence(r_data, xi_data, r_m, xi_m)
               if r_data is not None else np.nan)
        out[m] = dict(xy=xy, r=r_m, xi=xi_m, gamma=g_m, jsd=jsd, n=len(xy))
    return out


# ===========================================================================
# 8. TODO 3 — Population / métadonnée (marks)
# ===========================================================================
# Idée : le SP est une *hiérarchie*. Si l'on garde l'arbre complet (et pas
# seulement les feuilles), les nœuds des premiers niveaux sont rares et
# centraux → grandes villes/hubs ; les feuilles sont nombreuses → petites
# unités. On attribue donc une population décroissante avec le niveau, ce qui
# reproduit naturellement une distribution type loi de puissance (Zipf), et on
# vérifie que la métadonnée *corrèle avec la structure spatiale* (les nœuds à
# forte population sont plus agrégés), comme dans les données réelles.

def soneira_peebles_tree(lamb, eta, L, R, center, seed=None):
    """Génère l'arbre SP *complet* (tous les niveaux) avec le niveau de chaque nœud.

    Contrairement à soneira_peebles_border (feuilles seulement), on accumule
    les nœuds de tous les niveaux 0..L-1. Retourne (positions (n,2), levels (n,)).
    Le niveau 0 = premiers enfants de la racine (peu nombreux).
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
    """Population ∝ pop_max / eta^(slope·niveau) — hiérarchie type Zipf."""
    return pop_max / (eta ** (slope * levels.astype(float)))


def build_sp_with_population(ds, gdf_edge, L=None, pop_max=1e6, slope=1.0,
                             max_nodes=12_000, seed=SEED):
    """Construit un SP hiérarchique marqué par une population (TODO 3).

    Retourne dict(xy_norm, pop, levels) restreint aux nœuds intérieurs au
    territoire. L est choisi pour ≳ m* nœuds, mais **plafonné** pour que la
    taille de l'arbre (Σ η^i) reste ≤ max_nodes — sinon η élevé (ex. Suisse
    η=12) fait exploser le nombre de nœuds. On sous-échantillonne au besoin.
    """
    crs = ds["crs"]
    eta, gamma = ds["eta"], ds["gamma_data"]
    lamb = eta ** (1.0 / (2.0 - gamma))
    if L is None:
        L = max(int(np.ceil(np.log(max(ds["m_star"], eta)) / np.log(eta))), 4)
    # Plafond : plus grand L tel que Σ_{i=1}^{L} η^i ≤ max_nodes
    L_cap, cum = 1, eta
    while L_cap < L and cum + eta ** (L_cap + 1) <= max_nodes:
        L_cap += 1
        cum += eta ** L_cap
    L = L_cap
    center = gdf_edge.centroid.get_coordinates().to_numpy()[0]

    pos, lvl = soneira_peebles_tree(lamb, eta, L, ds["R_ref_m"], center, seed)
    pop = population_from_level(lvl, eta, pop_max, slope)

    # Filtre intérieur
    s = gpd.GeoDataFrame(geometry=gpd.points_from_xy(pos[:, 0], pos[:, 1]),
                         crs=crs)
    inside = gpd.sjoin(s, gdf_edge, how="left",
                       predicate="within")["index_right"].notna().to_numpy()
    pos, pop, lvl = pos[inside], pop[inside], lvl[inside]

    # Sécurité : borne le nombre de nœuds (cdist O(N²)).
    if len(pos) > max_nodes:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(pos), size=max_nodes, replace=False)
        pos, pop, lvl = pos[keep], pop[keep], lvl[keep]

    bounds = gdf_edge.total_bounds
    xy_norm, _ = normalize_coords(pos, bounds)
    return dict(xy_norm=xy_norm, pop=pop, levels=lvl)


def fit_zipf_exponent(pop, a=1, b=None):
    """Exposant s de la loi rang-taille : population ∝ rang^(−s) (fit log-log)."""
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
    """Choisit le `slope` du SP hiérarchique dont l'exposant rang-taille colle
    à celui des données (au lieu d'un défaut). Retourne (slope, zipf_sp)."""
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
    """Hawkes spatial **marqué** : la marque (population) module l'auto-excitation
    (TODO 3, vrai concurrent du SP sur la population).

    Un nœud de forte population engendre **davantage** d'enfants proches
    (nombre ∝ marque/⟨marque⟩), qui héritent d'une marque réduite (petites
    unités). → les fortes populations sont spatialement corrélées entre elles,
    comme dans les données. Les marques du fond sont tirées de `marks_pool`
    (≈ distribution empirique des populations → calibré sur les données).

    Retourne (xy (n,2), marks (n,)).
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
        all_xy.append(bg); all_m.append(bg_m)
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

    xy = np.vstack(all_xy); marks = np.concatenate(all_m)
    if len(xy) > N_target:
        idx = rng.choice(len(xy), size=N_target, replace=False)
        xy, marks = xy[idx], marks[idx]
    return xy, marks


def split_by_population(xy, pop, frac_high=0.2):
    """Sépare les nœuds en (haute, basse) population **par rang**.

    Le split par rang (et non par seuil de valeur) garantit deux groupes
    non-vides même quand la métadonnée comporte beaucoup d'ex æquo (fréquent
    pour 'population_density' qui prend des valeurs discrètes).
    """
    n = len(pop)
    n_hi = max(1, min(n - 1, int(round(frac_high * n))))
    order = np.argsort(pop)[::-1]           # tri décroissant
    return xy[order[:n_hi]], xy[order[n_hi:]]


def _cap_pts(xy, max_pts, seed=SEED):
    """Sous-échantillonne un nuage à max_pts points (borne cdist O(N²))."""
    if len(xy) <= max_pts:
        return xy
    rng = np.random.default_rng(seed)
    return xy[rng.choice(len(xy), size=max_pts, replace=False)]


def marked_clustering_2pcf(xy, pop, geom_norm, frac_high=0.2, min_pts=20,
                           max_pts=6000):
    """2PCF des nœuds haute-pop vs basse-pop (corrélation métadonnée–structure).

    Retourne dict(r_hi, xi_hi, r_lo, xi_lo, gamma_hi, gamma_lo). Si la
    métadonnée corrèle avec la structure, les nœuds haute-population sont plus
    agrégés (gamma_hi > gamma_lo). Renvoie des gamma NaN si un groupe est trop
    petit (< min_pts). Chaque groupe est plafonné à max_pts (cdist O(N²)).
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


CACHE_DIR = ROOT / "data" / "synthese_cache"      # symlink → fake_world/Synthese/cache


def cache_slug(name):
    return name.lower().replace(" ", "_")


def load_cached(name):
    """Charge le .npz précalculé par compute_all.py pour un dataset (ou None)."""
    f = CACHE_DIR / f"{cache_slug(name)}.npz"
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    out = {}
    for k in d.files:
        v = d[k]
        if v.ndim == 0:                      # scalaire / chaîne stockés en 0-d
            out[k] = v.item()
        else:
            out[k] = v
    return out


def available_cached():
    """Liste des datasets présents dans le cache (dans l'ordre de DATASETS)."""
    return [n for n in DATASETS if (CACHE_DIR / f"{cache_slug(n)}.npz").exists()]


def mark_correlation_function(xy, marks, geom_norm=None, n_bins=18,
                              max_pts=6000, seed=SEED):
    """Fonction de corrélation de marque k_mm(r) (Stoyan) — couplage continu.

    Au lieu d'un split binaire haute/basse population, on pondère **chaque
    paire** par le produit des marques (populations). Pour les paires séparées
    d'une distance ~r :

        k_mm(r) = ⟨ m_i · m_j ⟩_{paires à r}  /  ⟨m⟩²

    Interprétation :
      - k_mm(r) = 1  → la marque est indépendante de la position (pas de couplage)
      - k_mm(r) > 1  → les fortes marques sont proches entre elles à l'échelle r
                       (la métadonnée corrèle avec la structure spatiale)
      - k_mm(r) < 1  → les fortes marques se repoussent

    Mesure purement intrinsèque (aucun catalogue aléatoire requis) : c'est un
    ratio sur les paires de données. Retourne (r_centers, k_mm).
    """
    marks = np.asarray(marks, dtype=float)
    if len(xy) > max_pts:                       # borne cdist O(N²)
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
    """Distances au plus proche voisin (1-NN) de chaque point."""
    if len(xy) < 2:
        return np.zeros(0)
    d, _ = cKDTree(xy).query(xy, k=2)
    return d[:, 1]


def _jsd_hist(p, q):
    """Divergence de Jensen-Shannon entre deux histogrammes (densités)."""
    eps = 1e-12
    p = np.asarray(p, float) + eps
    q = np.asarray(q, float) + eps
    p /= p.sum(); q /= q.sum(); m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(a * np.log(a / b))
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def nn_comparison(data_xy, model_xys, n_bins=25):
    """Compare la distribution des distances au plus proche voisin (métrique
    NEUTRE : le SP n'est pas calibré dessus, contrairement à la 2PCF).

    Retourne (dict de distributions {r, data, <modèle>}, dict de JSD par modèle).
    Toutes les distributions partagent les mêmes bins (dérivés des données).
    """
    dd = nn_distances(data_xy)
    if len(dd) < 3:
        return None, {}
    rmin, rmax = np.percentile(dd, 0.5), np.percentile(dd, 99.5)
    edges = np.logspace(np.log10(max(rmin, 1e-9)), np.log10(rmax), n_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    hist = lambda x: (lambda h: h / max(h.sum(), 1))(
        np.histogram(x, bins=edges)[0])
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


def sweep_constant_N(ds, ld, n_list, models=("SP", "Poisson", "LGCP",
                     "Hawkes", "Cox Hawkes"), seed=SEED, bstpp_pools=None):
    """Balaie N (TODO 1) : JSD de chaque modèle à plusieurs N constants.

    Réutilise les mêmes pools BSTPP (simulés une seule fois) à chaque N.
    Retourne dict(N=[...], jsd={model: [...]}) en ne gardant que les N ≤ N_real.
    """
    ns = sorted({min(n, ld["N_real"]) for n in n_list})
    jsd = {m: [] for m in models}
    for N in ns:
        res = compare_at_constant_N(ds, ld, N_target=N, models=models,
                                    seed=seed, bstpp_pools=bstpp_pools)
        for m in models:
            jsd[m].append(res[m]["jsd"])
    return dict(N=np.array(ns), jsd={m: np.array(v) for m, v in jsd.items()})


if __name__ == "__main__":
    # Petit test de fumée sur le plus petit dataset (Belgium, N=815).
    name = "Belgium Cities"
    ds = DATASETS[name]
    print(f"=== Test de fumée : {name} ===")
    ld = load_dataset(ds)
    res = compare_at_constant_N(ds, ld, models=("SP", "Poisson", "LGCP"))
    print(f"\nRéférence données : extrait de {res['data']['n']} points")
    print("Modèle        N      gamma    JSD")
    for m, r in res.items():
        if m == "data":
            continue
        g = f"{r['gamma']:.3f}" if r["gamma"] else "  -  "
        print(f"{m:<12} {r['n']:>5}   {g}   {r['jsd']:.4f}")
