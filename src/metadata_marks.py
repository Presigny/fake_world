"""
metadata_marks.py
=================
Briques pour les deux nouveaux chantiers "métadonnées" (juillet 2026) :

TODO 1 — Induire les métadonnées dans le modèle génératif
  On génère un SP *mixte* (fractal + fond aléatoire, cf build_sp_mixture),
  puis on attribue une population à chaque nœud par **densité locale** :
  population d'un nœud ∝ densité locale de points autour de lui (nombre de
  voisins dans un rayon, ou inverse de la distance au k-ième voisin). Ça
  encode « les grandes villes sont là où c'est dense » sans passer par
  l'arbre du tout — donc ça marche même sur le nuage final complet
  (fractal + fond), là où l'option « niveau » (population_from_level) ne
  s'applique qu'à la partie fractale.
  (Cette approche est désignée « Option C » dans les notes de travail.)

TODO 2 — Induire la métadonnée dans une nouvelle observable
  - profil de population cumulée M(r) : pour chaque point, la somme des
    métadonnées (populations) des points inclus dans un cercle de rayon r,
    en fonction de r ;
  - « 2PCF des gens » : on superpose à chaque point un nombre de nœuds
    proportionnel à sa métadonnée puis on mesure la 2PCF du nuage étendu.
    Implémentée exactement (superposition) et efficacement (estimateur de
    Landy-Szalay pondéré par les marques — équivalent sans le coût N²).

Toutes les coordonnées sont supposées normalisées dans [0,1]²
(cf synthese_utils.normalize_coords).
"""

from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import synthese_utils as su


# ===========================================================================
# TODO 1 — Densité locale → population
# ===========================================================================

def local_density_knn(xy, k=10):
    """Densité locale par l'inverse de la distance au k-ième voisin.

    ρ_k(i) = k / (π · d_k(i)²), où d_k(i) est la distance du point i à son
    k-ième plus proche voisin. Estimateur classique de densité, insensible
    au choix d'un rayon absolu.
    """
    d, _ = cKDTree(xy).query(xy, k=k + 1)          # k+1 : le 1er voisin est soi
    dk = d[:, -1]
    return k / (np.pi * np.maximum(dk, 1e-12) ** 2)


def local_density_radius(xy, h):
    """Densité locale par comptage des voisins dans un rayon h (soi exclu)."""
    tree = cKDTree(xy)
    counts = np.asarray(tree.query_ball_point(xy, r=h, return_length=True)) - 1
    return counts / (np.pi * h ** 2)


def default_radius(xy, k=10):
    """Rayon de comptage par défaut : distance k-NN médiane du nuage."""
    d, _ = cKDTree(xy).query(xy, k=k + 1)
    return float(np.median(d[:, -1]))


def pop_powerlaw(rho, beta, pop_total):
    """Mapping puissance : population ∝ ρ^β, normalisé à pop_total."""
    w = np.maximum(np.asarray(rho, dtype=float), 1e-12) ** beta
    return pop_total * w / w.sum()


def pop_rank_matching(rho, pop_empirical):
    """Appariement des rangs : le nœud le plus dense reçoit la plus grande
    population empirique, etc. La marginale des populations est donc
    exactement celle des données ; seul le *placement* vient du modèle.
    Si les tailles diffèrent, on échantillonne les quantiles empiriques."""
    rho = np.asarray(rho, dtype=float)
    p = np.sort(np.asarray(pop_empirical, dtype=float))[::-1]
    n = len(rho)
    if len(p) != n:                                 # quantiles interpolés
        q = np.linspace(0, 1, n)
        p = np.quantile(np.asarray(pop_empirical, dtype=float),
                        1 - q, method="linear")
    out = np.empty(n)
    out[np.argsort(rho)[::-1]] = p
    return out


def pop_split_sp_background(is_sp, pop_empirical, seed=0):
    """Attribution (c) — par le processus génératif lui-même (split SP/fond).

    Le mélange SP+fond sépare déjà les nœuds en deux familles au moment de la
    génération : les points *structurés* (issus du fractal SP, calibrés par
    m*) et le *fond aléatoire uniforme*. On attribue les |SP| plus grandes
    populations empiriques aux nœuds SP et les restantes au fond ; à
    l'intérieur de chaque groupe, l'attribution est **uniformément aléatoire**
    (aucune densité locale n'est utilisée — moyenner sur plusieurs seeds,
    l'attribution intra-groupe est stochastique).

    Ce modèle teste l'hypothèse « être un nœud structuré suffit » : le
    couplage induit ne peut venir que du fait que les nœuds SP sont plus
    agrégés que le fond, pas d'un lien densité→taille nœud par nœud comme
    dans les variantes (a) et (b).

    is_sp vient de build_sp_mixture(..., return_labels=True). Si les tailles
    diffèrent, la distribution empirique est échantillonnée par quantiles
    (comme pop_rank_matching). Retourne le vecteur des populations.
    """
    is_sp = np.asarray(is_sp, dtype=bool)
    n = len(is_sp)
    p = np.sort(np.asarray(pop_empirical, dtype=float))[::-1]
    if len(p) != n:                                 # quantiles interpolés
        q = np.linspace(0, 1, n)
        p = np.quantile(np.asarray(pop_empirical, dtype=float),
                        1 - q, method="linear")
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    n_sp = int(is_sp.sum())
    out[is_sp] = rng.permutation(p[:n_sp])          # les n_sp plus grandes → SP
    out[~is_sp] = rng.permutation(p[n_sp:])         # le reste → fond aléatoire
    return out


def calibrate_beta_on_zipf(rho, target_zipf, pop_total,
                           betas=np.linspace(0.25, 4.0, 16)):
    """Choisit β pour que l'exposant rang-taille (Zipf) de pop ∝ ρ^β colle à
    celui des données. Retourne (beta*, zipf(beta*))."""
    best = None
    for b in betas:
        z = su.fit_zipf_exponent(pop_powerlaw(rho, b, pop_total))
        err = abs(z - target_zipf) if np.isfinite(z) else np.inf
        if best is None or err < best[1]:
            best = (float(b), err, z)
    return best[0], best[2]


def delta_gamma(xy, pop, geom_norm, frac_high=0.2):
    """Couplage métadonnée–structure : Δγ = γ_hi − γ_lo (split par rang).

    Enrobe synthese_utils.marked_clustering_2pcf ; Δγ > 0 signifie que les
    nœuds à forte population sont plus agrégés que les autres (comme dans
    les données réelles). Retourne (dict complet, Δγ)."""
    res = su.marked_clustering_2pcf(xy, pop, geom_norm, frac_high=frac_high)
    g_hi, g_lo = res["gamma_hi"], res["gamma_lo"]
    dg = (g_hi - g_lo) if (np.isfinite(g_hi) and np.isfinite(g_lo)) else np.nan
    return res, dg


# ===========================================================================
# TODO 2 — Observable 1 : profil de population cumulée M(r)
# ===========================================================================

def cumulative_population_profile(xy, pop, r_grid, include_self=False):
    """M_i(r) = Σ_{j : d_ij ≤ r} pop_j pour chaque point i, sur la grille r_grid.

    Retourne un tableau (n_points, n_r) ; la moyenne sur l'axe 0 donne le
    profil moyen M(r). En log-log, la pente de M(r) est une « dimension
    massique » : l'analogue pondéré-par-la-population de la dimension de
    corrélation D₂ (M(r) ∝ r^D pour une distribution auto-similaire).

    Coût O(N² log N) — réservé aux nuages ≤ ~10 000 points.
    """
    d_sorted, order = _neighbour_sort(xy, include_self)
    return _profile_from_sort(d_sorted, order, np.asarray(pop, float), r_grid)


def _neighbour_sort(xy, include_self=False):
    """Trie les voisins de chaque point par distance croissante.

    Retourne (d_sorted, order). Ne dépend QUE des positions : à réutiliser
    tel quel quand seules les marques changent (null par permutation), ce qui
    évite de refaire le cdist O(N²) à chaque tirage.
    """
    xy = np.asarray(xy, dtype=float)
    d = cdist(xy, xy)
    if not include_self:
        np.fill_diagonal(d, np.inf)
    order = np.argsort(d, axis=1)
    return np.take_along_axis(d, order, axis=1), order


def _profile_from_sort(d_sorted, order, marks, r_grid):
    """Profil cumulé (n_points, n_r) à partir d'un tri de voisins précalculé.

    M_i(r) = Σ_{j : d_ij ≤ r} marks_j est une fonction **en escalier** : elle
    saute à chaque voisin franchi et reste constante entre deux voisins. On
    l'évalue exactement par recherche dichotomique. (Une interpolation
    linéaire entre voisins successifs anticiperait la marque du voisin suivant
    et surestimerait M(r) jusqu'à ~30 % aux petits r, où les sauts sont rares —
    et casserait l'identité ⟨M(r)⟩ = ⟨m⟩⟨N(r)⟩ + Cov(m_j, n_j(r)).)
    """
    m_sorted = np.cumsum(marks[order], axis=1)
    out = np.empty((len(order), len(r_grid)))
    for i in range(len(order)):
        # nb de voisins à distance ≤ r ; 0 → aucune marque cumulée
        idx = np.searchsorted(d_sorted[i], r_grid, side="right")
        out[i] = np.where(idx > 0, m_sorted[i][np.maximum(idx - 1, 0)], 0.0)
    return out


def mass_dimension(r_grid, M_mean, a=2, b=None):
    """Pente log-log de M(r) (fit sur [a:b]) : M(r) ~ r^D_M.

    ⚠️ D_M est dominé par la géométrie des *points* et ne détecte quasiment
    pas le couplage marque↔position : mélanger les marques le laisse inchangé
    (à ~0.5σ), alors que changer les positions le déplace nettement. Pour le
    couplage, utiliser normalized_population_profile().
    """
    if b is None:
        b = len(r_grid) - 2
    valid = (M_mean[a:b] > 0) & (r_grid[a:b] > 0)
    if valid.sum() < 3:
        return np.nan
    return float(np.polyfit(np.log(r_grid[a:b][valid]),
                            np.log(M_mean[a:b][valid]), 1)[0])


def normalized_population_profile(xy, pop, r_grid, include_self=False):
    """M_norm(r) = ⟨M(r)⟩ / (⟨pop⟩·⟨N(r)⟩) — isole le couplage marque↔position.

    ⟨M(r)⟩ se décompose en ⟨pop⟩·⟨N(r)⟩ + Cov(pop_j, n_j(r)) : le premier terme
    est de la géométrie pure (il ne dépend que des positions), le couplage
    n'entre que par la covariance. Diviser par ⟨pop⟩·⟨N(r)⟩ élimine donc la
    géométrie et ne laisse que le couplage :

      - M_norm(r) = 1 : la population dans un rayon r est exactement celle
        attendue vu le nombre de voisins → aucun couplage (c'est la valeur du
        null « marques mélangées », par construction) ;
      - M_norm(r) > 1 : les nœuds sont entourés de *plus* de population
        qu'attendu à l'échelle r → les fortes populations se regroupent ;
      - M_norm(r) < 1 : elles s'évitent.

    Contrairement à la pente D_M, cette observable **discrimine** le couplage
    (sur la Belgique : pic à ~1.6 vers r≈0.012, null plat à 1.00).
    Retourne (M_norm, M_mean, N_mean).
    """
    pop = np.asarray(pop, dtype=float)
    d_sorted, order = _neighbour_sort(xy, include_self)
    M = _profile_from_sort(d_sorted, order, pop, r_grid).mean(axis=0)
    N = _profile_from_sort(d_sorted, order, np.ones(len(pop)), r_grid).mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        M_norm = np.where(N > 0, M / (pop.mean() * N), np.nan)
    return M_norm, M, N


def covariance_profile(M_mean, N_mean, pop):
    """Cov(pop_j, n_j(r)) — le terme de couplage **brut**, non normalisé.

    Se déduit exactement des profils déjà calculés, par l'identité
    ⟨M(r)⟩ = ⟨pop⟩·⟨N(r)⟩ + Cov(pop_j, n_j(r)) : aucun calcul supplémentaire.

    C'est le couplage « nu » : M_norm − 1 n'en est que la version rééchelonnée,
    Cov/(⟨pop⟩·⟨N(r)⟩). Les deux contiennent donc la *même* information ; leurs
    défauts sont complémentaires, d'où l'intérêt de tracer les deux :

      - **Cov** est lisse aux deux bouts (→ 0 quand r→0 car n_j = 0 partout, et
        quand r→∞ car n_j = N−1 partout) mais elle est **dimensionnée**
        (habitants × voisins) : sa valeur dépend de l'unité des marques et de
        l'échelle du système, donc elle n'est **pas comparable** d'un dataset à
        l'autre. Sur la Belgique son pic vaut 8.4× celui de l'Autriche, dont
        3.9× ne sont que la taille moyenne des villes (15.7 k contre 4.0 k).
      - **M_norm** est sans dimension, comparable et lisible (« +74 % de
        population autour »), et suit la convention du domaine — même
        construction que ξ(r) = DD/RR − 1 ou que k_mm = ⟨m_i m_j⟩/⟨m⟩² — mais
        elle vaut 0/0 aux petits r, où la division amplifie le bruit.
    """
    return np.asarray(M_mean, float) - np.mean(pop) * np.asarray(N_mean, float)


def shuffled_null_profile(xy, pop, r_grid, n_shuffle=30, seed=0,
                          include_self=False):
    """Null de M_norm(r) : marques permutées, moyenné sur n_shuffle tirages.

    Une seule permutation est très bruitée quand la distribution des marques a
    une queue lourde (Zipf : Bruxelles ≈ 15 % de la population belge), d'où la
    moyenne sur plusieurs tirages. Les positions étant fixes, le tri des
    voisins est calculé une fois pour toutes.

    Retourne (norm_mean, norm_std, DM_mean, DM_std).
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
# TODO 2 — Observable 2 : la « 2PCF des gens »
# ===========================================================================

def superpose_people(xy, pop, n_total=6000, jitter=0.0, seed=0):
    """Superpose à chaque point n_i ∝ pop_i nœuds (échantillonnage multinomial).

    Implémentation *littérale* de l'idée : le nuage retourné contient
    ~n_total points, chaque point original étant répliqué proportionnellement
    à sa métadonnée. `jitter` > 0 ajoute un bruit gaussien (utile seulement
    pour la visualisation ; laisser 0 pour la 2PCF).
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
    """2PCF « des gens » : estimateur de Landy-Szalay pondéré par les marques.

    Équivalent exact (en espérance) de la superposition de n_i ∝ pop_i nœuds
    sur chaque point, sans le coût quadratique du nuage étendu :

        DD_w(r) = Σ_{i<j, d_ij∈bin} w_i·w_j / Σ_{i<j} w_i·w_j
        DR_w(r) = Σ_{i,k, d_ik∈bin} w_i     / (Σ_i w_i · N_R)
        RR (r)  = inchangé (catalogue aléatoire non marqué)

        ξ_people(r) = (DD_w − 2·DR_w + RR) / RR

    avec w_i = pop_i / ⟨pop⟩. Les paires « clones du même point » (d = 0)
    sont exclues, comme dans l'estimateur standard. Retourne (r_centers, xi).
    """
    xy = np.asarray(xy, dtype=float)
    w = np.asarray(pop, dtype=float)
    w = w / w.mean()

    iu = np.triu_indices(len(xy), k=1)
    d_pairs = cdist(xy, xy)[iu]
    w_pairs = w[iu[0]] * w[iu[1]]
    pos = d_pairs > 0
    d_pairs, w_pairs = d_pairs[pos], w_pairs[pos]

    # Même grille radiale que compute_2pcf_norm (rmin ~ ½ NN moyen, bins log)
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
    # Test de fumée : Belgique, densité locale + les deux observables.
    name = "Belgium Cities"
    ds = su.DATASETS[name]
    print(f"=== Test de fumée metadata_marks : {name} ===")
    ld = su.load_dataset(ds)
    xy, pop = ld["xy_norm"], ld["pop_data"]

    rho = local_density_knn(xy, k=10)
    zipf_data = su.fit_zipf_exponent(pop)
    beta, zipf_model = calibrate_beta_on_zipf(rho, zipf_data, pop.sum())
    print(f"Zipf données={zipf_data:.2f}  β*={beta:.2f}  Zipf modèle={zipf_model:.2f}")

    r_grid = np.logspace(-2.5, -0.3, 12)
    M_norm, M, N = normalized_population_profile(xy, pop, r_grid)
    d_m = mass_dimension(r_grid, M)
    null_m, null_s, dm_null, dm_null_s = shuffled_null_profile(
        xy, pop, r_grid, n_shuffle=10)
    i_pk = int(np.nanargmax(M_norm[1:-1]) + 1)
    print(f"D_M = {d_m:.2f}  vs null {dm_null:.2f}±{dm_null_s:.2f} "
          f"({abs(d_m - dm_null) / max(dm_null_s, 1e-9):.1f} sigma → non discriminant)")
    print(f"M_norm pic = {M_norm[i_pk]:.2f} à r={r_grid[i_pk]:.4f} "
          f"(null plat à {np.nanmean(null_m):.2f} → c'est ça, le couplage)")

    r_p, xi_p = compute_2pcf_people(xy, pop, ld["geom_norm"])
    g, _ = su.fit_powerlaw_norm(r_p, xi_p)
    print(f"gamma_people = {g:.3f}  (gamma_noeuds = {ld['gamma_fit']:.3f})")
