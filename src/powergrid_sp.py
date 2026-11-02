"""
powergrid_sp.py
===============
Réseau de transport électrique construit sur le **modèle SP mixte** de ce repo.

On reprend le modèle de croissance de `TransmissionGrowthModel`
(Emery, Bercegol, Aumaître — « A Power Grid Growth Model Based on Rural-Urban
Differentiation »), et on ne remplace **que la partie spatiale** :

    modèle du papier            →  ce module
    ------------------------------------------------------------------
    urbain : décroissance       →  urbain : points du fractal SP
    exponentielle autour de 11     (`is_sp = True` de build_sp_mixture)
    centres-villes
    rural  : uniforme dans le   →  rural  : **identique** (le fond
    territoire                     aléatoire du mélange SP)
    connectivité                →  **identique** (plus proche voisin en
                                   rural, attachement préférentiel parmi
                                   les `nn_nb` plus proches en urbain,
                                   2ᵉ arête avec probabilité q)

Les métadonnées (population) sont attribuées par
`metadata_marks.pop_split_sp_background` — la variante « (c) » du repo :
les |SP| plus grandes populations empiriques vont aux nœuds structurés, le
reste au fond, **au hasard à l'intérieur de chaque groupe**. Elles fixent la
consommation de chaque poste ; la production reprend la distribution
empirique du registre RTE. Le bilan est équilibré par construction.

Les pertes Joule sont calculées par un load-flow continu (DC) identique à
celui de `scripts/build_energy_estimates.py` du papier, avec deux
corrections documentées dans le notebook 06 (solveur creux, et tensions
attribuées par rang de flux au lieu d'une file triée arbitraire).

Auteur : Emile Emery, août 2026.
"""

from pathlib import Path
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import networkx as nx
import scipy.sparse as spmat
import scipy.sparse.linalg as spla

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(__file__).resolve().parent
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import synthese_utils as su                      # noqa: E402
import metadata_marks as mm                      # noqa: E402
from sp_helpers import (                          # noqa: E402
    build_sp_model, _random_in_territory)

#: dossier du repo TransmissionGrowthModel (symlink data/tgm)
TGM = ROOT / "data" / "tgm"

MIN_EDGE_LEN_M = 10.0     # garde-fou : deux nœuds SP peuvent coïncider


# ===========================================================================
# 1. Caractéristiques électriques des lignes (identiques au papier)
# ===========================================================================
# Régression linéaire de r(U) et x(U) sur les valeurs standard françaises
# (Van Cutsem, « Analyse et fonctionnement des systèmes d'énergie électrique »).
_U_REF = np.array([70e3, 110e3, 220e3, 380e3])
_R_REF = np.array([0.13, 0.12, 0.08, 0.03])       # Ω/km
_X_REF = np.array([0.40, 0.40, 0.30, 0.25])       # Ω/km
_AR, _BR = np.polyfit(_U_REF, _R_REF, 1)
_AX, _BX = np.polyfit(_U_REF, _X_REF, 1)


def r_lin(U):
    """Résistance linéique (Ω/m) d'une ligne de tension nominale U (V)."""
    return (_AR * np.asarray(U, dtype=float) + _BR) / 1e3


def x_lin(U):
    """Réactance linéique (Ω/m) d'une ligne de tension nominale U (V)."""
    return (_AX * np.asarray(U, dtype=float) + _BX) / 1e3


# ===========================================================================
# 2. Réseau RTE réel — la référence
# ===========================================================================

def load_rte_reference(tgm=TGM):
    """Charge le graphe RTE pondéré (`RTE_weighted.joblib`) et ses injections.

    Retourne un dict avec :
      G        : nx.Graph, arêtes portant `length` (m) et `Tension` (V)
      P        : injections nettes (W), somme nulle
      P_prod   : production installée par nœud (W), somme = 133.9 GW
      P_conso  : consommation par nœud (W)
      volt_mix : tensions des arêtes réelles, ordre décroissant
    """
    import joblib
    sys.path.insert(0, str(tgm / "src"))
    G_raw = joblib.load(tgm / "out" / "RTE_weighted.joblib")

    G = nx.Graph()
    for n, d in G_raw.nodes(data=True):
        G.add_node(n, Longitude=d.get("Longitude"), Latitude=d.get("Latitude"))
    for u, v, d in G_raw.edges(data=True):
        G.add_edge(u, v,
                   length=max(float(d["dist_min"]), MIN_EDGE_LEN_M),
                   Tension=float(d["Tension"]))

    nodes = list(G.nodes())
    P = np.array([G_raw.nodes[n]["Production_MW"] for n in nodes]) * 1e6
    P_prod = np.array([G_raw.nodes[n]["positive_Production_MW"]
                       for n in nodes]) * 1e6
    volt_mix = np.sort([d["Tension"] for _, _, d in G.edges(data=True)])[::-1]
    return dict(G=G, nodes=nodes, P=P, P_prod=P_prod, volt_mix=volt_mix,
                P_conso=P_prod - P)


#: pertes publiées par RTE, en % de la production installée (voir notebook 06)
RTE_PUBLISHED_LOSS_PCT = 3.3816425


# ===========================================================================
# 3. Positions des nœuds — les trois modèles spatiaux
# ===========================================================================

def _denormalize(xy_norm, bounds):
    """Inverse de su.normalize_coords : [0,1]² → coordonnées métriques."""
    xmin, ymin, xmax, ymax = bounds
    scale = max(xmax - xmin, ymax - ymin)
    return np.column_stack([xy_norm[:, 0] * scale + xmin,
                            xy_norm[:, 1] * scale + ymin])


def nodes_sp_mixture(ds, ld, N_target, seed=0):
    """Positions du **modèle SP mixte** : fractal SP + fond uniforme.

    Appelle `su.build_sp_mixture(..., return_labels=True)` sans rien changer :
    le nuage est exactement celui des notebooks 01-04. On dé-normalise
    simplement vers le CRS métrique du dataset pour mesurer des longueurs.

    Retourne (xy_m, is_urban) où is_urban == is_sp.
    """
    xy_norm, is_sp = su.build_sp_mixture(
        ds, ld["gdf_edge"], ld["bounds"], ld["geom_norm"],
        ld["r_data"], ld["xi_data"], N_target=N_target, seed=seed,
        return_labels=True)
    return _denormalize(xy_norm, ld["bounds"]), is_sp


def nodes_sp_mixture_fixed(ds, ld, N_target, n_sp, seed=0):
    """Mélange SP à **nombre de nœuds structurés imposé** — variante « seuil fixé ».

    Identique à `nodes_sp_mixture` sauf sur un point : la proportion de nœuds
    fractals n'est plus déduite de l'amplitude de la 2PCF (ρ), elle est
    **posée**. On tire `n_sp` points au hasard dans le fractal SP et on complète
    à `N_target` avec du fond uniforme.

    Motivation : dans `build_sp_mixture`, la fraction SP vaut √ρ et fluctue de
    0,15 à 0,21 d'un tirage à l'autre, alors que le modèle du papier pose
    `urban_rate = 0.2`. Passer `n_sp = 0.2·N_target` (ou le compte urbain tiré
    par `nodes_cities` à la même graine) met les deux modèles à **structure
    urbaine de même taille**, ce qui isole l'effet de la *géométrie* des amas
    de celui de leur *nombre*.

    Le fractal est celui de `nodes_sp_mixture` à graine égale (même
    `np.random.seed(seed)` avant `build_sp_model`) : les deux variantes ne
    diffèrent donc que par le nombre de points qu'on en retient.

    Retourne (xy_m, is_sp), coordonnées métriques, ordre mélangé — la croissance
    du réseau dépend de l'ordre d'insertion des nœuds.
    """
    np.random.seed(seed)                     # build_sp_model utilise le RNG global
    m_star_eff = ds["m_star"]
    if N_target < ds["N"]:
        m_star_eff = max(1, round(ds["m_star"] / ds["N"] * N_target))
    pos_sp, _, _ = build_sp_model(max(m_star_eff, n_sp), ds["eta"], ds["gamma_data"],
                                  ds["R_ref_m"], ld["gdf_edge"], ds["crs"])

    rng = np.random.default_rng(seed + 1)
    n_sp = min(int(n_sp), len(pos_sp), N_target)
    sp = pos_sp[rng.choice(len(pos_sp), size=n_sp, replace=False)]

    terr = ld["gdf_edge"].geometry.union_all()
    bg = _random_in_territory(N_target - n_sp, rng, ld["gdf_edge"], ds["crs"],
                              terr, terr.bounds)
    xy = np.vstack([sp, bg])
    is_sp = np.zeros(N_target, dtype=bool)
    is_sp[:n_sp] = True
    perm = rng.permutation(N_target)
    return xy[perm], is_sp[perm]


#: les 11 aires urbaines de plus de 200 000 habitants (papier, build_energy_estimates)
POP11 = {1: 2_133_111, 2: 873_076, 3: 522_250, 4: 504_078, 5: 348_085,
         6: 323_204, 7: 302_454, 8: 291_313, 9: 261_804, 10: 236_710,
         11: 225_081}


def nodes_cities(ld, N_target, popdict=POP11, city_radius_m=5_000,
                 alpha=0.85, urban_rate=0.2, seed=0):
    """Positions du modèle **du papier** : décroissance exponentielle autour de
    11 centres-villes tirés au hasard, plus un fond uniforme.

    Reprend `grown_graph_pref_attach_poly` (rayon tiré par inversion de la CDF
    exponentielle), mais en coordonnées métriques uniquement.
    """
    from shapely.geometry import Point
    rng = np.random.default_rng(seed)
    sumpop = sum(popdict.values())
    urban_dict = {k: v / sumpop for k, v in popdict.items()}
    dict_r = {0: (1 / (urban_rate * alpha)) - (1 / alpha)}
    for i in range(1, len(popdict) + 1):
        dict_r[i] = dict_r[i - 1] + popdict[i] / (sumpop * alpha)
    rmax = dict_r[len(popdict)]

    geom = ld["gdf_edge"].geometry.union_all()
    boundary = geom.boundary

    centres = []                       # à plus de 10 rayons de la frontière
    while len(centres) < len(popdict):
        for x, y in _uniform_in_geom(len(popdict), geom, rng):
            if Point(x, y).distance(boundary) > 10 * city_radius_m:
                centres.append((x, y))
            if len(centres) == len(popdict):
                break

    pool = _uniform_in_geom(N_target, geom, rng)      # réserve de nœuds ruraux
    xy = np.empty((N_target, 2))
    is_urban = np.zeros(N_target, dtype=bool)
    k_rural = 0
    for t in range(N_target):
        r_rand = rng.uniform(0, rmax)
        city = 0
        for i in range(len(popdict)):
            if dict_r[i] <= r_rand < dict_r[i + 1]:
                delta_r = r_rand - dict_r[i]
                dist = -(np.log(1 - alpha * delta_r / urban_dict[i + 1])
                         / alpha) * city_radius_m
                city = i + 1
                break
        if city == 0:
            xy[t] = pool[k_rural]
            k_rural += 1
        else:
            theta = rng.uniform(0, 2 * np.pi)
            cx, cy = centres[city - 1]
            xy[t] = [cx + dist * np.cos(theta), cy + dist * np.sin(theta)]
            is_urban[t] = True
    return xy, is_urban


def nodes_uniform(ld, N_target, seed=0):
    """Positions **purement aléatoires** — le témoin sans structure urbaine."""
    rng = np.random.default_rng(seed)
    xy = _uniform_in_geom(N_target, ld["gdf_edge"].geometry.union_all(), rng)
    return xy, np.zeros(N_target, dtype=bool)


def _uniform_in_geom(n, geom, rng, factor=6):
    """Tirage uniforme par rejet de n points dans un polygone métrique."""
    from shapely.vectorized import contains
    x0, y0, x1, y1 = geom.bounds
    out, got = [], 0
    while got < n:
        k = max(factor * (n - got), 1000)
        xs = rng.uniform(x0, x1, k)
        ys = rng.uniform(y0, y1, k)
        m = contains(geom, xs, ys)
        out.append(np.column_stack([xs[m], ys[m]]))
        got += int(m.sum())
    return np.vstack(out)[:n]


# ===========================================================================
# 4. Croissance du réseau — règles du papier, inchangées
# ===========================================================================

def grow_network(xy_m, is_urban, proba, nn_nb=3, seed=0):
    """Fait croître le réseau nœud par nœud (règles de `grown_graph_pref_attach`).

    - nœud **rural** : arête vers le plus proche voisin déjà placé ;
    - nœud **urbain** : arête vers le nœud de **degré maximal** parmi les
      `nn_nb` plus proches déjà placés (attachement préférentiel local) ;
    - dans les deux cas, une **2ᵉ arête** est ajoutée avec la probabilité
      `proba` selon la même règle, en excluant la 1ʳᵉ cible.

    Différence assumée avec le code du papier : toutes les distances sont
    métriques. `grown_graph_pref_attach_poly` compare les nœuds ruraux en
    degrés (lat/lon) et les urbains en mètres, ce qui rend le voisinage rural
    anisotrope (1° de longitude ≈ 0,67° de latitude en France).

    Retourne un nx.Graph dont les arêtes portent `length` (m) et `time`
    (rang de création, utilisé par la règle de tension du papier).
    """
    rng = np.random.default_rng(seed)
    N = len(xy_m)
    G = nx.Graph()
    deg = np.zeros(N, dtype=np.int64)

    def _add(t, other):
        d = float(np.hypot(*(xy_m[t] - xy_m[other])))
        G.add_edge(t, other, length=max(d, MIN_EDGE_LEN_M),
                   time=G.number_of_edges())
        deg[t] += 1
        deg[other] += 1

    def _pref(cand_idx, d2):
        """Parmi cand_idx (triés par distance), le premier de degré maximal."""
        dc = deg[cand_idx]
        return int(cand_idx[int(np.argmax(dc == dc.max()))])

    for t in range(N):
        G.add_node(t, x=xy_m[t, 0], y=xy_m[t, 1],
                   Urbanity="Urban" if is_urban[t] else "Rural")
        if t == 0:
            continue
        d2 = np.sum((xy_m[:t] - xy_m[t]) ** 2, axis=1)
        if not is_urban[t]:
            first = int(np.argmin(d2))
        else:
            k = min(nn_nb, t)
            cand = np.argpartition(d2, k - 1)[:k]
            first = _pref(cand[np.argsort(d2[cand])], d2)
        _add(t, first)

        if rng.random() < proba and t > 1:
            d2b = d2.copy()
            d2b[first] = np.inf
            if not is_urban[t]:
                second = int(np.argmin(d2b))
            else:
                k = min(nn_nb, t - 1)
                cand = np.argpartition(d2b, k - 1)[:k]
                second = _pref(cand[np.argsort(d2b[cand])], d2b)
            if second != first:
                _add(t, second)
    return G


# ===========================================================================
# 5. Métadonnées → injections de puissance
# ===========================================================================

def assign_population(is_sp, pop_empirical, seed=0):
    """Attribution « (c) » du repo : split SP / fond, aléatoire dans chaque groupe.

    Ré-export de `metadata_marks.pop_split_sp_background`, pour garder
    explicite la chaîne notebook 01 → notebook 06.
    """
    return mm.pop_split_sp_background(is_sp, pop_empirical, seed=seed)


def _place_production(N, prod_pool, P_prod_total, rng,
                      prod_where="random", is_urban=None):
    """Pose les valeurs du registre de production sur les nœuds du modèle.

    Facteur commun aux recettes `injections` et `injections_marginal_load` :
    les `len(prod_pool)` valeurs non nulles du registre sont permutées, posées
    sur autant de nœuds tirés dans le groupe `prod_where`, et renormalisées à
    `P_prod_total`.
    """
    n_prod = min(len(prod_pool), N)

    if prod_where == "random":
        cand = np.arange(N)
    elif prod_where == "rural":
        cand = np.flatnonzero(~np.asarray(is_urban, dtype=bool))
    elif prod_where == "urban":
        cand = np.flatnonzero(np.asarray(is_urban, dtype=bool))
    else:
        raise ValueError(f"prod_where inconnu : {prod_where}")
    if len(cand) < n_prod:                       # groupe trop petit : on complète
        rest = np.setdiff1d(np.arange(N), cand)
        cand = np.concatenate([cand, rng.permutation(rest)[:n_prod - len(cand)]])
    where = rng.choice(cand, size=n_prod, replace=False)

    prod = np.zeros(N)
    vals = rng.permutation(np.asarray(prod_pool, dtype=float))[:n_prod]
    prod[where] = vals * (P_prod_total / vals.sum())
    return prod


def injections(pop, prod_pool, P_prod_total, seed=0, kappa=1.0,
               prod_where="random", is_urban=None):
    """Construit le vecteur d'injections nettes P (W), de somme nulle.

    Reprend l'attribution du papier (`scripts/power_weight_attribution.py`) :
    P_i = production_i − consommation_i, la consommation totale étant
    renormalisée sur la production totale installée.

    - **production** : les valeurs non nulles du registre RTE (`prod_pool`, en
      W) sont posées sur un sous-ensemble de nœuds et renormalisées à
      `P_prod_total` ;
    - **consommation** : proportionnelle à `pop ** kappa`.

    Attention à ce que signifie `kappa = 0` : la consommation vaut alors
    `P_prod_total / N` sur **tous** les nœuds (CV = 0). Ce n'est *pas* la
    convention des données : l'Agence ORE ne publie qu'un total départemental,
    que le papier répartit à parts égales entre les postes d'un même
    département — ce qui donne un escalier à 94 marches (8,9 → 119,1 MW par
    poste, plus 766 postes à zéro, CV = 0,72), pas une constante. `kappa = 0`
    aplatit donc *plus* que les données ne le font ; sur la vraie topologie
    l'écart vaut +0,24 point de pertes (3,20 → 3,44 %), soit cinq fois moins
    que le coût de la décorrélation position ↔ injection. Pour la marginale
    réelle sans structure spatiale, voir `injections_marginal_load`.

    prod_where : "random" (défaut), "rural" ou "urban" — où poser les
    centrales ; `is_urban` est requis pour les deux derniers.

    Retourne (P, prod).
    """
    rng = np.random.default_rng(seed)
    N = len(pop)
    prod = _place_production(N, prod_pool, P_prod_total, rng,
                             prod_where, is_urban)
    w = np.maximum(np.asarray(pop, dtype=float), 1e-12) ** kappa
    conso = P_prod_total * w / w.sum()
    return prod - conso, prod


def injections_marginal_load(conso_pool, prod_pool, P_prod_total, N=None,
                             seed=0, prod_where="random", is_urban=None):
    """Injections avec la **marginale de consommation réelle**, placée au hasard.

    Recette intermédiaire entre `injections(kappa=0)` — qui aplatit la charge à
    une constante — et `injections_shuffled` — qui reprend le vecteur net réel
    tout entier. Ici on garde la production modélisée (registre permuté, posée
    au hasard) mais on remplace la charge uniforme par un tirage dans la
    **distribution empirique** de la consommation par poste (`conso_pool`, en
    W) : les 94 paliers départementaux et les postes à zéro sont présents, avec
    leur CV de 0,72, mais leur position est neutre.

    Elle sépare donc deux effets que `kappa = 0` confond : celui de la
    *marginale* de charge et celui de l'*appariement* production/consommation
    sur un même nœud.

    La consommation est renormalisée à `P_prod_total`, donc Σ P = 0.

    Retourne (P, prod).
    """
    rng = np.random.default_rng(seed)
    pool = np.asarray(conso_pool, dtype=float)
    N = len(pool) if N is None else int(N)
    prod = _place_production(N, prod_pool, P_prod_total, rng,
                             prod_where, is_urban)
    conso = (rng.permutation(pool) if N == len(pool)
             else rng.choice(pool, size=N, replace=True))
    conso = conso * (P_prod_total / conso.sum())
    return prod - conso, prod


def injections_shuffled(P_real, seed=0):
    """Injections nettes = le vecteur empirique **permuté au hasard**.

    Recette sans métadonnée intermédiaire : on ne modélise ni la population ni
    la consommation, on reprend la **distribution marginale réelle** de
    P_i = production_i − consommation_i (registre RTE + ORE) et on la pose au
    hasard sur les nœuds du modèle. Le modèle spatial ne porte alors plus que
    la *géométrie* ; tout ce qui touche à l'intensité des injections est
    exactement celui des données.

    Σ P est conservée par permutation (nulle à la machine près), donc le
    load-flow reste bien posé. Complément naturel du vecteur *rangé* utilisé au
    § 7 (`np.sort(P)[::-1]`, la recette implicite du papier) : même marginale,
    placement neutre au lieu de placement corrélé à l'indice de nœud.

    Parameters
    ----------
    P_real : array (N,) — injections nettes réelles (W), de somme nulle
    seed   : int

    Returns
    -------
    ndarray (N,)
    """
    P = np.asarray(P_real, dtype=float)
    return np.random.default_rng(seed).permutation(P)


# ===========================================================================
# 6. Load-flow continu et pertes Joule
# ===========================================================================

def _solve_dc(I, J, X, U, P, N):
    """Résout B'θ' = P' (nœud 0 en référence) et renvoie (flux en W, θ).

    **La susceptance vaut b = U²/X, pas 1/X.** Le réseau mélange quatre
    niveaux de tension : en per-unit, x_pu = X·S_base/U², donc
    b_pu = U²/(S_base·X) et le flux en watts vaut P = Δθ·U²/X. Prendre
    b = 1/X revient à supposer toutes les lignes à la même tension : le flux
    se répartit alors sur les lignes 63 kV au lieu du réseau 400 kV, et les
    pertes sont sur-estimées d'un facteur ~5 (voir notebook 06, § pièges).
    """
    b = U ** 2 / X
    B = spmat.csr_matrix(
        (np.r_[b, b, -b, -b], (np.r_[I, J, I, J], np.r_[I, J, J, I])),
        shape=(N, N))
    theta = np.zeros(N)
    theta[1:] = spla.spsolve(B[1:, 1:].tocsc(), P[1:])
    return (theta[I] - theta[J]) * U ** 2 / X, theta


def dc_losses(G, P, volt_mix, n_iter=4, voltage_rule="flow"):
    """Pertes Joule d'un réseau par load-flow continu.

    Modèle du papier : chaque arête est une ligne triphasée de résistance
    R = r(U)·ℓ et de réactance X = x(U)·ℓ, le flux actif se répartit selon
    B_ij = 1/X_ij, et les pertes valent J_ij = R_ij·P_ij²/U_ij².

    `volt_mix` est le **spectre de tensions** à répartir sur les arêtes (p.ex.
    les tensions réelles des 6 272 lignes RTE). `voltage_rule` fixe la façon
    de l'attribuer :

    - `"flow"` (défaut) : la ligne la plus chargée reçoit la plus haute
      tension — le dimensionnement réel d'un réseau ; point fixe en ~3
      itérations ;
    - `"length"` : la ligne la plus longue reçoit la plus haute tension ;
    - `"given"` : on lit la tension réelle portée par l'arête (attribut
      `Tension`) — seulement disponible pour le réseau RTE réel ;
    - `"time"` : les arêtes créées en premier reçoivent les plus hautes
      tensions — c'est *littéralement* la file triée du papier
      (`queue_voltage.pop()` dépile par la droite). Nécessite l'attribut
      d'arête `time` ;
    - `"random"` : tirage au hasard (spectre respecté, corrélation avec le
      flux détruite).

    Retourne dict(J_tot, J, flow, U, theta, L, edges).
    """
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    I = np.array([idx[u] for u, _ in G.edges()])
    Jn = np.array([idx[v] for _, v in G.edges()])
    L = np.array([max(d["length"], MIN_EDGE_LEN_M)
                  for _, _, d in G.edges(data=True)])
    M = len(L)

    mix = np.sort(np.asarray(volt_mix, dtype=float))[::-1]
    if len(mix) != M:                       # ré-échantillonnage par quantiles
        mix = np.quantile(mix, np.linspace(1, 0, M))

    if voltage_rule == "given":
        U = np.array([d["Tension"] for _, _, d in G.edges(data=True)],
                     dtype=float)
        flow, theta = _solve_dc(I, Jn, x_lin(U) * L, U, P, N)
    elif voltage_rule == "random":
        U = np.random.default_rng(0).permutation(mix)
        flow, theta = _solve_dc(I, Jn, x_lin(U) * L, U, P, N)
    elif voltage_rule == "time":
        t_e = np.array([d["time"] for _, _, d in G.edges(data=True)])
        U = np.empty(M)
        U[np.argsort(t_e)] = mix
        flow, theta = _solve_dc(I, Jn, x_lin(U) * L, U, P, N)
    elif voltage_rule == "length":
        U = np.empty(M)
        U[np.argsort(L)[::-1]] = mix
        flow, theta = _solve_dc(I, Jn, x_lin(U) * L, U, P, N)
    elif voltage_rule == "flow":
        U = np.full(M, float(np.median(mix)))
        for _ in range(n_iter):
            flow, theta = _solve_dc(I, Jn, x_lin(U) * L, U, P, N)
            U_new = np.empty(M)
            U_new[np.argsort(np.abs(flow))[::-1]] = mix
            if np.array_equal(U_new, U):
                break
            U = U_new
        flow, theta = _solve_dc(I, Jn, x_lin(U) * L, U, P, N)
    else:
        raise ValueError(f"voltage_rule inconnue : {voltage_rule}")

    J = r_lin(U) * L * flow ** 2 / U ** 2
    return dict(J_tot=float(J.sum()), J=J, flow=flow, U=U, theta=theta,
                L=L, edges=list(zip(I, Jn)))


def loss_pct(G, P, P_prod_total, volt_mix, **kw):
    """Pertes Joule en % de la production totale — la grandeur du papier."""
    return 100.0 * dc_losses(G, P, volt_mix, **kw)["J_tot"] / P_prod_total


# ===========================================================================
# 7. Observables structurelles
# ===========================================================================

def total_length_km(G):
    """Longueur totale de lignes (km)."""
    return sum(d["length"] for _, _, d in G.edges(data=True)) / 1e3


def q_of(G):
    """Redondance q = (M − N)/N."""
    return (G.number_of_edges() - G.number_of_nodes()) / G.number_of_nodes()


if __name__ == "__main__":
    print("— test de fumée powergrid_sp —")
    ref = load_rte_reference()
    print(f"RTE : {ref['G'].number_of_nodes()} nœuds, "
          f"{ref['G'].number_of_edges()} arêtes, q={q_of(ref['G']):.3f}, "
          f"L_tot={total_length_km(ref['G']):,.0f} km")
    Ptot = ref["P_prod"].sum()
    print(f"production installée : {Ptot / 1e9:.1f} GW")
    for rule in ("flow", "length", "random"):
        print(f"  pertes (spectre réel, règle {rule:7s}) : "
              f"{loss_pct(ref['G'], ref['P'], Ptot, ref['volt_mix'], voltage_rule=rule):6.2f} %")
