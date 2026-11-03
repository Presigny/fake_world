"""
powergrid_sp.py
===============
Transmission grid grown on the SP mixture model of this repository.

The growth model of `TransmissionGrowthModel` (Emery, Bercegol, Aumaitre,
"A Power Grid Growth Model Based on Rural-Urban Differentiation") is reused
and only its spatial part is replaced:

    growth-model paper            ->  this module
    ------------------------------------------------------------------
    urban: exponential decay      ->  urban: nodes of the SP fractal
    around 11 city centres            (is_sp = True in build_sp_mixture)
    rural: uniform in the         ->  rural: identical (uniform
    territory                         background of the SP mixture)
    connectivity                  ->  identical (nearest neighbour for
                                      rural nodes, preferential attachment
                                      among the nn_nb nearest for urban
                                      nodes, second edge with probability q)

Populations are assigned with metadata_marks.pop_split_sp_background: the
|SP| largest empirical populations go to the structured nodes, the others to
the background, at random inside each group. They set the consumption of
each substation, and production follows the empirical distribution of the
RTE register. The power balance is zero by construction.

Joule losses come from a DC load flow identical to the one of
`scripts/build_energy_estimates.py` in TransmissionGrowthModel, with two
corrections documented in notebook 06 (sparse solver, and voltages assigned
by flow rank instead of an arbitrary sorted queue).
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
# Only src/ is put on the path, first: the repository root holds an older
# methods_two_point_correlation.py that must not shadow the one of src/.
if str(SRC) in sys.path:
    sys.path.remove(str(SRC))
sys.path.insert(0, str(SRC))

import synthesis_utils as su                     # noqa: E402
import metadata_marks as mm                      # noqa: E402
from sp_helpers import (                          # noqa: E402
    build_sp_model, _random_in_territory)

#: TransmissionGrowthModel repository (expected at data/tgm, see the README)
TGM = ROOT / "data" / "tgm"

MIN_EDGE_LEN_M = 10.0     # safeguard, two SP nodes can coincide


# ===========================================================================
# 1. Electrical characteristics of the lines (as in the growth-model paper)
# ===========================================================================
# Linear regression of r(U) and x(U) on standard French values
# (Van Cutsem, "Analyse et fonctionnement des systemes d'energie electrique").
_U_REF = np.array([70e3, 110e3, 220e3, 380e3])
_R_REF = np.array([0.13, 0.12, 0.08, 0.03])       # Ω/km
_X_REF = np.array([0.40, 0.40, 0.30, 0.25])       # Ω/km
_AR, _BR = np.polyfit(_U_REF, _R_REF, 1)
_AX, _BX = np.polyfit(_U_REF, _X_REF, 1)


def r_lin(U):
    """Resistance per unit length (Ohm/m) of a line of nominal voltage U (V)."""
    return (_AR * np.asarray(U, dtype=float) + _BR) / 1e3


def x_lin(U):
    """Reactance per unit length (Ohm/m) of a line of nominal voltage U (V)."""
    return (_AX * np.asarray(U, dtype=float) + _BX) / 1e3


# ===========================================================================
# 2. Real RTE grid, the reference
# ===========================================================================

def load_rte_reference(tgm=TGM):
    """Load the weighted RTE graph (`RTE_weighted.joblib`) and its injections.

    Returns a dict with
      G        : nx.Graph, edges carry `length` (m) and `Tension` (V)
      P        : net injections (W), zero sum
      P_prod   : installed production per node (W), total 133.9 GW
      P_conso  : consumption per node (W)
      volt_mix : voltages of the real edges, decreasing order
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


#: losses published by RTE, in % of the installed production (see notebook 06)
RTE_PUBLISHED_LOSS_PCT = 3.3816425


# ===========================================================================
# 3. Node positions, the three spatial models
# ===========================================================================

def _denormalize(xy_norm, bounds):
    """Inverse of su.normalize_coords, from [0, 1]^2 back to metric coordinates."""
    xmin, ymin, xmax, ymax = bounds
    scale = max(xmax - xmin, ymax - ymin)
    return np.column_stack([xy_norm[:, 0] * scale + xmin,
                            xy_norm[:, 1] * scale + ymin])


def nodes_sp_mixture(ds, ld, N_target, seed=0):
    """Node positions of the SP mixture (SP fractal + uniform background).

    Calls su.build_sp_mixture(..., return_labels=True) unchanged, so the
    cloud is the one of notebooks 01-04. It is only mapped back to the
    metric CRS of the dataset to measure lengths.
    Returns (xy_m, is_urban) with is_urban == is_sp.
    """
    xy_norm, is_sp = su.build_sp_mixture(
        ds, ld["gdf_edge"], ld["bounds"], ld["geom_norm"],
        ld["r_data"], ld["xi_data"], N_target=N_target, seed=seed,
        return_labels=True)
    return _denormalize(xy_norm, ld["bounds"]), is_sp


def nodes_sp_mixture_fixed(ds, ld, N_target, n_sp, seed=0):
    """SP mixture with an imposed number of structured nodes ("fixed threshold").

    Same as nodes_sp_mixture except that the fraction of fractal nodes is
    not deduced from the 2PCF amplitude (rho) but imposed: n_sp points are
    drawn at random in the SP fractal and the rest, up to N_target, is a
    uniform background.

    In build_sp_mixture the SP fraction is sqrt(rho) and varies from 0.15 to
    0.21 between draws, while the growth-model paper uses urban_rate = 0.2.
    Setting n_sp = 0.2 N_target (or the urban count drawn by nodes_cities
    with the same seed) gives both models an urban structure of the same
    size, which separates the effect of the cluster geometry from the effect
    of their number.

    The fractal is the one of nodes_sp_mixture for the same seed (same
    np.random.seed(seed) before build_sp_model), so the two variants only
    differ by the number of points kept.

    Returns (xy_m, is_sp), metric coordinates in shuffled order (the network
    growth depends on the insertion order).
    """
    np.random.seed(seed)                     # build_sp_model uses the global RNG
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


#: the 11 urban areas above 200,000 inhabitants (growth-model paper, build_energy_estimates)
POP11 = {1: 2_133_111, 2: 873_076, 3: 522_250, 4: 504_078, 5: 348_085,
         6: 323_204, 7: 302_454, 8: 291_313, 9: 261_804, 10: 236_710,
         11: 225_081}


def nodes_cities(ld, N_target, popdict=POP11, city_radius_m=5_000,
                 alpha=0.85, urban_rate=0.2, seed=0):
    """Node positions of the growth-model paper.

    Exponential decay around 11 randomly placed city centres, plus a uniform
    background. Same as grown_graph_pref_attach_poly (radius drawn by
    inverting the exponential CDF), but in metric coordinates only.
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

    centres = []                       # more than 10 city radii from the border
    while len(centres) < len(popdict):
        for x, y in _uniform_in_geom(len(popdict), geom, rng):
            if Point(x, y).distance(boundary) > 10 * city_radius_m:
                centres.append((x, y))
            if len(centres) == len(popdict):
                break

    pool = _uniform_in_geom(N_target, geom, rng)      # pool of rural nodes
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
    """Uniform random positions, the control without urban structure."""
    rng = np.random.default_rng(seed)
    xy = _uniform_in_geom(N_target, ld["gdf_edge"].geometry.union_all(), rng)
    return xy, np.zeros(N_target, dtype=bool)


def _uniform_in_geom(n, geom, rng, factor=6):
    """Rejection sampling of n uniform points in a metric polygon."""
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
# 4. Network growth (rules of the growth-model paper, unchanged)
# ===========================================================================

def grow_network(xy_m, is_urban, proba, nn_nb=3, seed=0):
    """Grow the network node by node (rules of grown_graph_pref_attach).

    - rural node: edge to the nearest node already placed;
    - urban node: edge to the node of largest degree among the nn_nb nearest
      nodes already placed (local preferential attachment);
    - in both cases, a second edge is added with probability `proba` with
      the same rule, excluding the first target.

    Deliberate difference with the original code: all distances are metric.
    grown_graph_pref_attach_poly compares rural nodes in degrees (lat/lon)
    and urban nodes in metres, which makes the rural neighbourhood
    anisotropic (1 degree of longitude is about 0.67 degree of latitude in
    France).

    Returns an nx.Graph whose edges carry `length` (m) and `time` (creation
    rank, used by the voltage rule of the paper).
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
        """Among cand_idx (sorted by distance), the first one of maximal degree."""
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
# 5. Populations and power injections
# ===========================================================================

def assign_population(is_sp, pop_empirical, seed=0):
    """Population split between SP and background nodes, random inside each group.

    Re-export of metadata_marks.pop_split_sp_background, to keep the link
    between notebook 01 and notebook 06 explicit.
    """
    return mm.pop_split_sp_background(is_sp, pop_empirical, seed=seed)


def _place_production(N, prod_pool, P_prod_total, rng,
                      prod_where="random", is_urban=None):
    """Place the values of the production register on the model nodes.

    Shared by injections and injections_marginal_load: the len(prod_pool)
    non-zero values of the register are shuffled, placed on as many nodes
    drawn in the group `prod_where`, and rescaled to P_prod_total.
    """
    n_prod = min(len(prod_pool), N)

    if prod_where == "random":
        cand = np.arange(N)
    elif prod_where == "rural":
        cand = np.flatnonzero(~np.asarray(is_urban, dtype=bool))
    elif prod_where == "urban":
        cand = np.flatnonzero(np.asarray(is_urban, dtype=bool))
    else:
        raise ValueError(f"unknown prod_where: {prod_where}")
    if len(cand) < n_prod:                       # group too small, complete it
        rest = np.setdiff1d(np.arange(N), cand)
        cand = np.concatenate([cand, rng.permutation(rest)[:n_prod - len(cand)]])
    where = rng.choice(cand, size=n_prod, replace=False)

    prod = np.zeros(N)
    vals = rng.permutation(np.asarray(prod_pool, dtype=float))[:n_prod]
    prod[where] = vals * (P_prod_total / vals.sum())
    return prod


def injections(pop, prod_pool, P_prod_total, seed=0, kappa=1.0,
               prod_where="random", is_urban=None):
    """Net injection vector P (W), with zero sum.

    Same assignment as the growth-model paper
    (scripts/power_weight_attribution.py): P_i = production_i - consumption_i,
    with the total consumption rescaled to the total installed production.

    - production: the non-zero values of the RTE register (prod_pool, in W)
      are placed on a subset of nodes and rescaled to P_prod_total;
    - consumption: proportional to pop ** kappa.

    With kappa = 0 the consumption is P_prod_total / N on every node
    (CV = 0). This is not the convention of the data: Agence ORE only
    publishes a total per departement, which the paper splits equally
    between the substations of that departement. This gives a staircase of
    94 steps (8.9 to 119.1 MW per substation, plus 766 substations at zero,
    CV = 0.72), not a constant. kappa = 0 therefore flattens the load more
    than the data do. On the real topology the difference is +0.24 point of
    losses (3.20 to 3.44%), five times less than the cost of decorrelating
    positions and injections. For the real marginal without spatial
    structure, see injections_marginal_load.

    prod_where is "random" (default), "rural" or "urban" (where to place the
    power plants). is_urban is required for the last two.

    Returns (P, prod).
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
    """Injections with the real consumption marginal, placed at random.

    Intermediate between injections(kappa=0), which flattens the load to a
    constant, and injections_shuffled, which reuses the whole real net
    vector. The production is modelled (shuffled register, random placement)
    but the uniform load is replaced by draws from the empirical distribution
    of the consumption per substation (conso_pool, in W). The 94 departement
    steps and the zero substations are present, with their CV of 0.72, but
    their position is neutral.

    It separates two effects that kappa = 0 mixes, the load marginal and the
    pairing of production and consumption on the same node. The consumption
    is rescaled to P_prod_total, so sum(P) = 0.

    Returns (P, prod).
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
    """Net injections given by the empirical vector, randomly shuffled.

    Neither population nor consumption is modelled. The real marginal of
    P_i = production_i - consumption_i (RTE register + ORE) is placed at
    random on the model nodes, so the spatial model only carries the
    geometry and the injection intensities are exactly those of the data.

    sum(P) is preserved by the permutation (zero up to machine precision),
    so the load flow stays well posed. It complements the sorted vector used
    in section 7 of notebook 06 (np.sort(P)[::-1], the implicit recipe of
    the paper): same marginal, neutral placement instead of a placement
    correlated with the node index.

    Parameters
    ----------
    P_real : array (N,), real net injections (W) with zero sum
    seed   : int

    Returns
    -------
    ndarray (N,)
    """
    P = np.asarray(P_real, dtype=float)
    return np.random.default_rng(seed).permutation(P)


# ===========================================================================
# 6. DC load flow and Joule losses
# ===========================================================================

def _solve_dc(I, J, X, U, P, N):
    """Solve B' theta' = P' (node 0 is the reference), return (flows in W, theta).

    The susceptance is b = U^2 / X, not 1 / X. The grid mixes four voltage
    levels. In per unit, x_pu = X S_base / U^2, so b_pu = U^2 / (S_base X)
    and the flow in watts is P = dtheta U^2 / X. Taking b = 1 / X amounts to
    assuming that all lines have the same voltage. The flow then goes through
    the 63 kV lines instead of the 400 kV grid and the losses are
    overestimated by a factor of about 5 (see notebook 06, pitfalls section).
    """
    b = U ** 2 / X
    B = spmat.csr_matrix(
        (np.r_[b, b, -b, -b], (np.r_[I, J, I, J], np.r_[I, J, J, I])),
        shape=(N, N))
    theta = np.zeros(N)
    theta[1:] = spla.spsolve(B[1:, 1:].tocsc(), P[1:])
    return (theta[I] - theta[J]) * U ** 2 / X, theta


def dc_losses(G, P, volt_mix, n_iter=4, voltage_rule="flow"):
    """Joule losses of a network from a DC load flow.

    Each edge is a three-phase line of resistance R = r(U) l and reactance
    X = x(U) l. The active flow follows the susceptances (see _solve_dc) and
    the losses are J_ij = R_ij P_ij^2 / U_ij^2.

    volt_mix is the voltage spectrum to distribute over the edges (for
    instance the real voltages of the 6,272 RTE lines). voltage_rule sets how:
    - "flow" (default): the most loaded line gets the highest voltage, as in
      the real sizing of a grid (fixed point in about 3 iterations);
    - "length": the longest line gets the highest voltage;
    - "given": the real voltage carried by the edge (attribute `Tension`),
      only available for the real RTE grid;
    - "time": the edges created first get the highest voltages, which is
      literally the sorted queue of the paper (queue_voltage.pop() pops from
      the right). Requires the edge attribute `time`;
    - "random": random draw (same spectrum, no correlation with the flow).

    Returns dict(J_tot, J, flow, U, theta, L, edges).
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
    if len(mix) != M:                       # resample by quantiles
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
        raise ValueError(f"unknown voltage_rule: {voltage_rule}")

    J = r_lin(U) * L * flow ** 2 / U ** 2
    return dict(J_tot=float(J.sum()), J=J, flow=flow, U=U, theta=theta,
                L=L, edges=list(zip(I, Jn)))


def loss_pct(G, P, P_prod_total, volt_mix, **kw):
    """Joule losses in % of the total production (the quantity of the paper)."""
    return 100.0 * dc_losses(G, P, volt_mix, **kw)["J_tot"] / P_prod_total


# ===========================================================================
# 7. Structural observables
# ===========================================================================

def total_length_km(G):
    """Total line length (km)."""
    return sum(d["length"] for _, _, d in G.edges(data=True)) / 1e3


def q_of(G):
    """Redundancy q = (M - N) / N."""
    return (G.number_of_edges() - G.number_of_nodes()) / G.number_of_nodes()


if __name__ == "__main__":
    print("powergrid_sp smoke test")
    ref = load_rte_reference()
    print(f"RTE: {ref['G'].number_of_nodes()} nodes, "
          f"{ref['G'].number_of_edges()} edges, q={q_of(ref['G']):.3f}, "
          f"L_tot={total_length_km(ref['G']):,.0f} km")
    Ptot = ref["P_prod"].sum()
    print(f"installed production: {Ptot / 1e9:.1f} GW")
    for rule in ("flow", "length", "random"):
        print(f"  losses (real spectrum, rule {rule:7s}): "
              f"{loss_pct(ref['G'], ref['P'], Ptot, ref['volt_mix'], voltage_rule=rule):6.2f} %")
