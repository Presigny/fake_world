"""
compute_all.py
==============
Précalcule, pour TOUS les datasets, les résultats lourds consommés par
`synthese.ipynb` :

  - comparaison SP vs processus ponctuels à N constant (2PCF + JSD)
  - stochasticité du SP : N_INSTANCES tirages (moyenne ± écart-type)
  - TODO 3 : clustering marqué par la population (gamma_hi vs gamma_lo)

Les gros datasets (ENEDIS, Routes, Germany…) ont des CSV de ~1 M lignes :
chargement + 2PCF coûteux. Ce script est donc **incrémental et reprenable** :
chaque dataset est sauvegardé dans `cache/<slug>.npz` dès qu'il est calculé,
et un dataset déjà en cache est sauté. Relancer le script reprend là où il
s'est arrêté.

Usage :
    python compute_all.py              # tous les datasets, N=3000
    python compute_all.py --n 5000     # change le N de comparaison
    python compute_all.py --only "Belgium Cities" "Swiss Stops"
    python compute_all.py --force      # recalcule même si déjà en cache
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import synthese_utils as su

CACHE = su.CACHE_DIR                 # data/synthese_cache → fake_world/Synthese/cache
CACHE.mkdir(exist_ok=True)

N_COMPARE_DEFAULT = 3000          # N constant (plafonné à N_real et N_CAP_MODELS)
N_INSTANCES = 10                  # tirages pour la stochasticité
# Modèles : SP + Poisson (référence) + les 3 processus ponctuels calibrés par
# SVI BSTPP (LGCP / Hawkes / Cox Hawkes) — étalon-or, sans heuristique.
MODELS = ("SP", "Poisson", "LGCP", "Hawkes", "Cox Hawkes")

# TODO 1 — balayage de N (sauté pour les gros datasets : reload ~30 min).
SWEEP_NS = [200, 500, 1000, 2000, 4000]
GIANT = {"ENEDIS France", "France Routes", "Germany Stops"}


def slug(name):
    return name.lower().replace(" ", "_")


def compute_one(name, ds, n_compare):
    """Calcule tous les résultats d'un dataset et les renvoie en dict de tableaux."""
    ld = su.load_dataset(ds, verbose=True)
    N = min(n_compare, ld["N_real"], su.N_CAP_MODELS)

    # ── Modèles BSTPP (SVI) : recharge une fois (pools + stochasticité) ────
    print(f"  [{name}] rechargement + simulation BSTPP (SVI) ...", flush=True)
    bstpp_models = su.load_bstpp_models(name, ld)
    bstpp_pools = {mn: (su.bstpp_simulate_pool(m, ld["geom_norm"], n_sim=6)
                        if m is not None else np.zeros((0, 2)))
                   for mn, m in bstpp_models.items()}
    for mn, pool in bstpp_pools.items():
        print(f"    BSTPP {mn:11s}: pool = {len(pool):,} pts", flush=True)

    print(f"  [{name}] comparaison à N={N} ...", flush=True)

    # ── 1. Comparaison à N constant ───────────────────────────────────────
    res = su.compare_at_constant_N(ds, ld, N_target=N, models=MODELS,
                                   bstpp_pools=bstpp_pools)
    store = dict(name=name, N=N, N_real=ld["N_real"], N_full=ld["N_full"],
                 scale_m=ld["scale_m"], gamma_data=ds["gamma_data"])
    store["data_r"] = res["data"]["r"]
    store["data_xi"] = res["data"]["xi"]
    for m in MODELS:
        store[f"{m}_r"] = res[m]["r"]
        store[f"{m}_xi"] = res[m]["xi"]
        store[f"{m}_gamma"] = res[m]["gamma"] if res[m]["gamma"] else np.nan
        store[f"{m}_jsd"] = res[m]["jsd"]

    # ── 1bis. Métrique NEUTRE : distances au plus proche voisin (NN) ───────
    # Le SP est calibré sur la 2PCF, pas sur la NN → test moins biaisé.
    model_xys = {m: res[m]["xy"] for m in MODELS}
    nn_dists, nn_jsd = su.nn_comparison(res["data"]["xy"], model_xys)
    if nn_dists is not None:
        store["nn_r"] = nn_dists["r"]
        store["nn_data"] = nn_dists["data"]
        for m in MODELS:
            store[f"nn_{m}"] = nn_dists[m]
            store[f"nn_jsd_{m}"] = nn_jsd[m]

    # ── 2. TODO 2 : stochasticité de TOUS les modèles (reproductibilité) ──
    # Bande 2PCF (moyenne±σ), JSD moyenne±σ (incertitude), largeur de bande cv.
    print(f"  [{name}] stochasticité × {N_INSTANCES} (tous modèles) ...",
          flush=True)
    bands = su.stochastic_bands(ds, ld, N, n_instances=N_INSTANCES,
                                bstpp_models=bstpp_models, models=MODELS)
    for m in MODELS:
        b = bands[m]
        if b["r"] is not None and "band_r" not in store:
            store["band_r"] = b["r"]
        store[f"band_{m}_xi_mean"] = b["xi_mean"] if b["xi_mean"] is not None else np.array([])
        store[f"band_{m}_xi_std"] = b["xi_std"] if b["xi_std"] is not None else np.array([])
        store[f"band_{m}_jsd_mean"] = b["jsd_mean"]
        store[f"band_{m}_jsd_std"] = b["jsd_std"]
        store[f"band_{m}_cv"] = b["cv"]
    # Compat rétro : bande SP pour la grille §5 existante
    store["stoch_r"] = bands["SP"]["r"]
    store["stoch_xi_mean"] = bands["SP"]["xi_mean"]
    store["stoch_xi_std"] = bands["SP"]["xi_std"]

    # ── 3. TODO 1 : balayage de N (non-géants seulement) ──────────────────
    if name not in GIANT:
        print(f"  [{name}] balayage N {SWEEP_NS} ...", flush=True)
        sw = su.sweep_constant_N(ds, ld, SWEEP_NS, models=MODELS,
                                 bstpp_pools=bstpp_pools)
        store["sweep_N"] = sw["N"]
        for m in MODELS:
            store[f"sweep_jsd_{m}"] = sw["jsd"][m]

    # ── 4. TODO 3 : population/marks — SP hiérarchique vs Hawkes marqué ────
    if ld["pop_data"] is not None:
        print(f"  [{name}] TODO3 population (slope Zipf + Hawkes marqué) ...",
              flush=True)
        # Caler le slope du SP sur la vraie pente rang-taille (Zipf) des données
        zipf_d = su.fit_zipf_exponent(ld["pop_data"])
        sp_slope, zipf_sp = su.calibrate_sp_slope(ds, ld["gdf_edge"], zipf_d)
        store["zipf_data"] = zipf_d
        store["zipf_sp"] = zipf_sp
        store["sp_slope"] = sp_slope

        spp = su.build_sp_with_population(ds, ld["gdf_edge"], slope=sp_slope,
                                          seed=1)
        # Hawkes marqué : la population module l'auto-excitation
        xy_mh, m_mh = su.simulate_marked_hawkes(N, ld["geom_norm"],
                                                ld["pop_data"], ds["sigma"],
                                                seed=1)

        # split γ_hi/γ_lo : données, SP marqué, Hawkes marqué
        def _num(g):
            return float(g) if (g is not None and np.isfinite(g)) else np.nan
        for tag, xy, pop in [("data", ld["xy_norm"], ld["pop_data"]),
                             ("sp", spp["xy_norm"], spp["pop"]),
                             ("mh", xy_mh, m_mh)]:
            mk = su.marked_clustering_2pcf(xy, pop, ld["geom_norm"])
            store[f"{tag}_gamma_hi"] = _num(mk["gamma_hi"])
            store[f"{tag}_gamma_lo"] = _num(mk["gamma_lo"])
        store["pop_data_sorted"] = np.sort(ld["pop_data"])[::-1]
        store["sp_pop_sorted"] = np.sort(spp["pop"])[::-1]

        # mark-correlation k_mm(r) : données vs SP vs Hawkes marqué
        r_km, kmm_d = su.mark_correlation_function(ld["xy_norm"],
                                                   ld["pop_data"], ld["geom_norm"])
        _, kmm_s = su.mark_correlation_function(spp["xy_norm"], spp["pop"],
                                                ld["geom_norm"])
        _, kmm_h = su.mark_correlation_function(xy_mh, m_mh, ld["geom_norm"])
        if r_km is not None:
            store["markcorr_r"] = r_km
            store["data_kmm"] = kmm_d
            store["sp_kmm"] = kmm_s if kmm_s is not None else np.full_like(r_km, np.nan)
            store["mh_kmm"] = kmm_h if kmm_h is not None else np.full_like(r_km, np.nan)
    return store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_COMPARE_DEFAULT)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    names = args.only if args.only else list(su.DATASETS)
    print(f"Datasets à traiter : {names}")
    print(f"N comparaison = {args.n}, instances = {N_INSTANCES}\n")

    for name in names:
        out = CACHE / f"{slug(name)}.npz"
        if out.exists() and not args.force:
            print(f"✓ {name} déjà en cache ({out.name}) — sauté")
            continue
        print(f"\n{'='*60}\n  {name}\n{'='*60}", flush=True)
        t0 = time.time()
        try:
            store = compute_one(name, su.DATASETS[name], args.n)
            np.savez(out, **{k: np.asarray(v, dtype=object) if isinstance(v, str)
                             else np.asarray(v) for k, v in store.items()})
            print(f"✓ {name} sauvé en {time.time()-t0:.0f}s → {out.name}",
                  flush=True)
        except Exception as e:
            import traceback
            print(f"✗ {name} ÉCHEC : {e}", flush=True)
            traceback.print_exc()

    print("\nTerminé.")


if __name__ == "__main__":
    main()
