"""
compute_all.py
==============
Precompute, for every dataset, the heavy results read by
`notebooks/07_model_comparison.ipynb` (Table 2 of the paper):

  - comparison of SP, Poisson and the BSTPP point processes at constant N
    (2PCF, JSD, nearest-neighbour distances)
  - variability over N_INSTANCES independent realisations of each model
  - JSD as a function of N (small datasets only)
  - population marks: hierarchical SP versus marked Hawkes

Large datasets (ENEDIS, roads, Germany) have CSV files of about 1 M rows, so
loading them and computing their 2PCF is slow. The script is incremental:
each dataset is saved in out/synthesis_cache/<slug>.npz as soon as it is
done, and datasets already in the cache are skipped.

The BSTPP models must have been fitted first (notebooks/09_bstpp_inference.ipynb).

Usage:
    python scripts/compute_all.py              # all datasets, N = 3000
    python scripts/compute_all.py --n 5000     # other comparison N
    python scripts/compute_all.py --only "Belgium Cities" "Swiss Stops"
    python scripts/compute_all.py --force      # recompute cached datasets
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import synthesis_utils as su

CACHE = su.CACHE_DIR                 # out/synthesis_cache
CACHE.mkdir(parents=True, exist_ok=True)

N_COMPARE_DEFAULT = 3000          # constant N (capped at N_real and N_CAP_MODELS)
N_INSTANCES = 10                  # independent realisations per model
# SP, Poisson (reference) and the three point processes fitted by SVI (BSTPP)
MODELS = ("SP", "Poisson", "LGCP", "Hawkes", "Cox Hawkes")

# N sweep, skipped for the largest datasets (BSTPP reload of about 30 min)
SWEEP_NS = [200, 500, 1000, 2000, 4000]
GIANT = {"ENEDIS France", "France Routes", "Germany Stops"}


def slug(name):
    return name.lower().replace(" ", "_")


def compute_one(name, ds, n_compare):
    """Compute all the results of one dataset, returned as a dict of arrays."""
    ld = su.load_dataset(ds, verbose=True)
    N = min(n_compare, ld["N_real"], su.N_CAP_MODELS)

    # BSTPP models (SVI), reloaded once for the pools and the variability
    print(f"  [{name}] reloading and simulating the BSTPP models ...", flush=True)
    bstpp_models = su.load_bstpp_models(name, ld)
    bstpp_pools = {mn: (su.bstpp_simulate_pool(m, ld["geom_norm"], n_sim=6)
                        if m is not None else np.zeros((0, 2)))
                   for mn, m in bstpp_models.items()}
    for mn, pool in bstpp_pools.items():
        print(f"    BSTPP {mn:11s}: pool = {len(pool):,} pts", flush=True)

    print(f"  [{name}] comparison at N={N} ...", flush=True)

    # 1. Comparison at constant N
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

    # 1b. Nearest-neighbour distances, an observable the SP is not calibrated on
    model_xys = {m: res[m]["xy"] for m in MODELS}
    nn_dists, nn_jsd = su.nn_comparison(res["data"]["xy"], model_xys)
    if nn_dists is not None:
        store["nn_r"] = nn_dists["r"]
        store["nn_data"] = nn_dists["data"]
        for m in MODELS:
            store[f"nn_{m}"] = nn_dists[m]
            store[f"nn_jsd_{m}"] = nn_jsd[m]

    # 2. Variability of every model: 2PCF band (mean +- std), JSD mean +- std,
    #    band width cv
    print(f"  [{name}] variability x {N_INSTANCES} (all models) ...",
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
    # SP band, stored separately for the band figure of notebook 07
    store["stoch_r"] = bands["SP"]["r"]
    store["stoch_xi_mean"] = bands["SP"]["xi_mean"]
    store["stoch_xi_std"] = bands["SP"]["xi_std"]

    # 3. N sweep (not for the largest datasets)
    if name not in GIANT:
        print(f"  [{name}] N sweep {SWEEP_NS} ...", flush=True)
        sw = su.sweep_constant_N(ds, ld, SWEEP_NS, models=MODELS,
                                 bstpp_pools=bstpp_pools)
        store["sweep_N"] = sw["N"]
        for m in MODELS:
            store[f"sweep_jsd_{m}"] = sw["jsd"][m]

    # 4. Population marks: hierarchical SP versus marked Hawkes
    if ld["pop_data"] is not None:
        print(f"  [{name}] population marks (Zipf slope + marked Hawkes) ...",
              flush=True)
        # SP slope calibrated on the rank-size (Zipf) exponent of the data
        zipf_d = su.fit_zipf_exponent(ld["pop_data"])
        sp_slope, zipf_sp = su.calibrate_sp_slope(ds, ld["gdf_edge"], zipf_d)
        store["zipf_data"] = zipf_d
        store["zipf_sp"] = zipf_sp
        store["sp_slope"] = sp_slope

        spp = su.build_sp_with_population(ds, ld["gdf_edge"], slope=sp_slope,
                                          seed=1)
        # Marked Hawkes: the population modulates the self-excitation
        xy_mh, m_mh = su.simulate_marked_hawkes(N, ld["geom_norm"],
                                                ld["pop_data"], ds["sigma"],
                                                seed=1)

        # gamma_hi / gamma_lo split for data, marked SP and marked Hawkes
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

        # mark correlation k_mm(r) for data, SP and marked Hawkes
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
    print(f"Datasets: {names}")
    print(f"comparison N = {args.n}, instances = {N_INSTANCES}\n")

    for name in names:
        out = CACHE / f"{slug(name)}.npz"
        if out.exists() and not args.force:
            print(f"{name} already cached ({out.name}), skipped")
            continue
        print(f"\n{'='*60}\n  {name}\n{'='*60}", flush=True)
        t0 = time.time()
        try:
            store = compute_one(name, su.DATASETS[name], args.n)
            np.savez(out, **{k: np.asarray(v, dtype=object) if isinstance(v, str)
                             else np.asarray(v) for k, v in store.items()})
            print(f"{name} saved in {time.time()-t0:.0f} s -> {out.name}",
                  flush=True)
        except Exception as e:
            import traceback
            print(f"{name} FAILED: {e}", flush=True)
            traceback.print_exc()

    print("\nDone.")


if __name__ == "__main__":
    main()
