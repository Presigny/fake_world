# Branch `emile-paper-2026`, code behind the FakeWorld manuscript

Author: Emile Emery. Branched from `main` at `d5a5afb` (worm / fruit-fly extraction).

This branch adds the code used for the manuscript *Replacing the random sprinkling:
a calibrated fractal point process* (SP + random mixture, comparison with
inferential point processes, power-grid application). It builds on the 2PCF
library of this repo. Every 2PCF computed on projected geodata goes through
`src/methods_two_point_correlation.py`.

The files of `main` (root scripts, `2pcf*.ipynb`, `fit_Soneira_Peebles*.ipynb`,
`generalized_2pcf.ipynb`, the extraction scripts) are not modified, except
`src/methods_two_point_correlation.py` (backward compatible, see below).

## Common structure (November 2026)

After the discussion with Charles, the code of this branch follows one structure,
shared with the pipeline of `main` (`scripts/readme.md`). Three points changed. The
state before the change is commit `8d18f31`.

**1. SP generation: the N-dimensional generator, also in 2D.**
`sp_helpers.soneira_peebles_nd(eta, lamb, L, R, center, domain, rng)` is the only
generator, called with `dim = 2` on the maps (`sp_helpers.build_sp_model(..., dim=2)`,
`calibration.SP_DIM = 2`) and `dim = 3` on the connectome. At the border, a child that
falls outside is drawn again alone, up to 64 times. The tree keeps its η^L leaves and
its geometry, and the generation ends whatever the size of the territory. The two
former options are worse: redrawing the whole tree until every point is inside
(`check_inside`) does not end when η^L is large, and keeping only the inside points of
a free tree (the former `build_sp_model`, now `build_sp_model_truncated`) removes whole
branches near the border and changes the number of leaves. The radial law of a child
is `r = R u^(1/dim)` (uniform in the disc, the definition of Soneira and Peebles);
`radial="linear"` gives back `r = R u` of the former 2D code.

**2. Amplitude ratio in log.**
`sp_helpers.log_amplitude_ratio`: log a = ⟨log ξ_data − log ξ_SP⟩ over the bins of the
fit window where both ξ are positive, then `n_rand = m_SP (a^(−1/2) − 1)`
(`dilution_from_a`). The former ρ = ⟨ξ_data/ξ_SP⟩ was a mean of ratios: one bin where
ξ_SP is close to zero could set it, it could be negative or above 1, and it was not
symmetric in data and model. The mean of the logs has none of these problems, and γ
and a are now measured on the same bins.

**3. One 2PCF chain, with a minimum number of pairs per bin.**
Every 2PCF goes through `methods_two_point_correlation.PCF_with_variance`: metres, 20
logarithmic edges from the mean nearest-neighbour distance to r_max, ξ at the upper
edge of the bin, Landy-Szalay with split random catalogues (5 × 7 000 points inside
the border), k = 3 repetitions, fit weighted by their spread on the bins 1 to
18. The comparison notebooks call it through `synthesis_utils.compute_2pcf(xy_norm, ld)`;
the former normalised estimator `compute_2pcf_norm` is removed. The fit window
(`sp_helpers.fit_window`) has one more rule: starting from the smallest r, a bin is
dropped as long as it holds fewer than `MIN_PAIRS = 100` data pairs DD. The first bin
with DD ≥ 100 opens the window and every bin above it is kept. The relative Poisson
noise of a bin is 1/√DD, 10 % at 100 pairs; below, ξ is set by a handful of pairs. The
pair counts come from `PCF_with_variance(..., return_counts=True)`, and the same window
is used for γ and for log a. The notebooks that compute a 2PCF on a map recall these
conventions in their second cell and print the window they used
(`su.describe_fit_window`).

What did not change: the files, the order of the calibration steps, the entry points
(`scripts/calibrate_all.py`, `calibration.calibrate(name)`, `su.load_dataset`,
`su.DATASETS`), and `methods_two_point_correlation.py` for existing callers. What
changed for a caller:

| Before | Now |
|--------|-----|
| `sh.build_sp_model(m, eta, gamma, R_ref, gdf_edge, crs)` (truncated tree, global `np.random`) | same call, now the N-dim generator; add `rng=np.random.default_rng(seed)`. Former behaviour: `build_sp_model_truncated` |
| `su.compute_2pcf_norm(xy, geom)` → r normalised | `su.compute_2pcf(xy, ld)` → r in metres; `full=True` returns counts, fit and window |
| `su.fit_powerlaw_norm(r, xi)` | `su.fit_gamma(r, xi, dd=None)` or `su.fit_powerlaw(pcf)` |
| `su.build_sp_mixture(ds, gdf_edge, bounds, geom, r_data, xi_data, N)` | `su.build_sp_mixture(ds, ld, pcf_data, N)` |
| `su.data_2pcf_at_N(ld, N)` → `(xy, r, xi)` | → `(xy, pcf)` with `pcf["r"]`, `pcf["xi"]`, `pcf["dd"]` |
| `su.marked_clustering_2pcf(xy, pop, geom)`, `mm.delta_gamma(xy, pop, geom)`, `mm.compute_2pcf_people(xy, pop, geom)` | same with `ld` in place of `geom` |
| `cal.dilution(...)` → `(rho, n_rand)` | → `(a, n_rand, info)`; columns `rho` → `a`, `log_a` in the summary |
| territory chosen notebook by notebook | `cal.load_territory(ds)`: whole multipolygon, or the mainland when the entry of `DATASETS` has `largest_only=True` (France, whose border file holds the overseas territories, and Germany) |
| `ld["r_data"]` normalised, `ld["r0_fit_norm"]` | `ld["r_data"]` in metres, `ld["r0_fit_m"]`, and the full dict `ld["pcf_data"]` |

## What is added

### Modules (`src/`)

| Path | Content |
|------|---------|
| `src/methods_two_point_correlation.py` | **modified**, backward compatible (see below) |
| `src/sp_helpers.py` | SP generation in N dimensions inside a border (`soneira_peebles_nd`, used with `dim=2` on the maps and `dim=3` on the connectome), fit window with the minimum-pairs criterion, power-law fit, amplitude ratio in log, cross-2PCF m-scan |
| `src/calibration.py` | per-dataset calibration of the SP mixture (γ, η, m*, λ, L, a), the parameters of Table 1 |
| `src/synthesis_utils.py` | dataset table (`DATASETS`), 2PCF of a normalised point set with the calibration chain (`compute_2pcf`), SP + random mixture, competing point processes (Poisson, LGCP, Hawkes, Cox-Hawkes), JSD, cache of `compute_all.py` |
| `src/powerlaw_fits.py` | estimators of γ compared in notebook 11 (weighted least squares, log-log, Poisson likelihood, `powerlaw` package) |
| `src/metadata_marks.py` | population marks from local density, mark-correlation profiles |
| `src/powergrid_sp.py` | grid growth on SP / city / uniform nodes, DC load flow and Joule losses |

### Notebooks (`notebooks/emile_2026/`)

The numbered notebooks of this branch are grouped in `notebooks/emile_2026/`, so
that they are separate from the notebooks of `main` (`notebooks/*.ipynb`). Each
one opens with its question, data, outputs and use in the paper, and ends with a
short list of results. Outputs are kept, except in notebook 09 (see below).

| Notebook | Result | Paper |
|----------|--------|-------|
| `calibration/*.ipynb` | per-dataset calibration, one notebook per dataset, all on `src/calibration.py` | Table 1 |
| `07_model_comparison` | SP mixture against Poisson, LGCP, Hawkes and Cox-Hawkes on the nine datasets | Table 2, Fig. 1 |
| `08_benchmark_complexity` | cost of each model as a function of N | Fig. 2 |
| `09_bstpp_inference` | SVI fits of LGCP, Hawkes and Cox-Hawkes with BSTPP | Table 2 |
| `10_gamma_consistency` | γ_cal against γ of the comparison extract with the common chain, effect of the minimum-pairs criterion, exponent and amplitude of the mixture over seeds, ranks over 10 realisations | Sec. 4, Limitations |
| `11_calibration_fits` | least squares against maximum likelihood for γ_cal, sensitivity of the amplitude ratio to the SP draw (former ρ against log a) | Sec. 3.2, Limitations |
| `01_metadata_generative_model` | population marks given to the SP nodes from the local density | not in the paper |
| `02_metadata_observables` | observables that include the population (M_norm, people 2PCF) | not in the paper |
| `03_estimator_comparison_belgium`, `_austria` | effect of the density estimator (k-NN or fixed radius) on the population coupling | not in the paper |
| `05_fly_2pcf` | 3D 2PCF of the fruit-fly neuron nuclei | not in the paper |
| `05bis_fly_sp3d` | 3D SP model of the fruit fly | not in the paper |
| `06_powergrid_sp_joule` | grid growth and Joule losses on SP nodes, against the RTE grid | not in the paper |

There is no notebook 00 or 04. The two notebooks 03 run the same code
(`DATASET_NAME`) on the two city catalogues with a real population.

### Austrian cities

The Austrian city catalogue is one of the nine datasets of the paper. Its raw
data (`data/cities/Austria_cities.csv`, `data/map/Austria.geojson`) are not
distributed with this branch, and are listed in `.gitignore`. The results are
kept. The entry of `DATASETS`, the calibration notebook
`calibration/austria_cities.ipynb`, the row of
`out/00_dataset_calibration_reference.csv` and the outputs of notebooks 03
(Austria), 07 and 10 come from local runs on these data. The other notebooks
skip a dataset whose files are missing.

The SVI fits of notebook 09 are long (5000 steps per model and per dataset). By default (`FORCE_FIT = False`) it reloads
the models stored in `out/bstpp_results` and only fits the missing ones. When
`out/bstpp_results` is a symbolic link, new fits are written to
`out/bstpp_reinferred` instead, so the shared models are never overwritten.

### Scripts (`scripts/`)

| Path | Content |
|------|---------|
| `scripts/calibrate_all.py` | runs `calibration.py` on every dataset whose files are present, writes `out/00_dataset_calibration.csv` |
| `scripts/compute_all.py` | incremental pre-computation for notebooks 07 and 10, one cache file per dataset in `out/synthesis_cache/`, or in `out/synthesis_cache_cp/` when `out/synthesis_cache` is a symbolic link |

### Data and results

| Path | Content |
|------|---------|
| `data/cities/Belgium_cities.csv`, `data/map/Belgium.geojson` | Belgian cities with population, and border |
| `data/clean_transport_dataset/France/` | `france_nat_earth.geojson` (border), `postes-electriques-rte.csv` (RTE substations), `noeuds_route500_gares.csv` and `noeuds_route500_commune.csv` (IGN ROUTE500 stations and municipalities) |
| `out/00_dataset_calibration_reference.csv` | calibration values used in the paper, with the notebook they come from |
| `out/*.csv` | summary tables written by the notebooks |
| `requirements.txt` | pinned versions of the environment used to run the notebooks |

Figures (`out/*.pdf`) and caches are not tracked, the notebooks regenerate them.

## Calibration of the datasets

The per-dataset analyses are in `notebooks/emile_2026/calibration/`, one notebook
per dataset. They all call the same chain, written once in `src/calibration.py`,
step by step. The original runs of March 2026, which give the values of Table 1 and
of `out/00_dataset_calibration_reference.csv`, are in the git history (commit
`8d18f31`).

1. Territory in a metric CRS and reference radius `R_ref = 0.5 × perimeter / 2π`.
2. Data 2PCF (Landy-Szalay, split random catalogues) and power-law fit, giving γ and r0.
3. Branching factor η, the number of KMeans clusters that maximises the silhouette score.
4. m-scan. The points are sorted by decreasing mark, the m first form A and the
   others B, and m* minimises `∫|ξ_AB(r)| dr`.
5. SP fractal with `λ = η^(1/(2−γ))` and at least m* nodes inside the territory,
   drawn by the N-dimensional generator with `dim = 2` (child-by-child rejection).
6. Dilution by uniform points, `log a = ⟨log ξ_data − log ξ_SP⟩` on the fit window,
   `n_rand = m_SP (a^(−1/2) − 1)`, then random deletion down to N nodes.

In step 2 the fit window is the bins 1 to 18 minus the small-r bins with fewer than
`MIN_PAIRS = 100` data pairs (see "Common structure").

```bash
python scripts/calibrate_all.py                              # all datasets present
python scripts/calibrate_all.py --only "Belgium Cities" "RTE France"
```

Two points matter when comparing with `out/00_dataset_calibration_reference.csv`.

- The random catalogues of the 2PCF are drawn without a fixed seed (as in
  `methods_two_point_correlation`), so γ and m* change slightly from one run
  to the next.
- The amplitude ratio depends on the SP draw, so the draw is seeded (`--seed`, 42
  by default). On Belgium, over ten draws, a goes from 0.08 to 0.19 with the log
  estimator (notebook 11). The former mean of the ratios ρ went from −0.29 to 0.35 on
  the same draws and had to be clipped on two of them, which is why it was replaced.

### Calibration exponent and exponent of the comparison extract

Table 1 gives γ_cal, measured on the complete dataset. The model comparison works on
an extract of 3 000 nodes. Both now use the same 2PCF chain, and notebook 10 checks
what is left between them. Five runs on the same extract have a standard deviation of
0.03 to 0.07 (0.12 on the 815 Belgian cities), which comes from the unseeded random
catalogues. Their mean gives back γ_cal within 0.06 on seven of the nine datasets
(+0.08 on Belgium, +0.13 on the railway stations). The gaps of up to 0.3 reported
before November 2026 came from the second estimator of the comparison, which is
removed. At N = 3 000 the first bin of the fit window holds 370 to 1 360 data pairs,
so `MIN_PAIRS = 100` drops no bin on these datasets; it protects the fits on smaller
samples.

The mixture reproduces γ of the extract within 0.08 on four datasets (RTE, railway
stations, ENEDIS, German stops) and is steeper by 0.14 to 0.30 on the five others,
with a spread of 0.03 to 0.17 over five seeds.

The calibration notebooks (`calibration/`) call `src/calibration.py` step by step.
Belgium, Austria, RTE and the railway stations were run again with the common chain
(November 2026); the five large sets (ENEDIS, municipalities, roads, German and Swiss
stops) have the new code but no outputs yet. The values of
`out/00_dataset_calibration_reference.csv` and of `DATASETS` are still those of the
runs of March 2026 (Table 1). The new runs give γ = 0.84 (Belgium, 0.81 in Table 1),
0.76 (Austria, 0.79), 0.91 (RTE, 0.93) and 0.76 (railway stations, 0.64). The notebooks
also print the exponent of the fractal alone ("SP γ"), fitted with the same estimator
on one realisation. Eq. (1) of the paper holds for the ensemble average and between
r_min and R, so the two need not agree.

## Changes to `src/methods_two_point_correlation.py`

All of `main` is kept, nothing is removed or renamed.

- `binning_data` keeps `np.bincount(..., minlength=nbins)` from `main`.
- `generate_random_point`: `crs` now defaults to `None`. On `main`,
  `compute_one_DR_RR(..., crs=None)` called `generate_random_point(gdf_edge, size)`
  without `crs`, which raised `TypeError`. The worm path now runs.
- `load_df_to_gdf(path, threshold, threshold_variable="population")`: the filter
  column is a parameter (default unchanged).
- `compute_DD` and `compute_one_DR_RR` are computed block by block to bound
  memory. Same output, same pair order as `pdist`, same `int32` truncation.
- `compute_two_point_correlation_2019`: `hist_DR` and `hist_RR` are cut or
  zero-padded to the length of `hist_DD` (any gap, not only ±1).
- `compute_two_point_correlation_2019`, `PCF_with_variance` and
  `compute_two_point_correlation_bootstrap` take `return_counts=False`. With `True`
  they also return the pair counts (`DD`, `DR`, `RR` per bin, and the sizes), which
  the minimum-pairs criterion needs. The default output is unchanged.
- Debug `print`s are commented out.
- New functions. `compute_two_point_correlation_bootstrap` (sub-sampling for
  large N), `compute_two_point_correlation_density_groups` (2PCF by metadata
  group), and a 3D block (`load_points_3d`, `coords_3d`, `sample_in_hull`,
  `pair_counts`, `compute_two_point_correlation_nd`, `PCF_with_variance_nd`)
  using KD-tree pair counts. The 3D functions use the same estimator,
  `compute_LS_correlation_2019`.

Checked on the Belgian cities with frozen random catalogues. `compute_DD`,
`compute_one_DR_RR` and `compute_two_point_correlation_2019` (rmax = None,
200 km, 400 km) give results identical to `main`, bit for bit.

## Correspondence with the functions of `main`

| Here | On `main` | Status |
|------|-----------|--------|
| `sp_helpers.soneira_peebles_border` | `soneira_peebles_border` (notebook `fit_Soneira_Peebles`) | same generator, centroid seed, `r ~ U(0, R/λ^i)`, `θ ~ U(0, 2π)` |
| `sp_helpers.build_sp_model` (`soneira_peebles_nd`, `dim=2`) | `check_inside` | **differs**. `check_inside` redraws the whole tree until every point is inside, `build_sp_model` redraws each outside child alone (the tree keeps its η^L leaves) and raises `L` until at least `m*` nodes |
| `λ = η^(1/(2-γ))` | same | same |
| `n_rand = m_SP (1/√a − 1)`, `log a = ⟨log ξ_data − log ξ_SP⟩` | `find_good_L`: `α = 1/⟨|ξ_SP/ξ_data|⟩`, `Nr = (1/√α − 1) η^L` | same dilution law, **different estimator of the amplitude ratio** (mean of the logs on the fit window against inverse of the mean of the inverse ratio on all bins) |
| `sp_helpers._run_cross_pcf` | `find_cross_2pcf` (notebook `generalized_2pcf`) | same formula, `[N² ξ − m² ξ_AA − (N−m)² ξ_BB] / (2m(N−m))` |
| `sp_helpers.fit_2pcf` | power-law fit in notebook `2pcf` and `scripts/fit_power_law.py` | same model `(r/r0)^-γ`, σ-weighted, bins `1:19`, adds bounds, NaN handling and the minimum-pairs criterion (`fit_power_law.py` drops the first bin, or the first two for three countries, by hand) |
| `synthesis_utils.compute_2pcf` | `PCF_with_variance` | same function, called after mapping the normalised points back to metres |

## Data not included

Too large for git, or not ours to redistribute. `synthesis_utils.DATASETS`
expects them at the paths below. A dataset whose files are missing is skipped
by `calibrate_all.py`, and each notebook only needs the files of the datasets
it runs on.

- `data/clean_transport_dataset/France/postes-electriques-enedis.csv` (ENEDIS
  substations, 42 MB) and `noeuds_route500_routes.csv` (IGN ROUTE500 road nodes, 42 MB)
- `data/clean_transport_dataset/Germany/localtransitgermany.csv` (19 MB) and
  `germany_nat_earth.geojson`
- `data/clean_transport_dataset/Switzerland/swiss_stops.csv` (3.5 MB) and
  `switzerland_nat_earth.geojson`
- `data/fly_neuron_net/neurons.zip` (notebooks 05 and 05bis, built by `scripts/extract_fruitfly.py`)
- `data/tgm/`, a copy of (or a link to) the TransmissionGrowthModel repository,
  which provides the RTE graph `out/RTE_weighted.joblib` for `powergrid_sp.py`
  and notebook 06
- `out/bstpp_results/`, the BSTPP models fitted by notebook 09, and
  `out/synthesis_cache/`, the caches of `compute_all.py`. Both can be regenerated.
  When `out/synthesis_cache` is a symbolic link (cache shared with another project,
  former chain), `compute_all.py` writes to and notebooks 07 and 10 read from
  `out/synthesis_cache_cp/`, so the shared cache is never overwritten.

## Running

```bash
pip install -r requirements.txt
pip install --no-deps bstpp==0.1.4   # notebooks 07, 09, compute_all; see requirements.txt
cd notebooks/emile_2026 && jupyter lab      # notebooks add ../../src to sys.path
```

For the model comparison, run notebook 09 (BSTPP fits), then
`scripts/compute_all.py`, then notebooks 07 and 10. Notebooks 01 to 06 and 08 are
independent of them. Notebooks 01 to 03 (Belgium) run on the data of this branch
alone.
