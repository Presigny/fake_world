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

## What is added

### Modules (`src/`)

| Path | Content |
|------|---------|
| `src/methods_two_point_correlation.py` | **modified**, backward compatible (see below) |
| `src/sp_helpers.py` | SP generation inside a border, power-law fit, cross-2PCF m-scan, SP in arbitrary dimension (3D connectome) |
| `src/calibration.py` | per-dataset calibration of the SP mixture (γ, η, m*, λ, L, ρ), the parameters of Table 1 |
| `src/synthesis_utils.py` | dataset table (`DATASETS`), normalised 2PCF, SP + random mixture, competing point processes (Poisson, LGCP, Hawkes, Cox-Hawkes), JSD, cache of `compute_all.py` |
| `src/metadata_marks.py` | population marks from local density, mark-correlation profiles |
| `src/powergrid_sp.py` | grid growth on SP / city / uniform nodes, DC load flow and Joule losses |

### Notebooks (`notebooks/emile_2026/`)

The numbered notebooks of this branch are grouped in `notebooks/emile_2026/`, so
that they are separate from the notebooks of `main` (`notebooks/*.ipynb`). Each
one opens with its question, data, outputs and use in the paper, and ends with a
short list of results. Outputs are kept, except in notebook 09 (see below).

| Notebook | Result | Paper |
|----------|--------|-------|
| `00_dataset_calibration` | calibration chain of `calibration.py`, step by step, on Belgium | Sec. 3.1 |
| `calibration/*.ipynb` | original per-dataset calibration runs (March 2026), outputs kept | Table 1 |
| `07_model_comparison` | SP mixture against Poisson, LGCP, Hawkes and Cox-Hawkes on the nine datasets | Table 2, Fig. 1 |
| `08_benchmark_complexity` | cost of each model as a function of N | Fig. 2 |
| `09_bstpp_inference` | SVI fits of LGCP, Hawkes and Cox-Hawkes with BSTPP | Table 2 |
| `10_gamma_consistency` | γ_cal against the exponent of the comparison settings, exponent of the mixture, ranks over 10 realisations | Sec. 4, Limitations |
| `01_metadata_generative_model` | population marks given to the SP nodes from the local density | not in the paper |
| `02_metadata_observables` | observables that include the population (M_norm, people 2PCF) | not in the paper |
| `03_estimator_comparison_belgium`, `_austria` | effect of the density estimator (k-NN or fixed radius) on the population coupling | not in the paper |
| `05_fly_2pcf` | 3D 2PCF of the fruit-fly neuron nuclei | not in the paper |
| `05bis_fly_sp3d` | 3D SP model of the fruit fly | not in the paper |
| `06_powergrid_sp_joule` | grid growth and Joule losses on SP nodes, against the RTE grid | not in the paper |

There is no notebook 04. The two notebooks 03 run the same code
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

Notebook 09 is stored without outputs, because the SVI fits are long (5000
steps per model and per dataset). By default (`FORCE_FIT = False`) it reloads
the models stored in `out/bstpp_results` and only fits the missing ones. When
`out/bstpp_results` is a symbolic link, new fits are written to
`out/bstpp_reinferred` instead, so the shared models are never overwritten.

### Scripts (`scripts/`)

| Path | Content |
|------|---------|
| `scripts/calibrate_all.py` | runs `calibration.py` on every dataset whose files are present, writes `out/00_dataset_calibration.csv` |
| `scripts/compute_all.py` | incremental pre-computation for notebooks 07 and 10, one cache file per dataset in `out/synthesis_cache/` |

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

The per-dataset analyses were first done in separate notebooks, one per
dataset. They are in `notebooks/emile_2026/calibration/` with their original outputs, which
give the values of Table 1 and of `out/00_dataset_calibration_reference.csv`.
Only the import paths and the dataset labels were changed (the helpers now come
from `src/sp_helpers.py`). These notebooks all repeat the same chain, which is
now written once in `src/calibration.py`.

1. Territory in a metric CRS and reference radius `R_ref = 0.5 × perimeter / 2π`.
2. Data 2PCF (Landy-Szalay, split random catalogues) and power-law fit, giving γ and r0.
3. Branching factor η, the number of KMeans clusters that maximises the silhouette score.
4. m-scan. The points are sorted by decreasing mark, the m first form A and the
   others B, and m* minimises `∫|ξ_AB(r)| dr`.
5. SP fractal with `λ = η^(1/(2−γ))` and at least m* nodes inside the territory.
6. Dilution by uniform points, `ρ = ⟨ξ_data/ξ_SP⟩`, `n_rand = m_SP (ρ^(−1/2) − 1)`,
   then random deletion down to N nodes.

```bash
python scripts/calibrate_all.py                              # all datasets present
python scripts/calibrate_all.py --only "Belgium Cities" "RTE France"
```

Two points matter when comparing with `out/00_dataset_calibration_reference.csv`.

- The random catalogues of the 2PCF are drawn without a fixed seed (as in
  `methods_two_point_correlation`), so γ and m* change slightly from one run
  to the next.
- The amplitude ratio ρ depends strongly on the SP draw. On Belgium, the mean
  ratio of ten SP draws spreads over about 0 to 0.3 (0.12 ± 0.09 in the stored
  run of notebook 00), and one draw gives a negative mean ratio, which is
  clipped to 1e-4. The SP draw is therefore seeded (`--seed`, 42 by
  default), `calibration.py` prints a warning when ρ is clipped, and section 6bis
  of notebook 00 shows the spread over ten seeds.

### Calibration exponent and comparison exponent

Table 1 gives γ_cal, from the calibration chain above. The model comparison
measures the 2PCF with other settings (`synthesis_utils.compute_2pcf_norm`, on
the extract of 3000 nodes), which give an exponent γ̂. The paper describes both
settings (Sec. 3.2 and 4.1). Notebook 10 shows that the extract is not what
separates them. The calibration chain is itself noisy, because its random
catalogues are not reproducible. Five runs on the same extract have a standard
deviation of 0.01 to 0.07, and their mean gives back γ_cal within 0.06 on seven
of the nine datasets (0.09 for the Austrian cities, 0.10 for the train
stations). On the same nodes, the gap with γ̂ is 6 to 10 standard deviations on
the two city catalogues (−0.31 and −0.38) and the French municipalities (+0.25),
4 to 5 on the RTE and ENEDIS substations, and within the noise elsewhere. It
comes from the estimator. The two 2PCFs are fitted on different windows, with
different bins and weights, and the empirical 2PCFs bend down at large r, so a
single exponent depends on the window. Setting λ from γ̂ instead of γ_cal was
tested. Both JSDs are then larger on five of the nine datasets, and the
calibration is kept.

The calibration notebooks (`calibration/`) apply exactly the method of the
paper, and their data exponent is γ_cal. They also print the exponent of the
fractal alone ("SP γ"), fitted with the same estimator on one realisation. It
differs from γ_cal by 0.25 to 0.33 on the municipalities, roads, German and
Swiss stops, and by less than 0.08 elsewhere. Eq. (1) of the paper holds for the
ensemble average and between r_min and R, whereas this check uses one clipped
realisation on a fixed window, so the two need not agree.

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
| `sp_helpers.build_sp_model` | `check_inside` | **differs**. `check_inside` redraws the whole tree until every point is inside, `build_sp_model` keeps the inside points and raises `L` until at least `m*` remain |
| `λ = η^(1/(2-γ))` | same | same |
| `n_rand = m_SP (1/√ρ − 1)`, `ρ = ⟨ξ_data/ξ_SP⟩` | `find_good_L`: `α = 1/⟨|ξ_SP/ξ_data|⟩`, `Nr = (1/√α − 1) η^L` | same dilution law, **different estimator of the amplitude ratio** (mean of the ratio against inverse of the mean of the inverse, all bins against bins with ξ_SP > 0) |
| `sp_helpers._run_cross_pcf` | `find_cross_2pcf` (notebook `generalized_2pcf`) | same formula, `[N² ξ − m² ξ_AA − (N−m)² ξ_BB] / (2m(N−m))` |
| `sp_helpers.fit_2pcf` | power-law fit in notebook `2pcf` | same model `(r/r0)^-γ`, σ-weighted, bins `1:19`, adds bounds and NaN handling |
| `synthesis_utils.compute_2pcf_norm` | `PCF_with_variance` | **different implementation**, see below |

### `compute_2pcf_norm` against `PCF_with_variance`

The model comparisons of the paper (JSD, constant-N protocol) use
`compute_2pcf_norm`, written for coordinates normalised to [0,1]². It computes
the same Landy-Szalay estimator with split random catalogues, with three
differences.

- Bins. `n_bins+1` log edges from `0.5 × mean NN distance` to `0.55 × max(DD)`,
  with ξ placed at the geometric bin centre. `main` uses `nbins` edges from the
  mean NN distance to `rmax`, with ξ placed at the upper edge of each bin
  (this moves `r0`, not `γ`).
- Random catalogue. One fixed seed, 5 × 5000 points, drawn inside the largest
  polygon only (islands dropped). `main` draws with `sample_points` in the whole
  multipolygon.
- Fit (`fit_powerlaw_norm`). Unweighted, positive bins only.

On the Belgian cities this gives γ̂ = 0.515, whereas `PCF_with_variance` with
the weighted fit gives γ ≈ 0.81–0.90. The γ used to calibrate λ (`gamma_data` in
`DATASETS`) comes from the `main` pipeline. The paper describes both settings in
Sec. 3.2 and 4.1. Table 1 only gives γ_cal, and γ̂ appears in the Limitations.

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

## Running

```bash
pip install -r requirements.txt
cd notebooks/emile_2026 && jupyter lab      # notebooks add ../../src to sys.path
```

For the model comparison, run notebook 09 (BSTPP fits), then
`scripts/compute_all.py`, then notebooks 07 and 10. Notebooks 00 to 06 and 08 are
independent of them. Notebooks 00 to 03 (Belgium) run on the data of this branch
alone.
