# Branch `emile-paper-2026` — code behind the FakeWorld manuscript

Author: Emile Emery. Branched from `main` at `d5a5afb` (worm / fruit-fly extraction).

This branch adds the code used for the manuscript *Replacing the random sprinkling:
a calibrated fractal point process* (SP + random mixture, comparison with
inferential point processes, power-grid application). It builds on the 2PCF
library of this repo: every 2PCF computed on projected geodata goes through
`src/methods_two_point_correlation.py`.

## What is added

| Path | Content |
|------|---------|
| `src/methods_two_point_correlation.py` | **modified**, backward compatible (see below) |
| `src/sp_helpers.py` | SP generation inside a border, power-law fit, cross-2PCF m-scan, SP in arbitrary dimension (3D connectome) |
| `src/synthese_utils.py` | dataset table (`DATASETS`), normalised 2PCF, SP + random mixture, competing point processes (Poisson, LGCP, Hawkes, Neyman–Scott), JSD |
| `src/metadata_marks.py` | population marks from local density, mark-correlation profiles |
| `src/powergrid_sp.py` | grid growth on SP / city / uniform nodes, DC load flow and Joule losses |
| `notebooks/01_…` – `06_…`, `05bis_…` | one notebook per result of the paper (outputs kept) |
| `scripts/compute_all.py` | incremental pre-computation for all datasets (cache per dataset) |
| `scripts/powergrid_data_build.py` | builds the power-grid inputs |
| `data/clean_transport_dataset/France/` | `france_nat_earth.geojson` (border), `postes-electriques-rte.csv` (RTE substations), `noeuds_route500_gares.csv` and `noeuds_route500_commune.csv` (IGN ROUTE500 stations and municipalities) |
| `out/*.csv` | summary tables written by the notebooks |
| `requirements.txt` | pinned versions, same as the headers of the 2PCF library |

## Changes to `src/methods_two_point_correlation.py`

All of `main` is kept; nothing is removed or renamed.

- `binning_data`: keeps `np.bincount(..., minlength=nbins)` from `main`.
- `generate_random_point`: `crs` now defaults to `None`. On `main`,
  `compute_one_DR_RR(..., crs=None)` called `generate_random_point(gdf_edge, size)`
  without `crs`, which raised `TypeError`; the worm path now runs.
- `load_df_to_gdf(path, threshold, threshold_variable="population")`: the filter
  column is a parameter (default unchanged).
- `compute_DD`, `compute_one_DR_RR`: computed block by block to bound memory.
  Same output, same pair order as `pdist`, same `int32` truncation.
- `compute_two_point_correlation_2019`: `hist_DR` and `hist_RR` are cut or
  zero-padded to the length of `hist_DD` (any gap, not only ±1).
- Debug `print`s are commented out.
- New: `compute_two_point_correlation_bootstrap` (sub-sampling for large N),
  `compute_two_point_correlation_density_groups` (2PCF by metadata group),
  and a 3D block (`load_points_3d`, `coords_3d`, `sample_in_hull`,
  `pair_counts`, `compute_two_point_correlation_nd`, `PCF_with_variance_nd`)
  using KD-tree pair counts. The 3D functions use the same estimator,
  `compute_LS_correlation_2019`.

Checked on the Belgian cities with frozen random catalogues: `compute_DD`,
`compute_one_DR_RR` and `compute_two_point_correlation_2019` (rmax = None,
200 km, 400 km) give results identical to `main`, bit for bit.

## Correspondence with the functions of `main`

| Here | On `main` | Status |
|------|-----------|--------|
| `sp_helpers.soneira_peebles_border` | `soneira_peebles_border` (notebook `fit_Soneira_Peebles`) | same generator: centroid seed, `r ~ U(0, R/λ^i)`, `θ ~ U(0, 2π)` |
| `sp_helpers.build_sp_model` | `check_inside` | **differs**: `check_inside` redraws the whole tree until every point is inside; `build_sp_model` keeps the inside points and raises `L` until at least `m*` remain |
| `λ = η^(1/(2-γ))` | same | same |
| `n_rand = m_SP (1/√ρ − 1)`, `ρ = ⟨ξ_data/ξ_SP⟩` | `find_good_L`: `α = 1/⟨|ξ_SP/ξ_data|⟩`, `Nr = (1/√α − 1) η^L` | same dilution law; **different estimator of the amplitude ratio** (mean of the ratio vs inverse of the mean of the inverse, all bins vs bins with ξ_SP > 0) |
| `sp_helpers._run_cross_pcf` | `find_cross_2pcf` (notebook `generalized_2pcf`) | same formula, `[N² ξ − m² ξ_AA − (N−m)² ξ_BB] / (2m(N−m))` |
| `sp_helpers.fit_2pcf` | power-law fit in notebook `2pcf` | same model `(r/r0)^-γ`, σ-weighted, bins `1:19`; adds bounds and NaN handling |
| `synthese_utils.compute_2pcf_norm` | `PCF_with_variance` | **different implementation**, see below |

### `compute_2pcf_norm` vs `PCF_with_variance`

The model comparisons of the paper (JSD, constant-N protocol) use
`compute_2pcf_norm`, written for coordinates normalised to [0,1]². It computes
the same Landy–Szalay estimator with split random catalogues, but:

- bins: `n_bins+1` log edges from `0.5 × mean NN distance` to `0.55 × max(DD)`,
  ξ placed at the geometric bin centre; `main` uses `nbins` edges from the
  mean NN distance to `rmax`, with ξ placed at the upper edge of each bin
  (this moves `r0`, not `γ`);
- random catalogue: one fixed seed, 5 × 5000 points, drawn inside the largest
  polygon only (islands dropped); `main` draws with `sample_points` in the whole
  multipolygon;
- fit (`fit_powerlaw_norm`): unweighted, positive bins only.

On the Belgian cities this gives γ̂ = 0.515, whereas `PCF_with_variance` +
weighted fit gives γ ≈ 0.81–0.90. The γ used to calibrate λ (`gamma_data` in
`DATASETS`) comes from the `main` pipeline; the paper reports both columns
(Table 1, γ_cal and γ̂).

## Data not included

Too large for git, or not ours to redistribute. `synthese_utils.DATASETS`
expects them at these paths:

- `data/map/Austria.geojson`, `data/cities/Austria_cities.csv` (same format as
  the Belgian files; notebooks 01–04)
- `data/clean_transport_dataset/France/`: `postes-electriques-enedis.csv` (42 MB),
  `noeuds_route500_routes.csv` (42 MB)
- `data/clean_transport_dataset/Germany/localtransitgermany.csv` (19 MB) + border
- `data/clean_transport_dataset/Switzerland/swiss_stops.csv` (3.5 MB) + border
- `data/fly_neuron_net/neurons.zip` (notebooks 05, 05bis; built by `scripts/extract_fruitfly.py`)
- `data/tgm/`: the TransmissionGrowthModel repo (RTE graph `out/RTE_weighted.joblib`),
  for `powergrid_sp.py` and notebook 06
- `data/bstpp_results/`, `data/synthese_cache/`: fitted BSTPP models and
  `compute_all.py` caches (regenerable)

## Running

```bash
pip install -r requirements.txt
cd notebooks && jupyter lab      # notebooks add ../src to sys.path
```

The Belgian parts of notebooks 03 and 04 run on the data of this branch alone;
the Austrian parts need the Austrian files listed above.
