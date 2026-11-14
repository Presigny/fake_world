# Synthetic world scripts description

# Synthetic world scripts description

## Principles
Ths document is meant to describe the process to fit, generate and compare structurally and dynamically the proposed SP+random sprinkle model with the data and random sprinkle of points. 

The process was designed for the datasets of cities and the C.elegans connectome.

Each script is performing a task and the serialization of the datasets are operated through bash files, unless otherwise mentionned

## extract_cities.py

Extract the cities from the countries in _l_country_ coming from the CNTR_RG_01M_2024_4326.geojson file.

Data type: geopandas dataframe

## create_map.py

Extract the border of the countries _l_country_ coming from the worldcities database.

## 2pcf.py

Compute the two-point correlation function (2pcf) of the input datasets based on a pandas dataframe of the nodes and a geopandas dataframe of their border.

Serialization oeprated through bash file.

N.B: Ideally, N_run*size > 20* number points in dataset
### Parameters
- "--name", type=str, default="Sweden": name of the dataset to load
- "--threshold", type=int, default=1: minimum population for cities to be included
- "--N_run", type=int, default=10: Size of the random catalog
- "--size", type=int, default=1000: Number of points of each random sprinkle in the catalog
- "--k", type=int, default=3: Number of times the process is repeated to build a statistical variance of the 2pcf
- "--rmax", type=float, default=None: maximal distance to consider, if None, take the maximal distance between two points+1
- "--scale", type=str, default="log": scale of the 2pcf (linear or log)
- "--nbins", type=int, default=20: number of bins to be considered in the 2pcf

### Output

Data type: python dictionnary

- "r_edges": binning for the 2pcf
- "xi": values of the 2pcf for the considered binning and repeated K times
- "distance_distribution": all pairs of distance between two points in the considered data

## fit_power_law.py

Fit the empirical 2pcf with powerlaws using non-linear regression (curve_fit from scipy). The first bin of the 2pcf is disregarded for the fit. For Czechia, Slovakia,Slovenia: it is the first two bins.

Serialization operated through bash file.

### Parameters
cf. _2pcf.py_

### Output
Data type: python dictionnary
On top of the output of _2pcf.py_

- "fit_SP": parameters for the fit (power law exponent and scale coefficient)
- "covariance_fit": covariance matrix of the fitted parameters

## compute_number_clusters.py

Compute the 5 best potential partition of the input datasets using K mean clustering, with quality assessed by silhouette score.

kmax=10, is the maximum number considered in the computation.

Serialization operated through bash file.

### Parameters
cf. _2pcf.py_

### Output
Data type: python dictionnary
On top of the output of _fit_power_law.py_

- "number_points": number of points in the input dataset
- "clustering": list of the 5 best partitions from Kmeans clustering
- "silhouette_score": list of the silhouett scores associated with the partitions

## fit_full_custom_model.py

Fit the Soneira Peebles + random with the input empirical datasets using the fitted power-law parameters and the previously computed partitions, assimilated to eta . For different values of L, the algorithm first generates a pure Soneira-Peebles and then compute the requires ratio SP points random points. Then it computes the euclidean distance between the generated 2pcf and the empirical 2pcf for each combination of generated 2pcf.

- diminish_radius = 0.75 # multiplication factor for the maximum radius, to be lowered if the SP does not fit inside the considered border quick enough

 - iteration_pure_SP = 1 number of try to set the basis of the Soneira-Peebles 2pcf
- number_of_try_for_each_parameter = 1 number of generated 2pcf SP+random by set of paraemters to compare with the reference

Serialization operated through bash file.

### Parameters
cf. _2pcf.py_

### Output
Data type: python dictionnary
On top of the output of _compute_number_clusters.py_

- "save_dist": list of euclidean distance between empirical and the best eta, L combination (as defined by the euclidean distance)
- "L": list of L (number of steps in the SP) associated with the distances (index-wise)
- "eta": list of L associated with the distances (index-wise)
- "save_random": list of number of random points to be added in the SP associated with the distances (index-wise)
- "iteration_pure_SP": number of try to set the basis of the Soneira-Peebles 2pcf
- "number_of_try_for_each_parameter": number of generated 2pcf SP+random by set of paraemters to compare with the reference

## compare_distance_distribution.py

Generate 100 Soneira Peebles + random with the best set of parameters as well as 100 purely random ones and calculate the Kullback-Leibler divergence between the distance distribution of the random sprinkles/SP+random and the empirical ones.

Serialization operated through bash file.

### Parameters
cf. _2pcf.py_

### Output
Data type: python dictionnary
On top of the output of _fit_full_custom_model.py_

- "KL_random": list of Kullback-Leibler divergence between the distance distibution of random sprinkle and the empirical one.
- "KLSP": list of Kullback-Leibler divergence between the distance distibution of Soneira-Peebles+ random sprinkle and the empirical one.
- "l_gdf": list of geopandas dataframe of generated sprinkles
- "l_gdf_random": list of geopandas dataframe of generated sprinkles
- "cut_off_distance_distribution": distance after 

## branching_city.py

For the city datasets, take the lists of generated sprinkles of random and SP+random points and compute their mobility matrix out using the radiation model out of it.

- pop_per_node=1000: synthetic population per nodes

Serialization operated through bash file.

### Parameters
cf. _2pcf.py_

### Output
Data type: list of  100 npy files

-"mobility_random": mobility matrix of the random sprinkle
-"mobility_generated":  mobility matrix of the SP+random sprinkle

## run_epidemics.py

Run SIR epidemics on the 100 Generated Soneira Peebles + random and the 100 randoms and compare them with the SIR epidemics of the empirical datasets.
- np.random.seed(40): for reproducibility of the random attribution of infected
- beta = 0.13
- gamma = 0.1
- mobility_rate = 0.01
- tmax=800
- I0[initial_patch] = 10: number of infected on a random patch
- n_jobs=10: number of workers for the parallelization for the SIR dynamics

Serialization operated through bash file.

### Parameters
cf. _2pcf.py_

### Output
Data type: python dictionnary
On top of the output of _fit_full_custom_model.py_

- "Delta_Ricovered_random": list of Kullback-Leibler divergence between the distance distibution of random sprinkle and the empirical one.
- "Delta_Ricovered_random": list of Kullback-Leibler divergence between the distance distibution of Soneira-Peebles+ random sprinkle and the empirical one.

## mobility_clustering.py
Compute the global weighted directed clustering coefficient for random and SP+random. We can compute globally or for the cutoff.
Make the boxplot of the ratio between the generated models and the clustering of the data.

- nocut = False: If True the clustering coefficnet is computed only on pairs whose distance is inferior to d_load["cut_off_distance_distribution"]
- n_jobs=10: number of workers for the parallelization for the SIR dynamics
### Parameters
cf. _2pcf.py_

### Output
Data type: python dictionnary
On top of the output of _fit_full_custom_model.py_

-"ratio_random_clustering": list of ratio between global clustering coefficient of random sprinkle and the one of the data

-"ratio_generated_clustering": list of ratio between global clustering coefficient of SP+random sprinkle and the one of the data
