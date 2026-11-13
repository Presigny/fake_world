import sys
from pathlib import Path
path = Path().cwd().parent / "src" #to add the src directory to the path regognized by Python
sys.path.append(str(path))
import methods_two_point_correlation as mtpc
import fit_custom_Soneira_Peebles as fcsp
import save_load_pickle as slp
import argparse
import numpy as np
import geopandas as gpd
import os
import matplotlib.pyplot as plt
from scipy.spatial import distance
import methods_epidemics as mepi
import pandas as pd
import gc
from joblib import Parallel, delayed
from scipy.stats import ttest_ind

def plot_SIR(t, S, I, R):

    plt.figure(figsize=(10, 6))

    plt.step(t, S, where="post", label="Susceptible")
    plt.step(t, I, where="post", label="Infected")
    plt.step(t, R, where="post", label="Recovered")

    plt.xlabel("Time")
    plt.ylabel("Number of patches")
    plt.title("SIR epidemic on a radiation mobility network")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def run_simulation(i):
    print(i)
    path_random = (
        name_load / "random" /
        Path(f"{i}_random_mobility_matrix_{threshold}_{N_run}_{size}_{k}__{rmax}_{scale}_{nbins}.npy")
    )

    path_generated = (
        name_load / "generated" /
        Path(f"{i}_generated_mobility_matrix_{threshold}_{N_run}_{size}_{k}__{rmax}_{scale}_{nbins}.npy")
    )

    # Load matrices
    mobility_random = np.load(path_random, mmap_mode="r")
    mobility_generated = np.load(path_generated, mmap_mode="r")

    # Random mobility SIR
    t, S, I, R_rand = mepi.metapopulation_SIR(
        pop_distribution=pop_distribution,
        mobility_matrix=mobility_random[0],
        beta=beta,
        gamma=gamma,
        mobility_rate=mobility_rate,
        initial_infected=I0,
        tmax=tmax
    )

    R_rand_total = R_rand.sum(axis=1)

    # Generated mobility SIR
    t, S, I, R_gen = mepi.metapopulation_SIR(
        pop_distribution=pop_distribution,
        mobility_matrix=mobility_generated[0],
        beta=beta,
        gamma=gamma,
        mobility_rate=mobility_rate,
        initial_infected=I0,
        tmax=tmax
    )

    R_gen_total = R_gen.sum(axis=1)

    # Release intermediate arrays
    del mobility_random, mobility_generated
    del S, I, R_rand, R_gen
    gc.collect()

    return i, np.sum(R_rand_total[-1]), np.sum(R_gen_total[-1])

    
# Bookkeeping of important path in the project
path_data = Path("/media/utente/LaCie/fake_world_data/")
data_dir = path_data#Path().cwd().parent/Path("data")
data_city = data_dir / "cities"
data_map = data_dir / "map"
data_2pcf = data_dir/"2pcf"
data_fit = data_dir /"fit"
out_dir = path_data #Path().cwd().parent/Path("out")
plot_dir = out_dir / Path("plot_infected_difference")
data_mobility = data_dir/ "mobility"
##Create directories and path to save data


parser = argparse.ArgumentParser(description="Compute KL divergence on distance distribution")

parser.add_argument("--name", type=str, default="Austria")
parser.add_argument("--threshold", type=int, default=1)
parser.add_argument("--N_run", type=int, default=10)
parser.add_argument("--size", type=int, default=4500)
parser.add_argument("--k", type=int, default=3)
parser.add_argument("--rmax", type=float, default=None)
parser.add_argument("--scale", type=str, default="log", choices=["log", "linear"])
parser.add_argument("--nbins", type=int, default=20)
parser.add_argument("--n_iteration", type=int, default=100)

args = parser.parse_args()

name = args.name
threshold = args.threshold
N_run = args.N_run
size = args.size
k = args.k
rmax = args.rmax
scale = args.scale
nbins = args.nbins
n_iteration = args.n_iteration

#Load the polygon file for border
path_border = data_map / f"{name}.geojson"
path_city = data_city / f"{name}_cities.csv"
gdf_data = mtpc.load_df_to_gdf(path_city,threshold)
gdf_edge = gpd.read_file(path_border)

name_load = data_mobility/ Path(name)
name_load_dico = data_fit/ Path(name) / Path(f"KL_{threshold}_{N_run}_{size}_{k}__{rmax}_{scale}_{nbins}")
name_save = data_fit/ Path(name) / Path(f"epidemics_{threshold}_{N_run}_{size}_{k}__{rmax}_{scale}_{nbins}")
d_load = slp.load_results(name_load_dico)
N = d_load["number_points"] #number of point sin the dataset
crs = mtpc.crs_selector(name) #cooridnate reference system associated with the datset
gdf_edge = gdf_edge.to_crs(crs) # Polygon of the border of the country
gdf_data = gdf_data.to_crs(crs)
cut = d_load["cut_off_distance_distribution"]
KL = d_load["KLSP"]
path_data = name_load /"data" / Path(f"data_mobility_matrix_{threshold}_{N_run}_{size}_{k}__{rmax}_{scale}_{nbins}.npy")
mobility_data = np.load(path_data)
coord_data = gdf_data.get_coordinates().to_numpy()
pop_per_node = 1000
pop_distribution = [pop_per_node for i in range(N)]

np.random.seed(40)

beta = 0.13
gamma = 0.1
mobility_rate = 0.01
tmax=800

initial_patch = np.random.randint(0, len(pop_distribution))

I0 = np.zeros(len(pop_distribution))
I0[initial_patch] = 10

t, S, I, R = mepi.metapopulation_SIR(
pop_distribution=pop_distribution,
mobility_matrix=mobility_data[0],
beta=beta,
gamma=gamma,
mobility_rate=mobility_rate,
initial_infected=I0,
tmax=tmax)

n_jobs = 10

results = Parallel(
    n_jobs=n_jobs,
    backend="loky",
    verbose=10
)(
    delayed(run_simulation)(i)
    for i in range(len(KL))
)

# Sort results by iteration index
results.sort(key=lambda x: x[0])

# Extract results
R_random_all = np.array([x[1] for x in results])
R_generated_all = np.array([x[2] for x in results])

R_random_all = [R_random_all[i] - np.sum(R[-1]) for i in range(len(R_random_all))]
R_generated_all = [R_generated_all[i] - np.sum(R[-1]) for i in range(len(R_generated_all))]

os.makedirs(plot_dir/f"{name}_epidemics_{threshold}_{N_run}_{size}_{k}__{rmax}_{scale}_{nbins}")
plt.boxplot([R_random_all,R_generated_all],labels=["random","Soneira-Peebles+random"])
plt.ylabel(r"$\Delta$ Infected models-data")
plt.savefig(plot_dir/f"{name}_epidemics_{threshold}_{N_run}_{size}_{k}__{rmax}_{scale}_{nbins}")
plt.show()
plt.close()

d_load["Delta_Ricovered_random"] = R_random_all
d_load["Delta_Ricovered_generated"] = R_random_all


print(ttest_ind(R_random_all,R_generated_all))
