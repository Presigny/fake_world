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


def run_simulation(i,d_load,nocut=False):
    path_random = (
        name_load / "random" /
        Path(f"{i}_random_mobility_matrix_{threshold}_{N_run}_{size}_{k}__{rmax}_{scale}_{nbins}.npy")
    )

    path_generated = (
        name_load / "generated" /
        Path(f"{i}_generated_mobility_matrix_{threshold}_{N_run}_{size}_{k}__{rmax}_{scale}_{nbins}.npy")
    )
    if nocut:
        cut=1e99
    else:
        cut = d_load["cut_off_distance_distribution"]
    gdf_random = d_load["l_gdf_random"][i]
    gdf= d_load["l_gdf"][i]
    gdf_random = gpd.GeoSeries(gdf_random.flatten(),crs=crs)
    
    # Convert 3D array to 2D
    arr_2d = gdf.reshape(-1, 3)
    
    # Create DataFrame
    df = pd.DataFrame(arr_2d, columns=['x', 'y', 'geometry'])
    
    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry='geometry',
        crs=crs  # Replace with your actual CRS
    )
    del df
    coord_random = gdf_random.get_coordinates().to_numpy()
    coord_gen = gdf.get_coordinates().to_numpy()
    distance_random = distance.cdist(coord_random,coord_random).astype(np.int32)
    distance_generated = distance.cdist(coord_gen,coord_gen).astype(np.int32)
    mobility_random = np.load(path_random)
    mobility_generated = np.load(path_generated)
    # Load matrices
    row,col = np.where(distance_random > cut)
    del distance_random
    mobility_random = mobility_random[0]
    mobility_generated = mobility_generated[0]
    mobility_random[row,col] = 0
    row,col = np.where(distance_generated > cut)
    del distance_generated
    gc.collect()
    mobility_generated[row,col] = 0
    # Random mobility SIR
    C_global_random,C_local_random = mepi.fagiolo_weighted_clustering(mobility_random)
    C_global_generated,C_local_generated = mepi.fagiolo_weighted_clustering(mobility_generated)

    return C_global_random, C_global_generated

    
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

parser.add_argument("--name", type=str, default="Belgium")
parser.add_argument("--threshold", type=int, default=1)
parser.add_argument("--N_run", type=int, default=10)
parser.add_argument("--size", type=int, default=5000)
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
nocut = False
name_load = data_mobility/ Path(name)
name_load_dico = data_fit/ Path(name) / Path(f"KL_{threshold}_{N_run}_{size}_{k}__{rmax}_{scale}_{nbins}")
name_save = data_fit/ Path(name) / Path(f"epidemics_{threshold}_{N_run}_{size}_{k}__{rmax}_{scale}_{nbins}")
d_load = slp.load_results(name_load_dico)
N = d_load["number_points"] #number of point sin the dataset
crs = mtpc.crs_selector(name) #cooridnate reference system associated with the datset
gdf_edge = gdf_edge.to_crs(crs) # Polygon of the border of the country
gdf_data = gdf_data.to_crs(crs)
cut = d_load["cut_off_distance_distribution"]
if nocut:
    cut = 1e99
KL = d_load["KLSP"]
path_data = name_load /"data" / Path(f"data_mobility_matrix_{threshold}_{N_run}_{size}_{k}__{rmax}_{scale}_{nbins}.npy")
mobility_data = np.load(path_data)
coord_data = gdf_data.get_coordinates().to_numpy()
distance_data = distance.cdist(coord_data,coord_data).astype(np.int32)
row,col = np.where(distance_data > cut)
mobility_data = mobility_data[0]
mobility_data[row,col] = 0

C_global_data,C_local_data = mepi.fagiolo_weighted_clustering(mobility_data)

n_jobs = 10

results = Parallel(
    n_jobs=n_jobs,
    backend="loky",
    verbose=10
)(
    delayed(run_simulation)(i,d_load,nocut)
    for i in range(len(KL))
)

# Sort results by iteration index
results.sort(key=lambda x: x[0])

# Extract results
Cglobal_random_all = np.array([results[i][0] for i in range(len(results))])
Cglobal_generated_all = np.array([results[i][1] for i in range(len(results))])
ratio_random = Cglobal_random_all/C_global_data
ratio_generated  = Cglobal_generated_all/C_global_data
# R_random_all = [R_random_all[i] - np.sum(R[-1]) for i in range(len(R_random_all))]
# R_generated_all = [R_generated_all[i] - np.sum(R[-1]) for i in range(len(R_generated_all))]

# os.makedirs(plot_dir/f"{name}_epidemics_{threshold}_{N_run}_{size}_{k}__{rmax}_{scale}_{nbins}")
plt.boxplot([ratio_random,ratio_generated],labels=["random","Soneira-Peebles+random"])
plt.ylabel(r"Ratio Weighted global clustering coefficient")
# plt.savefig(plot_dir/f"{name}_epidemics_{threshold}_{N_run}_{size}_{k}__{rmax}_{scale}_{nbins}")
# plt.show()
# plt.close()

# d_load["Delta_Ricovered_random"] = R_random_all
# d_load["Delta_Ricovered_generated"] = R_random_all
# #d_load[]



print(ttest_ind(ratio_random,ratio_generated))
