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

def median_and_min(arr):
    median_index = np.argmin(np.abs(arr - np.median(arr)))
    min_index =  np.where(arr == np.min(arr))[0]
    return median_index,min_index

def arrange(A,cut):
    A = np.triu(A)
    A = A.flatten()[A.flatten()<=cut]
    return np.sum(A)

def compute_ensemble_distance_matrix(dico,A,cut):
    l_gdf_random = d_load["l_gdf_random"] 
    l_gdf= d_load["l_gdf"] 
    l_rand,l_gen = [],[]
    for i in range(len(l_gdf)):
        arr_2d = l_gdf[i].reshape(-1, 3)
        # Create DataFrame
        df = pd.DataFrame(arr_2d, columns=['x', 'y', 'geometry'])
    
        # Convert to GeoDataFrame
        gdf_generated = gpd.GeoDataFrame(
            df,
            geometry='geometry',
            crs=crs  # Replace with your actual CRS
        )
        gdf_random = gpd.GeoSeries(l_gdf_random[i].flatten(),crs=crs)
        coord_generated = gdf_generated.get_coordinates().to_numpy()
        coord_random = gdf_random.get_coordinates().to_numpy()
        distance_generated = distance.cdist(coord_generated,coord_generated).astype(np.int32)
        distance_random = distance.cdist(coord_random,coord_random).astype(np.int32)
        sum_generated = arrange(distance_generated,cut)
        sum_random = arrange(distance_random,cut)
        l_rand.append(sum_random)
        l_gen.append(sum_generated)
    C = np.sum(A[A<=cut])
    l_rand = l_rand/C
    l_gen = l_gen/C
    return l_rand,l_gen
    

def compute_mobility_city(gdf,pop_distribution):
    coord = gdf.get_coordinates().to_numpy()
    distance_matrix = distance.cdist(coord,coord).astype(np.int32)
    mobility = mepi.multiprocessing_radiation_model(pop_distribution,distance_matrix)
    return mobility,distance_matrix

def compute_mobility_GPT(gdf,pop_distribution):
    coord = gdf.get_coordinates().to_numpy()
    distance_matrix = distance.cdist(coord,coord).astype(np.int32)
    mobility = mepi.radiation_model(pop_distribution,distance_matrix)
    return mobility,distance_matrix
    
# Bookkeeping of important path in the project
path_data = Path("/media/utente/LaCie/fake_world_data/")
data_dir = path_data#Path().cwd().parent/Path("data")
data_city = data_dir / "cities"
data_map = data_dir / "map"
data_2pcf = data_dir/"2pcf"
data_fit = data_dir /"fit"
out_dir = path_data #Path().cwd().parent/Path("out")
plot_dir = out_dir / Path("plot_ratio_distance_distribution")
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

diminish_radius = 1 # multiplication factor for the maximum radius, 
#to be lowered if the SP does not fit inside the considered border quick enough

#Load the polygon file for border
path_border = data_map / f"{name}.geojson"
path_city = data_city / f"{name}_cities.csv"
gdf_data = mtpc.load_df_to_gdf(path_city,threshold)
gdf_edge = gpd.read_file(path_border)

name_load = data_fit/ Path(name) / Path(f"KL_{threshold}_{N_run}_{size}_{k}__{rmax}_{scale}_{nbins}")
name_save = data_fit/ Path(name) / Path(f"_{threshold}_{N_run}_{size}_{k}__{rmax}_{scale}_{nbins}")
d_load = slp.load_results(name_load)
N = d_load["number_points"] #number of point sin the dataset
crs = mtpc.crs_selector(name) #cooridnate reference system associated with the datset
gdf_edge = gdf_edge.to_crs(crs) # Polygon of the border of the country
gdf_data = gdf_data.to_crs(crs)
cut = d_load["cut_off_distance_distribution"]
distance_matrix_data = d_load["distance_distribution"]

l_gdf_random = d_load["l_gdf_random"] 
l_gdf= d_load["l_gdf"] 
KL_random  = d_load["KL_random"]
KL_SP = d_load["KLSP"]
coord_data = gdf_data.get_coordinates().to_numpy()
pop_per_node = 1000
pop_distribution = [pop_per_node for i in range(N)]

l_rand,l_gen = compute_ensemble_distance_matrix(d_load,distance_matrix_data,cut)


plt.boxplot([l_rand,l_gen],labels=["random","Soneira-Peebles+random"])
plt.ylabel("Ratio of Cumulated distance below the cutoff ")
if not(name_save.parent):
    os.mkdir(name_save.parent)
plt.savefig(plot_dir / ("ratio_distance_distribution_"+name), format="png",transparent=True)

#mobility_data,distance_matrix_data  = compute_mobility_city(gdf_data,pop_distribution)
print("DATA DONE")

# median_random,min_random = median_and_min(KL_random)
# median_SP,min_SP= median_and_min(KL_SP)

# gdf_random_min = gpd.GeoSeries(l_gdf_random[min_random].flatten(),crs=crs)
# print("RANDOM DONE")

# # Convert 3D array to 2D
# arr_2d = l_gdf[min_SP].reshape(-1, 3)

# # Create DataFrame
# df = pd.DataFrame(arr_2d, columns=['x', 'y', 'geometry'])

# # Convert to GeoDataFrame
# gdf_generated_min = gpd.GeoDataFrame(
#     df,
#     geometry='geometry',
#     crs=crs  # Replace with your actual CRS
# )

# mobility_random_min, distance_matrix_random_min = compute_mobility_city(gdf_random_min,pop_distribution)
# mobility_generated_min,distance_matrix_generated_min = compute_mobility_city(gdf_generated_min,pop_distribution)

# # def compute_mobility_random_and_SP(l_gdf_generated,ind_SP,l_gdf_random,ind_random,pop_distribution):
# #     gdf_random_min = gpd.GeoSeries(l_gdf_random[ind_random].flatten(),crs=crs)

# #     # Convert 3D array to 2D
# #     arr_2d = l_gdf_generated[ind_SP].reshape(-1, 3)

# #     # Create DataFrame
# #     df = pd.DataFrame(arr_2d, columns=['x', 'y', 'geometry'])

# #     # Convert to GeoDataFrame
# #     gdf_generated_min = gpd.GeoDataFrame(
# #         df,
# #         geometry='geometry',
# #         crs=crs  # Replace with your actual CRS
# #     )

# #     mobility_random_min, distance_matrix_random_min = compute_mobility_city(gdf_random_min,pop_distribution)
# #     mobility_generated_min,distance_matrix_generated_min = compute_mobility_city(gdf_generated_min,pop_distribution)
# #     return mobility 
    


# A = arrange(distance_matrix_data,cut)
# B = arrange(distance_matrix_random_min,cut)
# C = arrange(distance_matrix_generated_min,cut)
