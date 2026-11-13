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


# def compute_mobility_city(gdf,pop_distribution):
#     coord = gdf.get_coordinates().to_numpy()
#     distance_matrix = distance.cdist(coord,coord).astype(np.int32)
#     mobility = mepi.multiprocessing_radiation_model(pop_distribution,distance_matrix)
#     return mobility,distance_matrix

def compute_mobility_GPT(gdf,pop_distribution):
    coord = gdf.get_coordinates().to_numpy()
    distance_matrix = distance.cdist(coord,coord).astype(np.int32)
    mobility = mepi.radiation_model_parallel(pop_distribution, distance_matrix)
    return mobility,#distance_matrix
    
# Bookkeeping of important path in the project
path_data = Path("/media/utente/LaCie/fake_world_data/")
data_dir = path_data#Path().cwd().parent/Path("data")
data_city = data_dir / "cities"
data_map = data_dir / "map"
data_2pcf = data_dir/"2pcf"
data_fit = data_dir /"fit"
data_mobility = data_dir/ "mobility"
out_dir = path_data #Path().cwd().parent/Path("out")
plot_dir = out_dir / Path("plot_ratio_distance_distribution")
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

diminish_radius = 1 # multiplication factor for the maximum radius, 
#to be lowered if the SP does not fit inside the considered border quick enough

#Load the polygon file for border
path_border = data_map / f"{name}.geojson"
path_city = data_city / f"{name}_cities.csv"
gdf_data = mtpc.load_df_to_gdf(path_city,threshold)
gdf_edge = gpd.read_file(path_border)

name_load = data_fit/ Path(name) / Path(f"KL_{threshold}_{N_run}_{size}_{k}__{rmax}_{scale}_{nbins}")
name_save = data_mobility/ Path(name) 
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


os.makedirs(name_save,exist_ok=True)
os.makedirs(name_save /"random",exist_ok=True)
os.makedirs(name_save/ "generated",exist_ok=True)
os.makedirs(name_save/"data",exist_ok=True)
mobility_data  = compute_mobility_GPT(gdf_data,pop_distribution)
path_data = name_save /"data" / Path(f"data_mobility_matrix_{threshold}_{N_run}_{size}_{k}__{rmax}_{scale}_{nbins}.npy")
np.save(path_data,mobility_data)
del mobility_data
for i in range(len(KL_random)):
    print(i)
    gdf_random_min = gpd.GeoSeries(l_gdf_random[i].flatten(),crs=crs)
    
    # Convert 3D array to 2D
    arr_2d = l_gdf[i].reshape(-1, 3)
    
    # Create DataFrame
    df = pd.DataFrame(arr_2d, columns=['x', 'y', 'geometry'])
    
    # Convert to GeoDataFrame
    gdf_generated_min = gpd.GeoDataFrame(
        df,
        geometry='geometry',
        crs=crs  # Replace with your actual CRS
    )
    del df
    mobility_random = compute_mobility_GPT(gdf_random_min,pop_distribution)
    mobility_generated = compute_mobility_GPT(gdf_generated_min,pop_distribution)
    path_random = name_save /"random" / Path(f"{i}_random_mobility_matrix_{threshold}_{N_run}_{size}_{k}__{rmax}_{scale}_{nbins}.npy")
    path_generated = name_save /"generated" / Path(f"{i}_generated_mobility_matrix_{threshold}_{N_run}_{size}_{k}__{rmax}_{scale}_{nbins}.npy")
    np.save(path_random,mobility_random)
    np.save(path_generated,mobility_generated)
    del mobility_random
    del mobility_generated
    gc.collect()
    #l_gen.append(mobility_generated)
# l_rand = np.array(l_rand)
# l_gen = np.array(l_gen)
