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
from scipy.stats import ttest_ind
    
# Bookkeeping of important path in the project
path_data = Path("/media/utente/LaCie/fake_world_data/")
data_dir = path_data#Path().cwd().parent/Path("data")
data_city = data_dir / "cities"
data_map = data_dir / "map"
data_2pcf = data_dir/"2pcf"
data_fit = data_dir /"fit"
out_dir = path_data #Path().cwd().parent/Path("out")
plot_dir = out_dir / Path("plot_KL_distribution")
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
gdf_edge = gpd.read_file(path_border)

name_load = data_fit/ Path(name) / Path(f"model_{threshold}_{N_run}_{size}_{k}__{rmax}_{scale}_{nbins}")
name_save = data_fit/ Path(name) / Path(f"KL_{threshold}_{N_run}_{size}_{k}__{rmax}_{scale}_{nbins}")
d_load = slp.load_results(name_load)
r_edges = d_load["r_edges"] 
l_xi = d_load["xi"]
distance_distribution = d_load["distance_distribution"] 
l_cluster = d_load["clustering"]
N = d_load["number_points"] #number of point sin the dataset
gamma, r0 = d_load["fit_SP"] #fitted gamma and R0
crs = mtpc.crs_selector(name) #cooridnate reference system associated with the datset
gdf_edge = gdf_edge.to_crs(crs) # Polygon of the border of the country


save_dist = d_load["save_dist"] 
save_L = d_load["L"]
l_cluster = d_load["eta"]
save_random = d_load["save_random"]

L = save_L[np.argmin(save_dist)] #np.argmin(save_dist)
eta = l_cluster[np.argmin(save_dist)]
random_point = save_random[np.argmin(save_dist)] # number of random points to generate on top of the SP to make SP+random
l = eta**(1/(2-gamma))
R = 0.5*gdf_edge.length[0]/(2*np.pi)*diminish_radius # radius max
start_off = r_edges[0]
# KLSP,KLrandom
mean_xi = np.mean(l_xi,axis=0)
index = next((i for i, x in enumerate(mean_xi) if x < 0.1), None) #find the index that corresponds to the distances range that matter for the SOneira Peebles
cut_off = r_edges[index]



#KLSP,KLrandom = fcsp.compare_log_distribution(start_off,cut_off,n_iteration,l,eta,L,R,gdf_edge,crs,N,random_point,nbins,distance_distribution,show_histplot=False)
KLSP,KLrandom,l_gdf,l_gdf_random = fcsp.compare_log_distribution_parallel(
    start_off, cut_off, n_iteration, l, eta, L, R,
    gdf_edge, crs, N, random_point, nbins,
    distance_distribution, show_histplot=False,
    max_workers=8, random_seed=None)
d_load["KL_random"] = KLrandom
d_load["KLSP"] = KLSP
d_load["l_gdf"] = l_gdf
d_load["l_gdf_random"] = l_gdf_random
d_load["cut_off_distance_distribution"] = cut_off
slp.save_results(name_save, d_load)
#print(KLSP,KLrandom)
print("SP,random",np.mean(KLSP),np.mean(KLrandom))

plt.boxplot([KLrandom,KLSP],labels=["random","Soneira-Peebles+random"])
plt.ylabel("Kullback-Leibler divergence")
if not(name_save.parent):
    os.mkdir(name_save.parent)
plt.savefig(plot_dir / ("KL_"+name), format="png",transparent=True)
print(ttest_ind(KLrandom,KLSP,equal_var=False))