import sys
from pathlib import Path
path = Path().cwd().parent / "src" #to add the src directory to the path regognized by Python
sys.path.append(str(path))
import methods_two_point_correlation as mtpc
import save_load_pickle as slp
import argparse
import numpy as np
import sklearn.cluster as sk
from sklearn.metrics import silhouette_score
    
# Bookkeeping of important path in the project
path_data = Path("/media/utente/LaCie/fake_world_data/")
data_dir = path_data#Path().cwd().parent/Path("data")
data_city = data_dir / "cities"
data_map = data_dir / "map"
data_2pcf = data_dir/"2pcf"
out_dir = path_data #Path().cwd().parent/Path("out")
plot_dir = out_dir / Path("plot_powerlaw")
##Create directories and path to save data


parser = argparse.ArgumentParser(description="Fit a power law to the 2pcf")

parser.add_argument("--name", type=str, default="Sweden")
parser.add_argument("--threshold", type=int, default=1)
parser.add_argument("--N_run", type=int, default=10)
parser.add_argument("--size", type=int, default=1000)
parser.add_argument("--k", type=int, default=3)
parser.add_argument("--rmax", type=float, default=None)
parser.add_argument("--scale", type=str, default="log", choices=["log", "linear"])
parser.add_argument("--nbins", type=int, default=20)

args = parser.parse_args()

name = args.name
threshold = args.threshold
N_run = args.N_run
size = args.size
k = args.k
rmax = args.rmax
scale = args.scale
nbins = args.nbins

###################################################################################
print(name)
##Preprocessing
crs = mtpc.crs_selector(name) # Coordinate reference system of the country considered
path_city = data_city / f"{name}_cities.csv" #Path of the cities 
path_border = data_map / f"{name}.geojson" # Path of the multipolygon of the territory of the country
gdf_city = mtpc.load_df_to_gdf(path_city,threshold) # Geopandas of the dataset
gdf_projected = gdf_city.to_crs(crs) # Projection in the right coordinate system
coord = gdf_projected.get_coordinates() # Coordinates of the points in dataset

name_load = data_2pcf/ Path(name) / Path(f"powerlawfit_{threshold}_{N_run}_{size}_{k}__{rmax}_{scale}_{nbins}")
name_save = data_2pcf/ Path(name) / Path(f"clusters_{threshold}_{N_run}_{size}_{k}__{rmax}_{scale}_{nbins}")
d_save = slp.load_results(name_load)

#Here we use the silhouette score to asess what is the most relevant paraemters eta to test in the SP 
#model, since teh KMeans clusetring is hierarichcal for each k we select the best clustering over n_init
#runß

kmax = 10 # maximum number of cluster to consider 
l_silhouette = []
silhouette_coefficients = []
for k in range(2, kmax):
    kmeans = sk.KMeans(n_clusters=k,tol=1e-8,max_iter=1000,n_init=100).fit(coord)
    score = silhouette_score(coord, kmeans.labels_)
    silhouette_coefficients.append(score)
l_silhouette.append(silhouette_coefficients)


l_cluster = np.argsort(np.mean(l_silhouette,axis=0))[-5:] #we select the 4 better clustering
l_cluster = np.flip(l_cluster)+2
#d_save = {} #save the final file with distance distribution,and the fit parameters
d_save["number_points"] = len(gdf_projected)
d_save["clustering"] = l_cluster
d_save["silhouette_score"] = l_silhouette
print(d_save.keys)
slp.save_results(name_save,d_save)

    
mtpc.plot_silhouette(name_save,plot_dir,kmax,l_silhouette)