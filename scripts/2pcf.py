import sys
from pathlib import Path
import geopandas as gpd
path = Path().cwd().parent / "src" #to add the src directory to the path regognized by Python
sys.path.append(str(path))
import methods_two_point_correlation as mtpc
import save_load_pickle as slp
import argparse
    
# Bookkeeping of important path in the project
path_data = Path("/media/utente/LaCie/fake_world_data/")
data_dir = path_data#Path().cwd().parent/Path("data")
data_city = data_dir / "cities"
data_map = data_dir / "map"
out_dir = path_data #Path().cwd().parent/Path("out")
plot_dir = out_dir / Path("plot")
##Create directories and path to save data



parser = argparse.ArgumentParser(description="Compute the 2PCF")

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


# name = "Sweden" #name of the system considered
# threshold = 1 # minimal number of inhabitants for the city to be considered
# N_run = 10 # number of random catalog generated for computing the 2PCF
# size = 1000 #number of points each random catalog has
# k = 3  #number of time the 2PCF computing process is repeated to estimate variance
# rmax = None#3e5 #maximal distance between two points considered, if None, taken the max from data
# scale = "log" #scale at which to bin the data, recommended "log"
# nbins = 20 # number of bins to consider in the computation of 2PCF (betweem rmin and rmax)


##Preprocessing
crs = mtpc.crs_selector(name) # Coordinate reference system of the country considered
path_city = data_city / f"{name}_cities.csv" #Path of the cities 
path_border = data_map / f"{name}.geojson" # Path of the multipolygon of the territory of the country
gdf_city = mtpc.load_df_to_gdf(path_city,threshold) # Geopandas of the dataset
print("number of cities considered: ",len(gdf_city))
print("size=",size)
gdf_edge = gpd.read_file(path_border) #Geopandas of the polygon border of the dataset
gdf_projected = gdf_city.to_crs(crs) # Projection in the right coordinate system
coord = gdf_projected.get_coordinates() # Coordinates of the points in dataset

path_save = out_dir /Path("2pcf") /Path(f"{name}/")
path_save.mkdir(parents=True,exist_ok=True)
name_save = path_save/Path(f"{threshold}_{N_run}_{size}_{k}__{rmax}_{scale}_{nbins}")
r_edges,l_xi = mtpc.PCF_with_variance(gdf_projected,gdf_edge,crs,N_run,size,k,rmax,scale,nbins=nbins)
distance_distribution = mtpc.compute_DD(gdf_projected)
#Store the 2pcf for the system under consideration
d_save = {}
d_save["r_edges"] = r_edges
d_save["xi"] = l_xi
d_save["distance_distribution"] = distance_distribution
slp.save_results(name_save,d_save)

mtpc.plot_2pcf(name_save,scale,plot_dir)
