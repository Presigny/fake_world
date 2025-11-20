import geopandas as gpd
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import methods_two_point_correlation as mtpc
path = Path().cwd().parent
sys.path.append(str(path /"src"))
import save_load_pickle as slp
save_dir = Path("data/two_point_correlation")
data_dir = Path("data")



N_run = 5
size = 7000
size_test = 7000
name = "Randomitaly"
rmin = 2000
nbins = 20
initial_guess = 150000
crs = mtpc.crs_selector("Italy")
path_border = Path.cwd().parent / data_dir / "map/Italy.geojson"

gdf_edge = gpd.read_file(path_border)
gdf_projected= mtpc.generate_random_point(gdf_edge,size_test,crs,check_gpd=True)

#radius = find_rmax_meaningful(gdf_edge,crs,initial_guess)
path_save = Path.cwd().parent / save_dir /Path(name+"_"+str(rmin)+"_"+str(nbins))
path_save.mkdir(exist_ok=True)
name_save = path_save/Path("lol_"+str(N_run)+"_"+str(size)+"_lol")
r_edges,xi = mtpc.compute_two_point_correlation_2019(gdf_projected,gdf_edge,crs,N_run,size,rmin,size_test,nbins=nbins)
distance_distribution = mtpc.compute_DD(gdf_projected)
d_save = {}
d_save["r_edges"] = r_edges
d_save["xi"] = xi
d_save["distance_distribution"] = distance_distribution

slp.save_results(name_save,d_save)


