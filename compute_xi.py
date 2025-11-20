import geopandas as gpd
import sys
from pathlib import Path
import methods_two_point_correlation as mtpc
path = Path().cwd().parent
sys.path.append(str(path /"src"))
import save_load_pickle as slp
save_dir = Path("data/two_point_correlation")
data_dir = Path("data")
number_xi = 10
N_run = 5
size = 8000
threshold = 1
name = "Italy"
rmin = 1000
nbins = 20
crs = mtpc.crs_selector(name)
path_city = Path.cwd().parent / data_dir / "italy_cities.csv"
path_border = Path.cwd().parent / data_dir / "map/Italy.geojson"


gdf_city = mtpc.load_df_to_gdf(path_city,threshold)
gdf_edge = gpd.read_file(path_border)
gdf_projected = gdf_city.to_crs(crs)
print(len(gdf_projected))
path_save = Path.cwd().parent / save_dir /Path(name+"_"+str(rmin)+"_"+str(nbins))
path_save.mkdir(exist_ok=True)

gdf_edge = gpd.read_file(path_border)


#radius = find_rmax_meaningful(gdf_edge,crs,initial_guess)
path_save = Path.cwd().parent / save_dir /Path(name+"_"+str(rmin)+"_"+str(nbins))
path_save.mkdir(exist_ok=True)
name_save = path_save/Path("lol_"+str(N_run)+"_"+str(size)+"_lol")
r_edges,xi = mtpc.compute_two_point_correlation_2019(gdf_projected,gdf_edge,crs,N_run,size,rmin,len(gdf_projected),nbins=nbins)
distance_distribution = mtpc.compute_DD(gdf_projected)
d_save = {}
d_save["r_edges"] = r_edges
d_save["xi"] = xi
d_save["distance_distribution"] = distance_distribution

slp.save_results(name_save,d_save)


