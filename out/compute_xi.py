import geopandas as gpd
import sys
from pathlib import Path
import methods_two_point_correlation as mtpc
sys.path.append("src")
import save_load_pickle as slp
save_dir = Path("data/two_point_correlation")
data_dir = Path("data")
number_xi = 10
N_run = 5
size = 10000
threshold = 5000
name = "France"
rmin = 1000
nbins = 20
crs = mtpc.crs_selector(name)
path_city = Path.cwd().parent / data_dir / "france_cities.csv"
path_border = Path.cwd().parent / data_dir / "map/France.geojson"

gdf_city = mtpc.load_df_to_gdf(path_city,threshold)
gdf_edge = gpd.read_file(path_border)
gdf_projected = gdf_city.to_crs(crs)

path_save = Path.cwd().parent / save_dir /Path(name+"_"+str(rmin)+"_"+str(nbins))
path_save.mkdir(exist_ok=True)

for i in range(10):
    name_save = path_save/Path(str(i)+"_"+str(N_run)+"_"+str(size)+"_"+str(threshold))
    r_edges,xi = mtpc.compute_two_point_correlation(gdf_projected,gdf_edge,crs,N_run,size,rmin,nbins=nbins)
    d_save = {}
    d_save["r_edges"] = r_edges
    d_save["xi"] = xi
    slp.save_results(name_save,d_save)


