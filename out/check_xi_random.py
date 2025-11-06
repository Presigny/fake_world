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
number_xi = 10
N_run = 20
size = 20000
size_test = 200
threshold = 5000
name = "Randombelgium"
rmin = 1000
nbins = 8

crs = mtpc.crs_selector("Belgium")
path_border = Path.cwd().parent / data_dir / "map/Belgium.geojson"

gdf_edge = gpd.read_file(path_border)
gdf_projected= mtpc.generate_random_point(gdf_edge,size_test,crs,check_gpd=True)
gdf_projected.plot()
plt.show()
path_save = Path.cwd().parent / save_dir /Path(name+"_"+str(rmin)+"_"+str(nbins))
path_save.mkdir(exist_ok=True)

for i in range(1):
    name_save = path_save/Path(str(i)+"_"+str(N_run)+"_"+str(size)+"_"+str(threshold))
    r_edges,xi = mtpc.compute_two_point_correlation_average(gdf_projected,gdf_edge,crs,N_run,size,rmin,nbins=nbins)
    distance_distribution = mtpc.compute_DD(gdf_projected)
    d_save = {}
    d_save["r_edges"] = r_edges
    d_save["xi"] = xi
    d_save["distance_distribution"] = distance_distribution
    
    slp.save_results(name_save,d_save)


