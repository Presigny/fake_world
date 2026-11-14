import sys
import pandas as pd
import geopandas as gpd
from pathlib import Path
path = Path().cwd().parent / "src" #to add the src directory to the path regognized by Python
sys.path.append(str(path))
import methods_two_point_correlation as mtpc
path_data = Path("/media/utente/LaCie/fake_world_data/cities")
#data_dir = Path.cwd().parent / Path("data/cities")
data_dir = path_data
data_map = Path("/media/utente/LaCie/fake_world_data/map")
l_country = [
    "Austria",
    "Belgium",
    "Bulgaria",
    "Croatia",
    "Cyprus",
    "Czechia",
    "Denmark",
    "Estonia",
    "Finland",
    "France",
    "Germany",
    "Greece",
    "Hungary",
    "Ireland",
    "Italy",
    "Latvia",
    "Lithuania",
    "Luxembourg",
    "Malta",
    "Netherlands",
    "Poland",
    "Portugal",
    "Romania",
    "Slovakia",
    "Slovenia",
    "Spain",
    "Sweden"
]
l_country = ["Poland"]
for name in l_country:
    path_border = data_map / f"{name}.geojson" 
    path_city = path_data / f"{name}_cities.csv" #Path of the cities 
    path_border = data_map / f"{name}.geojson"
    gdf_edge = gpd.read_file(path_border)
    gdf_city = mtpc.load_df_to_gdf(path_city,1) # Geopandas of the dataset
    print(f"{name}: ",len(gdf_city))
    ax=gdf_edge.plot()
    gdf_city.plot(ax=ax,color="red",markersize=0.1)