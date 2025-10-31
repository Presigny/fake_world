import pandas as pd
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
from methods_two_point_correlation import generate_random_point
###INPUT####
data_dir = Path("data")
data_map = Path("data/map")
load_file = data_dir / "italy_cities.csv"
gjson_country =  data_map / "Italy.geojson"
#################################################
df = pd.read_csv(load_file,low_memory=False)
df = df.loc[df["population"] >= 1000]

gdf = gpd.read_file(gjson_country)
gdf_cities = gpd.GeoDataFrame(
     df, geometry=gpd.points_from_xy(df.lng, df.lat), crs="EPSG:4326")
del df
fig, ax = plt.subplots()
gdf.plot(ax=ax)
gdf_cities.plot(ax=ax,markersize=0.01,color="red")
plt.show()

gdf_random = generate_random_point(gdf,20000,"EPSG:4326",check_gpd=True)
fig, ax = plt.subplots()
gdf.plot(ax=ax)
gdf_random.plot(ax=ax,markersize=0.01,color="red")
plt.show()