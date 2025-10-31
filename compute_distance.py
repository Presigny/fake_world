import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial import distance
import seaborn as sns
data_dir = Path("data")

load_file = data_dir / "france_cities.csv"

df = pd.read_csv(load_file,low_memory=False)
df = df.loc[df["population"] >= 10000]

gpd_fr = gpd.read_file("/home/utente/Documenti/FAKE_WORLD/code/data/CNTR_RG_01M_2024_4326.geojson")
gpd_fr = gpd_fr.loc[gpd_fr["NAME_ENGL"]=="France"]
geo_clip = gpd.clip(gpd_fr, (-5,40,10,52))
ax= geo_clip.plot()
df.plot(ax=ax,y="lat",x="lng",style="o",ms=0.1,color="red")
plt.show()
gdf = gpd.GeoDataFrame(
     df, geometry=gpd.points_from_xy(df.lng, df.lat), crs="EPSG:4326")
del df
del gpd_fr
fig, ax = plt.subplots()
geo_clip.plot(ax=ax)
gdf.plot(ax=ax,markersize=0.01,color="red")
plt.show()

gdf_projected = gdf.to_crs('EPSG:2154')
# #distance_matrix = gdf_projected.geometry.apply(lambda g: gdf_projected.geometry.distance(g))
# #df1 = df[['lat', 'lng']]
coord = gdf_projected.get_coordinates().to_numpy()
#distance_matrix_cdist = distance.cdist(coord,coord)
distance_matrix_cdist = np.triu(distance.cdist(coord,coord)).astype(np.int32)
distance_matrix_cdist = np.ravel(distance_matrix_cdist)
distance_matrix_cdist = distance_matrix_cdist[distance_matrix_cdist != 0]
sns.histplot(distance_matrix_cdist,stat="probability",binwidth=50000)
#distance_matrix_cdist = sp.csr_matrix(distance_matrix_cdist)
#distance_sk = euclidean_distances(coord)
# distance_matrix_cdist = pd.DataFrame(distance_matrix_cdist)
# distance_matrix_cdist.index = gdf_projected['city']
# distance_matrix_cdist.columns = gdf_projected['city']



#Need to use a CRS adapted to the place also for the random.