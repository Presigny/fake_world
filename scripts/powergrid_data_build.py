import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm import tqdm
import joblib
from shapely.geometry import MultiPolygon, Polygon, Point
from ast import literal_eval 

EPSG_DEGREE = 4326 #Google map =4326 #rte.france map =3857
EPSG_METER = 2154 #Lambert =2154
# RTE

substations_data = pd.read_csv('../data/postes-electriques-rte.csv',sep=';')

# Initial CRS

# latitude_rte = []
# longitude_rte = []
# for i in tqdm(range(len(substations_data))):
#     latitude_rte.append(substations_data['Latitude poste (DD)'].loc[i])
#     longitude_rte.append(substations_data['Longitude poste (DD)'].loc[i])

# CRS in meters

substations_data['geometry'] = substations_data.apply(lambda row: Point(row['Longitude poste (DD)'], row['Latitude poste (DD)']), axis=1)
substations_data = gpd.GeoDataFrame(substations_data, geometry='geometry', crs=4326)
substations_data = substations_data.to_crs(2154)

latitude_rte = []
longitude_rte = []
for i in tqdm(range(len(substations_data))):
    latitude_rte.append(substations_data['geometry'].loc[i].y)
    longitude_rte.append(substations_data['geometry'].loc[i].x)

joblib.dump(np.array([longitude_rte,latitude_rte]),'../out/position_in_meters_rte.joblib')


# Groupé les données par la colonne "Tension (kV)"
grouped_data = substations_data.groupby('Tension (kV)')

# Créer un dictionnaire pour stocker les valeurs de longitude et latitude pour chaque tension
position_dict = {}
for tension, group in grouped_data:
    longitude_rte = group['geometry'].apply(lambda point: point.x).tolist()
    latitude_rte = group['geometry'].apply(lambda point: point.y).tolist()
    position_dict[tension] = {'longitude_rte': longitude_rte, 'latitude_rte': latitude_rte}

# Enregistrer le dictionnaire dans un fichier
joblib.dump(position_dict, '../out/position_in_meters_per_voltage_rte.joblib')


# ENEDIS

gdf = gpd.read_file('../data/postes-electrique-enedis.geojson')
departements = gpd.read_file('../data/departements-shp')
# print(departements.crs)
def get_polygon_with_highest_latitude(multipolygon):
    if isinstance(multipolygon, Polygon):
        return multipolygon
    elif isinstance(multipolygon, MultiPolygon):
        return max(multipolygon.geoms, key=lambda polygon: polygon.bounds[3])
    else:
        raise TypeError("Input must be a Polygon or MultiPolygon")
        
clipped_gdf = gpd.GeoDataFrame(geometry=[get_polygon_with_highest_latitude(departements.geometry.union_all())], crs=departements.crs)

metropolitan_gdf = gpd.sjoin(gdf, clipped_gdf, predicate='within')

# Initial CRS

# distribnode2coordinates = dict(metropolitan_gdf['geo_point_2d'])

# longitude_enedis = []
# latitude_enedis = []

# for node in tqdm(distribnode2coordinates):
#     if node not in {10881,739676,698363,720643,739662,772930}:#{doublons} + nan (le dernier)
#         longitude_enedis.append(literal_eval(distribnode2coordinates[node])['lon'])
#         latitude_enedis.append(literal_eval(distribnode2coordinates[node])['lat'])

# CRS in meters

metropolitan_gdf = metropolitan_gdf.to_crs(2154)
distribnode2coordinates = dict(metropolitan_gdf['geometry'])


longitude_enedis = []
latitude_enedis = []

# for i in tqdm(range(len(metropolitan_gdf))):
#     if i not in {10881,739676,698363,720643,739662,772930}:#{doublons} + nan (le dernier)
#         longitude_enedis.append(metropolitan_gdf['geometry'].loc[i].x)
#         latitude_enedis.append(metropolitan_gdf['geometry'].loc[i].y)

for node in tqdm(distribnode2coordinates):
    if node not in {10881,739676,698363,720643,739662,772930}:#{doublons} + nan (le dernier)
        longitude_enedis.append(distribnode2coordinates[node].x)
        latitude_enedis.append(distribnode2coordinates[node].y)

joblib.dump(clipped_gdf,'../out/france_shape_gdf.joblib')

joblib.dump(np.array([longitude_enedis,latitude_enedis]),'../out/position_in_meters_enedis.joblib')
