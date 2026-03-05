"""This fucntion is used to create a polygon map for each country to be used after in the coputation of the two_poit correlation function for instance"""

import geopandas as gpd
from pathlib import Path

data_dir = Path.cwd().parent / Path("data/map")
#### INPUT######
l_country = ["Greece","Austria","Cyprus","United Kingdom","Germany","Luxembourg"]
l_country = ["Spain"]
##################
gpd = gpd.read_file(data_dir / "CNTR_RG_01M_2024_4326.geojson")
for country in l_country:
    gpd_country = gpd.loc[gpd["NAME_ENGL"]==country]
    
    if country == "France": #supress overseas territory
        gpd_country = gpd_country.clip(gpd, (-5,40,10,52))
        
    gpd_country.plot()
    gpd_country.to_file(str(data_dir) +"/"+ country+".geojson", driver='GeoJSON')
