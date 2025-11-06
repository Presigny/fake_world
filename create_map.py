"""This fucntion is used to create a polygon map for each country to be used after in the coputation of the two_poit correlation function for instance"""
import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial import distance
import seaborn as sns
data_dir = Path("data/map")
#### INPUT######
country = "Ukraine"
##################
gpd = gpd.read_file(data_dir / "CNTR_RG_01M_2024_4326.geojson")
gpd = gpd.loc[gpd["NAME_ENGL"]==country]
if country == "France":
    gpd = gpd.clip(gpd, (-5,40,10,52))
gpd.plot()
gpd.to_file(str(data_dir) +"/"+ country+".geojson", driver='GeoJSON')
