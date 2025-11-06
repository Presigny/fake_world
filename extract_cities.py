import pandas as pd
import geopandas as gpd
from pathlib import Path

data_dir = Path("data")

worldcities_file = data_dir / "worldcities.csv"
save_file = data_dir / "ukraine_cities.csv"

df = pd.read_csv(worldcities_file,low_memory=False)

df_fr = df.loc[df["country"] == "Ukraine"]

df_fr = df_fr.dropna(subset=['population'])


df_fr.to_csv(save_file)

# gpd_fr = gpd.read_file("/home/utente/Documenti/FAKE_WORLD/code/data/CNTR_RG_01M_2024_4326.geojson")
# gpd_fr = gpd_fr.loc[gpd_fr["NAME_ENGL"]=="France"]
# gpd_fr.plot()

#need to convert dataframe into geopandas
#need to compute distance efficiently 