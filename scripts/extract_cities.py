import pandas as pd
import geopandas as gpd
from pathlib import Path

data_dir = Path.cwd().parent / Path("data/cities")
country = "Switzerland"

worldcities_file = data_dir / "worldcities.csv"
save_file = data_dir / f"{country}_cities.csv"

df = pd.read_csv(worldcities_file,low_memory=False)

df_fr = df.loc[df["country"] == country]

df_fr = df_fr.dropna(subset=['population'])


df_fr.to_csv(save_file)
