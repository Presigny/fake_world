import pandas as pd
from pathlib import Path

data_dir = Path.cwd().parent / Path("data/cities")
l_country = ["Greece","Austria","Cyprus","United Kingdom","Germany","Luxembourg"]
l_country = ["Netherlands"]
worldcities_file = data_dir / "worldcities.csv"
df = pd.read_csv(worldcities_file,low_memory=False,encoding='unicode_escape')
for country in l_country:
    save_file = data_dir / f"{country}_cities.csv"
    
    df_fr = df.loc[df["country"] == country]
    
    df_fr = df_fr.dropna(subset=['population'])
    #maybe filter for where there is no inhabitants
    
    df_fr.to_csv(save_file)
    
    print(f"Number cities in {country}:", len(df_fr))
