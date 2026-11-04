import pandas as pd
from pathlib import Path
path_data = Path("/media/utente/LaCie/fake_world_data/cities")
#data_dir = Path.cwd().parent / Path("data/cities")
data_dir = path_data
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
worldcities_file = data_dir / "worldcities.csv"
df = pd.read_csv(worldcities_file,low_memory=False,encoding="utf-8")
for country in l_country:
    save_file = data_dir / f"{country}_cities.csv"
    
    df_fr = df.loc[df["country"] == country]
    
    df_fr = df_fr.dropna(subset=['population'])
    #maybe filter for where there is no inhabitants
    
    df_fr.to_csv(save_file)

    print(f"Number cities in {country}:", len(df_fr))
