# %%1. Installation (si nécessaire)
!pip install --upgrade pynytimes

# %%2. Imports
from pynytimes import NYTAPI
from datetime import datetime
import pandas as pd

#%% 3. Configuration de l'API
# IMPORTANT : On met parse_dates=False pour éviter l'erreur de format
API_KEY = "15MPlclMxKGuTa7WreSZ9IYcbjoo45sL"
nyt = NYTAPI(API_KEY, parse_dates=False) 

# --- CONFIGURATION DE LA PÉRIODE ---
START_YEAR, START_MONTH = 2019, 1
END_YEAR, END_MONTH = 2021, 1  # On peut maintenant aller jusqu'en 2026
# -----------------------------------

# 4. Récupération des données
all_data = []

print(f"Début de l'extraction de {START_MONTH}/{START_YEAR} à {END_MONTH}/{END_YEAR}...")

for year in range(START_YEAR, END_YEAR + 1):
    m_start = START_MONTH if year == START_YEAR else 1
    m_end = END_MONTH if year == END_YEAR else 12
    
    for month in range(m_start, m_end + 1):
        try:
            print(f"Extraction : Année {year} | Mois {month}")
            data = nyt.archive_metadata(date=datetime(year, month, 1))
            all_data.extend(data)
        except Exception as e:
            print(f"⚠️ Erreur sautée pour {month}/{year}: {e}")

# 5. Traitement avec Pandas
if all_data:
    df = pd.DataFrame(all_data)

    # On ne garde que les colonnes utiles
    df = df[['pub_date', 'lead_paragraph']]

    # Transformation robuste de la date par Pandas
    df['pub_date'] = pd.to_datetime(df['pub_date'], errors='coerce')
    
    # Formatage final en YYYY-DD-MM
    df['pub_date'] = df['pub_date'].dt.strftime('%Y-%d-%m')

    # --- AJOUT : Suppression des paragraphes de moins de 5 mots ---
    # On s'assure d'abord que le texte est une chaîne de caractères
    df['lead_paragraph'] = df['lead_paragraph'].astype(str)
    
    # On filtre : on ne garde que si le nombre de mots (séparés par des espaces) est >= 5
    df = df[df['lead_paragraph'].apply(lambda x: len(x.split()) >= 5)]
    # --------------------------------------------------------------

    # Nettoyage des valeurs manquantes restantes
    df = df.dropna(subset=['lead_paragraph', 'pub_date'])

    # 6. Sauvegarde
    filename = f"nyt_articles_{START_YEAR}_{END_YEAR}.csv"
    df.to_csv(filename, index=False)
    
    print("-" * 30)
    print(f"Succès ! Fichier sauvegardé : {filename}")
    print(f"Total articles (après filtrage > 5 mots) : {len(df)}")
# %%
