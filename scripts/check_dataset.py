from pathlib import Path

import pandas as pd


DATASET_PATH = Path("data/processed/events_lille_clean.csv")


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Fichier introuvable : {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    print("\n=== Aperçu du dataset ===")
    print(df.head())

    print("\n=== Informations générales ===")
    print(df.info())

    print("\n=== Dimensions ===")
    print(f"Nombre de lignes : {df.shape[0]}")
    print(f"Nombre de colonnes : {df.shape[1]}")

    print("\n=== Colonnes ===")
    print(df.columns.tolist())

    print("\n=== Valeurs manquantes par colonne ===")
    print(df.isna().sum())

    print("\n=== Nombre de doublons sur event_id ===")
    if "event_id" in df.columns:
        print(df["event_id"].duplicated().sum())
    else:
        print("Colonne event_id absente")

    print("\n=== Villes présentes ===")
    if "city" in df.columns:
        print(df["city"].dropna().unique())
    else:
        print("Colonne city absente")

    print("\n=== Plage de dates ===")
    if "date_start" in df.columns:
        date_series = pd.to_datetime(df["date_start"], errors="coerce")
        print(f"Date min : {date_series.min()}")
        print(f"Date max : {date_series.max()}")
    else:
        print("Colonne date_start absente")

    print("\n=== Exemples de descriptions ===")
    if "description" in df.columns:
        print(df["description"].dropna().head(5).to_string(index=False))
    else:
        print("Colonne description absente")


if __name__ == "__main__":
    main()