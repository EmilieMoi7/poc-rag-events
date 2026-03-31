from pathlib import Path

import pandas as pd


DATASET_PATH = Path("data/processed/events_lille_clean.csv")


def load_dataset():
    assert DATASET_PATH.exists(), "Le fichier CSV n'existe pas"
    df = pd.read_csv(DATASET_PATH)
    return df


def test_dataset_not_empty():
    df = load_dataset()
    assert len(df) > 0, "Le dataset est vide"


def test_no_duplicate_event_id():
    df = load_dataset()
    assert "event_id" in df.columns
    duplicates = df["event_id"].duplicated().sum()
    assert duplicates == 0, "Des doublons sont présents"


def test_title_not_empty():
    df = load_dataset()
    assert "title" in df.columns
    empty_titles = df["title"].isna().sum()
    assert empty_titles == 0, "Titres manquants détectés"


def test_date_valid():
    df = load_dataset()
    assert "date_start" in df.columns

    dates = pd.to_datetime(df["date_start"], errors="coerce")
    invalid = dates.isna().sum()

    assert invalid == 0, "Dates invalides détectées"


def test_city_column_exists():
    df = load_dataset()
    assert "city" in df.columns