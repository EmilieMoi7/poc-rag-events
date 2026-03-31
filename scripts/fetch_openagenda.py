import os
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv


load_dotenv()

API_KEY = os.getenv("OPENAGENDA_API_KEY")
AGENDA_UID = "la-culture-en-continu"
BASE_URL = f"https://api.openagenda.com/v2/agendas/{AGENDA_UID}/events"

RAW_OUTPUT = Path("data/raw/openagenda_lille.json")
PROCESSED_OUTPUT = Path("data/processed/events_lille_clean.csv")

PAGE_SIZE = 100
TARGET_CITY: Optional[str] = None
PAST_DAYS = 365
FUTURE_DAYS = 365


def fetch_events() -> dict:
    """
    Récupère tous les événements de l'agenda OpenAgenda
    en gérant la pagination.
    """
    if not API_KEY:
        raise ValueError("Clé OpenAgenda introuvable dans le fichier .env")

    all_events = []
    offset = 0
    total = None

    while True:
        params = {
            "key": API_KEY,
            "size": PAGE_SIZE,
            "offset": offset
        }

        response = requests.get(BASE_URL, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        events = data.get("events", [])
        pagination = data.get("pagination", {})

        if total is None:
            total = pagination.get("total", len(events))

        if not events:
            break

        all_events.extend(events)
        offset += PAGE_SIZE

        if len(all_events) >= total:
            break

    full_data = {
        "agenda_uid": AGENDA_UID,
        "total_events_fetched": len(all_events),
        "events": all_events
    }

    RAW_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(RAW_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(full_data, f, ensure_ascii=False, indent=2)

    return full_data


def extract_text_field(field) -> str:
    """
    Extrait proprement un champ texte multilingue.
    Priorité au français, sinon chaîne vide.
    """
    if isinstance(field, dict):
        return field.get("fr", "") or ""
    return ""


def extract_category_field(keywords) -> str:
    """
    Transforme le champ keywords en chaîne de texte.
    """
    if isinstance(keywords, dict):
        fr_keywords = keywords.get("fr", [])
        if isinstance(fr_keywords, list):
            return ", ".join(fr_keywords)
    return ""


def clean_events(data: dict) -> pd.DataFrame:
    """
    Nettoie, structure et filtre les événements.
    """
    events = data.get("events", [])
    rows = []

    for event in events:
        title = extract_text_field(event.get("title"))
        description = extract_text_field(event.get("description"))
        long_description = extract_text_field(event.get("longDescription"))

        full_description = " ".join(
            part.strip() for part in [description, long_description] if part
        ).strip()

        first_timing = event.get("firstTiming", {}) or {}
        date_start = first_timing.get("begin", "")
        date_end = first_timing.get("end", "")

        location = event.get("location", {}) or {}
        city = location.get("city", "") or ""
        location_name = location.get("name", "") or ""

        category = extract_category_field(event.get("keywords"))

        event_id = event.get("uid", "") or ""
        event_url = event.get("canonicalUrl", "") or ""

        rows.append({
            "event_id": event_id,
            "title": title,
            "date_start": date_start,
            "date_end": date_end,
            "city": city,
            "location_name": location_name,
            "category": category,
            "description": full_description,
            "url": event_url
        })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df = df.drop_duplicates(subset=["event_id"])
    df = df[df["title"].notna() & (df["title"].str.strip() != "")]
    df = df[df["date_start"].notna() & (df["date_start"].astype(str).str.strip() != "")]

    df["date_start"] = pd.to_datetime(df["date_start"], errors="coerce", utc=True)
    df["date_end"] = pd.to_datetime(df["date_end"], errors="coerce", utc=True)

    df = df[df["date_start"].notna()]

    today = pd.Timestamp.now(tz="UTC")
    past_limit = today - pd.Timedelta(days=PAST_DAYS)
    future_limit = today + pd.Timedelta(days=FUTURE_DAYS)

    df = df[
        (df["date_start"] >= past_limit) &
        (df["date_start"] <= future_limit)
    ]

    if TARGET_CITY:
        df = df[df["city"].str.contains(TARGET_CITY, case=False, na=False)]

    df = df.sort_values(by="date_start").reset_index(drop=True)

    return df


def save_dataset(df: pd.DataFrame) -> None:
    """
    Sauvegarde le dataset nettoyé en CSV.
    """
    PROCESSED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_OUTPUT, index=False, encoding="utf-8")

    print(f"Dataset créé : {PROCESSED_OUTPUT}")
    print(f"Nombre de lignes : {len(df)}")
    print(df.head())


if __name__ == "__main__":
    data = fetch_events()
    df = clean_events(data)
    save_dataset(df)