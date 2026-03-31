from pathlib import Path

import faiss
import pandas as pd
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


DATASET_PATH = Path("data/processed/events_lille_clean.csv")
VECTORSTORE_DIR = Path("vectorstore/faiss_index_chunks")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def clean_value(value) -> str:
    """
    Convertit proprement les valeurs manquantes en chaîne vide.
    """
    if pd.isna(value):
        return ""
    return str(value).strip()


def load_dataset() -> pd.DataFrame:
    """
    Charge le dataset nettoyé depuis le CSV.
    """
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Fichier introuvable : {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    if df.empty:
        raise ValueError("Le dataset est vide")

    return df


def build_event_text(row: pd.Series) -> str:
    """
    Construit un texte complet pour un événement.
    """
    parts = [
        f"Titre : {clean_value(row.get('title', ''))}",
        f"Date de début : {clean_value(row.get('date_start', ''))}",
        f"Date de fin : {clean_value(row.get('date_end', ''))}",
        f"Ville : {clean_value(row.get('city', ''))}",
        f"Lieu : {clean_value(row.get('location_name', ''))}",
        f"Catégorie : {clean_value(row.get('category', ''))}",
        f"Description : {clean_value(row.get('description', ''))}",
        f"URL : {clean_value(row.get('url', ''))}",
    ]

    return "\n".join(parts).strip()


def create_documents(df: pd.DataFrame) -> list[Document]:
    """
    Crée un document LangChain par événement.
    """
    documents = []

    for _, row in df.iterrows():
        content = build_event_text(row)

        metadata = {
            "event_id": clean_value(row.get("event_id", "")),
            "title": clean_value(row.get("title", "")),
            "city": clean_value(row.get("city", "")),
            "date_start": clean_value(row.get("date_start", "")),
            "location_name": clean_value(row.get("location_name", "")),
            "category": clean_value(row.get("category", "")),
            "url": clean_value(row.get("url", "")),
        }

        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """
    Découpe les documents en chunks et ajoute des métadonnées de chunk.
    """
    if not documents:
        raise ValueError("Aucun document à découper")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunked_documents = text_splitter.split_documents(documents)

    for i, doc in enumerate(chunked_documents):
        doc.metadata["chunk_id"] = i

    return chunked_documents


def verify_indexing(df: pd.DataFrame, documents: list[Document], chunked_documents: list[Document]) -> None:
    """
    Vérifie que tous les événements du dataset ont bien produit au moins un chunk.
    """
    dataset_event_ids = set(df["event_id"].astype(str).tolist())
    document_event_ids = {doc.metadata.get("event_id", "") for doc in documents}
    chunk_event_ids = {doc.metadata.get("event_id", "") for doc in chunked_documents}

    print("\n===== Vérification de l'indexation =====")
    print(f"Événements dans le dataset : {len(dataset_event_ids)}")
    print(f"Documents créés : {len(documents)}")
    print(f"Chunks créés : {len(chunked_documents)}")
    print(f"event_id uniques dans les documents : {len(document_event_ids)}")
    print(f"event_id uniques dans les chunks : {len(chunk_event_ids)}")

    missing_in_documents = dataset_event_ids - document_event_ids
    missing_in_chunks = dataset_event_ids - chunk_event_ids

    if missing_in_documents:
        raise ValueError(f"Événements non convertis en documents : {missing_in_documents}")

    if missing_in_chunks:
        raise ValueError(f"Événements non présents dans les chunks : {missing_in_chunks}")

    print("Tous les événements ont bien été pris en compte.")


def create_vectorstore(chunked_documents: list[Document]) -> FAISS:
    """
    Crée explicitement un index FAISS FlatL2 à partir des chunks.
    """
    if not chunked_documents:
        raise ValueError("Aucun chunk à indexer")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )

    sample_vector = embeddings.embed_query("test")
    dimension = len(sample_vector)

    # Index FAISS simple et rapide, adapté à un petit POC
    index = faiss.IndexFlatL2(dimension)

    vectorstore = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )

    vectorstore.add_documents(chunked_documents)
    return vectorstore


def save_vectorstore(vectorstore: FAISS) -> None:
    """
    Sauvegarde l'index FAISS sur disque.
    """
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))

def rebuild_vectorstore() -> None:
    """
    Reconstruit complètement l'index vectoriel FAISS à partir du dataset nettoyé.
    """
    print("Chargement du dataset...")
    df = load_dataset()

    print("Création des documents...")
    documents = create_documents(df)
    print(f"Nombre de documents créés : {len(documents)}")

    print("Découpage en chunks...")
    chunked_documents = split_documents(documents)
    print(f"Nombre de chunks créés : {len(chunked_documents)}")

    verify_indexing(df, documents, chunked_documents)

    print("\nCréation de l'index vectoriel FAISS...")
    vectorstore = create_vectorstore(chunked_documents)

    print("Sauvegarde de l'index...")
    save_vectorstore(vectorstore)

    print(f"Index FAISS sauvegardé dans : {VECTORSTORE_DIR}")


def main() -> None:
    rebuild_vectorstore()


if __name__ == "__main__":
    main()