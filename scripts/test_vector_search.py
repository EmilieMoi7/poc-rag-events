from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


VECTORSTORE_DIR = Path("vectorstore/faiss_index_chunks")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_vectorstore():
    """
    Charge l'index FAISS sauvegardé.
    """
    if not VECTORSTORE_DIR.exists():
        raise FileNotFoundError("Index FAISS introuvable")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )

    vectorstore = FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return vectorstore


def search(query: str, k: int = 3):
    """
    Recherche les k documents les plus proches.
    """
    vectorstore = load_vectorstore()

    results = vectorstore.similarity_search(query, k=k)

    print("\n===== QUESTION =====")
    print(query)

    print("\n===== RESULTATS =====")

    for i, doc in enumerate(results, start=1):
        print(f"\n--- Resultat {i} ---")
        print(doc.page_content)
        print("\nMetadata :", doc.metadata)


if __name__ == "__main__":
    question = "concert musique lille"
    search(question)