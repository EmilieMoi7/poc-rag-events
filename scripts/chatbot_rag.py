import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from mistralai.client import Mistral

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
VECTORSTORE_DIR = Path("vectorstore/faiss_index_chunks")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "mistral-small-latest"
TOP_K = 3


def load_vectorstore() -> FAISS:
    """
    Charge l'index FAISS sauvegardé.
    """
    if not VECTORSTORE_DIR.exists():
        raise FileNotFoundError(f"Index FAISS introuvable : {VECTORSTORE_DIR}")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    vectorstore = FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore


def retrieve_context(question: str, k: int = TOP_K):
    """
    Recherche les chunks les plus pertinents dans FAISS.
    """
    vectorstore = load_vectorstore()
    results = vectorstore.similarity_search(question, k=k)
    return results


def build_context(documents) -> str:
    """
    Construit le contexte textuel à envoyer au LLM.
    """
    context_parts = []

    for i, doc in enumerate(documents, start=1):
        context_parts.append(
            f"""Document {i}
Titre : {doc.metadata.get('title', '')}
Date : {doc.metadata.get('date_start', '')}
Ville : {doc.metadata.get('city', '')}
Lieu : {doc.metadata.get('location_name', '')}
Catégorie : {doc.metadata.get('category', '')}
URL : {doc.metadata.get('url', '')}

Contenu :
{doc.page_content}
"""
        )

    return "\n\n".join(context_parts)


def build_prompt(question: str, context: str) -> str:
    """
    Construit le prompt final pour Mistral.
    """
    prompt = f"""
Tu es un assistant spécialisé dans les événements culturels à Lille.

Réponds uniquement à partir des informations fournies dans le contexte.
Si l'information n'est pas présente dans le contexte, dis clairement que tu ne sais pas.
Réponds en français, de manière claire, utile et naturelle.

Contexte :
{context}

Question utilisateur :
{question}

Réponse :
"""
    return prompt.strip()


def ask_mistral(prompt: str) -> str:
    if not MISTRAL_API_KEY:
        raise ValueError("Clé Mistral manquante")

    with Mistral(api_key=MISTRAL_API_KEY) as client:
        response = client.chat.complete(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

    return response.choices[0].message.content

def ask_rag(question: str) -> str:
    documents = retrieve_context(question, k=TOP_K)
    context = build_context(documents)
    prompt = build_prompt(question, context)
    answer = ask_mistral(prompt)
    return answer

def main():
    while True:
        question = input("\nPose ta question (ou 'quit' pour arrêter) : ").strip()

        if question.lower() in ["quit", "exit"]:
            print("Fin du chatbot.")
            break

        if not question:
            print("Aucune question saisie.")
            continue

        documents = retrieve_context(question, k=TOP_K)

        print("\n===== DOCUMENTS RETROUVÉS =====")
        for i, doc in enumerate(documents, start=1):
            print(f"\n--- Document {i} ---")
            print(doc.page_content[:500])
            print("Metadata :", doc.metadata)

        answer = ask_rag(question)

        print("\n===== RÉPONSE DU CHATBOT =====")
        print(answer)


if __name__ == "__main__":
    main()