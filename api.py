from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from scripts.chatbot_rag import ask_rag

from scripts.build_vector_index_chunks import rebuild_vectorstore

app = FastAPI(
    title="API RAG Événements",
    description="API REST pour interroger le système RAG sur les événements indexés",
    version="1.0.0"
)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    question: str
    response: str
    status: str


@app.get("/")
def root() -> dict:
    return {"message": "API RAG opérationnelle"}


@app.post("/ask", response_model=AskResponse, summary="Poser une question au système RAG")
def ask_endpoint(request: AskRequest) -> AskResponse:
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="La question ne peut pas être vide."
        )

    try:
        answer = ask_rag(question)
        return AskResponse(
            question=question,
            response=answer,
            status="success"
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la génération de la réponse : {str(exc)}"
        )
    
@app.post("/rebuild", summary="Reconstruire la base vectorielle FAISS")
def rebuild_endpoint() -> dict:
    try:
        rebuild_vectorstore()
        return {
            "status": "success",
            "message": "Base vectorielle reconstruite avec succès."
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la reconstruction : {str(exc)}"
        )