import os
from fastapi import FastAPI, HTTPException

from app.rag.schemas import IngestRequest, IngestResponse, QueryRequest, QueryResponse, Citation
from app.rag.ingest import load_documents, chunk_documents
from app.rag.retrieve import build_vectorstore, load_vectorstore, retrieve
from app.rag.pipeline import answer_question

PERSIST_DIR = "data/vectorstore"

app = FastAPI(title="PaperPulse RAG")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    docs = load_documents(req.input_dir)
    if not docs:
        raise HTTPException(status_code=400, detail="No documents found in input_dir")

    chunks = chunk_documents(docs, req.chunk_size, req.chunk_overlap)
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks created")

    build_vectorstore(chunks, PERSIST_DIR)
    return IngestResponse(docs_ingested=len(docs), chunks_created=len(chunks))

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not os.path.exists(PERSIST_DIR):
        raise HTTPException(status_code=400, detail="Vectorstore not found. Run /ingest first.")

    vs = load_vectorstore(PERSIST_DIR)
    docs = retrieve(vs, req.question, req.top_k, req.similarity_threshold)
    answer, cites = answer_question(req.question, req.mode, docs)

    return QueryResponse(
        answer=answer,
        citations=[Citation(**c) for c in cites],
        metrics={"retrieved_chunks": len(docs)}
    )
