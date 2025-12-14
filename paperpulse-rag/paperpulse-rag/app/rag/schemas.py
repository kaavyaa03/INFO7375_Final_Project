from pydantic import BaseModel
from typing import List, Optional, Dict, Literal

Mode = Literal["ask", "summarize"]

class IngestRequest(BaseModel):
    input_dir: str = "data/kb_samples"
    chunk_size: int = 900
    chunk_overlap: int = 180

class IngestResponse(BaseModel):
    docs_ingested: int
    chunks_created: int

class QueryRequest(BaseModel):
    question: str
    mode: Mode = "ask"
    top_k: int = 8
    similarity_threshold: float = 0.25
    history: Optional[List[Dict[str, str]]] = None

class Citation(BaseModel):
    source: str
    chunk_id: str

class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    metrics: Dict
