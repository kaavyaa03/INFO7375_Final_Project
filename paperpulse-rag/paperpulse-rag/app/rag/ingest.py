import os, uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def load_documents(input_dir: str):
    docs = []
    for file in os.listdir(input_dir):
        path = os.path.join(input_dir, file)
        if file.lower().endswith(".txt"):
            with open(path, encoding="utf-8") as f:
                docs.append(Document(page_content=f.read(), metadata={"source": file}))
    return docs

def chunk_documents(docs, chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for doc in docs:
        for text in splitter.split_text(doc.page_content):
            chunks.append(Document(
                page_content=text,
                metadata={"source": doc.metadata["source"], "chunk_id": str(uuid.uuid4())[:8]}
            ))
    return chunks
