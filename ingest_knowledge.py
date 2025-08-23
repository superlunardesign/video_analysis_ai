# ingest_knowledge.py — pure Numpy index (no FAISS)
import os, re, glob, pickle
from pathlib import Path
from typing import List, Dict
import numpy as np
from pypdf import PdfReader
from docx import Document as DocxDocument
from openai import OpenAI
from config import OPENAI_API_KEY

INDEX_DIR  = "knowledge"
EMB_PATH   = f"{INDEX_DIR}/embeddings.npy"
META_PATH  = f"{INDEX_DIR}/meta.pkl"
EMBED_MODEL = "text-embedding-3-small"

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

client = OpenAI(api_key=OPENAI_API_KEY)

def read_txt(p):  return Path(p).read_text(encoding="utf-8", errors="ignore")
def read_pdf(p):
    r = PdfReader(p)
    return "\n".join((pg.extract_text() or "") for pg in r.pages)
def read_docx(p):
    d = DocxDocument(p)
    return "\n".join(p.text for p in d.paragraphs)

def load_file(p):
    ext = Path(p).suffix.lower()
    if ext in [".txt", ".md"]: return read_txt(p)
    if ext == ".pdf":          return read_pdf(p)
    if ext == ".docx":         return read_docx(p)
    return read_txt(p)

def clean_text(s: str) -> str:
    s = re.sub(r"\r", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def chunk_text(s: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    s = s.strip()
    chunks, start = [], 0
    while start < len(s):
        end = min(len(s), start + size)
        segment = s[start:end]
        chunks.append(segment)
        start = end - overlap
        if start < 0: start = 0
        if end == len(s): break
    return [c.strip() for c in chunks if c.strip()]

def embed_batch(texts: List[str]) -> List[List[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def main():
    os.makedirs(INDEX_DIR, exist_ok=True)
    files = sorted(
        glob.glob("knowledge/*.pdf")
        + glob.glob("knowledge/*.docx")
        + glob.glob("knowledge/*.txt")
        + glob.glob("knowledge/*.md")  # Fixed: added the missing closing quote and file extension
    )
    
    if not files:
        print("No files found in knowledge/ directory")
        return
    
    all_chunks = []
    all_meta = []
    
    for fpath in files:
        print(f"Processing: {fpath}")
        try:
            content = load_file(fpath)
            content = clean_text(content)
            chunks = chunk_text(content)
            
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_meta.append({
                    "file": fpath,
                    "chunk_id": i,
                    "text": chunk
                })
        except Exception as e:
            print(f"Error processing {fpath}: {e}")
    
    if all_chunks:
        print(f"Embedding {len(all_chunks)} chunks...")
        embeddings = embed_batch(all_chunks)
        embeddings_array = np.array(embeddings)
        
        np.save(EMB_PATH, embeddings_array)
        with open(META_PATH, 'wb') as f:
            pickle.dump(all_meta, f)
        
        print(f"Saved {len(all_chunks)} chunks to {EMB_PATH} and {META_PATH}")
    else:
        print("No chunks to process")

if __name__ == "__main__":
    main()