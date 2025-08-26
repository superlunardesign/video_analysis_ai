# ingest_knowledge.py — Fixed to save metadata as dictionary
import os, re, glob, pickle
from pathlib import Path
from typing import List, Dict
import numpy as np
from pypdf import PdfReader
from docx import Document as DocxDocument
from openai import OpenAI
##from config import OPENAI_API_KEY

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("[ERROR] OPENAI_API_KEY environment variable not set!")
    exit(1)

INDEX_DIR  = "knowledge"
EMB_PATH   = f"{INDEX_DIR}/embeddings.npy"
META_PATH  = f"{INDEX_DIR}/meta.pkl"
EMBED_MODEL = "text-embedding-3-small"

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

client = OpenAI(api_key=OPENAI_API_KEY)

def read_txt(p):  
    return Path(p).read_text(encoding="utf-8", errors="ignore")

def read_pdf(p):
    r = PdfReader(p)
    return "\n".join((pg.extract_text() or "") for pg in r.pages)

def read_docx(p):
    d = DocxDocument(p)
    return "\n".join(p.text for p in d.paragraphs)

def load_file(p):
    ext = Path(p).suffix.lower()
    if ext in [".txt", ".md"]: 
        return read_txt(p)
    if ext == ".pdf":          
        return read_pdf(p)
    if ext == ".docx":         
        return read_docx(p)
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
        if start < 0: 
            start = 0
        if end == len(s): 
            break
    return [c.strip() for c in chunks if c.strip()]

def embed_batch(texts: List[str], batch_size: int = 100) -> List[List[float]]:
    """Embed texts in batches to avoid API limits"""
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"  Embedding batch {i+1}-{min(i+batch_size, len(texts))} of {len(texts)}...")
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        all_embeddings.extend([d.embedding for d in resp.data])
    
    return all_embeddings

def main():
    os.makedirs(INDEX_DIR, exist_ok=True)
    files = sorted(
        glob.glob("knowledge/*.pdf")
        + glob.glob("knowledge/*.docx")
        + glob.glob("knowledge/*.txt")
        + glob.glob("knowledge/*.md")
    )
    
    if not files:
        print("No files found in knowledge/ directory")
        return
    
    print(f"Found {len(files)} knowledge files")
    
    all_chunks = []
    all_meta = {}  # CHANGED: Dictionary instead of list
    idx = 0  # Index counter
    
    for fpath in files:
        fname = os.path.basename(fpath)
        print(f"Processing: {fname}")
        
        try:
            content = load_file(fpath)
            content = clean_text(content)
            chunks = chunk_text(content)
            print(f"  Created {len(chunks)} chunks")
            
            for chunk_id, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                # CHANGED: Store with integer key
                all_meta[idx] = {
                    "file": fname,
                    "chunk_id": chunk_id,
                    "text": chunk
                }
                idx += 1
                
        except Exception as e:
            print(f"  Error processing {fname}: {e}")
    
    if all_chunks:
        print(f"\nTotal chunks to embed: {len(all_chunks)}")
        print("Starting embedding process...")
        
        embeddings = embed_batch(all_chunks, batch_size=100)
        
        # Convert to numpy array and normalize
        embeddings_array = np.array(embeddings, dtype="float32")
        
        # Normalize each embedding
        print("Normalizing embeddings...")
        for i in range(embeddings_array.shape[0]):
            norm = np.linalg.norm(embeddings_array[i])
            if norm > 0:
                embeddings_array[i] = embeddings_array[i] / norm
        
        # Save embeddings and metadata
        np.save(EMB_PATH, embeddings_array)
        with open(META_PATH, 'wb') as f:
            pickle.dump(all_meta, f)  # Now saving dictionary
        
        print(f"\nSuccess!")
        print(f"Saved embeddings: {EMB_PATH} ({embeddings_array.shape})")
        print(f"Saved metadata: {META_PATH} ({len(all_meta)} entries)")
        
        # Show file sizes
        emb_size = os.path.getsize(EMB_PATH) / 1024 / 1024
        meta_size = os.path.getsize(META_PATH) / 1024 / 1024
        print(f"File sizes: {emb_size:.1f} MB (embeddings) + {meta_size:.1f} MB (metadata)")
    else:
        print("No chunks to process")

if __name__ == "__main__":
    main()