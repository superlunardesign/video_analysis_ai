import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# ingest_knowledge.py
import os, re, pickle, glob
from pathlib import Path
from typing import List, Dict, Tuple
from pypdf import PdfReader
from docx import Document as DocxDocument
from openai import OpenAI
import faiss
from rag_config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, INDEX_DIR, INDEX_PATH, META_PATH
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

def read_txt(path):  # .txt, .md
    return Path(path).read_text(encoding="utf-8", errors="ignore")

def read_pdf(path):
    reader = PdfReader(path)
    pages = [p.extract_text() or "" for p in reader.pages]
    return "\n".join(pages)

def read_docx(path):
    doc = DocxDocument(path)
    return "\n".join(p.text for p in doc.paragraphs)

def load_file(path):
    ext = Path(path).suffix.lower()
    if ext in [".txt", ".md"]:
        return read_txt(path)
    if ext == ".pdf":
        return read_pdf(path)
    if ext in [".docx"]:
        return read_docx(path)
    # fallback: try read as text
    return read_txt(path)

def clean_text(s: str) -> str:
    s = re.sub(r"\r", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def chunk_text(s: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    s = s.strip()
    chunks = []
    start = 0
    while start < len(s):
        end = min(len(s), start + size)
        # try to end on a sentence boundary if possible
        segment = s[start:end]
        m = re.search(r"(?s)^(.+?)([.!?])(\s|$)", segment[::-1])
        if m and end < len(s):
            # can keep default; but to keep simple, accept raw window
            pass
        chunks.append(segment)
        start = end - overlap
        if start < 0:
            start = 0
        if end == len(s):
            break
    return [c.strip() for c in chunks if c.strip()]

def embed_texts(texts: List[str]) -> List[List[float]]:
    # OpenAI embeddings: returns list of vectors
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def main():
    os.makedirs(INDEX_DIR, exist_ok=True)
    files = sorted(
        glob.glob("knowledge/*.pdf")
        + glob.glob("knowledge/*.docx")
        + glob.glob("knowledge/*.txt")
        + glob.glob("knowledge/*.md")
    )
    if not files:
        print("No files found in knowledge/. Add PDFs, DOCX, TXT, or MD.")
        return

    all_chunks: List[str] = []
    meta: Dict[int, Dict] = {}
    idx = 0

    for f in files:
        print(f"[ingest] reading {f}")
        text = clean_text(load_file(f))
        chunks = chunk_text(text)
        print(f"  -> {len(chunks)} chunks")
        for i, ch in enumerate(chunks):
            all_chunks.append(ch)
            meta[idx] = {"file": os.path.basename(f), "chunk_id": i, "path": f, "text": ch}
            idx += 1

    if not all_chunks:
        print("No chunks to embed.")
        return

    print(f"[ingest] embedding {len(all_chunks)} chunks...")
    vecs = []
    batch = 64
    for i in range(0, len(all_chunks), batch):
        vecs.extend(embed_texts(all_chunks[i:i+batch]))

    dim = len(vecs[0])
    index = faiss.IndexFlatIP(dim)  # cosine-like if we normalize
    # normalize vectors for cosine similarity equivalence
    import numpy as np
    mat = np.array(vecs, dtype="float32")
    faiss.normalize_L2(mat)
    index.add(mat)

    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

    print(f"[ingest] saved index to {INDEX_PATH} and meta to {META_PATH}")

if __name__ == "__main__":
    main()
