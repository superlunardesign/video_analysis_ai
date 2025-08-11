# rag_helper.py — Numpy cosine top-k + all-knowledge concat
import os, pickle
from typing import List, Tuple
import numpy as np
from openai import OpenAI
from config import OPENAI_API_KEY

EMB_PATH   = "knowledge/embeddings.npy"
META_PATH  = "knowledge/meta.pkl"
EMBED_MODEL = "text-embedding-3-small"

client = OpenAI(api_key=OPENAI_API_KEY)

def _load_matrix_and_meta():
    if not (os.path.exists(EMB_PATH) and os.path.exists(META_PATH)):
        return None, None
    mat = np.load(EMB_PATH).astype("float32")  # already normalized
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    return mat, meta

def _embed_query(q: str) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=[q])
    v = np.array(resp.data[0].embedding, dtype="float32")
    v = v / (np.linalg.norm(v) + 1e-12)       # normalize
    return v

def retrieve_context(query: str, top_k: int = 8, max_chars: int = 3000) -> Tuple[str, List[dict]]:
    mat, meta = _load_matrix_and_meta()
    if mat is None:
        return "", []

    v = _embed_query(query)                   # (d,)
    sims = mat @ v                            # cosine similarity
    if top_k <= 0 or top_k > len(sims): top_k = len(sims)
    idxs = np.argsort(-sims)[:top_k]

    ctx_parts = []
    cits = []
    total = 0
    for rank, idx in enumerate(idxs, start=1):
        m = meta.get(int(idx))
        if not m: continue
        snippet = (m["text"] or "").strip()
        if not snippet: continue
        piece = f"[{rank}] {snippet}\n\n"
        if total + len(piece) > max_chars: break
        ctx_parts.append(piece); total += len(piece)
        cits.append({
            "rank": rank,
            "file": m["file"],
            "chunk_id": m["chunk_id"],
            "preview": snippet[:180] + ("..." if len(snippet) > 180 else "")
        })

    return "".join(ctx_parts).strip(), cits

def retrieve_all_context(max_chars: int = 16000) -> Tuple[str, List[dict]]:
    if not os.path.exists(META_PATH):
        return "", []

    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)

    from collections import defaultdict
    by_file = defaultdict(list)
    for idx, m in meta.items():
        by_file[m["file"]].append((m["chunk_id"], m["text"]))

    for k in by_file:
        by_file[k].sort(key=lambda x: x[0])

    ctx_parts = []
    citations = []
    total = 0
    rank = 1

    for file_name, chunks in sorted(by_file.items()):
        header = f"\n\n=== {file_name} ===\n"
        if total + len(header) > max_chars: break
        ctx_parts.append(header); total += len(header)

        for chunk_id, text in chunks:
            segment = (text or "").strip() + "\n\n"
            if not segment.strip(): continue
            if total + len(segment) > max_chars: break
            ctx_parts.append(segment); total += len(segment)

        citations.append({
            "rank": rank, "file": file_name, "chunk_id": "all",
            "preview": f"{file_name} (multiple chunks)"
        })
        rank += 1
        if total >= max_chars: break

    return "".join(ctx_parts).strip(), citations
