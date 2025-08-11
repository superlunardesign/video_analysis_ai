# rag_helper.py — vector search (top-k) + all-knowledge concat mode

import os
import pickle
from typing import List, Tuple
import faiss
import numpy as np
from openai import OpenAI

from rag_config import INDEX_PATH, META_PATH, EMBEDDING_MODEL
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------
# Internal helpers
# -------------------------
def _load_index_and_meta():
    if not (os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
        return None, None
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    return index, meta


def _embed_query(q: str) -> np.ndarray:
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=[q])
    v = np.array(resp.data[0].embedding, dtype="float32")
    faiss.normalize_L2(v.reshape(1, -1))
    return v


# -------------------------
# Public: top-k retrieval
# -------------------------
def retrieve_context(query: str, top_k: int = 8, max_chars: int = 3000) -> Tuple[str, List[dict]]:
    """
    Vector search over ingested chunks. Returns (context_text, citations)
    where citations = [{file, chunk_id, preview}].
    """
    index, meta = _load_index_and_meta()
    if index is None:
        return "", []

    v = _embed_query(query)
    D, I = index.search(v.reshape(1, -1), top_k)

    ctx_parts = []
    cits = []
    total = 0
    rank = 1
    for idx in I[0]:
        if idx == -1:
            continue
        m = meta.get(int(idx))
        if not m:
            continue
        snippet = m["text"].strip()
        if not snippet:
            continue
        piece = f"[{rank}] {snippet}\n\n"
        if total + len(piece) > max_chars:
            break
        ctx_parts.append(piece)
        cits.append({
            "rank": rank,
            "file": m["file"],
            "chunk_id": m["chunk_id"],
            "preview": snippet[:180] + ("..." if len(snippet) > 180 else "")
        })
        total += len(piece)
        rank += 1

    context = "".join(ctx_parts).strip()
    return context, cits


# -------------------------
# Public: ALL-knowledge concat (no vector search)
# -------------------------
def retrieve_all_context(max_chars: int = 16000) -> Tuple[str, List[dict]]:
    """
    Returns ALL knowledge chunks concatenated (up to max_chars) + simple citations.
    Bypasses vector search entirely and reads META_PATH only.
    """
    if not os.path.exists(META_PATH):
        return "", []

    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)

    from collections import defaultdict
    by_file = defaultdict(list)
    for idx, m in meta.items():
        by_file[m["file"]].append((m["chunk_id"], m["text"]))

    # Sort files + chunks for coherent output
    for k in by_file:
        by_file[k].sort(key=lambda x: x[0])

    ctx_parts = []
    citations = []
    total = 0
    rank = 1

    for file_name, chunks in sorted(by_file.items()):
        header = f"\n\n=== {file_name} ===\n"
        if total + len(header) > max_chars:
            break
        ctx_parts.append(header)
        total += len(header)

        for chunk_id, text in chunks:
            segment = (text or "").strip() + "\n\n"
            if not segment.strip():
                continue
            if total + len(segment) > max_chars:
                break
            ctx_parts.append(segment)
            total += len(segment)

        citations.append({
            "rank": rank,
            "file": file_name,
            "chunk_id": "all",
            "preview": f"{file_name} (multiple chunks)"
        })
        rank += 1
        if total >= max_chars:
            break

    context = "".join(ctx_parts).strip()
    return context, citations
