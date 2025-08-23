# rag_helper.py — Optimized for maximum relevant knowledge retrieval
import os, pickle
from typing import List, Tuple, Dict
import numpy as np
from openai import OpenAI
from config import OPENAI_API_KEY

EMB_PATH   = "knowledge/embeddings.npy"
META_PATH  = "knowledge/meta.pkl"
EMBED_MODEL = "text-embedding-3-small"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY)

def _load_matrix_and_meta():
    if not (os.path.exists(EMB_PATH) and os.path.exists(META_PATH)):
        print("[WARNING] Embeddings not found. Run ingest_knowledge.py first.")
        return None, None
    mat = np.load(EMB_PATH).astype("float32")
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    return mat, meta

def _embed_query(q: str) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=[q])
    v = np.array(resp.data[0].embedding, dtype="float32")
    v = v / (np.linalg.norm(v) + 1e-12)
    return v

def retrieve_context(query: str, top_k: int = 30, max_chars: int = 75000) -> Tuple[str, List[dict]]:
    """
    OPTIMIZED: Retrieves up to 75K chars of the most relevant content
    Increased from 3K to 75K chars and 8 to 30 top chunks
    """
    mat, meta = _load_matrix_and_meta()
    if mat is None:
        return "", []

    v = _embed_query(query)
    sims = mat @ v
    
    # Get more chunks to have better coverage
    if top_k <= 0 or top_k > len(sims): 
        top_k = min(len(sims), 50)  # Cap at 50 for performance
    
    idxs = np.argsort(-sims)[:top_k]

    ctx_parts = []
    cits = []
    total = 0
    
    # Add relevance threshold - only include highly relevant chunks
    min_similarity = 0.65  # Adjust based on your needs
    
    for rank, idx in enumerate(idxs, start=1):
        similarity = sims[idx]
        
        # Skip low relevance chunks
        if similarity < min_similarity and rank > 10:  # Always include top 10
            continue
            
        m = meta.get(int(idx))
        if not m: 
            continue
            
        snippet = (m["text"] or "").strip()
        if not snippet: 
            continue
            
        # Include similarity score in output for transparency
        piece = f"[Relevance: {similarity:.2f}] {snippet}\n\n"
        
        if total + len(piece) > max_chars: 
            break
            
        ctx_parts.append(piece)
        total += len(piece)
        
        cits.append({
            "rank": rank,
            "similarity": float(similarity),
            "file": m["file"],
            "chunk_id": m["chunk_id"],
            "preview": snippet[:180] + ("..." if len(snippet) > 180 else "")
        })
    
    print(f"[RAG] Retrieved {len(ctx_parts)} chunks, {total} chars (query: {query[:50]}...)")
    return "".join(ctx_parts).strip(), cits

def retrieve_all_context(max_chars: int = 100000) -> Tuple[str, List[dict]]:
    """
    OPTIMIZED: Returns up to 100K chars of knowledge
    Increased from 16K to 100K chars
    """
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

    print(f"[RAG] Retrieved all context: {total} chars from {rank-1} files")
    return "".join(ctx_parts).strip(), citations

def retrieve_smart_context(
    transcript: str, 
    frames: str, 
    creator_note: str, 
    goal: str,
    max_chars: int = 75000
) -> Tuple[str, List[dict]]:
    """
    NEW: Smart retrieval based on video context
    Builds intelligent query and retrieves most relevant knowledge
    """
    
    # Build a comprehensive query from all context
    query_parts = []
    
    # Performance context is CRITICAL
    if creator_note:
        query_parts.append(f"Creator situation: {creator_note}")
        
        # Add specific performance keywords
        note_lower = creator_note.lower()
        if any(word in note_lower for word in ["300", "low", "poor", "didn't", "failed"]):
            query_parts.append("low performing video weak hooks failed underperformed fixes improvements")
        elif any(word in note_lower for word in ["million", "viral", "blew up", "exploded"]):
            query_parts.append("high performing viral successful patterns what worked millions of views")
    
    # Content type detection
    combined_text = f"{transcript} {frames}".lower()
    
    if "unboxing" in combined_text or "package" in combined_text:
        query_parts.append("unboxing package reveal mystery box opening product review")
    elif "skincare" in combined_text or "beauty" in combined_text or "routine" in combined_text:
        query_parts.append("skincare beauty routine morning night aesthetic self-care")
    elif "cooking" in combined_text or "recipe" in combined_text or "food" in combined_text:
        query_parts.append("cooking recipe food kitchen tutorial process")
    elif not transcript or len(transcript.strip()) < 20:
        query_parts.append("visual only no speech ambient audio satisfying process ASMR")
    
    # Add goal context
    goal_keywords = {
        "viral_reach": "viral explosive growth millions of views shareable",
        "follower_growth": "followers retention loyalty community building",
        "sales": "conversion monetization selling products services",
        "engagement": "comments shares saves interaction community"
    }
    query_parts.append(goal_keywords.get(goal, goal))
    
    # Add actual content samples
    if transcript and len(transcript.strip()) > 20:
        query_parts.append(f"Transcript: {transcript[:200]}")
    if frames:
        query_parts.append(f"Visuals: {frames[:200]}")
    
    # Build final query
    smart_query = " ".join(query_parts)
    
    print(f"[SMART RAG] Query built from context: {smart_query[:200]}...")
    
    # Use higher top_k for comprehensive results
    return retrieve_context(smart_query, top_k=40, max_chars=max_chars)

def get_category_weights(transcript: str, frames: str, creator_note: str) -> Dict[str, float]:
    """
    NEW: Calculate relevance weights for different knowledge categories
    """
    weights = {}
    combined = f"{transcript} {frames} {creator_note}".lower()
    
    # Weight based on content type
    if "unboxing" in combined:
        weights["unboxing"] = 2.0
        weights["package"] = 1.8
        weights["reveal"] = 1.5
    
    if "skincare" in combined or "beauty" in combined:
        weights["skincare"] = 2.0
        weights["beauty"] = 1.8
        weights["routine"] = 1.5
    
    # Weight based on performance
    if creator_note:
        if "low" in creator_note.lower() or "300" in creator_note:
            weights["fix"] = 2.5
            weights["improve"] = 2.0
            weights["weak"] = 1.8
            weights["problem"] = 1.5
        elif "million" in creator_note.lower():
            weights["viral"] = 2.5
            weights["successful"] = 2.0
            weights["pattern"] = 1.5
    
    # Always weight hooks highly
    weights["hook"] = 1.8
    weights["opening"] = 1.5
    weights["first"] = 1.3
    
    return weights