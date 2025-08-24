# rag_helper.py — Baseline knowledge + specific context retrieval
import os, pickle
from typing import List, Tuple, Dict
import numpy as np
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("[ERROR] OPENAI_API_KEY environment variable not set!")
    exit(1)

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

def get_baseline_knowledge(mat, meta, max_chars: int = 30000) -> Tuple[str, List[dict]]:
    """
    ALWAYS retrieve fundamental video psychology knowledge
    These patterns apply to ALL videos regardless of type
    """
    # Check if we have valid data
    if mat is None or meta is None:
        return "", []
    
    # Core concepts that EVERY video needs
    baseline_queries = [
        "hook first 3 seconds opening attention scroll stopping",
        "retention watch time completion rate drop off points",
        "engagement psychology curiosity gaps pattern interrupts",
        "viral mechanics shareable moments social triggers", 
        "visual storytelling frame composition text overlays",
        "promise payoff satisfaction completion desire",
        "psychological triggers emotional response viewer state",
        "platform algorithm optimization signals metrics",
        "content structure timing pacing rhythm flow",
        "audience psychology attention spans viewing behavior"
    ]
    
    baseline_chunks = []
    baseline_citations = []
    seen_chunks = set()  # Avoid duplicates
    total_chars = 0
    
    for query in baseline_queries:
        if total_chars >= max_chars:
            break
            
        v = _embed_query(query)
        sims = mat @ v
        top_idxs = np.argsort(-sims)[:5]  # Top 5 for each concept
        
        for idx in top_idxs:
            if total_chars >= max_chars:
                break
                
            # Skip if we've seen this chunk
            chunk_key = int(idx)
            if chunk_key in seen_chunks:
                continue
                
            m = meta.get(chunk_key)
            if not m:
                continue
                
            snippet = (m["text"] or "").strip()
            if not snippet:
                continue
                
            # Only include highly relevant baseline knowledge
            if sims[idx] < 0.3:
                continue
                
            piece = f"{snippet}\n\n"
            if total_chars + len(piece) > max_chars:
                break
                
            baseline_chunks.append(piece)
            seen_chunks.add(chunk_key)
            total_chars += len(piece)
            
            baseline_citations.append({
                "type": "baseline",
                "concept": query.split()[0],  # First word as concept label
                "file": m["file"],
                "chunk_id": m["chunk_id"],
                "similarity": float(sims[idx])
            })
    
    print(f"[BASELINE RAG] Retrieved {len(baseline_chunks)} fundamental chunks ({total_chars} chars)")
    return "".join(baseline_chunks).strip(), baseline_citations

def get_specific_knowledge(
    mat, meta, 
    transcript: str, 
    frames: str, 
    creator_note: str,
    goal: str,
    max_chars: int = 45000
) -> Tuple[str, List[dict]]:
    """
    Retrieve knowledge specific to this video's context
    More targeted than baseline, but still comprehensive
    """
    # Check if we have valid data
    if mat is None or meta is None:
        return "", []
    
    specific_queries = []
    
    # Performance-based queries (CRITICAL)
    if creator_note:
        note_lower = creator_note.lower()
        
        # Extract view count for targeted search
        import re
        view_patterns = re.findall(r'(\d+\.?\d*)\s*(k|m|million|thousand)', note_lower)
        
        if view_patterns:
            number, unit = view_patterns[0]
            num = float(number)
            
            if unit in ['m', 'million'] or (unit in ['k', 'thousand'] and num >= 500):
                # High performing - learn what worked
                specific_queries.extend([
                    "viral success patterns million views breakthrough",
                    "what makes videos explode viral triggers mechanics",
                    "high performing content successful examples"
                ])
            elif unit in ['k', 'thousand'] and num < 100:
                # Low performing - learn fixes
                specific_queries.extend([
                    "low performing videos common mistakes fixes",
                    "weak hooks poor retention problems solutions",
                    "underperforming content improvements optimization"
                ])
    
    # Content type queries (but not too specific)
    combined = f"{transcript} {frames}".lower()
    
    # Broader content categories
    if any(word in combined for word in ["product", "unbox", "package", "review"]):
        specific_queries.append("product reveals unboxing reviews demonstrations")
    
    if any(word in combined for word in ["routine", "morning", "night", "daily", "skincare", "beauty"]):
        specific_queries.append("routines processes daily habits lifestyle content")
    
    if any(word in combined for word in ["cook", "recipe", "food", "kitchen", "bake"]):
        specific_queries.append("cooking food tutorials process videos recipes")
    
    if any(word in combined for word in ["diy", "craft", "make", "build", "create"]):
        specific_queries.append("DIY crafts making building creative process")
    
    # Visual-only detection
    if not transcript or len(transcript.strip()) < 20:
        specific_queries.append("visual only content no speech satisfying ASMR ambient")
    
    # Goal-based queries
    goal_queries = {
        "viral_reach": "viral growth explosive reach millions shareable",
        "follower_growth": "building audience followers community loyalty retention",
        "sales": "selling conversion buyers psychology purchasing decisions",
        "engagement": "comments interaction discussion community response"
    }
    if goal in goal_queries:
        specific_queries.append(goal_queries[goal])
    
    # Process type queries (from frames analysis)
    if "drawing" in combined or "art" in combined:
        specific_queries.append("art process creative visual satisfaction completion")
    
    if "transform" in combined or "before" in combined or "after" in combined:
        specific_queries.append("transformation before after reveal dramatic change")
    
    # Now retrieve specific knowledge
    specific_chunks = []
    specific_citations = []
    seen_chunks = set()
    total_chars = 0
    
    for query in specific_queries:
        if total_chars >= max_chars:
            break
            
        v = _embed_query(query)
        sims = mat @ v
        top_idxs = np.argsort(-sims)[:8]  # Top 8 for each specific query
        
        for idx in top_idxs:
            if total_chars >= max_chars:
                break
                
            chunk_key = int(idx)
            if chunk_key in seen_chunks:
                continue
                
            m = meta.get(chunk_key)
            if not m:
                continue
                
            snippet = (m["text"] or "").strip()
            if not snippet:
                continue
                
            # Slightly lower threshold for specific content
            if sims[idx] < 0.65:
                continue
                
            piece = f"{snippet}\n\n"
            if total_chars + len(piece) > max_chars:
                break
                
            specific_chunks.append(piece)
            seen_chunks.add(chunk_key)
            total_chars += len(piece)
            
            specific_citations.append({
                "type": "specific",
                "query": query[:30],
                "file": m["file"],
                "chunk_id": m["chunk_id"],
                "similarity": float(sims[idx])
            })
    
    print(f"[SPECIFIC RAG] Retrieved {len(specific_chunks)} context chunks ({total_chars} chars)")
    return "".join(specific_chunks).strip(), specific_citations

# In rag_helper.py, find the retrieve_smart_context function and replace it with this:

def get_baseline_knowledge(meta, max_chars: int = 30000):
    """
    ALWAYS include viral/retention essentials for ALL videos
    These are the foundation of ANY successful video
    """
    # ESSENTIAL for ALL videos - hooks, virality, retention
    universal_essentials = [
        "50_Hook_Examples.pdf",
        "HookWritingGuide_Download.pdf", 
        "Trial Reels Guide.pdf",
        "x8u4vlfmj1n62gdem7rbpyq52jcg.pdf",
        "video_retention.txt",
        "architecture_of_retention.txt"
    ]
    
    baseline_chunks = []
    baseline_citations = []
    total_chars = 0
    
    # GUARANTEE these documents are loaded first
    for target_file in universal_essentials:
        # Get all chunks from this essential file
        file_chunks = []
        for idx, m in meta.items():
            if m["file"] == target_file:
                file_chunks.append((m["chunk_id"], m["text"]))
        
        # Sort by chunk_id to maintain document flow
        file_chunks.sort(key=lambda x: x[0])
        
        # Add all chunks from this essential file
        for chunk_id, text in file_chunks:
            if total_chars + len(text) > max_chars:
                break
            baseline_chunks.append(f"[Essential: {target_file}]\n{text}")
            baseline_citations.append({
                "file": target_file,
                "type": "universal_essential",
                "chunk_id": chunk_id
            })
            total_chars += len(text)
    
    print(f"[BASELINE] Loaded {len(baseline_chunks)} essential chunks ({total_chars} chars)")
    return "\n\n".join(baseline_chunks), baseline_citations

def get_specific_knowledge(meta, mat, transcript, frames, creator_note, goal, max_chars: int = 45000):
    """
    Add goal-specific and context-aware knowledge ON TOP of essentials
    """
    # [Copy the entire get_specific_knowledge function from above]
    # ... (the full function code)

def retrieve_smart_context(transcript: str, frames: str, creator_note: str, goal: str, max_chars: int = 75000) -> Tuple[str, List[
) -> Tuple[str, List[dict]]:
    """
    IMPROVED: Baseline + Specific knowledge retrieval
    Always includes fundamental principles + context-specific insights
    """
    
    mat, meta = _load_matrix_and_meta()
    if mat is None:
        # Fallback to retrieve_all_context if embeddings don't exist
        print("[WARNING] No embeddings found, trying fallback to all context")
        return retrieve_all_context(max_chars)
    
    # Step 1: Get baseline knowledge (30K chars)
    baseline_text, baseline_cits = get_baseline_knowledge(
        mat, meta, max_chars=30000
    )
    
    # Step 2: Get specific knowledge (45K chars)
    specific_text, specific_cits = get_specific_knowledge(
        mat, meta, transcript, frames, creator_note, goal, max_chars=45000
    )
    
    # Combine with clear sections
    combined_knowledge = f"""
=== FUNDAMENTAL VIDEO PRINCIPLES (Apply to ALL content) ===
{baseline_text}

=== SPECIFIC PATTERNS FOR THIS CONTENT ===
{specific_text}
""".strip()
    
    # Combine citations
    all_citations = baseline_cits + specific_cits
    
    # Trim if over max_chars
    if len(combined_knowledge) > max_chars:
        combined_knowledge = combined_knowledge[:max_chars]
    
    print(f"[SMART RAG TOTAL] {len(combined_knowledge)} chars ({len(baseline_cits)} baseline + {len(specific_cits)} specific chunks)")
    
    return combined_knowledge, all_citations

def retrieve_all_context(max_chars: int = 100000) -> Tuple[str, List[dict]]:
    """
    Returns ALL knowledge organized by importance
    First baseline, then everything else
    """
    # First try with embeddings for better organization
    mat, meta = _load_matrix_and_meta()
    
    if mat is not None and meta is not None:
        # Get baseline first using embeddings
        baseline_text, baseline_cits = get_baseline_knowledge(
            mat, meta, max_chars=40000
        )
        
        # Then get remaining content
        with open(META_PATH, "rb") as f:
            meta_full = pickle.load(f)
        
        from collections import defaultdict
        by_file = defaultdict(list)
        
        # Collect all chunks not in baseline
        baseline_chunk_ids = {c["chunk_id"] for c in baseline_cits}
        
        for idx, m in meta_full.items():
            if m["chunk_id"] not in baseline_chunk_ids:
                by_file[m["file"]].append((m["chunk_id"], m["text"]))
        
        for k in by_file:
            by_file[k].sort(key=lambda x: x[0])
        
        additional_parts = []
        additional_citations = []
        total = len(baseline_text)
        
        for file_name, chunks in sorted(by_file.items()):
            if total >= max_chars:
                break
                
            header = f"\n\n=== {file_name} ===\n"
            if total + len(header) > max_chars:
                break
            additional_parts.append(header)
            total += len(header)
            
            for chunk_id, text in chunks:
                segment = (text or "").strip() + "\n\n"
                if not segment.strip():
                    continue
                if total + len(segment) > max_chars:
                    break
                additional_parts.append(segment)
                total += len(segment)
            
            additional_citations.append({
                "file": file_name,
                "type": "additional"
            })
        
        combined = f"""
=== FUNDAMENTAL PRINCIPLES ===
{baseline_text}

=== ADDITIONAL KNOWLEDGE ===
{"".join(additional_parts)}
""".strip()
        
        all_citations = baseline_cits + additional_citations
        
        print(f"[RAG ALL] Retrieved {len(combined)} chars total")
        return combined, all_citations
    
    # Fallback: Just load metadata and return all text if no embeddings
    if not os.path.exists(META_PATH):
        print("[WARNING] No metadata found. Run ingest_knowledge.py first.")
        return "", []
    
    with open(META_PATH, "rb") as f:
        meta_full = pickle.load(f)
    
    from collections import defaultdict
    by_file = defaultdict(list)
    for idx, m in meta_full.items():
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
    
    print(f"[RAG ALL - Fallback] Retrieved {total} chars from {rank-1} files")
    return "".join(ctx_parts).strip(), citations

def retrieve_context(query: str, top_k: int = 30, max_chars: int = 75000) -> Tuple[str, List[dict]]:
    """
    Original function maintained for compatibility
    But now includes baseline knowledge automatically
    """
    mat, meta = _load_matrix_and_meta()
    if mat is None:
        return "", []
    
    # Get baseline first (20K)
    baseline_text, baseline_cits = get_baseline_knowledge(
        mat, meta, max_chars=20000
    )
    
    # Then get query-specific (55K)
    v = _embed_query(query)
    sims = mat @ v
    
    if top_k <= 0 or top_k > len(sims):
        top_k = min(len(sims), 50)
    
    idxs = np.argsort(-sims)[:top_k]
    
    query_parts = []
    query_cits = []
    total = len(baseline_text)
    
    seen_baseline = {c["chunk_id"] for c in baseline_cits}
    
    for rank, idx in enumerate(idxs, start=1):
        m = meta.get(int(idx))
        if not m or m["chunk_id"] in seen_baseline:
            continue
            
        snippet = (m["text"] or "").strip()
        if not snippet:
            continue
            
        piece = f"{snippet}\n\n"
        if total + len(piece) > max_chars:
            break
            
        query_parts.append(piece)
        total += len(piece)
        
        query_cits.append({
            "rank": rank,
            "similarity": float(sims[idx]),
            "file": m["file"],
            "chunk_id": m["chunk_id"]
        })
    
    combined = f"""
=== FUNDAMENTAL PRINCIPLES ===
{baseline_text}

=== QUERY-SPECIFIC KNOWLEDGE ===
{"".join(query_parts)}
""".strip()
    
    return combined, baseline_cits + query_cits