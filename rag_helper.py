# rag_helper.py — Baseline knowledge + specific context retrieval
import os, pickle
from typing import List, Tuple, Dict
import numpy as np
from openai import OpenAI
from collections import defaultdict

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

def get_baseline_knowledge(meta, max_chars: int = 30000) -> Tuple[str, List[dict]]:
    """
    ALWAYS include viral/retention essentials for ALL videos
    These are the foundation of ANY successful video
    """
    # ESSENTIAL for ALL videos - hooks, virality, retention
    universal_essentials = [
        "master.txt",
        "x8u4vlfmj1n62gdem7rbpyq52jcg.pdf",
        "video_retention.txt",
        "architecture_of_retention.txt",
        "hook_mechanisms.txt",
        "thisvsthat.txt",
        "failuretofix.txt",
        "50_Hook_Examples.pdf",
        "HookWritingGuide_Download.pdf", 
        "Trial Reels Guide.pdf",
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
    essential_files = list(set(c["file"] for c in baseline_citations))
    print(f"[BASELINE] Files: {', '.join(essential_files)}")
    
    return "\n\n".join(baseline_chunks), baseline_citations

def get_specific_knowledge(meta, mat, transcript, frames, creator_note, goal, max_chars: int = 45000) -> Tuple[str, List[dict]]:
    """
    Add goal-specific and context-aware knowledge ON TOP of essentials
    """
    specific_chunks = []
    specific_citations = []
    total_chars = 0
    
    # GOAL-BASED DOCUMENT SELECTION
    goal_document_map = {
        "sales_conversions": [
            "sales-psychology.txt",
            "sales_backed_content.txt",
            "messaging_to_sell_your_offer.txt",
            "overcoming_objections.txt",
            "types_of_buyers_and_how_to_sell.txt",
            "what_makes_them_buy.txt",
            "specificity_sells.txt",
            "example_scripts_for_growth_and_sales.txt"
        ],
        "lead_generation": [
            "how_to_warm_up_your_audience.txt",
            "messaging_to_sell_your_offer.txt",
            "example_scripts_for_growth_and_sales.txt"
        ],
        "follower_growth": [
            "Audience_Strategies.txt",
            "jazmedia_framework.txt",
            "example_scripts_for_growth_and_sales.txt"
        ],
        "viral_reach": [
            "jazmedia_framework.txt",
            "Audience_Strategies.txt"
        ],
        "engagement": [
            "Audience_Strategies.txt",
            "how_to_warm_up_your_audience.txt"
        ]
    }
    
    # Load goal-specific documents
    if goal in goal_document_map:
        print(f"[SPECIFIC] Loading {goal} knowledge")
        target_docs = goal_document_map[goal]
        
        for target_file in target_docs:
            if total_chars >= max_chars * 0.7:  # Use 70% for goal docs
                break
                
            # Get all chunks from this file
            file_chunks = []
            for idx, m in meta.items():
                if m["file"] == target_file:
                    file_chunks.append((m["chunk_id"], m["text"]))
            
            # Sort and add
            file_chunks.sort(key=lambda x: x[0])
            for chunk_id, text in file_chunks:
                if total_chars + len(text) > max_chars * 0.7:
                    break
                specific_chunks.append(f"[{goal.replace('_', ' ').title()}: {target_file}]\n{text}")
                specific_citations.append({
                    "file": target_file,
                    "type": f"{goal}_specific",
                    "chunk_id": chunk_id
                })
                total_chars += len(text)
    
    # Use remaining space for contextual semantic search if mat is available
    if mat is not None and total_chars < max_chars:
        remaining_chars = max_chars - total_chars
        combined_context = f"{transcript} {frames} {creator_note}".lower()
        
        # Build intelligent context queries
        context_queries = []
        
        # Add queries based on detected content type
        if "unbox" in combined_context or "package" in combined_context:
            context_queries.append("unboxing reveal surprise product review")
        
        if "tutorial" in combined_context or "how to" in combined_context:
            context_queries.append("educational tutorial teaching guide")
        
        if goal in ["sales", "lead_generation"]:
            context_queries.append("conversion selling services client acquisition")
        
        # Semantic search for additional relevant content
        for query in context_queries:
            if remaining_chars <= 0:
                break
                
            v = _embed_query(query)
            sims = mat @ v
            top_idxs = np.argsort(-sims)[:3]  # Top 3 per query
            
            for idx in top_idxs:
                if remaining_chars <= 0:
                    break
                    
                if sims[idx] < 0.4:
                    continue
                    
                m = meta.get(int(idx))
                if m:
                    # Check if we already have this chunk
                    already_added = any(
                        c["file"] == m["file"] and c["chunk_id"] == m["chunk_id"] 
                        for c in specific_citations
                    )
                    
                    if not already_added:
                        chunk_text = f"[Contextual: {m['file']}]\n{m['text']}"
                        if len(chunk_text) <= remaining_chars:
                            specific_chunks.append(chunk_text)
                            specific_citations.append({
                                "file": m["file"],
                                "type": "contextual",
                                "chunk_id": m["chunk_id"],
                                "similarity": float(sims[idx])
                            })
                            remaining_chars -= len(chunk_text)
    
    print(f"[SPECIFIC] Loaded {len(specific_chunks)} goal/context chunks ({total_chars} chars)")
    return "\n\n".join(specific_chunks), specific_citations

def retrieve_smart_context(transcript: str, frames: str, creator_note: str, goal: str, max_chars: int = 75000) -> Tuple[str, List[dict]]:
    """
    ALWAYS: Hooks + Virality + Retention (30K)
    PLUS: Goal-specific knowledge (45K)
    """
    mat, meta = _load_matrix_and_meta()
    if mat is None or meta is None:
        print("[WARNING] No embeddings found, trying fallback to all context")
        return retrieve_all_context(max_chars)
    
    # Step 1: ALWAYS get hooks, virality, retention (30K chars)
    baseline_text, baseline_cits = get_baseline_knowledge(meta, 30000)
    
    # Step 2: Add goal-specific knowledge (45K chars)
    specific_text, specific_cits = get_specific_knowledge(
        meta, mat, transcript, frames, creator_note, goal, 45000
    )
    
    # Determine section label based on goal
    specific_label = {
        "sales": "SALES & CONVERSION STRATEGIES",
        "lead_generation": "LEAD GENERATION & WARMING",
        "follower_growth": "AUDIENCE GROWTH TACTICS",
        "viral_reach": "VIRAL AMPLIFICATION STRATEGIES",
        "engagement": "ENGAGEMENT OPTIMIZATION"
    }.get(goal, "TARGETED STRATEGIES")
    
    combined = f"""
=== ESSENTIAL VIDEO FOUNDATIONS (ALWAYS APPLY) ===
Every video needs strong hooks, retention mechanics, and viral principles:

{baseline_text}

=== {specific_label} ===
Specific strategies for your {goal.replace('_', ' ')} goal:

{specific_text}
""".strip()
    
    all_citations = baseline_cits + specific_cits
    
    # Summary output
    essential_files = set(c["file"] for c in baseline_cits if c["type"] == "universal_essential")
    specific_files = set(c["file"] for c in specific_cits)
    
    print(f"[SMART RAG] Essential docs: {len(essential_files)} files")
    print(f"[SMART RAG] Goal-specific docs: {len(specific_files)} files for '{goal}'")
    print(f"[SMART RAG TOTAL] {len(combined)} chars ({len(baseline_cits)} essential + {len(specific_cits)} specific chunks)")
    
    return combined[:max_chars], all_citations

def retrieve_all_context(max_chars: int = 100000) -> Tuple[str, List[dict]]:
    """
    Returns ALL knowledge organized by importance
    First baseline, then everything else
    """
    # First try with embeddings for better organization
    mat, meta = _load_matrix_and_meta()
    
    if mat is not None and meta is not None:
        # Get baseline first using file-based approach
        baseline_text, baseline_cits = get_baseline_knowledge(meta, 40000)
        
        # Then get remaining content
        by_file = defaultdict(list)
        
        # Collect all chunks not in baseline
        baseline_chunk_ids = {(c["file"], c["chunk_id"]) for c in baseline_cits}
        
        for idx, m in meta.items():
            if (m["file"], m["chunk_id"]) not in baseline_chunk_ids:
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
    if mat is None or meta is None:
        return "", []
    
    # Get baseline first (20K)
    baseline_text, baseline_cits = get_baseline_knowledge(meta, 20000)
    
    # Then get query-specific (55K) using semantic search
    v = _embed_query(query)
    sims = mat @ v
    
    if top_k <= 0 or top_k > len(sims):
        top_k = min(len(sims), 50)
    
    idxs = np.argsort(-sims)[:top_k]
    
    query_parts = []
    query_cits = []
    total = len(baseline_text)
    
    seen_baseline = {(c["file"], c["chunk_id"]) for c in baseline_cits}
    
    for rank, idx in enumerate(idxs, start=1):
        m = meta.get(int(idx))
        if not m or (m["file"], m["chunk_id"]) in seen_baseline:
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