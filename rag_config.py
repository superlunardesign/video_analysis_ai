# rag_config.py
CHUNK_SIZE = 1200          # characters per chunk
CHUNK_OVERLAP = 200        # characters overlap
EMBEDDING_MODEL = "text-embedding-3-small"  # cheap + good
INDEX_DIR = "knowledge"
INDEX_PATH = f"{INDEX_DIR}/faiss.index"
META_PATH  = f"{INDEX_DIR}/meta.pkl"