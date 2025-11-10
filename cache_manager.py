"""Smart caching with 24-hour expiration."""
import hashlib
import pickle
import os
import time
from pathlib import Path
from typing import Any, Optional


class AnalysisCache:
    def __init__(self, cache_dir: str = "./cache", max_size_gb: float = 1.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        self.expiry_seconds = 86400  # 24 hours

    def get_video_hash(self, url: str) -> str:
        """Create unique hash for video URL."""
        return hashlib.md5(url.encode()).hexdigest()

    def get_cached_analysis(self, url: str, analysis_type: str = 'full') -> Optional[Any]:
        """Retrieve cached analysis if fresh."""
        video_hash = self.get_video_hash(url)
        cache_file = self.cache_dir / f"{video_hash}_{analysis_type}.pkl"

        if cache_file.exists():
            # Check age
            age = time.time() - cache_file.stat().st_mtime
            if age < self.expiry_seconds:
                try:
                    with open(cache_file, 'rb') as f:
                        print(f"[CACHE HIT] Using cached {analysis_type} (age: {age/3600:.1f}h)")
                        return pickle.load(f)
                except Exception as e:
                    print(f"[CACHE] Error loading cache: {e}")
                    return None
        return None

    def save_analysis(self, url: str, analysis_type: str, data: Any):
        """Save analysis to cache."""
        try:
            video_hash = self.get_video_hash(url)
            cache_file = self.cache_dir / f"{video_hash}_{analysis_type}.pkl"

            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)

            print(f"[CACHE] Saved {analysis_type} analysis")
            self._manage_cache_size()
        except Exception as e:
            print(f"[CACHE] Error saving cache: {e}")

    def _manage_cache_size(self):
        """Remove oldest files if cache exceeds limit."""
        try:
            total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.pkl"))

            if total_size > self.max_size_bytes:
                # Sort by modification time, delete oldest
                files = sorted(self.cache_dir.glob("*.pkl"), key=lambda f: f.stat().st_mtime)
                while total_size > self.max_size_bytes and files:
                    oldest = files.pop(0)
                    total_size -= oldest.stat().st_size
                    oldest.unlink()
                    print(f"[CACHE] Removed old cache: {oldest.name}")
        except Exception as e:
            print(f"[CACHE] Error managing cache size: {e}")

    def clear_cache(self):
        """Clear all cached files."""
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            print("[CACHE] All cache cleared")
        except Exception as e:
            print(f"[CACHE] Error clearing cache: {e}")
