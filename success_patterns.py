"""
Success Pattern Learning System
Stores and retrieves patterns from high-performing videos
"""
import os
import pickle
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class SuccessPatternStore:
    def __init__(self, storage_dir: str = "./success_patterns"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.patterns_file = self.storage_dir / "patterns.pkl"
        self.embeddings_file = self.storage_dir / "embeddings.npy"
        self.metadata_file = self.storage_dir / "metadata.pkl"

        # Load existing data
        self.patterns = self._load_patterns()
        self.embeddings = self._load_embeddings()
        self.metadata = self._load_metadata()

    def _load_patterns(self) -> List[Dict]:
        if self.patterns_file.exists():
            with open(self.patterns_file, 'rb') as f:
                return pickle.load(f)
        return []

    def _load_embeddings(self) -> Optional[np.ndarray]:
        if self.embeddings_file.exists():
            return np.load(self.embeddings_file)
        return None

    def _load_metadata(self) -> List[Dict]:
        if self.metadata_file.exists():
            with open(self.metadata_file, 'rb') as f:
                return pickle.load(f)
        return []

    def _save_all(self):
        """Save all data to disk"""
        with open(self.patterns_file, 'wb') as f:
            pickle.dump(self.patterns, f)

        if self.embeddings is not None:
            np.save(self.embeddings_file, self.embeddings)

        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)

    def add_successful_video(
        self,
        analysis_text: str,
        video_url: str,
        metrics: Dict,
        niche: str = "general",
        platform: str = "tiktok"
    ):
        """
        Store patterns from a successful video.

        Args:
            analysis_text: The full analysis from the AI
            video_url: URL of the video
            metrics: Performance metrics (views, engagement_rate, etc.)
            niche: Video niche/category
            platform: Social platform
        """
        # Extract key patterns from the analysis
        pattern_summary = self._extract_patterns(analysis_text, metrics)

        # Create embedding for similarity search
        embedding = self._create_embedding(pattern_summary)

        # Store pattern
        self.patterns.append({
            'summary': pattern_summary,
            'full_analysis': analysis_text,
            'timestamp': datetime.now().isoformat()
        })

        # Store embedding
        if self.embeddings is None:
            self.embeddings = np.array([embedding])
        else:
            self.embeddings = np.vstack([self.embeddings, embedding])

        # Store metadata
        self.metadata.append({
            'video_url': video_url,
            'metrics': metrics,
            'niche': niche,
            'platform': platform,
            'timestamp': datetime.now().isoformat()
        })

        self._save_all()
        print(f"[SUCCESS PATTERN] Stored pattern from {video_url}")
        print(f"  Views: {metrics.get('views', 'N/A')}, Engagement: {metrics.get('engagement_rate', 'N/A')}")

    def _extract_patterns(self, analysis_text: str, metrics: Dict) -> str:
        """Extract factual format patterns (not causal success factors)"""
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a video format pattern analyzer. Extract FACTUAL STRUCTURAL PATTERNS, not assumptions about why it succeeded.

Focus on:
- Observable formats (e.g., "this vs that", "before/after", "reaction", "tutorial")
- Hook structures (question, shock value, promise, etc.)
- Content organization (reveal timing, pacing, segment structure)
- Visual patterns (text placement, editing style, transitions)
- Script formulas (opening line pattern, call-to-action placement)

DO NOT make assumptions about causation. Report what IS present, not why it worked."""
                    },
                    {
                        "role": "user",
                        "content": f"""This video had:
- Views: {metrics.get('views', 'N/A')}
- Engagement Rate: {metrics.get('engagement_rate', 'N/A')}
- Average Watch Time: {metrics.get('watch_time', 'N/A')}

Analysis:
{analysis_text}

Extract OBSERVABLE FORMAT PATTERNS:
1. Content Format Type (e.g., "this vs that", "tutorial", "storytime")
2. Hook Structure (first 3s pattern)
3. Script Formula (opening/body/closing pattern)
4. Visual Organization (text, editing, pacing)
5. Structural Elements (reveals, callbacks, CTAs)

Be FACTUAL. Report patterns present, not assumed reasons for success."""
                    }
                ],
                max_tokens=800,
                temperature=0.2  # Lower for more factual
            )

            return response.choices[0].message.content
        except Exception as e:
            print(f"[ERROR] Pattern extraction failed: {e}")
            # Fallback to simple extraction
            return f"High-performing video patterns (Views: {metrics.get('views', 'N/A')})\n\n{analysis_text[:1000]}"

    def _create_embedding(self, text: str) -> np.ndarray:
        """Create embedding for similarity search"""
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            print(f"[ERROR] Embedding creation failed: {e}")
            return np.zeros(1536)  # Default dimension for text-embedding-3-small

    def find_similar_patterns(
        self,
        analysis_text: str,
        top_k: int = 3,
        niche_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Find similar successful video patterns.

        Args:
            analysis_text: Current video analysis
            top_k: Number of similar patterns to return
            niche_filter: Optional niche to filter by

        Returns:
            List of similar patterns with metadata
        """
        if self.embeddings is None or len(self.patterns) == 0:
            return []

        # Create embedding for current analysis
        query_embedding = self._create_embedding(analysis_text)

        # Calculate similarities
        similarities = np.dot(self.embeddings, query_embedding)

        # Filter by niche if specified
        valid_indices = range(len(self.patterns))
        if niche_filter:
            valid_indices = [
                i for i in valid_indices
                if self.metadata[i].get('niche', '').lower() == niche_filter.lower()
            ]

        # Get top-k most similar
        if not valid_indices:
            valid_indices = range(len(self.patterns))

        valid_similarities = [(i, similarities[i]) for i in valid_indices]
        valid_similarities.sort(key=lambda x: x[1], reverse=True)
        top_indices = [i for i, _ in valid_similarities[:top_k]]

        # Return patterns with metadata
        results = []
        for idx in top_indices:
            results.append({
                'pattern': self.patterns[idx],
                'metadata': self.metadata[idx],
                'similarity': float(similarities[idx])
            })

        return results

    def get_pattern_insights(
        self,
        current_analysis: str,
        niche: Optional[str] = None,
        include_formats_only: bool = True
    ) -> str:
        """
        Get formatted insights from similar videos (BETA FEATURE).

        Args:
            current_analysis: Current video analysis
            niche: Optional niche filter
            include_formats_only: If True, focus on format patterns only (recommended)
        """
        similar = self.find_similar_patterns(current_analysis, top_k=3, niche_filter=niche)

        if not similar:
            return ""  # Don't show anything if no patterns

        insights = ["=" * 80]
        insights.append("🧪 BETA: FORMAT PATTERNS FROM SIMILAR VIDEOS")
        insights.append("=" * 80)
        insights.append("\nNote: These are OBSERVED PATTERNS, not proven success formulas.")
        insights.append("Use as creative inspiration, not rigid rules.\n")

        for i, result in enumerate(similar, 1):
            meta = result['metadata']
            pattern = result['pattern']

            insights.append(f"\nPattern {i} (from video with {meta['metrics'].get('views', 'N/A')} views):")
            insights.append(f"Platform: {meta.get('platform', 'N/A')} | Similarity: {result['similarity']:.1%}")
            insights.append("\nObserved Format:")
            insights.append(f"{pattern['summary']}")
            insights.append("\n" + "-" * 80)

        insights.append("\n💡 TIP: Look for recurring formats across multiple patterns.")
        insights.append("Common patterns ≠ guaranteed success, but they're worth testing!\n")

        return "\n".join(insights)


# Example usage in app.py:
# success_store = SuccessPatternStore()
#
# # When user marks a video as successful:
# success_store.add_successful_video(
#     analysis_text=final_analysis,
#     video_url=video_url,
#     metrics={'views': 150000, 'engagement_rate': 0.12, 'watch_time': '45s'},
#     niche=niche,
#     platform='tiktok'
# )
#
# # During new analysis:
# pattern_insights = success_store.get_pattern_insights(current_analysis, niche=niche)
# final_analysis += f"\n\n{pattern_insights}"
