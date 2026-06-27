"""
Improved Success Pattern Learning System with Context Awareness
Stores patterns WITH their context, constraints, and applicability
"""
import hashlib
import pickle
import os
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Dict, List
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class ContextAwarePatternStore:
    """
    Stores video patterns with full context about WHEN and WHY they work.

    Key improvements:
    1. Patterns include context variables (niche, audience, video type, etc.)
    2. Constraints captured (when NOT to use the pattern)
    3. Applicability rules (what scenarios this works for)
    4. Preview before storing
    5. Context-aware retrieval (match context similarity, not just pattern)
    """

    def __init__(self, storage_dir: str = "./success_patterns"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.patterns_file = self.storage_dir / "patterns_v2.pkl"
        self.embeddings_file = self.storage_dir / "embeddings_v2.npy"

        # Load existing data
        self.patterns = self._load_patterns()
        self.embeddings = self._load_embeddings()

    def _load_patterns(self) -> List[Dict]:
        if self.patterns_file.exists():
            with open(self.patterns_file, 'rb') as f:
                return pickle.load(f)
        return []

    def _load_embeddings(self) -> Optional[np.ndarray]:
        if self.embeddings_file.exists():
            return np.load(self.embeddings_file)
        return None

    def _save_all(self):
        """Save all data to disk"""
        with open(self.patterns_file, 'wb') as f:
            pickle.dump(self.patterns, f)

        if self.embeddings is not None:
            np.save(self.embeddings_file, self.embeddings)

    def preview_pattern(
        self,
        analysis_text: str,
        video_url: str,
        metrics: Dict,
        curator_notes: str,
        niche: str = "general",
        platform: str = "tiktok",
        audience: str = "",
        video_type: str = ""
    ) -> Dict:
        """
        Generate a preview of what will be stored WITHOUT actually storing it.
        User can review and edit before confirming.

        Returns:
            Dict with extracted pattern, context, constraints, and applicability
        """
        # Extract pattern with full context
        pattern_data = self._extract_contextual_pattern(
            analysis_text=analysis_text,
            metrics=metrics,
            curator_notes=curator_notes,
            niche=niche,
            platform=platform,
            audience=audience,
            video_type=video_type
        )

        return {
            'preview': pattern_data,
            'will_store': {
                'pattern_summary': pattern_data.get('pattern_summary', ''),
                'context': pattern_data.get('context', {}),
                'constraints': pattern_data.get('constraints', []),
                'applicability': pattern_data.get('applicability', {}),
                'metrics': metrics,
                'curator_notes': curator_notes
            },
            'similarity_examples': self._find_similar_existing_patterns(pattern_data, top_k=3)
        }

    def _extract_contextual_pattern(
        self,
        analysis_text: str,
        metrics: Dict,
        curator_notes: str,
        niche: str,
        platform: str,
        audience: str,
        video_type: str
    ) -> Dict:
        """
        Extract pattern WITH full context about when/why it works.
        """
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a pattern extraction expert. Extract video patterns WITH their context.

CRITICAL: Patterns are NOT universal rules. They work in SPECIFIC contexts.

Your job:
1. Identify the PATTERN (format, structure, technique)
2. Identify the CONTEXT (what variables made this work HERE)
3. Identify CONSTRAINTS (when NOT to use this)
4. Identify APPLICABILITY (what scenarios this could work for)

Be specific and factual. Avoid causal assumptions."""
                    },
                    {
                        "role": "user",
                        "content": f"""Extract a contextual pattern from this video:

VIDEO CONTEXT:
- Niche: {niche}
- Platform: {platform}
- Audience: {audience or 'Not specified'}
- Video Type: {video_type or 'Not specified'}
- Views: {metrics.get('views', 'N/A')}
- Engagement Rate: {metrics.get('engagement_rate', 'N/A')}

CURATOR NOTES (What to learn from this):
{curator_notes}

FULL ANALYSIS:
{analysis_text[:2000]}

Extract and return in this EXACT JSON format:
{{
    "pattern_summary": "Brief description of the core pattern (e.g., 'This vs That format with visual contrast')",
    "pattern_details": {{
        "format_type": "e.g., comparison, tutorial, storytelling, etc.",
        "hook_structure": "What the hook does in first 3s",
        "content_flow": "How content is organized",
        "visual_elements": "Key visual patterns",
        "text_strategy": "How text overlays are used",
        "audio_approach": "Audio/music strategy"
    }},
    "context": {{
        "niche": "{niche}",
        "target_audience": "Who this resonated with",
        "platform": "{platform}",
        "video_type": "Type of content",
        "why_it_worked_here": "Specific reasons this pattern worked in THIS context"
    }},
    "constraints": [
        "Requires X to work",
        "Won't work if Y",
        "Needs Z audience type"
    ],
    "applicability": {{
        "works_best_for": ["scenario 1", "scenario 2"],
        "could_work_for": ["scenario 3", "scenario 4"],
        "avoid_for": ["scenario 5", "scenario 6"]
    }},
    "variables": {{
        "required": ["What's absolutely needed"],
        "optional": ["What's nice to have"],
        "adaptable": ["What can be changed while keeping the pattern"]
    }}
}}"""
                    }
                ],
                max_tokens=1500,
                temperature=0.3,
                response_format={"type": "json_object"}
            )

            import json
            pattern_data = json.loads(response.choices[0].message.content)
            return pattern_data

        except Exception as e:
            print(f"[ERROR] Contextual pattern extraction failed: {e}")
            # Fallback to simple extraction
            return {
                'pattern_summary': curator_notes[:200],
                'context': {'niche': niche, 'platform': platform},
                'constraints': [],
                'applicability': {},
                'variables': {}
            }

    def store_pattern(
        self,
        pattern_data: Dict,
        video_url: str,
        metrics: Dict,
        curator_notes: str
    ):
        """
        Store a pattern after user confirmation.

        Args:
            pattern_data: The pattern data from preview (possibly edited by user)
            video_url: Original video URL
            metrics: Performance metrics
            curator_notes: Curator's notes
        """
        # Create embedding from pattern summary + context
        embedding_text = f"{pattern_data.get('pattern_summary', '')} {pattern_data.get('context', {}).get('niche', '')} {pattern_data.get('context', {}).get('target_audience', '')}"
        embedding = self._create_embedding(embedding_text)

        # Store pattern with full context
        self.patterns.append({
            'pattern_data': pattern_data,
            'video_url': video_url,
            'metrics': metrics,
            'curator_notes': curator_notes,
            'timestamp': datetime.now().isoformat()
        })

        # Store embedding
        if self.embeddings is None:
            self.embeddings = np.array([embedding])
        else:
            self.embeddings = np.vstack([self.embeddings, embedding])

        self._save_all()
        print(f"[PATTERN STORED] {pattern_data.get('pattern_summary', 'Pattern')}")
        print(f"  Context: {pattern_data.get('context', {}).get('niche', 'N/A')}")
        print(f"  Applicability: {len(pattern_data.get('applicability', {}).get('works_best_for', []))} scenarios")

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
            return np.zeros(1536)

    def _find_similar_existing_patterns(self, new_pattern: Dict, top_k: int = 3) -> List[Dict]:
        """Find similar patterns already in the store (for preview comparison)"""
        if not self.patterns or self.embeddings is None:
            return []

        # Create embedding for new pattern
        embedding_text = f"{new_pattern.get('pattern_summary', '')} {new_pattern.get('context', {}).get('niche', '')}"
        query_embedding = self._create_embedding(embedding_text)

        # Calculate similarities
        similarities = np.dot(self.embeddings, query_embedding)

        # Get top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        similar = []
        for idx in top_indices:
            if similarities[idx] > 0.7:  # Only show if reasonably similar
                similar.append({
                    'pattern': self.patterns[idx]['pattern_data'].get('pattern_summary', ''),
                    'similarity': float(similarities[idx]),
                    'context': self.patterns[idx]['pattern_data'].get('context', {})
                })

        return similar

    def find_applicable_patterns(
        self,
        current_video_context: Dict,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Find patterns that are APPLICABLE to the current video context.

        This is smarter than just similarity - it considers:
        1. Context match (niche, audience, video type)
        2. Applicability rules
        3. Constraints

        Args:
            current_video_context: Dict with 'niche', 'audience', 'video_type', etc.

        Returns:
            List of applicable patterns with context-aware scoring
        """
        if not self.patterns:
            return []

        applicable = []

        for idx, pattern_entry in enumerate(self.patterns):
            pattern_data = pattern_entry['pattern_data']

            # Calculate context match score
            context_score = self._calculate_context_match(
                pattern_data.get('context', {}),
                current_video_context
            )

            # Calculate applicability score
            applicability_score = self._calculate_applicability_score(
                pattern_data.get('applicability', {}),
                current_video_context
            )

            # Combined score
            relevance_score = (context_score * 0.6) + (applicability_score * 0.4)

            if relevance_score > 0.4:  # Threshold for relevance
                applicable.append({
                    'pattern': pattern_data,
                    'relevance_score': relevance_score,
                    'context_match': context_score,
                    'applicability_match': applicability_score,
                    'metrics': pattern_entry['metrics'],
                    'why_suggested': self._explain_suggestion(pattern_data, current_video_context)
                })

        # Sort by relevance
        applicable.sort(key=lambda x: x['relevance_score'], reverse=True)
        return applicable[:top_k]

    def _calculate_context_match(self, pattern_context: Dict, current_context: Dict) -> float:
        """Calculate how well pattern context matches current context"""
        score = 0.0
        total_factors = 0

        # Niche match (most important)
        if pattern_context.get('niche') == current_context.get('niche'):
            score += 0.4
        total_factors += 0.4

        # Audience match
        if pattern_context.get('target_audience', '').lower() in current_context.get('audience', '').lower():
            score += 0.3
        total_factors += 0.3

        # Platform match
        if pattern_context.get('platform') == current_context.get('platform'):
            score += 0.2
        total_factors += 0.2

        # Video type match
        if pattern_context.get('video_type', '').lower() in current_context.get('video_type', '').lower():
            score += 0.1
        total_factors += 0.1

        return score / total_factors if total_factors > 0 else 0.0

    def _calculate_applicability_score(self, applicability: Dict, current_context: Dict) -> float:
        """Calculate if pattern is applicable based on its rules"""
        video_type = current_context.get('video_type', '').lower()
        niche = current_context.get('niche', '').lower()

        # Check avoid_for list
        avoid_for = [x.lower() for x in applicability.get('avoid_for', [])]
        if any(avoid in video_type or avoid in niche for avoid in avoid_for):
            return 0.0  # Don't suggest if explicitly avoided

        # Check works_best_for
        works_best = [x.lower() for x in applicability.get('works_best_for', [])]
        if any(best in video_type or best in niche for best in works_best):
            return 1.0

        # Check could_work_for
        could_work = [x.lower() for x in applicability.get('could_work_for', [])]
        if any(could in video_type or could in niche for could in could_work):
            return 0.6

        return 0.3  # Default moderate score

    def _explain_suggestion(self, pattern_data: Dict, current_context: Dict) -> str:
        """Generate explanation for why this pattern is being suggested"""
        reasons = []

        pattern_context = pattern_data.get('context', {})

        if pattern_context.get('niche') == current_context.get('niche'):
            reasons.append(f"Same niche ({current_context.get('niche')})")

        if pattern_context.get('platform') == current_context.get('platform'):
            reasons.append(f"Same platform ({current_context.get('platform')})")

        applicability = pattern_data.get('applicability', {})
        video_type = current_context.get('video_type', '').lower()

        for scenario in applicability.get('works_best_for', []):
            if scenario.lower() in video_type:
                reasons.append(f"Pattern works best for {scenario}")
                break

        return " | ".join(reasons) if reasons else "Pattern similarity"
