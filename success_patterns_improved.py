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
                'key_insights': pattern_data.get('key_insights', []),
                'context': pattern_data.get('context', {}),
                'when_to_apply': pattern_data.get('when_to_apply', []),
                'cautions': pattern_data.get('cautions', []),
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
                        "content": """You are synthesizing video strategy knowledge from a curator's specific observation about a video into a GENERAL, UNIVERSALLY APPLICABLE principle.

YOUR JOB:
1. Read the curator's note about what they observed in THIS specific video
2. Read the full analysis context to understand the video
3. SYNTHESIZE both into a generalized knowledge pattern that applies to ALL future videos, not just this one
4. Write it as strategic knowledge — like advice from a growth strategist, not a description of one video
5. The output should read like a chapter from a content strategy playbook

CRITICAL: Do NOT just repeat what the curator said. GENERALIZE it.
- Curator says "this designer's strong opinion caused pushback" → You write about how strong opinions in ANY niche drive engagement through debate
- Curator says "the controversy drove comments" → You write about the mechanics of polarizing content and audience alignment"""
                    },
                    {
                        "role": "user",
                        "content": f"""Synthesize a universal content strategy principle from this curator observation + video analysis.

VIDEO CONTEXT:
- Niche: {niche}
- Platform: {platform}
- Audience: {audience or 'Not specified'}
- Video Type: {video_type or 'Not specified'}
- Views: {metrics.get('views', 'N/A')}
- Engagement Rate: {metrics.get('engagement_rate', 'N/A')}

CURATOR'S OBSERVATION (their specific insight about this video):
{curator_notes}

FULL ANALYSIS CONTEXT:
{analysis_text[:3000]}

Synthesize this into GENERAL knowledge. Write it as if you're adding a principle to a content strategy guidebook that will be referenced when analyzing ANY future video.

Return JSON:
{{
    "pattern_summary": "2-4 sentences. A generalized strategic principle written in universal terms — NOT about this specific video. Should read like: 'Videos that [do X] create [Y effect] because [Z psychology]. This leads to [measurable outcome].' Write it so it applies to any creator in any niche.",
    "key_insights": [
        "Each insight should be a standalone general principle, not tied to this specific video",
        "Write as universal truths about content strategy backed by the curator's observation",
        "Example: 'Polarizing opinions drive engagement through two mechanisms: agreement (follows, saves, shares) and disagreement (comments, debate). Both increase algorithmic reach.'",
        "3-5 insights"
    ],
    "context": {{
        "niche": "{niche}",
        "platform": "{platform}",
        "observed_in": "Brief description of the specific video this was observed in",
        "why_it_matters": "The universal principle — why any creator should know this"
    }},
    "when_to_apply": [
        "Specific scenarios where this principle should be considered in future analyses",
        "Example: 'When a video takes a strong stance or controversial position'",
        "Example: 'When comment sentiment is highly polarized'"
    ],
    "cautions": [
        "Only include if relevant — genuine risks of applying this principle poorly",
        "Empty array if none"
    ]
}}

IMPORTANT: The pattern_summary and key_insights must be GENERALIZED — they should never mention this specific creator, video, or niche. Write universal principles."""
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
        Auto-detects if pattern is similar to existing ones and aggregates learnings.

        Args:
            pattern_data: The pattern data from preview (possibly edited by user)
            video_url: Original video URL
            metrics: Performance metrics
            curator_notes: Curator's notes
        """
        # Create embedding from pattern summary (without niche for better cross-niche matching)
        embedding_text = pattern_data.get('pattern_summary', '')
        embedding = self._create_embedding(embedding_text)

        # Check if this is a similar pattern we've seen before
        similar_patterns = self._find_similar_existing_patterns(pattern_data, top_k=1)

        if similar_patterns and similar_patterns[0]['similarity'] > 0.85:
            # Very similar pattern exists - this might be the same format in a different niche
            print(f"[PATTERN INSIGHT] Similar pattern detected (similarity: {similar_patterns[0]['similarity']:.2f})")
            print(f"  Existing: {similar_patterns[0]['pattern']} in {similar_patterns[0]['context'].get('niche')}")
            print(f"  New: {pattern_data.get('pattern_summary')} in {pattern_data.get('context', {}).get('niche')}")

            # Check if different niches → might be universal pattern
            existing_niche = similar_patterns[0]['context'].get('niche', '').lower()
            new_niche = pattern_data.get('context', {}).get('niche', '').lower()

            if existing_niche != new_niche and existing_niche and new_niche:
                print(f"  → LEARNING: This format works across multiple niches!")

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

        # Analyze pattern universality
        universality = self._analyze_pattern_universality(pattern_data)

        print(f"[PATTERN STORED] {pattern_data.get('pattern_summary', 'Pattern')}")
        print(f"  Context: {pattern_data.get('context', {}).get('niche', 'N/A')}")
        print(f"  Universality: {universality['type']} ({universality['confidence']})")
        print(f"  Seen in niches: {', '.join(universality['niches'])}")

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
        1. Pattern universality (universal patterns suggested regardless of niche)
        2. Context match (niche, audience, video type)
        3. Applicability rules
        4. Constraints

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

            # Analyze pattern universality
            universality = self._analyze_pattern_universality(pattern_data)

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

            # Adjust scoring based on universality
            if universality['type'] == 'universal':
                # Universal patterns get a boost even with low context match
                context_weight = 0.3  # Less weight on exact niche match
                universal_boost = 0.3  # Boost for being universal
            elif universality['type'] == 'cross-niche':
                context_weight = 0.5
                universal_boost = 0.15
            else:
                # Niche-specific patterns need strong context match
                context_weight = 0.7
                universal_boost = 0.0

            # Combined score with universality consideration
            relevance_score = (context_score * context_weight) + \
                            (applicability_score * (1 - context_weight)) + \
                            universal_boost

            if relevance_score > 0.4:  # Threshold for relevance
                applicable.append({
                    'pattern': pattern_data,
                    'relevance_score': relevance_score,
                    'context_match': context_score,
                    'applicability_match': applicability_score,
                    'universality': universality,
                    'metrics': pattern_entry['metrics'],
                    'why_suggested': self._explain_suggestion_with_universality(
                        pattern_data,
                        current_video_context,
                        universality
                    )
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

    def _explain_suggestion_with_universality(
        self,
        pattern_data: Dict,
        current_context: Dict,
        universality: Dict
    ) -> str:
        """Generate explanation including universality insights"""
        reasons = []

        # Add universality context
        if universality['type'] == 'universal':
            reasons.append(f"Universal format (works across {len(universality['niches'])} niches)")
        elif universality['type'] == 'cross-niche':
            reasons.append(f"Cross-niche format (seen in: {', '.join(universality['niches'][:3])})")

        pattern_context = pattern_data.get('context', {})

        if pattern_context.get('niche') == current_context.get('niche'):
            reasons.append(f"Same niche ({current_context.get('niche')})")

        if pattern_context.get('platform') == current_context.get('platform'):
            reasons.append(f"Same platform ({current_context.get('platform')})")

        applicability = pattern_data.get('applicability', {})
        video_type = current_context.get('video_type', '').lower()

        for scenario in applicability.get('works_best_for', []):
            if scenario.lower() in video_type:
                reasons.append(f"Works best for {scenario}")
                break

        return " | ".join(reasons) if reasons else "Pattern similarity"

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

    def _analyze_pattern_universality(self, new_pattern: Dict) -> Dict:
        """
        Analyze if this pattern is universal or niche-specific.
        Checks how many different niches have similar patterns.

        Returns:
            Dict with 'type' (universal/cross-niche/niche-specific),
            'confidence' (high/medium/low),
            'niches' (list of niches this pattern appears in)
        """
        if not self.patterns:
            return {
                'type': 'niche-specific',
                'confidence': 'low (first example)',
                'niches': [new_pattern.get('context', {}).get('niche', 'unknown')]
            }

        # Find all similar patterns (high similarity on format, not context)
        pattern_summary = new_pattern.get('pattern_summary', '')
        embedding = self._create_embedding(pattern_summary)

        # Calculate similarities to all existing patterns
        if self.embeddings is not None and len(self.embeddings) > 0:
            similarities = np.dot(self.embeddings, embedding)

            # Find patterns with >0.80 similarity (same format family)
            similar_indices = np.where(similarities > 0.80)[0]

            # Collect niches where this pattern appears
            niches_seen = set()
            niches_seen.add(new_pattern.get('context', {}).get('niche', 'unknown').lower())

            for idx in similar_indices:
                niche = self.patterns[idx]['pattern_data'].get('context', {}).get('niche', '').lower()
                if niche:
                    niches_seen.add(niche)

            # Determine universality
            unique_niches = [n for n in niches_seen if n and n != 'unknown' and n != 'general']
            niche_count = len(unique_niches)

            if niche_count >= 4:
                return {
                    'type': 'universal',
                    'confidence': 'high',
                    'niches': sorted(unique_niches),
                    'example_count': len(similar_indices) + 1
                }
            elif niche_count >= 2:
                return {
                    'type': 'cross-niche',
                    'confidence': 'medium',
                    'niches': sorted(unique_niches),
                    'example_count': len(similar_indices) + 1
                }
            else:
                return {
                    'type': 'niche-specific',
                    'confidence': 'low' if len(similar_indices) < 2 else 'medium',
                    'niches': sorted(unique_niches) if unique_niches else ['unknown'],
                    'example_count': len(similar_indices) + 1
                }

        return {
            'type': 'niche-specific',
            'confidence': 'low',
            'niches': [new_pattern.get('context', {}).get('niche', 'unknown')],
            'example_count': 1
        }
