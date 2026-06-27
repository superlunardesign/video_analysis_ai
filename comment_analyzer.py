"""
Video Comment Analysis System
Extracts insights from video comments to understand audience engagement
"""
import os
import re
from typing import Dict, List, Optional
from openai import OpenAI
from collections import Counter

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class CommentAnalyzer:
    def __init__(self):
        self.client = client

    def analyze_comments(
        self,
        comments: List[Dict[str, str]],
        video_transcript: Optional[str] = None,
        frame_analysis: Optional[str] = None
    ) -> Dict:
        """
        Analyze video comments to extract engagement insights with like-weighted consensus.

        Args:
            comments: List of comment dicts with 'text', 'likes', 'timestamp' (optional)
            video_transcript: Optional transcript for context matching
            frame_analysis: Optional frame analysis for visual reference matching

        Returns:
            Dict with insights about what resonated with viewers
        """
        if not comments:
            return {"error": "No comments provided"}

        # Categorize comments by type
        categorized = self._categorize_comments(comments)

        # Sort comments by engagement (likes) for analysis
        sorted_comments = sorted(
            comments,
            key=lambda c: int(c.get('likes', 0)),
            reverse=True
        )

        # Focus on high-value comments (100+ likes get priority)
        high_value = [c for c in sorted_comments if int(c.get('likes', 0)) >= 100]
        medium_value = [c for c in sorted_comments if 10 <= int(c.get('likes', 0)) < 100]

        # Analyze top comments (prioritize high-liked ones)
        top_comments = (high_value[:30] + medium_value[:20])[:50]

        # Detect consensus patterns (cluster similar comments, sum their likes)
        consensus_patterns = self._detect_consensus_patterns(sorted_comments)

        # Analyze patterns
        analysis = {
            'total_comments': len(comments),
            'categorization': categorized,
            'top_comments_analyzed': len(top_comments),
            'consensus_patterns': consensus_patterns,  # NEW: Like-weighted themes
            'timestamp_mentions': self._extract_timestamp_mentions(top_comments),
            'emotion_analysis': self._analyze_emotions(top_comments),
            'key_moments': self._identify_key_moments(top_comments, video_transcript),
            'engagement_themes': self._extract_themes(top_comments),
            'ai_insights': self._get_ai_insights(
                top_comments,
                video_transcript,
                frame_analysis,
                consensus_patterns  # Pass consensus to AI for better insights
            )
        }

        return analysis

    def _extract_timestamp_mentions(self, comments: List[Dict]) -> List[Dict]:
        """Extract timestamp mentions like '0:23' or '1:45'"""
        timestamp_pattern = r'(\d{1,2}):(\d{2})'
        mentions = []

        for comment in comments:
            text = comment.get('text', '')
            matches = re.finditer(timestamp_pattern, text)

            for match in matches:
                minutes = int(match.group(1))
                seconds = int(match.group(2))
                total_seconds = minutes * 60 + seconds

                mentions.append({
                    'timestamp': match.group(0),
                    'seconds': total_seconds,
                    'comment': text,
                    'likes': comment.get('likes', 0)
                })

        # Sort by frequency and likes
        if mentions:
            timestamp_counter = Counter([m['timestamp'] for m in mentions])
            return [
                {
                    'timestamp': ts,
                    'mentions': count,
                    'example_comments': [m['comment'] for m in mentions if m['timestamp'] == ts][:3]
                }
                for ts, count in timestamp_counter.most_common(10)
            ]

        return []

    def _analyze_emotions(self, comments: List[Dict]) -> Dict:
        """Analyze emotional reactions in comments"""
        emotion_keywords = {
            'excitement': ['omg', 'wow', 'amazing', 'incredible', 'insane', '🔥', '😱', '🤯', '!!!'],
            'humor': ['lol', 'lmao', 'haha', 'funny', 'dead', '💀', '😂', '😭'],
            'surprise': ['wait', 'what', 'no way', 'seriously', '😮', '🤯', 'wtf'],
            'inspiration': ['love this', 'needed this', 'saving', 'thank you', '🙏', '❤️', '💯'],
            'curiosity': ['how', 'why', 'where', 'tutorial', 'explain', '?'],
            'relatability': ['same', 'me too', 'relatable', 'fr fr', 'real', 'facts']
        }

        emotion_scores = {emotion: 0 for emotion in emotion_keywords}

        for comment in comments:
            text = comment.get('text', '').lower()
            likes = int(comment.get('likes', 0))

            for emotion, keywords in emotion_keywords.items():
                if any(keyword in text for keyword in keywords):
                    # Weight by likes (popular comments indicate stronger sentiment)
                    emotion_scores[emotion] += 1 + (likes / 10)

        # Normalize and return top emotions
        total_score = sum(emotion_scores.values())
        if total_score > 0:
            emotion_scores = {
                emotion: round((score / total_score) * 100, 1)
                for emotion, score in emotion_scores.items()
            }

        return dict(sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True))

    def _identify_key_moments(
        self,
        comments: List[Dict],
        transcript: Optional[str] = None
    ) -> List[Dict]:
        """Identify specific moments or quotes people are reacting to"""
        # Look for quoted text in comments
        quote_pattern = r'["\']([^"\']{10,})["\']'
        key_moments = []

        for comment in comments:
            text = comment.get('text', '')
            quotes = re.findall(quote_pattern, text)

            for quote in quotes:
                # Check if quote appears in transcript
                in_transcript = False
                if transcript and quote.lower() in transcript.lower():
                    in_transcript = True

                key_moments.append({
                    'quote': quote,
                    'in_transcript': in_transcript,
                    'comment': text,
                    'likes': comment.get('likes', 0)
                })

        # Sort by likes
        key_moments.sort(key=lambda x: x['likes'], reverse=True)
        return key_moments[:10]

    def _categorize_comments(self, comments: List[Dict]) -> Dict:
        """
        Categorize comments by type to understand engagement patterns.
        Returns counts and percentages for each category.
        """
        categories = {
            'substantive': 0,  # Context-rich comments (timestamps, quotes, specific observations)
            'emoji_only': 0,   # Just emojis or very short reactions
            'generic': 0,      # Generic praise/reactions without context
            'spam': 0          # Spam, self-promotion, off-topic
        }

        emoji_pattern = r'^[\s\U0001F000-\U0001F9FF]+$'  # Just emojis
        spam_patterns = ['follow', 'check out', 'my channel', 'subscribe', 'dm me', 'link in bio']

        for comment in comments:
            text = comment.get('text', '').strip()

            if not text:
                continue

            text_lower = text.lower()

            # Check for spam
            if any(pattern in text_lower for pattern in spam_patterns):
                categories['spam'] += 1
            # Check for emoji-only
            elif re.match(emoji_pattern, text) or len(text) <= 5:
                categories['emoji_only'] += 1
            # Check for generic (short reactions without context)
            elif len(text) < 15 and not any(char.isdigit() or char in ['"', "'", '?'] for char in text):
                categories['generic'] += 1
            # Everything else is substantive
            else:
                categories['substantive'] += 1

        total = len(comments)
        percentages = {
            cat: round((count / total * 100), 1) if total > 0 else 0
            for cat, count in categories.items()
        }

        return {
            'counts': categories,
            'percentages': percentages,
            'total': total,
            'insight': self._categorization_insight(percentages)
        }

    def _categorization_insight(self, percentages: Dict) -> str:
        """Generate insight from comment categorization"""
        substantive = percentages.get('substantive', 0)
        emoji_only = percentages.get('emoji_only', 0)

        if substantive > 40:
            return "High substantive engagement - viewers are articulating specific moments and reactions"
        elif emoji_only > 30:
            return "High emoji-only engagement suggests strong emotional response but viewers aren't verbalizing WHY - may indicate visually-driven content"
        elif substantive < 20:
            return "Low substantive engagement - consider adding elements that prompt specific viewer reactions"
        else:
            return "Balanced engagement mix"

    def _detect_consensus_patterns(self, comments: List[Dict]) -> List[Dict]:
        """
        Detect consensus patterns by clustering similar comments and summing their likes.
        This shows what the AUDIENCE collectively agreed on, not just individual opinions.
        """
        # Group comments by similarity (simple keyword clustering)
        patterns = []

        # Common engagement themes to look for
        theme_keywords = {
            'background_element': ['background', 'behind', 'in the back', 'spotted', 'noticed'],
            'specific_moment': ['part when', 'moment', 'second', 'at the', ':', 'timestamp'],
            'visual_element': ['outfit', 'hair', 'makeup', 'shirt', 'wearing', 'ring', 'nails', 'accessories'],
            'audio': ['song', 'music', 'sound', 'audio', 'voice', 'singing'],
            'transition': ['transition', 'edit', 'effect', 'cut', 'switch'],
            'pet': ['cat', 'dog', 'pet', 'puppy', 'kitten'],
            'tutorial_request': ['how', 'tutorial', 'teach', 'explain', 'show me'],
            'end_reveal': ['end', 'ending', 'reveal', 'final', 'wait until', 'stayed for'],
        }

        # Track mentions and total likes for each theme
        theme_data = {theme: {'count': 0, 'total_likes': 0, 'examples': []} for theme in theme_keywords}

        for comment in comments:
            text = comment.get('text', '').lower()
            likes = int(comment.get('likes', 0))

            for theme, keywords in theme_keywords.items():
                if any(keyword in text for keyword in keywords):
                    theme_data[theme]['count'] += 1
                    theme_data[theme]['total_likes'] += likes
                    if len(theme_data[theme]['examples']) < 3:
                        theme_data[theme]['examples'].append({
                            'text': comment.get('text', ''),
                            'likes': likes
                        })

        # Convert to sorted list of patterns (by total likes = consensus strength)
        for theme, data in theme_data.items():
            if data['count'] > 0 and data['total_likes'] > 0:
                patterns.append({
                    'theme': theme.replace('_', ' ').title(),
                    'mentions': data['count'],
                    'total_likes': data['total_likes'],
                    'consensus_strength': data['total_likes'] / max(data['count'], 1),  # Avg likes per mention
                    'examples': data['examples']
                })

        # Sort by total likes (strongest consensus first)
        patterns.sort(key=lambda x: x['total_likes'], reverse=True)

        return patterns[:10]  # Top 10 consensus patterns

    def _extract_themes(self, comments: List[Dict]) -> List[str]:
        """Extract common themes using simple NLP"""
        # Combine top comments
        combined_text = ' '.join([c.get('text', '') for c in comments[:30]])

        # Simple word frequency (excluding common words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'is', 'it', 'this', 'that', 'was', 'are', 'be', 'have', 'has',
                     'with', 'from', 'by', 'i', 'you', 'me', 'my', 'your', 'he', 'she', 'they'}

        words = re.findall(r'\b[a-z]{4,}\b', combined_text.lower())
        filtered_words = [w for w in words if w not in stop_words]

        word_freq = Counter(filtered_words)
        return [word for word, count in word_freq.most_common(15)]

    def _get_ai_insights(
        self,
        comments: List[Dict],
        transcript: Optional[str] = None,
        frame_analysis: Optional[str] = None,
        consensus_patterns: Optional[List[Dict]] = None
    ) -> str:
        """Use GPT-4 to extract deeper insights from comments with consensus awareness"""
        try:
            # Prepare top comments for analysis
            top_comments_text = "\n".join([
                f"[{c.get('likes', 0)} likes] {c.get('text', '')}"
                for c in comments[:30]
            ])

            # Add consensus patterns (like-weighted themes)
            consensus_text = ""
            if consensus_patterns:
                consensus_text = "\n\nCONSENSUS PATTERNS (what viewers collectively agreed on):\n"
                for pattern in consensus_patterns[:5]:
                    consensus_text += f"- {pattern['theme']}: {pattern['mentions']} mentions, {pattern['total_likes']} total likes\n"
                    if pattern['examples']:
                        consensus_text += f"  Example: \"{pattern['examples'][0]['text']}\" ({pattern['examples'][0]['likes']} likes)\n"

            context = ""
            if transcript:
                context += f"\n\nVideo Transcript:\n{transcript[:500]}..."
            if frame_analysis:
                context += f"\n\nFrame Analysis:\n{frame_analysis[:500]}..."

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert at analyzing video comments to understand ACTUAL audience engagement.

CRITICAL: Comments reveal what viewers ACTUALLY noticed, which may differ from video intent.
- If comments focus on background elements more than main content → that's the real hook
- If comments mention specific timestamps → those are retention drivers
- High consensus (many likes on similar comments) = strong pattern

Your job:
1. Identify what ACTUALLY drove engagement (intended or unintentional)
2. Spot gaps between intent and attention
3. Find hidden hooks (accidental elements that worked)
4. Provide actionable insights based on viewer behavior, not assumptions"""
                    },
                    {
                        "role": "user",
                        "content": f"""Analyze these video comments to understand what ACTUALLY engaged viewers:

{top_comments_text}
{consensus_text}
{context}

Provide analysis (300 words max) focusing on:
1. ACTUAL engagement drivers (what viewers noticed most)
2. Intent vs Reality (did main content drive engagement, or something else?)
3. Hidden hooks (unintentional elements that worked)
4. Moments that kept viewers watching
5. Actionable insights for creators"""
                    }
                ],
                max_tokens=700,
                temperature=0.4
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"[ERROR] AI insights generation failed: {e}")
            return "Unable to generate AI insights from comments."

    def format_analysis(self, analysis: Dict) -> str:
        """Format analysis into readable text"""
        output = ["=" * 80]
        output.append("COMMENT ANALYSIS - AUDIENCE INSIGHTS")
        output.append("=" * 80)

        output.append(f"\nTotal Comments Analyzed: {analysis.get('total_comments', 0)}")
        output.append(f"Top Comments Reviewed: {analysis.get('top_comments_analyzed', 0)}")

        # Timestamp mentions
        if analysis.get('timestamp_mentions'):
            output.append("\n📍 SPECIFIC MOMENTS VIEWERS HIGHLIGHTED:")
            for mention in analysis['timestamp_mentions'][:5]:
                output.append(f"  • {mention['timestamp']} - {mention['mentions']} mentions")
                output.append(f"    Example: \"{mention['example_comments'][0][:100]}...\"")

        # Emotion analysis
        if analysis.get('emotion_analysis'):
            output.append("\n😊 EMOTIONAL REACTIONS:")
            emotions = analysis['emotion_analysis']
            for emotion, score in list(emotions.items())[:5]:
                if score > 5:  # Only show significant emotions
                    output.append(f"  • {emotion.title()}: {score}%")

        # Key moments
        if analysis.get('key_moments'):
            output.append("\n💬 QUOTED MOMENTS (What Stood Out):")
            for moment in analysis['key_moments'][:5]:
                output.append(f"  • \"{moment['quote']}\" ({moment['likes']} likes)")

        # Themes
        if analysis.get('engagement_themes'):
            output.append("\n🔑 RECURRING THEMES:")
            themes = ', '.join(analysis['engagement_themes'][:10])
            output.append(f"  {themes}")

        # AI Insights
        if analysis.get('ai_insights'):
            output.append("\n🤖 AI-POWERED INSIGHTS:")
            output.append(f"\n{analysis['ai_insights']}")

        output.append("\n" + "=" * 80)

        return "\n".join(output)


# Example usage:
"""
# In your app.py or processing flow:

from comment_analyzer import CommentAnalyzer

# Fetch comments (you'd implement this based on platform)
comments = [
    {'text': 'The part at 0:23 was insane! 🔥', 'likes': 245},
    {'text': 'I can\'t believe "nobody saw this coming" - so relatable', 'likes': 189},
    {'text': 'This is exactly what I needed to see today 🙏', 'likes': 156},
    # ... more comments
]

analyzer = CommentAnalyzer()

# Analyze comments
analysis = analyzer.analyze_comments(
    comments=comments,
    video_transcript=transcript_text,
    frame_analysis=frame_analysis_text
)

# Get formatted output
insights = analyzer.format_analysis(analysis)

# Add to your main analysis
final_analysis += f"\n\n{insights}"
"""
