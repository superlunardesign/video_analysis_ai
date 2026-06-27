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
        Analyze video comments to extract engagement insights.

        Args:
            comments: List of comment dicts with 'text', 'likes', 'timestamp' (optional)
            video_transcript: Optional transcript for context matching
            frame_analysis: Optional frame analysis for visual reference matching

        Returns:
            Dict with insights about what resonated with viewers
        """
        if not comments:
            return {"error": "No comments provided"}

        # Sort comments by engagement (likes)
        sorted_comments = sorted(
            comments,
            key=lambda c: int(c.get('likes', 0)),
            reverse=True
        )

        # Extract top comments
        top_comments = sorted_comments[:50]  # Focus on most-liked comments

        # Analyze patterns
        analysis = {
            'total_comments': len(comments),
            'top_comments_analyzed': len(top_comments),
            'timestamp_mentions': self._extract_timestamp_mentions(top_comments),
            'emotion_analysis': self._analyze_emotions(top_comments),
            'key_moments': self._identify_key_moments(top_comments, video_transcript),
            'engagement_themes': self._extract_themes(top_comments),
            'ai_insights': self._get_ai_insights(top_comments, video_transcript, frame_analysis)
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
        frame_analysis: Optional[str] = None
    ) -> str:
        """Use GPT-4 to extract deeper insights from comments"""
        try:
            # Prepare top comments for analysis
            top_comments_text = "\n".join([
                f"[{c.get('likes', 0)} likes] {c.get('text', '')}"
                for c in comments[:30]
            ])

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
                        "content": """You are an expert at analyzing video comments to understand audience engagement.
Identify:
1. What specific moments, quotes, or elements viewers are reacting to
2. What kept them watching (hooks, curiosity gaps, payoffs)
3. What emotions drove engagement (humor, surprise, inspiration, etc.)
4. What patterns emerge in viewer feedback
5. Actionable insights for content creators"""
                    },
                    {
                        "role": "user",
                        "content": f"""Analyze these top video comments to understand what resonated with viewers:

{top_comments_text}
{context}

Provide a concise analysis (300 words max) focusing on:
- Key moments that drove engagement
- What kept viewers watching
- Emotional drivers
- Actionable patterns for creators"""
                    }
                ],
                max_tokens=600,
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
