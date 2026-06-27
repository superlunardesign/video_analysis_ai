"""
Comment Fetcher for Different Platforms
Fetches comments from TikTok, Instagram, YouTube, etc.
"""
import os
import re
import requests
from typing import List, Dict, Optional


class CommentFetcher:
    """
    Fetches comments from various social media platforms.

    Note: Comment fetching requires different approaches per platform:
    - YouTube: Official API (requires API key)
    - TikTok: Unofficial APIs or scraping (TikTok doesn't have official comment API)
    - Instagram: Graph API (requires business account)
    """

    def __init__(self):
        self.youtube_api_key = os.getenv("YOUTUBE_API_KEY")
        self.rapidapi_key = os.getenv("RAPIDAPI_KEY")  # For TikTok/IG unofficial APIs

    def fetch_comments(
        self,
        video_url: str,
        max_comments: int = 100,
        platform: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Fetch comments from a video URL.

        Args:
            video_url: URL of the video
            max_comments: Maximum number of comments to fetch
            platform: Optional platform override (auto-detected from URL)

        Returns:
            List of comment dicts with 'text', 'likes', 'author', 'timestamp'
        """
        if not platform:
            platform = self._detect_platform(video_url)

        print(f"[COMMENTS] Fetching from {platform}...")

        if platform == 'youtube':
            return self._fetch_youtube_comments(video_url, max_comments)
        elif platform == 'tiktok':
            return self._fetch_tiktok_comments(video_url, max_comments)
        elif platform == 'instagram':
            return self._fetch_instagram_comments(video_url, max_comments)
        else:
            print(f"[WARNING] Platform {platform} not supported for comment fetching")
            return []

    def _detect_platform(self, url: str) -> str:
        """Detect platform from URL"""
        url_lower = url.lower()

        if 'youtube.com' in url_lower or 'youtu.be' in url_lower:
            return 'youtube'
        elif 'tiktok.com' in url_lower:
            return 'tiktok'
        elif 'instagram.com' in url_lower:
            return 'instagram'
        elif 'twitter.com' in url_lower or 'x.com' in url_lower:
            return 'twitter'
        else:
            return 'unknown'

    def _extract_youtube_id(self, url: str) -> Optional[str]:
        """Extract YouTube video ID from URL"""
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]{11})',
            r'youtube\.com\/embed\/([a-zA-Z0-9_-]{11})',
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        return None

    def _fetch_youtube_comments(self, video_url: str, max_comments: int) -> List[Dict]:
        """
        Fetch YouTube comments using official API.

        Requires: YOUTUBE_API_KEY environment variable
        Get your key: https://console.cloud.google.com/apis/credentials
        """
        if not self.youtube_api_key:
            print("[WARNING] YOUTUBE_API_KEY not set. Cannot fetch comments.")
            return []

        video_id = self._extract_youtube_id(video_url)
        if not video_id:
            print("[ERROR] Could not extract YouTube video ID")
            return []

        try:
            comments = []
            next_page_token = None

            while len(comments) < max_comments:
                # YouTube API endpoint
                url = "https://www.googleapis.com/youtube/v3/commentThreads"
                params = {
                    'part': 'snippet',
                    'videoId': video_id,
                    'key': self.youtube_api_key,
                    'maxResults': min(100, max_comments - len(comments)),
                    'order': 'relevance'  # Get most relevant (most liked) first
                }

                if next_page_token:
                    params['pageToken'] = next_page_token

                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()

                data = response.json()

                # Extract comments
                for item in data.get('items', []):
                    snippet = item['snippet']['topLevelComment']['snippet']
                    comments.append({
                        'text': snippet['textDisplay'],
                        'likes': snippet.get('likeCount', 0),
                        'author': snippet.get('authorDisplayName', 'Unknown'),
                        'timestamp': snippet.get('publishedAt', '')
                    })

                # Check if there are more pages
                next_page_token = data.get('nextPageToken')
                if not next_page_token or len(comments) >= max_comments:
                    break

            print(f"[SUCCESS] Fetched {len(comments)} YouTube comments")
            return comments[:max_comments]

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                print("[ERROR] YouTube API quota exceeded or invalid API key")
            else:
                print(f"[ERROR] YouTube API error: {e}")
            return []
        except Exception as e:
            print(f"[ERROR] Failed to fetch YouTube comments: {e}")
            return []

    def _fetch_tiktok_comments(self, video_url: str, max_comments: int) -> List[Dict]:
        """
        Fetch TikTok comments using unofficial API.

        Option 1: Use RapidAPI's TikTok API (requires RAPIDAPI_KEY)
        Option 2: Use alternative scraping methods

        Note: TikTok doesn't provide official comment API
        """
        if not self.rapidapi_key:
            print("[WARNING] RAPIDAPI_KEY not set. Cannot fetch TikTok comments.")
            print("  Get API key at: https://rapidapi.com/yi-tang-tang-default/api/tiktok-scraper7")
            return []

        try:
            # Extract video ID from TikTok URL
            video_id_match = re.search(r'/video/(\d+)', video_url)
            if not video_id_match:
                print("[ERROR] Could not extract TikTok video ID")
                return []

            video_id = video_id_match.group(1)

            # RapidAPI TikTok endpoint (example - may vary by API)
            url = "https://tiktok-scraper7.p.rapidapi.com/video/comments"

            headers = {
                "X-RapidAPI-Key": self.rapidapi_key,
                "X-RapidAPI-Host": "tiktok-scraper7.p.rapidapi.com"
            }

            params = {
                "video_id": video_id,
                "count": min(max_comments, 100)
            }

            response = requests.get(url, headers=headers, params=params, timeout=15)
            response.raise_for_status()

            data = response.json()

            # Parse response (structure varies by API provider)
            comments = []
            for comment in data.get('data', {}).get('comments', []):
                comments.append({
                    'text': comment.get('text', ''),
                    'likes': comment.get('digg_count', 0),
                    'author': comment.get('user', {}).get('nickname', 'Unknown'),
                    'timestamp': comment.get('create_time', '')
                })

            print(f"[SUCCESS] Fetched {len(comments)} TikTok comments")
            return comments[:max_comments]

        except Exception as e:
            print(f"[ERROR] Failed to fetch TikTok comments: {e}")
            print("  Note: TikTok comment fetching requires RapidAPI subscription")
            return []

    def _fetch_instagram_comments(self, video_url: str, max_comments: int) -> List[Dict]:
        """
        Fetch Instagram comments.

        Instagram requires Graph API with proper authentication.
        This is a placeholder - implementation depends on your IG setup.
        """
        print("[WARNING] Instagram comment fetching not fully implemented")
        print("  Requires: Instagram Graph API with Business Account")
        print("  See: https://developers.facebook.com/docs/instagram-api/reference/ig-media/comments")

        # Placeholder for Instagram implementation
        return []


# Example usage:
"""
# In your app.py:

from comment_fetcher import CommentFetcher
from comment_analyzer import CommentAnalyzer

# Fetch comments
fetcher = CommentFetcher()
comments = fetcher.fetch_comments(video_url, max_comments=100)

if comments:
    # Analyze comments
    analyzer = CommentAnalyzer()
    analysis = analyzer.analyze_comments(
        comments=comments,
        video_transcript=transcript_text,
        frame_analysis=frame_analysis_text
    )

    # Format and add to report
    comment_insights = analyzer.format_analysis(analysis)
    final_analysis += f"\n\n{comment_insights}"
else:
    print("[INFO] No comments available for analysis")
"""
