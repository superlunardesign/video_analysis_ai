# How YouTube Data API Extracts Video Information

## Overview

The YouTube Data API is Google's **official REST API** for accessing YouTube data. It doesn't "scrape" - it provides structured access to YouTube's database.

---

## How It Works

### 1. Authentication

You get an **API key** from Google Cloud Console:
```
https://console.cloud.google.com/apis/credentials
```

This key authenticates your requests to YouTube's servers.

### 2. API Endpoints

YouTube provides different endpoints for different data:

#### **Video Details** (`/videos`)
```http
GET https://www.googleapis.com/youtube/v3/videos
  ?part=snippet,statistics,contentDetails
  &id=VIDEO_ID
  &key=YOUR_API_KEY
```

**Returns**:
```json
{
  "items": [
    {
      "snippet": {
        "title": "Video title",
        "description": "Video description",
        "publishedAt": "2024-01-15T10:00:00Z",
        "channelTitle": "Channel name"
      },
      "statistics": {
        "viewCount": "150000",
        "likeCount": "12000",
        "commentCount": "850"
      },
      "contentDetails": {
        "duration": "PT45S"  // 45 seconds (ISO 8601 format)
      }
    }
  ]
}
```

#### **Comments** (`/commentThreads`)
```http
GET https://www.googleapis.com/youtube/v3/commentThreads
  ?part=snippet
  &videoId=VIDEO_ID
  &maxResults=100
  &order=relevance
  &key=YOUR_API_KEY
```

**Returns**:
```json
{
  "items": [
    {
      "snippet": {
        "topLevelComment": {
          "snippet": {
            "textDisplay": "This is amazing! 🔥",
            "authorDisplayName": "John Doe",
            "likeCount": 245,
            "publishedAt": "2024-01-16T14:30:00Z"
          }
        }
      }
    }
  ],
  "nextPageToken": "CAUQAA"  // For pagination
}
```

---

## What You Can Extract

### ✅ **Available via Data API** (Public data)

| Data Point | Endpoint | Notes |
|------------|----------|-------|
| **View count** | `/videos` | Public for all videos |
| **Like count** | `/videos` | Public |
| **Comment count** | `/videos` | Total count |
| **Comments text** | `/commentThreads` | Up to 100 per request |
| **Comment likes** | `/commentThreads` | Per comment |
| **Comment authors** | `/commentThreads` | Username |
| **Upload date** | `/videos` | Exact timestamp |
| **Duration** | `/videos` | In ISO 8601 format |
| **Title/Description** | `/videos` | Full text |
| **Channel info** | `/videos` or `/channels` | Name, ID, etc. |
| **Thumbnails** | `/videos` | Multiple sizes |

### ⚠️ **Requires Analytics API** (Creator-only)

These need **OAuth 2.0** authentication as the video owner:

| Data Point | Why Creator-Only |
|------------|------------------|
| **Average view duration** | Privacy - shows how long viewers actually watch |
| **Audience retention graph** | Privacy - second-by-second drop-off data |
| **Traffic sources** | Privacy - where viewers came from |
| **Demographics** | Privacy - age, gender, location of viewers |
| **Revenue data** | Privacy - AdSense earnings |
| **Subscriber changes** | Privacy - gained/lost from video |

---

## How Our Comment Fetcher Uses It

From `comment_fetcher.py`:

```python
def _fetch_youtube_comments(self, video_url: str, max_comments: int):
    # Extract video ID from URL
    video_id = self._extract_youtube_id(video_url)
    # e.g., "dQw4w9WgXcQ" from "youtube.com/watch?v=dQw4w9WgXcQ"

    # Call YouTube API
    url = "https://www.googleapis.com/youtube/v3/commentThreads"
    params = {
        'part': 'snippet',
        'videoId': video_id,
        'key': self.youtube_api_key,
        'maxResults': 100,
        'order': 'relevance'  # Get most-liked comments first
    }

    response = requests.get(url, params=params)
    data = response.json()

    # Extract comment data
    comments = []
    for item in data['items']:
        snippet = item['snippet']['topLevelComment']['snippet']
        comments.append({
            'text': snippet['textDisplay'],
            'likes': snippet['likeCount'],
            'author': snippet['authorDisplayName'],
            'timestamp': snippet['publishedAt']
        })

    return comments
```

---

## API Quotas & Limits

YouTube API uses a **quota system**:

### Free Tier Limits:
- **10,000 units per day** (default)
- Different operations cost different amounts:
  - `videos.list`: **1 unit** per request
  - `commentThreads.list`: **1 unit** per request
  - `search.list`: **100 units** per request (expensive!)

### What This Means:
```
10,000 units/day ÷ 1 unit per comment fetch = 
Up to 10,000 comment fetches per day

OR

10,000 units ÷ 100 requests per page (100 comments) = 
Up to 1,000,000 comments per day (if only doing comments)
```

**In practice**: You can analyze ~100-200 videos with comments per day on free tier.

---

## Why This is Better Than Scraping

### YouTube Data API (Official):
✅ **Reliable**: Structured data format  
✅ **Legal**: Google provides it officially  
✅ **Fast**: Optimized API servers  
✅ **Stable**: Won't break when YouTube updates their site  
✅ **Complete**: Includes non-visible data (exact counts, etc.)  

### Web Scraping (Unofficial):
❌ **Fragile**: Breaks when HTML changes  
❌ **Slower**: Parse entire pages  
❌ **Against TOS**: Violates YouTube terms  
❌ **Incomplete**: Some data hidden in JavaScript  
❌ **Can get IP banned**: YouTube detects scraping  

---

## What About TikTok?

TikTok doesn't have a public comment API like YouTube. Options:

### 1. **RapidAPI** (What we use)
- Third-party service that accesses TikTok data
- Paid subscription required
- Less reliable than official APIs
- Example: `tiktok-scraper7.p.rapidapi.com`

```python
# RapidAPI call (simplified)
headers = {
    "X-RapidAPI-Key": api_key,
    "X-RapidAPI-Host": "tiktok-scraper7.p.rapidapi.com"
}
response = requests.get(
    "https://tiktok-scraper7.p.rapidapi.com/video/comments",
    headers=headers,
    params={'video_id': video_id}
)
```

### 2. **TikTok For Developers** (Official, but restricted)
- Requires application & approval
- Limited to specific use cases
- Not available for general comment access
- Primarily for business/creator tools

---

## Practical Example: Full Flow

### User analyzes YouTube video:
```
1. User submits: youtube.com/watch?v=dQw4w9WgXcQ

2. yt-dlp extracts:
   - View count: 150,000
   - Like count: 12,000
   - Video metadata

3. YouTube Data API fetches comments:
   GET /commentThreads?videoId=dQw4w9WgXcQ&key=...
   
4. Returns top 100 comments with:
   - Comment text
   - Like counts
   - Authors
   - Timestamps

5. Our comment_analyzer.py analyzes:
   - "0:23 was insane!" → timestamp mention
   - "🔥🔥🔥" → excitement emotion
   - Quotes → key moments
   - Common words → themes

6. GPT-4 synthesizes:
   - What moments resonated
   - Why viewers engaged
   - Emotional drivers
```

---

## Setup Steps (YouTube)

### 1. Get API Key:
```
1. Go to https://console.cloud.google.com/
2. Create new project (or use existing)
3. Enable "YouTube Data API v3"
4. Go to Credentials → Create API Key
5. Copy key
```

### 2. Add to .env:
```bash
YOUTUBE_API_KEY=AIzaSyC...your_key_here
```

### 3. Test:
```python
from comment_fetcher import CommentFetcher

fetcher = CommentFetcher()
comments = fetcher.fetch_comments(
    'https://youtube.com/watch?v=dQw4w9WgXcQ',
    max_comments=50
)

print(f"Fetched {len(comments)} comments")
for c in comments[:3]:
    print(f"- {c['text']} ({c['likes']} likes)")
```

---

## Cost & Scaling

### Free Tier:
- 10,000 units/day
- ~100-200 videos with comments
- **Perfect for your use case**

### If you need more:
- Request quota increase (form in Google Cloud Console)
- Can get up to 1,000,000 units/day
- Usually approved for legitimate business use

### Alternative:
- Cache aggressively (we already do this)
- Only fetch comments when explicitly requested
- Use comment analysis as opt-in feature

---

## Summary

**YouTube Data API**:
- Official REST API from Google
- Makes HTTP requests with API key
- Returns structured JSON data
- Free tier: 10k units/day (~100-200 videos)
- No scraping, no breaking, reliable

**Our Implementation**:
1. Extract video ID from URL
2. Call YouTube API with your key
3. Parse JSON response
4. Analyze with GPT-4
5. Show insights to user

**Why it's safe**:
- Google provides it officially
- Within their terms of service
- Won't get blocked/banned
- Structured, reliable data

Just need to add `YOUTUBE_API_KEY` to your .env file and you're ready! 🚀
