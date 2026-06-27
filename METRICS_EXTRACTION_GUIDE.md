# Video Metrics Extraction Guide

## Current State: What We Can Extract

### ✅ Fully Automated (No Setup Required)

These metrics are automatically extracted from video URLs using `yt-dlp`:

| Metric | TikTok | YouTube | Instagram | Source |
|--------|--------|---------|-----------|--------|
| **View Count** | ✅ | ✅ | ✅ | yt-dlp metadata |
| **Like Count** | ✅ | ✅ | ✅ | yt-dlp metadata |
| **Comment Count** | ✅ | ✅ | ✅ | yt-dlp metadata |
| **Upload Date** | ✅ | ✅ | ✅ | yt-dlp metadata |
| **Duration** | ✅ | ✅ | ✅ | yt-dlp metadata |
| **Title/Description** | ✅ | ✅ | ✅ | yt-dlp metadata |
| **Share Count** | ✅ | ✅ | ⚠️ | yt-dlp (varies) |

**Current Implementation**: `processing.py` → `extract_video_metadata()`

---

### 📝 Available with API Setup

These require API keys but don't need creator access:

#### **Comment Analysis**
| Platform | Status | API Needed | Cost |
|----------|--------|------------|------|
| YouTube | ✅ Ready | YouTube Data API (free) | 10k units/day |
| TikTok | ✅ Ready | RapidAPI ($) | Varies by plan |
| Instagram | ⚠️ Partial | Graph API (free) | Requires business account |

**Implementation**: `comment_fetcher.py` + `comment_analyzer.py`

**What You Get**:
- Comment text
- Comment likes/engagement
- Timestamp mentions
- Emotional reactions
- Quoted moments
- Recurring themes

**Setup**:
```bash
# .env file
YOUTUBE_API_KEY=your_key_here
RAPIDAPI_KEY=your_key_here  # For TikTok
```

---

### ⚠️ Estimated (No Direct API Access)

These can be **estimated** from available metrics:

#### **Engagement Rate**
```python
engagement_rate = (likes + comments + shares) / views
```
**Status**: ✅ Can calculate automatically
**Accuracy**: High (uses real data)

#### **Average Watch Time** (Estimated)
```python
# Based on engagement correlation
if engagement_rate > 0.15:  # Viral
    avg_retention = 75%
elif engagement_rate > 0.08:  # Good
    avg_retention = 50%
else:
    avg_retention = 30%

estimated_watch_time = duration * avg_retention
```
**Status**: ⚠️ Estimation only
**Accuracy**: Medium (industry averages)

**Implementation**: `metrics_fetcher.py` → `estimate_metrics_from_available()`

---

### ❌ Requires Creator Access (Not Available Publicly)

These metrics require being logged in as the video creator:

| Metric | Why Unavailable | Where to Get It |
|--------|-----------------|-----------------|
| **Exact Watch Time** | Privacy/creator-only | YouTube Studio, TikTok Analytics |
| **Retention Curve** | Privacy/creator-only | YouTube Studio retention graph |
| **Drop-off Points** | Privacy/creator-only | TikTok Analytics |
| **Traffic Sources** | Privacy/creator-only | Platform analytics |
| **Demographics** | Privacy/creator-only | Platform analytics |
| **Saves (TikTok)** | Some 3rd party APIs | TikTok Creator Center |
| **Re-watches** | Platform-specific | Not publicly exposed |

**Why**: Privacy policies prevent exposing viewer-level data to non-creators.

---

## Implementation Options

### Option 1: Manual Entry (Current - Best for Beta)

**For your admin learning feature**, users manually enter metrics:

```javascript
// results.html form
Views: 150000
Engagement Rate: 12% (or 0.12)
Watch Time: "58s average" (optional note)
```

**Pros**:
- Works for any video
- No API limitations
- User can add context

**Cons**:
- Manual work
- Relies on user having access to creator analytics

---

### Option 2: Automatic Calculation (Recommended Next Step)

Add to `processing.py`:

```python
def calculate_engagement_metrics(metadata: Dict) -> Dict:
    """Calculate derived metrics from available data"""
    views = metadata.get('view_count', 0)
    likes = metadata.get('like_count', 0)
    comments = metadata.get('comment_count', 0)
    shares = metadata.get('repost_count', 0)
    duration = metadata.get('duration', 30)

    metrics = {}

    # Engagement rate
    if views > 0:
        metrics['engagement_rate'] = round(
            (likes + comments + shares) / views, 4
        )
    else:
        metrics['engagement_rate'] = 0

    # Estimated retention (based on engagement patterns)
    eng_rate = metrics['engagement_rate']
    if eng_rate > 0.15:
        retention = 0.75
    elif eng_rate > 0.08:
        retention = 0.50
    elif eng_rate > 0.03:
        retention = 0.30
    else:
        retention = 0.15

    metrics['estimated_retention_pct'] = round(retention * 100, 1)
    metrics['estimated_avg_watch_time_sec'] = int(duration * retention)

    # Completion rate estimate
    if eng_rate > 0.15:
        completion = 0.60
    elif eng_rate > 0.08:
        completion = 0.35
    elif eng_rate > 0.03:
        completion = 0.20
    else:
        completion = 0.10

    metrics['estimated_completion_rate'] = round(completion * 100, 1)

    return metrics
```

**Then use in analysis**:
```python
# In process() endpoint
auto_metrics = calculate_engagement_metrics(metadata)

# Add to template vars
template_vars['auto_calculated_metrics'] = auto_metrics
template_vars['engagement_rate'] = auto_metrics['engagement_rate']
template_vars['estimated_watch_time'] = auto_metrics['estimated_avg_watch_time_sec']
```

**Pros**:
- Fully automated
- No API keys needed
- Works for all videos

**Cons**:
- Estimates, not exact
- Can't show retention curve

---

### Option 3: Creator Analytics Integration (Advanced)

For videos you own/manage:

```python
from metrics_fetcher import AdvancedMetricsFetcher

fetcher = AdvancedMetricsFetcher()

# Try to get real analytics (requires setup)
youtube_analytics = fetcher.get_youtube_analytics(video_id)

if 'error' not in youtube_analytics:
    # Use real creator data
    watch_time = youtube_analytics['average_view_duration_seconds']
    retention = youtube_analytics['average_view_percentage']
else:
    # Fallback to estimates
    estimated = fetcher.estimate_metrics_from_available(basic_metrics)
    watch_time = estimated['estimated_avg_watch_time_seconds']
```

**Setup Required**:
- YouTube: OAuth 2.0 + YouTube Analytics API
- TikTok: TikTok For Developers account + approval
- Instagram: Business account + Facebook Graph API

**Pros**:
- Exact analytics for your videos
- Retention curves
- Traffic sources
- Demographics

**Cons**:
- Complex setup
- Only works for videos you own
- Not scalable for analyzing others' content

---

## Recommended Approach

For your current use case (analyzing successful videos, learning patterns):

### Phase 1: Auto-Calculate (Easy Win) ✅
```python
# Add to processing.py
1. Auto-calculate engagement rate from metadata
2. Estimate avg watch time from engagement patterns
3. Show estimates with disclaimers
```

**Implementation**: ~30 lines of code, works immediately

---

### Phase 2: Manual Enhancement (Current Beta) ✅
```python
# Already implemented
1. User submits video for learning
2. Manually enters view count, engagement rate
3. Adds curator notes about patterns
```

**Status**: Already working!

---

### Phase 3: Comment Analysis (Optional) 🎯
```python
# Enable comment fetching
1. Set up YouTube API key (free)
2. Optionally: RapidAPI for TikTok
3. Analyze comments for engagement insights
```

**Value**: Understand what moments resonated with viewers

---

## Quick Setup: Auto-Calculate Metrics

Add this to your `app.py` right after metadata extraction:

```python
# Around line 2040, after extract_video_metadata()
def enhance_metadata_with_calculations(metadata: Dict) -> Dict:
    """Add calculated metrics to metadata"""
    views = metadata.get('view_count', 0)
    likes = metadata.get('like_count', 0)
    comments = metadata.get('comment_count', 0)
    shares = metadata.get('repost_count', 0)
    duration = metadata.get('duration', 30)

    # Engagement rate
    if views > 0:
        engagement_rate = (likes + comments + shares) / views
        metadata['engagement_rate'] = round(engagement_rate, 4)
        metadata['engagement_rate_pct'] = round(engagement_rate * 100, 2)

        # Estimated retention based on engagement
        if engagement_rate > 0.15:
            retention = 0.75
            perf = "viral"
        elif engagement_rate > 0.08:
            retention = 0.50
            perf = "strong"
        elif engagement_rate > 0.03:
            retention = 0.30
            perf = "average"
        else:
            retention = 0.15
            perf = "low"

        metadata['estimated_retention_pct'] = round(retention * 100, 1)
        metadata['estimated_avg_watch_time_sec'] = int(duration * retention)
        metadata['performance_tier'] = perf
    else:
        metadata['engagement_rate'] = 0
        metadata['estimated_retention_pct'] = 0
        metadata['estimated_avg_watch_time_sec'] = 0

    return metadata

# Use it:
metadata = extract_video_metadata(video_url)
metadata = enhance_metadata_with_calculations(metadata)
```

Then in your template, show:
```html
<div class="metric-card">
  <div class="metric-value">{{ engagement_rate_pct }}%</div>
  <div class="metric-label">Engagement Rate</div>
</div>

<div class="metric-card">
  <div class="metric-value">~{{ estimated_avg_watch_time_sec }}s</div>
  <div class="metric-label">Est. Avg Watch Time</div>
  <div class="metric-note">*Estimated from engagement</div>
</div>
```

---

## Summary

| Metric | Status | How to Get |
|--------|--------|------------|
| Views, Likes, Comments | ✅ Auto | Already working via yt-dlp |
| Engagement Rate | ✅ Auto | Calculate from above |
| Est. Watch Time | ⚠️ Estimated | Calculate from engagement |
| Comments Text/Analysis | 🔧 API Setup | YouTube API + comment_analyzer.py |
| Exact Watch Time | ❌ Creator Only | Manual entry or creator API |
| Retention Curve | ❌ Creator Only | Manual entry or creator API |
| Saves (TikTok) | ⚠️ Some APIs | RapidAPI or manual |

**Next Steps**:
1. ✅ Already done: Manual entry for learning feature
2. 🎯 Easy win: Auto-calculate engagement + estimated watch time
3. 🔧 Optional: Set up comment analysis APIs

Let me know if you want me to implement the auto-calculation!
