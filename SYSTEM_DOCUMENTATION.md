# Video Analysis AI - System Documentation

## Overview
This is a Flask-based video analysis system that uses AI (GPT-4) to analyze social media videos (TikTok, YouTube, Instagram) for content creators. It provides deep insights into hooks, engagement, viral mechanics, and replication formulas.

---

## Core Architecture

### Data Flow
```
Video URL Input
    ↓
yt-dlp Extraction (metadata, audio, video frames)
    ↓
Parallel Processing:
    - Audio → Whisper (transcription)
    - Frames → GPT-4 Vision (visual analysis)
    - Comments → Fetch & Analyze (audience insights)
    ↓
Knowledge Base Retrieval (RAG with embeddings)
    ↓
Main GPT-4 Analysis (comprehensive insights)
    ↓
Template Variables Preparation
    ↓
Two Storage Paths:
    1. PDF Cache (full data with frames) - temporary
    2. Database (lightweight data) - permanent
    ↓
Results Display (results.html or analysis_summary.html)
```

---

## Critical Data Structures

### Lightweight Data (Saved to Database)
**Location:** `app.py` → `complete_analysis()` → `lightweight_data` dict

**CRITICAL FIELDS TO ALWAYS SAVE:**
```python
lightweight_data = {
    # Basic Info
    'video_title': str,
    'creator': str,
    'platform': str,  # 'tiktok', 'youtube', 'instagram'
    'target_duration': str,
    'goal': str,  # 'follower_growth', 'engagement', etc.
    'niche': str,
    
    # Metrics
    'view_count': str,
    'like_count': int,
    'comment_count': int,
    'share_count': int,
    
    # Content (CRITICAL - needed for history view)
    'video_description': str,
    'hashtags': list,
    'audio_type': str,
    'music_info': str,
    'transcript': str,  # ⚠️ MUST SAVE - displays in history
    
    # Hook and Loop Analysis
    'hook': str,  # ⚠️ MUST SAVE - displays in analysis_summary.html
    'loop': str,  # ⚠️ MUST SAVE - displays in analysis_summary.html
    
    # Comment Analysis (NEW in v2.6.0)
    'comment_insights': dict or None,  # ⚠️ SAVE if available - displays comment section
    # Structure: {
    #   'total_comments': int,
    #   'categorization': {
    #     'counts': dict,
    #     'percentages': dict,
    #     'insight': str
    #   },
    #   'consensus_patterns': list[dict],  # Ranked by total likes
    #   'emotion_analysis': dict,          # Emotion → percentage
    #   'timestamp_mentions': list[dict],  # Timestamp → mentions
    #   'ai_insights': str                 # GPT-4 analysis of comments
    # }
    
    # Analysis Results
    'what_this_video_is': str,
    'why_it_performed': str,
    'replication_formula': dict,  # See structure below
    'improvements': str,
    'viral_mechanics': str,
    'performance_prediction': str,
    'scores': dict,  # See structure below
    'exact_hook_breakdown': dict,
    'all_hooks_identified': dict,
    
    # Legacy Fields (backwards compatibility)
    'goal_analysis': str,
    'overall_assessment': str,
    'primary_strengths': list,
    'areas_for_improvement': list,
}
```

### Scores Structure
```python
scores = {
    'hook_strength': int (1-10),
    'promise_clarity': int (1-10),
    'retention_design': int (1-10),
    'engagement_potential': int (1-10),
    'viral_potential': int (1-10),
    'satisfaction_delivery': int (1-10),
    'goal_alignment': int (1-10)
}
```

### Replication Formula Structure
```python
replication_formula = {
    'formula_name': str,
    'structure': str,
    'scenarios_for_same_niche': list[str],  # List of scenario markdown blocks
    'why_it_works': str,
    'text_template': str,  # Multi-line text template
    'visual_requirements': str,
    # Timeline entries
    '0-3s': str,
    '3-8s': str,
    '8-20s': str,
    # ... more timeline entries
}
```

---

## Template System

### Two Main Templates

#### 1. results.html
- Used for: **Real-time analysis** (just completed)
- Has access to: Full template_vars, PDF cache, frame gallery
- Expects fields:
  - `video_thumbnail` (not thumbnail_url)
  - `metadata.url` (for "View Original Video" link)
  - `transcript_quality.transcript` (for transcript text)
  - All scores, analysis fields

#### 2. analysis_summary.html
- Used for: **Saved analyses** (from history)
- Has access to: Lightweight DB data only (no frames)
- Expects fields:
  - `thumbnail_url`
  - `video_url`
  - `transcript`
  - `hook`
  - `loop`
  - All scores, analysis fields

### Template Field Mapping
When loading saved analyses, view_analysis must map DB fields to template expectations:

```python
# For results.html compatibility:
template_vars['video_thumbnail'] = analysis.thumbnail_url
template_vars['metadata']['url'] = analysis.video_url
template_vars['transcript_quality']['transcript'] = template_vars.get('transcript', '')

# For analysis_summary.html compatibility:
template_vars['video_url'] = analysis.video_url
template_vars['thumbnail_url'] = analysis.thumbnail_url
```

---

## Storage Systems

### 1. PDF Cache (Temporary)
**Location:** `PdfCache` class in `app.py`

**Purpose:** Store complete analysis data including base64 frame images

**Lifetime:** In-memory + disk cache, purged on deployment

**Storage:**
```python
pdf_cache[cache_key] = {
    'template_vars': {
        # ALL template variables including:
        'frame_gallery': [base64_images],  # Large data
        'transcript_quality': full_dict,
        # ... everything
    },
    'frames_dir': path,
    'timestamp': datetime
}
```

**Why:** Allows instant re-access to complete analysis with frames for PDF generation and re-viewing

### 2. Database (Permanent)
**Location:** SQLAlchemy `Analysis` model

**Purpose:** Store lightweight analysis data for history

**Columns:**
- `id`: Primary key
- `user_id`: Foreign key to User
- `video_url`: Original video URL
- `video_title`: Extracted title
- `thumbnail_url`: First frame as thumbnail
- `analysis_data`: JSON field with `lightweight_data` dict
- `pdf_cache_key`: Reference to PDF cache (may be expired)
- `status`: 'processing', 'completed', 'failed'
- `created_at`: Timestamp
- `completed_at`: Timestamp

**Why:** Permanent storage without large images, enables history browsing

---

## Analysis Flow Details

### 1. Video Extraction
**Tool:** yt-dlp

**Extracts:**
- Metadata (title, creator, views, likes, comments, shares)
- Audio track
- Video file for frame extraction

**Key Fields:**
- `view_count`, `like_count`, `repost_count`, `comment_count`
- `uploader`, `track`, `description`
- Performance level calculation based on views

### 2. Parallel Processing

#### Audio Analysis
- **Whisper:** Transcribes speech → transcript text
- **Librosa:** Analyzes tempo, energy, music detection
- **ACRCloud:** Identifies viral sounds (optional)

#### Frame Analysis
- **OpenCV:** Extracts frames based on strategy (smart/uniform)
- **GPT-4 Vision:** Analyzes each frame for:
  - Text overlays (exact text on screen)
  - Visual elements
  - Scene changes
  - Satisfaction elements

#### Comment Analysis (Optional)
- **TikTok API:** Fetches comments via RapidAPI
- **YouTube API:** Fetches comments via official API
- **Analyzer:** Categorizes, detects consensus patterns, extracts:
  - Why viewers watched
  - What intrigued them
  - What irritated them
  - Questions asked

### 3. Knowledge Base (RAG)
**Embeddings:** text-embedding-3-small (OpenAI)

**Storage:** ChromaDB + PDFs in knowledge/

**Retrieval Strategy:**
1. **Baseline:** Load 20 essential chunks from core patterns
2. **Goal-specific:** Load 21 chunks matching goal (follower_growth, etc.)
3. **Total:** ~60-64KB of relevant knowledge context

**Improves:** Pattern matching, formula suggestions, context-aware insights

### 4. Main Analysis
**Model:** GPT-4 (gpt-4o)

**Input:**
- Transcript text
- Frame summaries (concatenated)
- Creator notes
- Knowledge base context
- Comment insights
- Performance level

**Output Format:** Delimiter-based sections (===SECTION_NAME===)

**Parsed Sections:**
- WHAT_THIS_VIDEO_IS
- WHY_IT_PERFORMED
- ALL_HOOKS_TEXT/VISUAL/VERBAL/PSYCHOLOGICAL
- EXACT_HOOK_BREAKDOWN
- SCORES
- REPLICATION_FORMULA
- VIRAL_MECHANICS
- PERFORMANCE_PREDICTION
- IMPROVEMENTS

### 5. Response Parsing
**Function:** `parse_delimited_response()` in `app.py`

**Handles:**
- Multi-line values (replication formula scenarios, text templates)
- Nested structures (scores dict, hook breakdown dict)
- List parsing (hooks, knowledge patterns)

**Critical:** REPLICATION_FORMULA parsing splits on `**Scenario` markers for scenarios list

---

## Pattern Learning System (Admin Only)

### Access Control
**User:** christina@superlunardesign.com only

### Purpose: Train the AI to Notice Success Nuances

**CRITICAL CONCEPT:** This system is NOT about creating universal rules. It's about **training the AI** to notice the subtle patterns and principles that make videos successful.

**How It Works:**
- Curator adds notes about what they learned from analyzing a video
- AI extracts the **teaching** from those notes (not assumptions)
- These learnings accumulate to help the AI suggest better insights in future analyses
- Over time, AI learns to notice: "Oh, general topics reach wider audiences", "Agitation can keep viewers watching", "Agreement drives engagement"

**What to Include in Curator Notes:**
- ✅ Specific observations: "comments proved people stayed even through frustration"
- ✅ Principles discovered: "strong curiosity gap kept people watching"
- ✅ Nuances noticed: "agitation isn't necessarily bad for views"
- ✅ Counter-intuitive findings: "they were agitated but stayed to the end"
- ❌ NOT general assumptions: "this works best for B2B"
- ❌ NOT categorizations: "avoid for entertainment"

### Two-Step Process

#### 1. Preview Pattern
**Endpoint:** `/preview_pattern` (POST)

**Purpose:** Extract learnings from curator notes and show preview

**What Gets Extracted:**
```python
pattern_data = {
    'pattern_summary': str,  # What curator is teaching AI
    'key_insights': [
        "Direct principle from curator notes",
        "Another insight from curator notes"
    ],
    'context': {
        'niche': str,
        'platform': str,
        'what_happened': str,  # What curator observed
        'why_it_matters': str   # The lesson
    },
    'when_to_notice': [
        # Only if curator specifies when to look for this
    ],
    'cautions': [
        # Only if curator explicitly warns about something
    ],
    'observed_elements': [
        # Factual things that were present in video
    ]
}
```

**Shows:** Preview of what will be stored - curator can verify it extracted their teaching correctly

**IMPORTANT:** If preview shows assumptions not in your notes, the extraction is wrong

### Examples of Good vs Bad Curator Notes

#### ✅ GOOD - Specific Insights
```
"Comments proved that people stayed to watch the whole video but were agitated 
at the creator not getting to the point. But it proves that the hook created a 
strong enough gap that people stayed even through their frustration. The tension 
built didn't provide value, entertainment, or satisfaction, but it still built tension.

Other commenters agreed with the video's overall take. Fonts are also a general 
enough concept that people outside of professional spheres will stay to know what 
fonts are no-nos, even if they don't have a logo."
```

**What AI learns:**
- General topics include wider pool of viewers
- Agitation isn't necessarily bad if curiosity gap is strong
- Strong curiosity gap can keep people watching even through frustration
- Agreement drives engagement (likes, shares, comments)
- Tension can work even without providing value

#### ❌ BAD - Vague Generalizations
```
"This video works because of good pacing and a strong hook."
```

**Problem:** Too generic - AI can't learn anything specific to notice

#### ❌ BAD - Incorrect Assumptions from Preview
```
Preview shows: "Works best for: B2B services, Branding advice"
But curator notes said: "Fonts are general enough that anyone will watch"
```

**Problem:** AI invented a niche restriction that contradicts the actual insight

### How to Write Effective Curator Notes

1. **State What You Observed**
   - "Comments showed people were frustrated but stayed"
   - "Video got high engagement despite not delivering satisfaction"

2. **Extract the Principle**
   - "This proves curiosity gap can override frustration"
   - "General topics reach beyond target audience"

3. **Note Nuances**
   - "Agitation isn't always bad for retention"
   - "Agreement increases sharing behavior"

4. **Don't Over-Categorize**
   - ❌ "This only works for professional content"
   - ✅ "Professional topics can attract casual viewers if general enough"

5. **State Cautions if Relevant**
   - "Could cross into rage bait territory if overdone"
   - "Requires strong hook to offset frustration"

#### 2. Store Pattern
**Endpoint:** `/submit_for_learning` (POST)

**Storage:** ChromaDB with embeddings

**Universality Detection:**
- **Universal:** Pattern found in 4+ niches → always retrieved
- **Cross-niche:** Pattern in 2-3 niches → high priority
- **Niche-specific:** Pattern in 1 niche → contextual retrieval

**Why:** Patterns stored WITH context, not as universal rules. System learns which formats work across niches vs. niche-specific patterns.

---

## Comment Analysis System

### Fetching
**Platforms:**
- TikTok: RapidAPI (tiktok-api15)
- YouTube: Official YouTube Data API v3
- Instagram: Not implemented (requires Graph API)

**Fetch Limit:** 100 comments (configurable)

### Analysis Features

#### 1. Categorization
```python
categories = {
    'substantive': 0,  # Context-rich (timestamps, quotes)
    'emoji_only': 0,   # Just emojis
    'generic': 0,      # Generic praise
    'spam': 0          # Self-promotion
}
```

#### 2. Consensus Patterns (KEY FEATURE)
**Groups similar comments by theme keywords:**
```python
theme_keywords = {
    'background_element': ['background', 'behind', 'spotted'],
    'specific_moment': ['part when', 'moment', 'timestamp'],
    'visual_element': ['outfit', 'ring', 'nails'],
    'pet': ['cat', 'dog', 'puppy'],
    'end_reveal': ['end', 'reveal', 'wait until'],
}
```

**Like-Weighted Scoring:**
- Sums total likes for each theme
- `consensus_strength = total_likes / mentions`
- Reveals what audience COLLECTIVELY agreed on

**Example:** If 50 comments about "cat in background" have 10k total likes, that's stronger consensus than 200 comments about "music" with 2k total likes.

#### 3. Hidden Hook Discovery
Detects when viewers engage with **unintentional elements**:
- Background elements noticed more than main content
- Accidental elements that drove engagement
- Gap between creator intent and viewer attention

### Integration with Main Analysis
Comment insights added to GPT prompt:
```
VIEWER COMMENT ANALYSIS:
- What viewers ACTUALLY noticed
- Consensus patterns (ranked by likes)
- Intent vs. Reality gap
- Hidden hooks
```

**Influences:** Performance analysis, improvement suggestions, viral mechanics identification

### Comment Analysis Display (NEW in v2.6.0)

**Location:** Both results.html and analysis_summary.html

**Purpose:** Show what comments reveal about viewer engagement

**Display Components:**

1. **Comment Types Breakdown**
   - Grid showing percentages: Substantive, Emoji-only, Generic, Spam
   - Insight text explaining engagement quality
   - Example: "High substantive engagement - viewers are articulating specific moments"

2. **Consensus Patterns (Ranked by Total Likes)**
   - Top 5 patterns sorted by like-weighted consensus
   - Shows: Theme name, # mentions, total likes
   - Displays example comment with like count
   - Reveals what audience COLLECTIVELY agreed on

3. **Emotional Reactions**
   - Pills/badges showing top emotions > 5%
   - Example: "Excitement: 25%", "Humor: 18%"
   - Color-coded with solar-lime highlighting

4. **Moments Viewers Highlighted**
   - Top 5 timestamp mentions from comments
   - Shows exact timestamp, # mentions, example comment
   - Example: "0:23 (12 mentions)"

5. **AI Analysis of Comments**
   - GPT-4 generated insights from comment patterns
   - Identifies: What drove engagement, hidden hooks, intent vs reality gap
   - Highlighted in purple box with solar-lime border

**Styling Guidelines:**
- Use `rgba(106, 116, 207, 0.15)` for section backgrounds
- Use `var(--solar-lime)` for highlights and borders
- Use `var(--celestial-lavender)` for secondary text
- Use `var(--stellar-cream)` for primary text

**Conditional Display:**
- Only shows if `comment_insights` exists AND `total_comments > 0`
- Each subsection conditional on data availability
- Gracefully handles missing data (no errors if partial data)

**Data Flow:**
```
Comment Fetcher → Comment Analyzer → comment_insights dict
    ↓
prepare_template_variables (adds to template_vars)
    ↓
lightweight_data (saves to database)
    ↓
results.html or analysis_summary.html (displays sections)
```

**CRITICAL:** Always check `{% if comment_insights and comment_insights.total_comments > 0 %}` before displaying

---

## Key Routes

### User-Facing
- `GET /` → index.html (analysis form)
- `POST /analyze_async` → progress.html (shows processing status)
- `POST /process` → Complete analysis flow
- `GET /history` → List saved analyses
- `GET /analysis/<id>` → View saved analysis

### Admin-Only
- `POST /preview_pattern` → Preview pattern before storing
- `POST /submit_for_learning` → Store confirmed pattern

### API/Utility
- `GET /download_pdf/<cache_key>` → Download PDF report
- `DELETE /analysis/<id>` → Delete saved analysis
- `GET /check_analysis/<id>` → Check if analysis exists
- `GET /api/analysis_status/<id>` → Get analysis status (for polling)

---

## Critical Bugs Fixed

### Bug 1: Transcript Lost in History
**Symptom:** Saved analyses show "(No verbal content)" even though transcript loaded in real-time

**Cause:** `transcript` field not saved to lightweight_data

**Fix:** Added `'transcript': template_vars.get('transcript', '')` to lightweight_data dict

### Bug 2: Missing Thumbnail/Link in History
**Symptom:** Saved analyses don't show video thumbnail or "View Original Video" link

**Cause:** results.html expects `video_thumbnail` and `metadata.url`, but view_analysis wasn't setting them

**Fix:** Added field mapping in view_analysis:
```python
template_vars['video_thumbnail'] = analysis.thumbnail_url
template_vars['metadata']['url'] = analysis.video_url
```

### Bug 3: Replication Formula Empty Sections
**Symptom:** "Scenarios for Your Niche" and "Text Template" sections empty

**Cause:** Parser expected bullet points but GPT returned `**Scenario 1:**` markdown headings

**Fix:** Rewrote parser to split on `**Scenario` markers and preserve multi-line values

### Bug 4: Light Green Text Unreadable
**Symptom:** Solar-lime (#F5FFA0) text on solar-lime background

**Cause:** "High Repost Rate" and "Viral Sound" boxes used `background: rgba(245, 255, 160, 0.1)` with `color: var(--solar-lime)`

**Fix:** Changed to purple background with cream text:
```css
background: rgba(106, 116, 207, 0.2);
color: var(--stellar-cream);
```

---

## Environment Variables

### Required
```bash
OPENAI_API_KEY=sk-...           # GPT-4, Whisper, embeddings
DATABASE_URL=postgresql://...    # PostgreSQL database
SECRET_KEY=random_string         # Flask session secret
```

### Optional (For Full Features)
```bash
YOUTUBE_API_KEY=AIza...         # YouTube comment fetching
RAPIDAPI_KEY=...                # TikTok comment fetching
ACRCLOUD_ACCESS_KEY=...         # Viral sound detection
ACRCLOUD_ACCESS_SECRET=...      # Viral sound detection
```

---

## Color Scheme
```css
--cosmic-indigo: #1E0A3C;      /* Dark purple background */
--lunar-blush: #FF7AA2;         /* Pink accent */
--stellar-cream: #FFF8E1;       /* Cream text */
--solar-lime: #F5FFA0;          /* Yellow-green highlights */
--celestial-lavender: #B7B4ED;  /* Lavender secondary text */
--nebula-purple: #6A74CF;       /* Purple accents */
```

**Usage Guidelines:**
- ✅ solar-lime for: Borders, accent text on dark backgrounds, buttons
- ❌ solar-lime NOT for: Text on light backgrounds (use stellar-cream or celestial-lavender)

---

## Testing Saved Analyses

### Checklist for History View
- [ ] Video thumbnail displays
- [ ] "View Original Video" link works
- [ ] Transcript shows (not "No verbal content")
- [ ] Hook analysis displays
- [ ] Loop analysis displays
- [ ] Scores display (all 7 scores)
- [ ] Replication formula displays with scenarios and text template
- [ ] **Comment analysis section displays** (if comments were fetched)
  - [ ] Comment types breakdown shows
  - [ ] Consensus patterns ranked by likes
  - [ ] Emotional reactions display
  - [ ] Timestamp mentions show
  - [ ] AI insights from comments display
- [ ] All sections have content (not empty)
- [ ] No light green text on light backgrounds
- [ ] Re-run button autofills URL in form field (not appended to URL)

### How to Test
1. Analyze a video (ensure it has speech)
2. Go to History tab
3. Click on the saved analysis
4. Verify all checklist items

---

## Future Enhancements

### Planned Features
- Instagram comment fetching (requires Graph API setup)
- More granular pattern universality (track which specific elements work cross-niche)
- Export patterns as JSON for sharing
- Pattern visualization dashboard
- Batch analysis for multiple videos

### Technical Debt
- Consolidate results.html and analysis_summary.html (too much duplication)
- Move parsing functions to separate module
- Add comprehensive error recovery for API failures
- Implement proper retry logic for external APIs

---

## Development Guidelines

### When Adding New Analysis Fields

1. **Add to GPT prompt** (in `run_main_analysis()`)
   - Add section delimiter: `===NEW_FIELD===`
   - Add instructions for what GPT should return

2. **Add to parser** (in `parse_delimited_response()`)
   - Handle the new section
   - Parse into appropriate data structure

3. **Add to result dict** (in `run_main_analysis()`)
   - Include in `result = {...}` return value

4. **Add to template_vars** (in `prepare_template_variables()`)
   - Map from gpt_result to template_vars

5. **Add to lightweight_data** (in `complete_analysis()`)
   - ⚠️ CRITICAL: Add to `lightweight_data = {...}` for history to work

6. **Add to template** (in results.html or analysis_summary.html)
   - Display the new field with proper styling

7. **Test both views:**
   - Real-time analysis (results.html)
   - Saved analysis (history → analysis_summary.html)

### When Removing Fields
❌ **NEVER remove fields from lightweight_data without:**
1. Checking all templates for usage
2. Adding backwards compatibility defaults
3. Testing old saved analyses don't break

---

## Common Issues & Solutions

### "Analysis not found" Error
**Cause:** Trying to view analysis that doesn't belong to current user

**Solution:** Check user_id matches in database

### Template Rendering Errors
**Cause:** Missing required field in template_vars

**Solution:**
1. Add safe defaults in view_analysis
2. Use `{% if field %}` conditionals in templates
3. Add `.get('field', '')` for dict access

### PDF Download Fails
**Cause:** PDF cache expired (purged on deploy)

**Solution:**
1. Check if pdf_cache_key exists in cache
2. If not, offer "Re-run analysis" instead
3. Consider longer cache TTL or persistent storage

### Slow Analysis
**Cause:** Too many frames or large knowledge base

**Solution:**
1. Reduce cap (default 40 frames)
2. Optimize knowledge retrieval (fewer chunks)
3. Use smart extraction over uniform

### Re-run Button Appends to URL
**Cause:** Using `href="/?url=..."` appends parameter instead of populating form

**Solution:** Added JavaScript in index.html to auto-populate form field from URL parameter
```javascript
// On page load, check for ?url= parameter
const urlParams = new URLSearchParams(window.location.search);
const urlParam = urlParams.get('url');
if (urlParam) {
    document.getElementById('tiktok_url').value = decodeURIComponent(urlParam);
}
```

---

## Monitoring & Logging

### Key Log Patterns
```python
[DEBUG] gpt_result has replication_formula: True/False
[TEMPLATE_PREP] Has replication_formula: True/False
[DB] Saving analysis {id}
[DB] lightweight_data keys: [...]
[VIEW] Using lightweight DB data for analysis {id}
[VIEW] DB data keys: [...]
```

**Use these to trace data flow from GPT → template_vars → database → view**

### Performance Metrics
- Extraction time: ~30-40s
- Frame analysis: ~60-90s (depends on frame count)
- Main analysis: ~40-60s
- **Total:** ~2-4 minutes per video

---

## Deployment (Render.com)

### Build Command
```bash
pip install -r requirements.txt
```

### Start Command
```bash
gunicorn app:app
```

### Environment
- Python 3.11
- PostgreSQL addon (database)
- 512MB RAM minimum (recommended 1GB for concurrent analyses)

### Important Notes
- PDF cache purged on each deploy
- Database persists across deploys
- Set all environment variables in Render dashboard
- Use persistent disk for knowledge/ directory (optional)

---

## File Structure
```
video_analysis_ai/
├── app.py                          # Main Flask application
├── comment_analyzer.py             # Comment analysis logic
├── comment_fetcher.py              # Fetch comments from platforms
├── success_patterns_improved.py    # Pattern learning system
├── requirements.txt                # Python dependencies
├── knowledge/                      # PDF knowledge base
│   ├── x8u4vlfmj1n62gdem7rbpyq52jcg.pdf
│   └── ...
├── templates/                      # Jinja2 templates
│   ├── base.html                   # Base layout
│   ├── index.html                  # Analysis form
│   ├── results.html                # Real-time analysis display
│   ├── analysis_summary.html       # Saved analysis display
│   ├── history.html                # Analysis history list
│   ├── processing.html             # Progress indicator
│   └── login.html                  # User authentication
├── static/                         # Static assets (if any)
└── SYSTEM_DOCUMENTATION.md         # This file
```

---

## Maintenance Checklist

### Weekly
- [ ] Check Render logs for errors
- [ ] Monitor database size
- [ ] Review failed analyses

### Monthly
- [ ] Update knowledge base PDFs
- [ ] Review pattern learning accuracy
- [ ] Check API quota usage (OpenAI, YouTube, RapidAPI)
- [ ] Optimize ChromaDB performance

### Before Major Updates
- [ ] Test on staging environment
- [ ] Backup database
- [ ] Document new features in this file
- [ ] Update test checklist
- [ ] Verify backwards compatibility

---

## Version History

### v2.7.0 (Current)
- ✅ **FIXED**: Text hooks now filtered to remove captions (cross-checks with transcript)
- ✅ **FIXED**: Transcript now displays in saved analyses (field properly preserved)
- ✅ **FIXED**: Video description displays in history
- ✅ **FIXED**: Pattern learning extracts actual insights (no more invented assumptions)
- ✅ Added debug logging for transcript and description in view_analysis

### v2.6.0
- ✅ **NEW**: Added Comment Analysis section to both results.html and analysis_summary.html
- ✅ **NEW**: Comment insights saved to lightweight_data (permanent storage)
- ✅ **NEW**: Display categorization, consensus patterns, emotions, timestamp mentions, AI insights
- ✅ Fixed re-run button to properly autofill form field (not append to URL)
- ✅ Updated SYSTEM_DOCUMENTATION.md with comprehensive comment analysis details

### v2.5.0
- ✅ Fixed transcript display in saved analyses
- ✅ Fixed thumbnail and link display in history
- ✅ Fixed light green text readability
- ✅ Added hook and loop fields to lightweight_data
- ✅ Improved view_analysis field mapping

### v2.4.0
- Added comment analysis integration
- Implemented like-weighted consensus patterns
- Added hidden hook discovery

### v2.3.0
- Added context-aware pattern learning
- Implemented pattern universality detection
- Added preview before storing patterns

### v2.2.0
- Fixed replication formula parsing for multi-line values
- Improved delimiter-based response parsing

### v2.1.0
- Added knowledge base RAG system
- Implemented smart retrieval (baseline + goal-specific)

---

**Last Updated:** June 27, 2026
**Maintained By:** Claude + christina@superlunardesign.com
**Critical:** Keep this file updated with ALL system changes
