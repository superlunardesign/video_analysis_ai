import os
import random
import time as _time
import json
import re
from collections import Counter
from flask import Flask, request, render_template
from openai import OpenAI

from processing import (
    extract_audio_and_frames,
    transcribe_audio,
    analyze_frames_batch,
    download_video,
    probe_duration,
    extract_audio,
    scene_change_times,
    motion_event_times,
    extract_frames_at_times,
    is_blurry,
    dedupe_frames_by_phash,
    keep_text_heavy_frames,
    extract_frames_uniform,
    _ensure_dirs
)
from rag_helper import retrieve_context, retrieve_all_context

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=600.0)


def _api_retry(callable_fn, *args, **kwargs):
    max_tries = 4
    base = 1.25
    for attempt in range(1, max_tries + 1):
        try:
            return callable_fn(*args, **kwargs)
        except Exception as e:
            if attempt == max_tries:
                raise
            sleep_s = (base ** attempt) + random.uniform(0, 0.5)
            print(f"[retry] OpenAI call failed ({attempt}/{max_tries}): {e}. Retrying in {sleep_s:.1f}s")
            _time.sleep(sleep_s)


# ==============================
# ENHANCED TEXT ANALYSIS - Distinguish on-screen text from captions
# ==============================

def analyze_text_synchronization(frames_summaries_text, transcript_text, frame_timestamps=None):
    """
    Distinguish between on-screen text (graphics/overlays) and spoken captions.
    """
    
    # Extract text mentions from frame analysis
    frame_texts = []
    frames_blocks = frames_summaries_text.split('\n\n')
    
    for i, block in enumerate(frames_blocks):
        if 'text' in block.lower() or 'overlay' in block.lower() or 'caption' in block.lower():
            # Extract potential text content
            text_patterns = [
                r'"([^"]+)"',  # Text in quotes
                r'text[:\s]+([^\n.]+)',  # Text following "text:"
                r'overlay[:\s]+([^\n.]+)',  # Text following "overlay:"
                r'caption[:\s]+([^\n.]+)',  # Text following "caption:"
                r'reads[:\s]+([^\n.]+)',  # Text following "reads:"
                r'says[:\s]+([^\n.]+)',  # Text following "says:"
            ]
            
            extracted_texts = []
            for pattern in text_patterns:
                matches = re.findall(pattern, block, re.IGNORECASE)
                extracted_texts.extend([match.strip() for match in matches])
            
            if extracted_texts:
                timestamp = frame_timestamps[i] if frame_timestamps and i < len(frame_timestamps) else i * 2  # Assume 2s intervals
                frame_texts.append({
                    'timestamp': timestamp,
                    'texts': extracted_texts,
                    'frame_block': block
                })
    
    # Analyze synchronization with transcript
    transcript_words = transcript_text.lower().split()
    
    synchronized_texts = []
    onscreen_graphics = []
    
    for frame_text_data in frame_texts:
        timestamp = frame_text_data['timestamp']
        texts = frame_text_data['texts']
        
        for text in texts:
            text_lower = text.lower()
            
            # Check if this text appears in the transcript
            text_words = text_lower.split()
            
            # Calculate similarity to transcript
            matching_words = sum(1 for word in text_words if word in transcript_words)
            similarity_ratio = matching_words / len(text_words) if text_words else 0
            
            # Classify as synchronized caption vs on-screen graphic
            if similarity_ratio > 0.7 and len(text_words) > 2:
                synchronized_texts.append({
                    'text': text,
                    'timestamp': timestamp,
                    'type': 'caption',
                    'similarity': similarity_ratio
                })
            else:
                onscreen_graphics.append({
                    'text': text,
                    'timestamp': timestamp,
                    'type': 'graphic',
                    'similarity': similarity_ratio
                })
    
    return {
        'synchronized_captions': synchronized_texts,
        'onscreen_graphics': onscreen_graphics,
        'has_graphics': len(onscreen_graphics) > 0,
        'has_captions': len(synchronized_texts) > 0,
        'text_analysis_summary': f"Found {len(synchronized_texts)} synchronized captions and {len(onscreen_graphics)} on-screen graphics"
    }


# ==============================
# ENHANCED PSYCHOLOGICAL ANALYSIS FUNCTIONS
# ==============================

# Replace the broken detect_specific_niche function with this corrected version:

def detect_specific_niche(transcript_text, frames_summaries_text):
    """Detect specific niche and content type for targeted analysis."""
    
    combined_text = f"{transcript_text} {frames_summaries_text}".lower()
    
    # Detailed niche detection
    niche_indicators = {
        'beauty_skincare': {
            'keywords': ['skincare', 'routine', 'glow', 'skin', 'moisturizer', 'serum', 'cleanser'],
            'content_types': ['routine', 'transformation', 'product_review', 'tips'],
            'psychology_focus': 'self-improvement, confidence, transformation, aspirational identity'
        },
        'fitness_wellness': {
            'keywords': ['workout', 'fitness', 'exercise', 'healthy', 'diet'],
            'content_types': ['routine', 'transformation', 'tips', 'motivation'],
            'psychology_focus': 'discipline, transformation, health anxiety, comparison'
        },
        'lifestyle_productivity': {
            'keywords': ['morning routine', 'productive', 'organized', 'habits'],
            'content_types': ['routine', 'tips', 'lifestyle'],
            'psychology_focus': 'control, optimization, aspiration, self-improvement'
        },
        'food_cooking': {
            'keywords': ['recipe', 'cooking', 'food', 'kitchen', 'meal'],
            'content_types': ['tutorial', 'recipe', 'process'],
            'psychology_focus': 'comfort, creativity, nourishment, sharing'
        }
    }
    
    # Determine primary niche
    niche_scores = {}
    for niche, data in niche_indicators.items():
        score = sum(1 for keyword in data['keywords'] if keyword in combined_text)
        if score > 0:
            niche_scores[niche] = score
    
    primary_niche = max(niche_scores, key=niche_scores.get) if niche_scores else 'general'
    
    # Specialized analysis focus based on niche
    analysis_focuses = {
        'beauty_skincare': """
BEAUTY/SKINCARE PSYCHOLOGY FOCUS:
- Transformation aspiration: How does this tap into desires for change/improvement?
- Routine psychology: What makes routines psychologically satisfying?
- Before/after anticipation: How is transformation tension built and resolved?
- Product authority: How does the creator establish skincare credibility?
- Aspirational identity: What version of themselves does this help viewers imagine?
""",
        'general': """
GENERAL CONTENT PSYCHOLOGY FOCUS:
- Universal appeal mechanisms
- Cross-demographic engagement triggers
- Broad psychological satisfaction elements
"""
    }
    
    return {
        'primary_niche': primary_niche,
        'content_type': 'routine' if 'routine' in combined_text else 'transformation',
        'analysis_focus': analysis_focuses.get(primary_niche, analysis_focuses['general'])
    }



def create_visual_enhanced_fallback(frames_summaries_text, transcript_data, goal):
    """Enhanced fallback for when GPT analysis fails on visual content."""
    
    visual_analysis = create_visual_content_description(frames_summaries_text, None)
    text_sync_analysis = analyze_text_synchronization(frames_summaries_text, transcript_data.get('transcript', ''))
    
    analysis = f"Visual content analysis: {visual_analysis['description']}. "
    
    if not transcript_data['is_reliable']:
        analysis += f"Audio consists of ambient activity sounds rather than speech. "
    
    if text_sync_analysis['has_graphics'] or text_sync_analysis['has_captions']:
        analysis += f"{text_sync_analysis['text_analysis_summary']}. "
    
    analysis += "The content uses visual engagement and process satisfaction to maintain viewer attention."
    
    # Generate appropriate hooks based on content type
    if visual_analysis.get('content_type') == 'visual_process':
        if 'coloring' in frames_summaries_text.lower():
            hooks = [
                "this drawing technique is blowing everyone's mind",
                "watch this simple outline become something amazing",
                "the way this transforms will shock you",
                "POV: you discover the most satisfying art method",
                "this drawing hack changed everything for me"
            ]
        elif 'skincare' in frames_summaries_text.lower():
            hooks = [
                "this skincare routine is going viral for a reason",
                "watch my skin transform with these 3 steps",
                "the glow up is real with this routine",
                "POV: you finally find a routine that works",
                "this is why my skin looks like this"
            ]
        else:
            hooks = [
                "this process is oddly satisfying",
                "watch this transformation happen",
                "the end result will amaze you",
                "POV: you discover the perfect method",
                "this technique is pure satisfaction"
            ]
    else:
        hooks = [
            "wait until you see how this ends",
            "this process is mesmerizing",
            "the transformation is incredible",
            "you won't believe the final result",
            "this is so satisfying to watch"
        ]
    
    return {
        "analysis": analysis,
        "hooks": hooks,
        "scores": {
            "hook_strength": 7,
            "promise_clarity": 8 if visual_analysis.get('has_strong_visual_narrative') else 6,
            "retention_design": 8,
            "engagement_potential": 9 if visual_analysis.get('satisfaction_analysis', {}).get('highly_satisfying') else 7,
            "goal_alignment": 7
        },
        "timing_breakdown": "Visual progression builds anticipation from setup through completion",
        "formula": "Visual hook → Process demonstration → Transformation delivery → Satisfying conclusion",
        "basic_formula": "1. Show engaging setup 2. Demonstrate process 3. Build anticipation 4. Deliver satisfying result",
        "timing_formula": "0-3s: Visual hook, 3-10s: Process setup, Middle: Transformation, End: Final result",
        "template_formula": "[Visual Hook] → [Process Setup] → [Transformation] → [Satisfying Result]",
        "psychology_formula": "Visual attention → Process fascination → Anticipation → Completion satisfaction",
        "improvements": f"Enhance visual clarity, consider adding text overlays for context, optimize pacing for {goal}",
        "performance_prediction": "Strong visual retention expected from satisfying process content",
        "visual_content_analysis": visual_analysis,
        "transcript_quality": transcript_data,
        "text_sync_analysis": text_sync_analysis
    }

# Add these missing functions to your app.py:

def create_enhanced_analysis_prompt(transcript_text, frames_summaries_text, creator_note, video_description, content_themes, goal, performance_data, performance_context, dual_engagement_note, text_sync_analysis):
    """Create a much more sophisticated analysis prompt that delivers richer insights."""
    
    # Detect specific niche and content type for targeted analysis
    niche_context = detect_specific_niche(transcript_text, frames_summaries_text)
    
    # Add text synchronization context
    text_context = ""
    if text_sync_analysis['has_graphics'] or text_sync_analysis['has_captions']:
        text_context = f"""
TEXT ANALYSIS:
{text_sync_analysis['text_analysis_summary']}
On-screen Graphics: {[g['text'] for g in text_sync_analysis['onscreen_graphics']]}
Synchronized Captions: {[c['text'] for c in text_sync_analysis['synchronized_captions']]}
This affects retention through visual-verbal coordination and information layering.
"""
    
    prompt = f"""
You are a world-class retention psychology expert and viral content strategist with deep understanding of human behavior, social media psychology, and platform-specific mechanics. Your analysis should rival the depth of top marketing psychologists.

VIDEO CONTENT ANALYSIS:
TRANSCRIPT (What they say): {transcript_text}
VISUAL FRAMES (What viewers see): {frames_summaries_text}
CREATOR NOTE: {creator_note}
VIDEO DESCRIPTION: {video_description}
NICHE: {niche_context['primary_niche']} | CONTENT TYPE: {niche_context['content_type']}
GOAL: {goal} | PERFORMANCE: {performance_data.get('success_level', 'unknown')}
{text_context}
{dual_engagement_note}
{performance_context}

FRAMEWORK FOR DEEP PSYCHOLOGICAL ANALYSIS:

1. HOOK PSYCHOLOGY DECONSTRUCTION:
- What SPECIFIC psychological trigger does the opening use? (Curiosity gap, pattern interrupt, social proof, controversy, transformation promise)
- How does the opening work psychologically? Break down each word's impact
- What deeper emotional need does this tap into? (confidence, control, transformation, belonging)
- Why is this particular phrasing effective for this audience vs generic alternatives?

2. PROMISE STRUCTURE ANALYSIS:
- What's the explicit promise vs the implicit emotional payoff?
- How does the promise create "must-watch" psychology?
- What anticipation loops are created and how are they resolved?
- How does the promise align with the audience's deepest desires?

3. RETENTION MECHANISM DEEP DIVE:
- VISUAL SATISFACTION: What makes the process visually addictive? (completion, precision, transformation)
- DUAL ENGAGEMENT: How do satisfying visuals work WITH verbal content?
- TEXT COORDINATION: How do on-screen graphics enhance or compete with spoken content?
- PACING PSYCHOLOGY: How does information release create dopamine hits?
- SOCIAL PROOF SIGNALS: What authority/credibility markers are embedded?

4. EMOTIONAL ARCHITECTURE:
- What emotional journey does the viewer experience? (Map second by second)
- How does the creator make viewers feel about THEMSELVES?
- What aspirational identity is triggered?
- How does this content make viewers feel seen/understood?

5. PLATFORM-NATIVE COMMUNICATION:
- Why does this sound authentically TikTok vs scripted?
- What vocabulary/energy choices create trust?
- How does casualness enhance rather than diminish authority?
- What makes this shareable in authentic peer-to-peer conversations?

6. REPLICATION BLUEPRINT:
- What are the EXACT structural elements that must be preserved?
- Which surface elements can be varied while keeping psychology intact?
- How do timing, energy, and delivery impact effectiveness?
- What would break the formula if changed?

7. ADVANCED ENGAGEMENT PSYCHOLOGY:
- Comment triggers: What specific elements make people want to engage?
- Save psychology: What makes this feel valuable enough to return to?
- Share motivation: Why would someone send this to a friend?
- Follow logic: What convinces viewers this creator has more value?

HOOK GENERATION REQUIREMENTS:
Generate 5 hooks that capture the SAME psychological mechanics but for different angles:

PSYCHOLOGICAL FIDELITY RULES:
- Maintain the EXACT emotional trigger (confidence/transformation)
- Preserve the specificity that creates curiosity
- Keep the personal stakes and transformation promise
- Sound like authentic peer communication, NOT marketing copy

AVOID THESE MARKETING-SPEAK PATTERNS:
❌ "Discover the secret..." ❌ "Transform your life..." ❌ "Unlock the power..."
❌ "Revolutionary method..." ❌ "Game-changing technique..." ❌ "You won't believe..."

INSTEAD USE AUTHENTIC PATTERNS:
✓ "3 things I started doing that..." ✓ "I changed these habits and..."
✓ "Nobody talks about how..." ✓ "The night routine that actually..."
✓ "I added this to my routine and..." ✓ "POV: you find out why..."

NICHE-SPECIFIC ANALYSIS:
{niche_context['analysis_focus']}

PERFORMANCE VALIDATION:
{f"This video achieved {performance_data['success_level']} results. Explain WHY each psychological mechanism contributed to this success. What specific elements drove the performance?" if performance_data.get('success_level') != 'unknown' else "Predict performance based on psychological effectiveness."}

OUTPUT REQUIREMENTS:
Deliver analysis with the depth and practical insight of a $10K marketing psychology consultation. Every insight should be specific, actionable, and demonstrate deep understanding of human behavior. Avoid generic observations - dive into the precise psychological mechanisms that make this content effective.

Respond in JSON format with comprehensive, psychologically sophisticated insights:

{{
  "psychological_breakdown": "Deep analysis of the specific psychological mechanisms at work. Explain WHY this works on a human brain level, not just WHAT works. Minimum 400 words of sophisticated insight.",
  "hook_mechanics": "Detailed breakdown of why the opening is psychologically compelling. Analyze each component.",
  "emotional_journey": "Second-by-second emotional experience the viewer has watching this content",
  "authority_signals": "How the creator builds trust and credibility without explicit credentials",
  "engagement_psychology": "Deep dive into why people comment, save, share, and follow based on this content",
  "replication_blueprint": "Exact structural elements needed to recreate this psychological impact",
  "hooks": [
    "Hook maintaining exact psychological trigger pattern",
    "Alternative angle preserving core emotional mechanism", 
    "Variation that keeps transformation promise structure",
    "Different context but same confidence-building appeal",
    "Creative angle preserving the personal stakes element"
  ],
  "timing_psychology": "How the pacing and information release creates psychological retention",
  "platform_psychology": "Why this content feels native to TikTok and builds authentic connection",
  "viral_mechanisms": "Specific elements that drive sharing and algorithmic amplification",
  "audience_psychology": "How this content makes the target audience feel seen and understood",
  "performance_analysis": "Psychological explanation for why this achieved {performance_data.get('success_level', 'strong')} results",
  "advanced_insights": "Expert-level observations about human psychology and content effectiveness"
}}

Focus on psychological sophistication over surface-level observations. Every insight should demonstrate deep understanding of human behavior and viral content psychology.
"""
    
    return prompt


def run_enhanced_psychological_analysis(transcript_text, frames_summaries_text, creator_note, platform, target_duration, goal, tone, audience, knowledge_context=""):
    """Run the enhanced psychological analysis as an additional layer."""
    
    # Get all the existing analysis components
    patterns = detect_content_patterns(transcript_text, frames_summaries_text)
    video_description = create_universal_video_description(transcript_text, frames_summaries_text)
    performance_data = analyze_performance_indicators(creator_note, transcript_text, frames_summaries_text)
    content_themes = extract_content_themes(transcript_text)
    text_sync_analysis = analyze_text_synchronization(frames_summaries_text, transcript_text)
    
    # Build performance context
    performance_context = ""
    if performance_data['success_level'] != "unknown":
        performance_context = f"""
PERFORMANCE CONTEXT:
This video achieved: {', '.join(performance_data['success_reasons'])}
Success Level: {performance_data['success_level']}
Use this performance data to validate your analysis - explain WHY this video achieved this level of success based on psychological mechanisms.
        """
    
    # Build dual engagement note
    dual_engagement_note = ""
    if patterns.get('dual_engagement', False):
        dual_engagement_note = "\n🎯 DUAL ENGAGEMENT DETECTED: This video combines satisfying visual processes with verbal content delivery - analyze how both channels work together for retention."
    
    # Create the enhanced prompt
    prompt = create_enhanced_analysis_prompt(
        transcript_text, frames_summaries_text, creator_note, 
        video_description, content_themes, goal, performance_data, 
        performance_context, dual_engagement_note, text_sync_analysis
    )
    
    try:
        print(f"Sending enhanced psychological analysis prompt to GPT-4o...")
        gpt_response = _api_retry(
            client.chat.completions.create,
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Lower temperature for more focused analysis
            max_tokens=4000   # More tokens for richer analysis
        )

        response_text = gpt_response.choices[0].message.content.strip()
        
        # Parse JSON response
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        parsed = json.loads(response_text)
        
        # Return enhanced analysis structure
        return {
            "analysis": parsed.get("psychological_breakdown", "Analysis not available").strip(),
            "hooks": parsed.get("hooks", []),
            "scores": {
                "hook_strength": 9,  # Higher default for successful content
                "promise_clarity": 8,
                "retention_design": 9,
                "engagement_potential": 9,
                "goal_alignment": 8
            },
            "timing_breakdown": parsed.get("timing_psychology", "").strip(),
            "formula": parsed.get("replication_blueprint", "").strip(),
            "basic_formula": parsed.get("replication_blueprint", "").strip(),
            "timing_formula": parsed.get("timing_psychology", "").strip(),
            "template_formula": parsed.get("replication_blueprint", "").strip(),
            "psychology_formula": parsed.get("platform_psychology", "").strip(),
            "improvements": parsed.get("advanced_insights", "").strip(),
            "performance_prediction": parsed.get("performance_analysis", "").strip(),
            "video_description": video_description,
            "content_patterns": patterns,
            "performance_data": performance_data,
            "text_sync_analysis": text_sync_analysis,
            
            # Enhanced analysis fields
            "psychological_breakdown": parsed.get("psychological_breakdown", "").strip(),
            "hook_mechanics": parsed.get("hook_mechanics", "").strip(),
            "emotional_journey": parsed.get("emotional_journey", "").strip(),
            "authority_signals": parsed.get("authority_signals", "").strip(),
            "engagement_psychology": parsed.get("engagement_psychology", "").strip(),
            "viral_mechanisms": parsed.get("viral_mechanisms", "").strip(),
            "audience_psychology": parsed.get("audience_psychology", "").strip(),
            "multimodal_insights": parsed.get("platform_psychology", "").strip(),
            "engagement_triggers": parsed.get("engagement_psychology", "").strip(),
            "improvement_opportunities": parsed.get("advanced_insights", "").strip(),
            "viral_potential_factors": parsed.get("viral_mechanisms", "").strip()
        }
        
    except Exception as e:
        print(f"Enhanced analysis error: {e}")
        return None  # Return None so original analysis can be used

# ==============================
# FLASK ROUTES
# ==============================

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze_async", methods=["POST"])
def analyze_async():
    return render_template("progress.html", form_data=request.form.to_dict())


@app.route("/process", methods=["POST"])
def process():
    try:
        # --- Form data ---
        tiktok_url = request.form.get("tiktok_url", "").strip()
        creator_note = request.form.get("creator_note", "").strip()
        strategy = request.form.get("strategy", "smart").strip()
        frames_per_minute = int(request.form.get("frames_per_minute", 24))
        cap = int(request.form.get("cap", 60))
        scene_threshold = float(request.form.get("scene_threshold", 0.24))
        platform = request.form.get("platform", "tiktok").strip()
        target_duration = request.form.get("target_duration", "30").strip()
        goal = request.form.get("goal", "follower_growth").strip()
        tone = request.form.get("tone", "confident, friendly").strip()
        audience = request.form.get("audience", "creators and small business owners").strip()

        if not tiktok_url:
            return "Error: TikTok URL is required", 400

        print(f"Processing: {tiktok_url}")
        print(f"Strategy: {strategy}, Goal: {goal}")

        # --- Enhanced Video processing with better frame distribution ---
        try:
            audio_path, frames_dir, frame_paths = enhanced_extract_audio_and_frames(
                tiktok_url,
                strategy=strategy,
                frames_per_minute=frames_per_minute,
                cap=cap,
                scene_threshold=scene_threshold,
            )
            print(f"Extracted {len(frame_paths)} frames with improved distribution")
        except Exception as e:
            print(f"Video processing error: {e}")
            return f"Error processing video: {str(e)}", 500

        # --- Enhanced Transcription with quality analysis ---
        try:
            transcript_data = enhanced_transcribe_audio(audio_path)
            print(f"Transcript quality: {transcript_data['quality']} - {transcript_data['quality_reason']}")
            
            if not transcript_data['is_reliable']:
                print(f"[warning] Transcript appears unreliable: {transcript_data['quality_reason']}")
        except Exception as e:
            print(f"Transcription error: {e}")
            transcript_data = {
                'transcript': "(Transcription failed)",
                'quality': 'error',
                'quality_reason': str(e),
                'is_reliable': False
            }

        # --- Frame analysis ---
        try:
            frames_summaries_text, gallery_data_urls = analyze_frames_batch(frame_paths)
            print(f"Frame analysis complete, gallery has {len(gallery_data_urls)} images")
            print(f"Frame analysis preview: {frames_summaries_text[:200]}...")
        except Exception as e:
            print(f"Frame analysis error: {e}")
            frames_summaries_text = "(Frame analysis failed)"
            gallery_data_urls = []

        # --- Skip RAG for now to focus on retention analysis ---
        knowledge_context = ""
        knowledge_citations = []

        # --- Enhanced AI Analysis for visual content ---
        try:
            # Generate audio description if transcript is unreliable
            audio_description = None
            if not transcript_data['is_reliable']:
                audio_description = generate_inferred_audio_description(
                    frames_summaries_text, 
                    (transcript_data['quality'], transcript_data['quality_reason'])
                )
            
            # Create enhanced visual content description
            visual_content_analysis = create_visual_content_description(
                frames_summaries_text, 
                audio_description
            )
            
            # Use inferred description or original transcript
            transcript_for_analysis = audio_description if audio_description else transcript_data['transcript']
            
            # Run enhanced analysis (tries psychological first, falls back to comprehensive)
            gpt_result = run_enhanced_gpt_analysis(
                transcript_for_analysis,
                frames_summaries_text,
                creator_note,
                platform,
                target_duration,
                goal,
                tone,
                audience,
                knowledge_context
            )
            
            # Add visual analysis results
            gpt_result['visual_content_analysis'] = visual_content_analysis
            gpt_result['transcript_quality'] = transcript_data
            gpt_result['audio_description'] = audio_description
            
            print("Enhanced retention analysis complete")
            
        except Exception as e:
            print(f"GPT analysis error: {e}")
            gpt_result = create_visual_enhanced_fallback(
                frames_summaries_text,
                transcript_data,
                goal
            )

        # --- Extract ALL results ---
        analysis_text = gpt_result.get("analysis", "Analysis not available")
        hooks_list = gpt_result.get("hooks", [])
        scores = gpt_result.get("scores", {})
        timing_breakdown = gpt_result.get("timing_breakdown", "")
        formula = gpt_result.get("formula", "")
        basic_formula = gpt_result.get("basic_formula", "")
        timing_formula = gpt_result.get("timing_formula", "")
        template_formula = gpt_result.get("template_formula", "")
        psychology_formula = gpt_result.get("psychology_formula", "")
        improvements = gpt_result.get("improvements", "")
        video_description = gpt_result.get("video_description", "Video analysis")
        content_patterns = gpt_result.get("content_patterns", {})
        performance_data = gpt_result.get("performance_data", {})
        
        # Enhanced fields
        visual_content_analysis = gpt_result.get("visual_content_analysis", {})
        transcript_quality = gpt_result.get("transcript_quality", {})
        audio_description = gpt_result.get("audio_description", "")
        text_sync_analysis = gpt_result.get("text_sync_analysis", {})
        
        # Enhanced psychological analysis fields
        psychological_breakdown = gpt_result.get("psychological_breakdown", "")
        hook_mechanics = gpt_result.get("hook_mechanics", "")
        emotional_journey = gpt_result.get("emotional_journey", "")
        authority_signals = gpt_result.get("authority_signals", "")
        engagement_psychology = gpt_result.get("engagement_psychology", "")
        viral_mechanisms = gpt_result.get("viral_mechanisms", "")
        audience_psychology = gpt_result.get("audience_psychology", "")
        
        # Rich analysis fields (available in both enhanced and original)
        multimodal_insights = gpt_result.get("multimodal_insights", "")
        engagement_triggers = gpt_result.get("engagement_triggers", "")
        improvement_opportunities = gpt_result.get("improvement_opportunities", "")
        viral_potential_factors = gpt_result.get("viral_potential_factors", "")
        
        # Update video description with enhanced analysis
        if visual_content_analysis and visual_content_analysis.get('description'):
            video_description = visual_content_analysis.get('description', video_description)
        
        # Use enhanced transcript for analysis
        transcript_for_analysis = audio_description if audio_description else transcript_data.get('transcript', '')
        
        if isinstance(hooks_list, str):
            hooks_list = [hooks_list]

        # --- Prepare frame summaries for template ---
        frame_summaries = []
        if frames_summaries_text:
            blocks = frames_summaries_text.split('\n\n')
            frame_summaries = [block.strip() for block in blocks if block.strip()]
        
        if not frame_summaries and frames_summaries_text:
            frame_summaries = [frames_summaries_text]

        print("Rendering enhanced results template")
        return render_template(
            "results.html",
            tiktok_url=tiktok_url,
            creator_note=creator_note,
            transcript=transcript_for_analysis,  # This will be the cleaned/inferred version
            frame_summary=frames_summaries_text,
            frame_summaries=frame_summaries,
            frame_gallery=gallery_data_urls,
            strategy=strategy,
            frames_per_minute=frames_per_minute,
            cap=cap,
            scene_threshold=scene_threshold,
            frames_count=len(frame_paths) if frame_paths else 0,
            platform=platform,
            target_duration=target_duration,
            goal=goal,
            tone=tone,
            audience=audience,
            knowledge_citations=knowledge_citations,
            knowledge_context=knowledge_context,
            frames_dir=frames_dir,
            frame_paths=frame_paths,
            analysis=analysis_text,
            hooks=hooks_list,
            scores=scores,
            timing_breakdown=timing_breakdown,
            formula=formula,
            basic_formula=basic_formula,
            timing_formula=timing_formula,
            template_formula=template_formula,
            psychology_formula=psychology_formula,
            improvements=improvements,
            video_description=video_description,
            content_patterns=content_patterns,
            performance_data=performance_data,
            
            # Enhanced fields
            visual_content_analysis=visual_content_analysis,
            transcript_quality=transcript_quality,
            audio_description=audio_description,
            transcript_original=transcript_data.get('transcript', ''),
            transcript_for_analysis=transcript_for_analysis,
            text_sync_analysis=text_sync_analysis,
            
            # Enhanced psychological analysis fields
            psychological_breakdown=psychological_breakdown,
            hook_mechanics=hook_mechanics,
            emotional_journey=emotional_journey,
            authority_signals=authority_signals,
            engagement_psychology=engagement_psychology,
            viral_mechanisms=viral_mechanisms,
            audience_psychology=audience_psychology,
            
            # Rich analysis fields (available in both modes)
            multimodal_insights=multimodal_insights,
            engagement_triggers=engagement_triggers,
            improvement_opportunities=improvement_opportunities,
            viral_potential_factors=viral_potential_factors,
            
            # Keep these for backward compatibility
            gpt_response=analysis_text
        )

    except Exception as e:
        print(f"Unexpected error in process(): {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Unexpected error: {str(e)}", 500
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)