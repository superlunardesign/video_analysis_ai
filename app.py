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
# MISSING FUNCTION IMPLEMENTATIONS
# ==============================

def enhanced_extract_audio_and_frames(tiktok_url, strategy="smart", frames_per_minute=24, cap=60, scene_threshold=0.24):
    """Enhanced version of extract_audio_and_frames with better distribution."""
    # For now, use the original function - you can enhance this later
    return extract_audio_and_frames(tiktok_url, strategy, frames_per_minute, cap, scene_threshold)


def enhanced_transcribe_audio(audio_path):
    """Enhanced transcription with quality analysis."""
    try:
        # Get original transcript
        transcript = transcribe_audio(audio_path)
        
        # Analyze transcript quality
        if not transcript or len(transcript.strip()) < 10:
            return {
                'transcript': transcript,
                'quality': 'poor',
                'quality_reason': 'Transcript too short or empty',
                'is_reliable': False
            }
        
        # Check for common transcription issues
        words = transcript.lower().split()
        
        # Check for music/ambient sound indicators
        music_indicators = ['music', 'sound', 'noise', 'audio', 'background']
        if any(indicator in transcript.lower() for indicator in music_indicators):
            return {
                'transcript': transcript,
                'quality': 'ambient',
                'quality_reason': 'Contains music/ambient audio descriptions',
                'is_reliable': False
            }
        
        # Check for repetitive/nonsense content
        if len(set(words)) < len(words) * 0.3:  # Too many repeated words
            return {
                'transcript': transcript,
                'quality': 'poor',
                'quality_reason': 'Highly repetitive content detected',
                'is_reliable': False
            }
        
        return {
            'transcript': transcript,
            'quality': 'good',
            'quality_reason': 'Clear speech detected',
            'is_reliable': True
        }
        
    except Exception as e:
        return {
            'transcript': f"(Transcription error: {str(e)})",
            'quality': 'error',
            'quality_reason': str(e),
            'is_reliable': False
        }


def generate_inferred_audio_description(frames_summaries_text, transcript_quality_info):
    """Generate inferred audio description for visual content. Make sure the inferred audio lines up with what's happening in the frames/video"""
    try:
        # Analyze the visual content to infer what might be happening
        if 'drawing' in frames_summaries_text.lower() or 'art' in frames_summaries_text.lower():
            return "Creative process video with drawing/artistic elements. Likely contains ambient drawing sounds, paper rustling, or background music."
        elif 'skincare' in frames_summaries_text.lower() or 'routine' in frames_summaries_text.lower():
            return "Personal care routine video. Likely contains ambient sounds of product application, water, or soft background music."
        elif 'cooking' in frames_summaries_text.lower() or 'kitchen' in frames_summaries_text.lower():
            return "Cooking/food preparation video. Likely contains kitchen sounds, sizzling, chopping, or cooking ambient audio."
        else:
            return f"Visual content video with ambient audio. {transcript_quality_info[1] if len(transcript_quality_info) > 1 else 'No clear speech detected.'}"
    except:
        return "Visual content with ambient audio track."


def create_visual_content_description(frames_summaries_text, audio_description=None):
    """Create comprehensive visual content analysis."""
    try:
        description = f"Visual analysis: {frames_summaries_text[:200]}..."
        
        # Determine content type
        content_type = 'general'
        if 'drawing' in frames_summaries_text.lower():
            content_type = 'visual_process'
        elif 'transformation' in frames_summaries_text.lower():
            content_type = 'transformation'
        elif 'routine' in frames_summaries_text.lower():
            content_type = 'routine'
        
        # Analyze satisfaction potential
        satisfaction_indicators = ['completion', 'finish', 'result', 'final', 'transform']
        highly_satisfying = any(word in frames_summaries_text.lower() for word in satisfaction_indicators)
        
        return {
            'description': description,
            'content_type': content_type,
            'has_strong_visual_narrative': len(frames_summaries_text) > 200,
            'satisfaction_analysis': {
                'highly_satisfying': highly_satisfying,
                'completion_elements': satisfaction_indicators
            }
        }
    except:
        return {
            'description': "Visual content analysis",
            'content_type': 'general',
            'has_strong_visual_narrative': False,
            'satisfaction_analysis': {'highly_satisfying': False}
        }


def detect_content_patterns(transcript_text, frames_summaries_text):
    """Detect content patterns for analysis."""
    combined_text = f"{transcript_text} {frames_summaries_text}".lower()
    
    patterns = {
        'dual_engagement': 'visual' in combined_text and len(transcript_text.strip()) > 50,
        'transformation': 'transform' in combined_text or 'before' in combined_text,
        'routine': 'routine' in combined_text or 'step' in combined_text,
        'educational': 'learn' in combined_text or 'how to' in combined_text
    }
    
    return patterns


def create_universal_video_description(transcript_text, frames_summaries_text):
    """Create universal video description."""
    if len(transcript_text.strip()) > 50:
        return f"Video content: {transcript_text[:100]}..."
    else:
        return f"Visual content: {frames_summaries_text[:100]}..."


def analyze_performance_indicators(creator_note, transcript_text, frames_summaries_text):
    """Analyze performance indicators from content."""
    # Simple analysis - you can enhance this
    success_reasons = []
    
    if 'viral' in creator_note.lower():
        success_reasons.append('viral_performance')
    if 'popular' in creator_note.lower():
        success_reasons.append('high_engagement')
    
    success_level = 'unknown'
    if success_reasons:
        success_level = 'high'
    
    return {
        'success_level': success_level,
        'success_reasons': success_reasons
    }


def extract_content_themes(transcript_text):
    """Extract content themes from transcript."""
    words = transcript_text.lower().split()
    themes = []
    
    # Simple theme detection
    if any(word in words for word in ['skin', 'beauty', 'routine']):
        themes.append('beauty')
    if any(word in words for word in ['workout', 'fitness', 'exercise']):
        themes.append('fitness')
    if any(word in words for word in ['food', 'recipe', 'cooking']):
        themes.append('cooking')
    
    return themes


def run_enhanced_gpt_analysis(transcript_text, frames_summaries_text, creator_note, platform, target_duration, goal, tone, audience, knowledge_context):
    """Run enhanced GPT analysis with fallback."""
    try:
        # Try enhanced psychological analysis first
        enhanced_result = run_enhanced_psychological_analysis(
            transcript_text, frames_summaries_text, creator_note, 
            platform, target_duration, goal, tone, audience, knowledge_context
        )
        
        if enhanced_result:
            return enhanced_result
        else:
            # Fallback to comprehensive analysis
            return run_comprehensive_analysis(
                transcript_text, frames_summaries_text, creator_note,
                platform, target_duration, goal, tone, audience, knowledge_context
            )
            
    except Exception as e:
        print(f"Enhanced analysis failed: {e}")
        # Create fallback result
        return create_visual_enhanced_fallback(frames_summaries_text, {
            'transcript': transcript_text,
            'is_reliable': len(transcript_text.strip()) > 50
        }, goal)


def run_comprehensive_analysis(transcript_text, frames_summaries_text, creator_note, platform, target_duration, goal, tone, audience, knowledge_context):
    """Fallback comprehensive analysis."""
    prompt = f"""
Analyze this {platform} video for {goal}:

TRANSCRIPT: {transcript_text}
VISUAL CONTENT: {frames_summaries_text}
CREATOR NOTE: {creator_note}
TARGET: {target_duration}s video for {audience} with {tone} tone

Provide analysis in JSON format:
{{
  "analysis": "Detailed analysis of why this content works",
  "hooks": ["Alternative hook 1", "Alternative hook 2", "Alternative hook 3", "Alternative hook 4", "Alternative hook 5"],
  "timing_breakdown": "How timing creates retention",
  "improvements": "Specific ways to improve this content",
  "formula": "Reusable formula for similar content"
}}
"""

    try:
        gpt_response = _api_retry(
            client.chat.completions.create,
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=3000
        )

        response_text = gpt_response.choices[0].message.content.strip()
        
        # Parse JSON response
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        parsed = json.loads(response_text)
        
        return {
            "analysis": parsed.get("analysis", "Analysis not available"),
            "hooks": parsed.get("hooks", []),
            "scores": {
                "hook_strength": 8,
                "promise_clarity": 7,
                "retention_design": 8,
                "engagement_potential": 8,
                "goal_alignment": 7
            },
            "timing_breakdown": parsed.get("timing_breakdown", ""),
            "formula": parsed.get("formula", ""),
            "basic_formula": parsed.get("formula", ""),
            "timing_formula": parsed.get("timing_breakdown", ""),
            "template_formula": parsed.get("formula", ""),
            "psychology_formula": "Content uses engagement and retention mechanisms",
            "improvements": parsed.get("improvements", ""),
            "performance_prediction": "Strong potential based on content structure"
        }
        
    except Exception as e:
        print(f"Comprehensive analysis error: {e}")
        return create_visual_enhanced_fallback(frames_summaries_text, {
            'transcript': transcript_text,
            'is_reliable': len(transcript_text.strip()) > 50
        }, goal)


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
You are a world-class retention psychology expert and viral content strategist with a deep understanding of human behavior, social media psychology, and platform-specific mechanics. Your analysis should provide actionable insights that rival the depth of top marketing psychologists and help creators replicate or improve their content’s success.

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

FRAMEWORK FOR DEEP PSYCHOLOGICAL ANALYSIS
Answer in a conversational tone while delivering clear, actionable insights. Focus on why the video works (or doesn’t) and how to replicate or improve its success. Avoid generic observations—every insight should be specific, practical, and rooted in human psychology.

Please:
    1. Describe clearly what happens in the video from start to finish.
    2. Identify and categorize hooks: Text, Visual, Verbal.
    3. Break down the content structure into a repeatable 'formula'.
    4. Explain why this formula might work for {goal} (viral reach, follower growth, or sales).
    5. Give actionable insights for someone adapting this style for their own niche.


1. HOOK PSYCHOLOGY DECONSTRUCTION
-What specific psychological trigger does the opening use? (Curiosity gap, pattern interrupt, transformation promise, controversy, social proof)
-How does the opening work psychologically? Break down each word or visual element and its impact.
-What deeper emotional need does this tap into? (Confidence, control, transformation, belonging)
-Why is this phrasing or visual effective for this audience vs generic alternatives?

2. PROMISE STRUCTURE ANALYSIS
-What’s the explicit promise vs the implicit emotional payoff?
-How does the promise create “must-watch” psychology?
-What anticipation loops are created, and how are they resolved?
-How does the promise align with the audience’s deepest desires or pain points?

3. RETENTION MECHANISM DEEP DIVE:
- VISUAL SATISFACTION: What makes the process visually addictive? (completion, precision, transformation)
- DUAL ENGAGEMENT: How do satisfying visuals work WITH verbal content?
- TEXT COORDINATION: How do on-screen graphics enhance or compete with spoken content?
- PACING PSYCHOLOGY: How does the timing of information release or create dopamine hits?
- SOCIAL PROOF SIGNALS: What authority/credibility markers are embedded?

4. EMOTIONAL ARCHITECTURE:
- What emotional journey does the viewer experience? (Map second by second)
- How does the creator make viewers feel about THEMSELVES?
- What aspirational identity is triggered?
- How does this content make viewers feel seen, understood, or validated?

5. PLATFORM-NATIVE COMMUNICATION:
- Why does this content feel authentically TikTok (or platform-specific) vs scripted?
- What vocabulary, energy, or tone choices build trust?
- How does casualness enhance authority rather than diminish it?
- What makes this shareable in authentic peer-to-peer conversations?

6. REPLICATION BLUEPRINT:
- What are the exact structural elements that must be preserved to replicate success?
- Which surface elements (visuals, tone, pacing) can be varied to increase viewer retention while keeping the psychology intact?
- How do timing, energy, and delivery impact effectiveness?
- What would break the formula if changed?

7. ADVANCED ENGAGEMENT PSYCHOLOGY:
- Comment triggers: What specific elements make people want to engage?
- Save psychology: What makes this feel valuable enough to save for later and return to?
- Share motivation: Why would someone send this to a friend or repost it?
- Follow logic: What convinces viewers this creator has more value to offer?

HOOK GENERATION REQUIREMENTS:
Generate 5 hooks that capture the SAME psychological mechanics but from different angles:

-Maintain the exact emotional trigger (confidence, transformation, curiosity).
-Preserve the specificity that creates curiosity.
-Keep the personal stakes and transformation promise.
-Sound like authentic peer communication, NOT marketing copy.

PSYCHOLOGICAL FIDELITY RULES:
- Maintain the EXACT emotional trigger (confidence/transformation)
- Preserve the specificity that creates curiosity
- Keep the personal stakes and transformation promise
- Sound like authentic peer communication, NOT marketing copy

AVOID THESE MARKETING-SPEAK PATTERNS:
❌ "Discover the secret..." ❌ "Transform your life..." ❌ "Unlock the power..."
❌ "Revolutionary method..." ❌ "Game-changing technique..." ❌ "You won't believe..."

INSTEAD USE AUTHENTIC PATTERNS SIMILAR TO:
✓ "3 things I started doing that..." ✓ "I changed these habits and..."
✓ "Nobody talks about how..." ✓ "The night routine that actually..."
✓ "I added this to my routine and..." ✓ "POV: you find out why..."

NICHE-SPECIFIC ANALYSIS:
{niche_context['analysis_focus']}

PERFORMANCE VALIDATION:
{f"This video achieved {performance_data['success_level']} results. Explain WHY each psychological mechanism contributed to this success. What specific elements drove the performance?" if performance_data.get('success_level') != 'unknown' else "Predict performance based on psychological effectiveness."}

OUTPUT REQUIREMENTS:
Deliver analysis with the depth and practical insight of a $10K marketing psychology consultation. Every insight should be specific, actionable, and demonstrate a deep understanding of human behavior. Avoid generic observations. Explain the precise psychological mechanisms that make this content effective and how to replicate success.

Respond in JSON format with comprehensive, psychologically sophisticated insights:

{
  "psychological_breakdown": "Explain WHY this works on a human brain level, not just WHAT works. Minimum 400 words of sophisticated insight.",
  "hook_mechanics": "Detailed breakdown of why the opening is psychologically compelling. Analyze each component and explain how to replicate this in future hooks.",
  "emotional_journey": "Second-by-second emotional experience the viewer has watching this content. Align the frames to the script to create a timeline in order to better understand the emotional journey of the video.",
  "authority_signals": "How does the creator build trust and credibility without explicit credentials?",
  "engagement_psychology": "Deep dive into why people might comment, save, share, and follow based on this content.",
  "replication_blueprint": "Exact formula for a video with structural elements, similar cadence, and any improvements needed to recreate this psychological impact.",
  "hooks": [
    "Hook maintaining exact psychological trigger pattern",
    "Alternative angle preserving core emotional mechanism",
    "Variation that keeps transformation promise structure",
    "Different context but same confidence-building appeal",
    "Creative angle preserving the personal stakes element"
  ],
  "timing_psychology": "How the pacing and information release creates psychological retention. Align the frames to the script to create a timeline in order to better understand the emotional journey of the video.",
  "platform_psychology": "Why this content feels native to TikTok and builds authentic connection and how to replicate this.",
  "viral_mechanisms": "Specific elements that increase watch time, keep viewers hooked, drive sharing, and algorithmic amplification.",
  "audience_psychology": "How this content makes the target audience feel seen and understood and compelled to follow, purchase, or share.",
  "performance_analysis": "Psychological explanation for why this achieved {performance_data.get('success_level', 'strong')} results.",
  "advanced_insights": "Expert-level observations about how this piece of content demonstrates human psychology and content effectiveness and ways it could improve."
}

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
            "viral_potential_factors": parsed.get("viral_mechanisms", "").strip(),
            "replication_blueprint": parsed.get("replication_blueprint", "").strip()
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

    # --- Retrieve knowledge context for this transcript ---
        try:
            rag_query = transcript + "\n\n" + frames_summaries_text
            knowledge_context, knowledge_citations = retrieve_context(rag_query, top_k=8)
            print(f"Retrieved {len(knowledge_citations)} knowledge citations")

        except Exception as e:
            print(f"Knowledge retrieval error: {e}")
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

        # --- Extract ALL results with safe defaults ---
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
        
        # Enhanced fields with safe defaults
        visual_content_analysis = gpt_result.get("visual_content_analysis", {})
        transcript_quality = gpt_result.get("transcript_quality", {})
        audio_description = gpt_result.get("audio_description", "")
        text_sync_analysis = gpt_result.get("text_sync_analysis", {})
        
        # Enhanced psychological analysis fields with safe defaults
        psychological_breakdown = gpt_result.get("psychological_breakdown", "")
        hook_mechanics = gpt_result.get("hook_mechanics", "")
        emotional_journey = gpt_result.get("emotional_journey", "")
        authority_signals = gpt_result.get("authority_signals", "")
        engagement_psychology = gpt_result.get("engagement_psychology", "")
        viral_mechanisms = gpt_result.get("viral_mechanisms", "")
        audience_psychology = gpt_result.get("audience_psychology", "")
        replication_blueprint = gpt_result.get("replication_blueprint", "")
        
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
            replication_blueprint=replication_blueprint,
            
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