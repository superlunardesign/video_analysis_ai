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
from rag_helper import retrieve_smart_context, retrieve_all_context

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=600.0)


def validate_dependencies():
    """Check if all required dependencies are available"""
    missing_deps = []
    
    try:
        from pypdf import PdfReader
    except ImportError:
        missing_deps.append("pypdf (for PDF processing)")
    
    try:
        from openai import OpenAI
    except ImportError:
        missing_deps.append("openai")
    
    if not os.getenv("OPENAI_API_KEY"):
        missing_deps.append("OPENAI_API_KEY environment variable")
    
    if missing_deps:
        print(f"WARNING: Missing dependencies: {', '.join(missing_deps)}")
        print("Some features may not work properly.")
    
    return len(missing_deps) == 0


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
# AUDIO/VIDEO ANALYSIS HELPERS
# ==============================

def analyze_audio_with_visual_context(transcript_text, frames_summaries_text):
    """
    Intelligently analyze audio by considering visual context
    Don't assume random sounds - correlate with what's happening visually
    """
    
    has_meaningful_speech = False
    transcript_quality = 'unknown'
    likely_sound_source = None
    audio_visual_correlation = {}
    
    if transcript_text and len(transcript_text.strip()) > 20:
        transcript_lower = transcript_text.lower()
        words = transcript_lower.split()
        unique_words = set(words)
        
        # Check for repetitive non-speech patterns
        repetitive_patterns = ['roar', 'ah', 'um', 'oh', 'hmm', 'purr', 'meow', 'woof', 
                              'buzz', 'hiss', 'click', 'tap', 'thud', 'swoosh', 'whoosh']
        
        # If transcript is mostly repetitive sounds
        if len(unique_words) <= 5 and any(pattern in transcript_lower for pattern in repetitive_patterns):
            has_meaningful_speech = False
            transcript_quality = 'ambient_sounds'
            
            # Now correlate with visual content
            frames_lower = frames_summaries_text.lower() if frames_summaries_text else ""
            
            # Visual-audio correlation mapping
            visual_audio_correlations = {
                # Art/Drawing activities
                ('drawing', 'marker', 'pen', 'pencil', 'sketch', 'art', 'illustrat', 'coloring'): 
                    'marker/pen on paper sounds',
                ('paint', 'brush', 'canvas', 'watercolor', 'acrylic'): 
                    'brush strokes and paint sounds',
                
                # Crafting activities
                ('cutting', 'scissor', 'paper', 'craft'): 
                    'cutting/crafting sounds',
                ('sewing', 'fabric', 'thread', 'stitch'): 
                    'sewing machine or fabric sounds',
                
                # Cooking activities
                ('cooking', 'sizzl', 'pan', 'stove', 'fry'): 
                    'cooking/sizzling sounds',
                ('mixing', 'bowl', 'whisk', 'stir'): 
                    'mixing/stirring sounds',
                
                # Beauty/Grooming
                ('makeup', 'brush', 'powder', 'foundation', 'skincare', 'routine'): 
                    'makeup/skincare application sounds',
                ('hair', 'brush', 'style', 'dry'): 
                    'hair styling sounds',
                
                # Packaging/Unboxing
                ('unbox', 'package', 'open', 'tape', 'box'): 
                    'packaging/unwrapping sounds',
                
                # Nature/Outdoor
                ('nature', 'outdoor', 'tree', 'wind'): 
                    'natural ambient sounds',
                
                # Cleaning/Organizing
                ('cleaning', 'organizing', 'folding', 'tidy'): 
                    'cleaning/organizing sounds'
            }
            
            # Find matching visual-audio correlation
            for visual_keywords, sound_description in visual_audio_correlations.items():
                if any(keyword in frames_lower for keyword in visual_keywords):
                    likely_sound_source = sound_description
                    audio_visual_correlation = {
                        'detected_activity': visual_keywords[0],
                        'likely_sound': sound_description,
                        'confidence': 'high'
                    }
                    break
            
            # If no specific match, make a general inference
            if not likely_sound_source:
                if any(word in frames_lower for word in ['process', 'making', 'creating', 'building']):
                    likely_sound_source = 'process/activity sounds'
                    audio_visual_correlation = {
                        'detected_activity': 'general process',
                        'likely_sound': 'activity-related sounds',
                        'confidence': 'medium'
                    }
                else:
                    likely_sound_source = 'ambient sounds'
                    audio_visual_correlation = {
                        'detected_activity': 'unknown',
                        'likely_sound': 'ambient sounds',
                        'confidence': 'low'
                    }
        
        # Check for actual speech patterns
        elif len(unique_words) > 10 and len(words) > 15:
            # Check if it might be quotes or viral audio
            if any(indicator in transcript_lower for indicator in 
                   ['he said', 'she said', 'they said', 'pov:', 'when you', 'that moment']):
                has_meaningful_speech = True
                transcript_quality = 'viral_audio_possible'
            else:
                has_meaningful_speech = True
                transcript_quality = 'original_speech'
        else:
            # Short but potentially meaningful
            has_meaningful_speech = True
            transcript_quality = 'brief_speech'
    else:
        has_meaningful_speech = False
        transcript_quality = 'no_audio_detected'
        
        # Check what visual activity might produce sounds
        if frames_summaries_text:
            frames_lower = frames_summaries_text.lower()
            if any(word in frames_lower for word in ['drawing', 'writing', 'sketching', 'painting', 'coloring']):
                likely_sound_source = 'visual activity sounds (drawing/writing)'
            elif any(word in frames_lower for word in ['cooking', 'mixing', 'preparing']):
                likely_sound_source = 'cooking/preparation sounds'
            elif any(word in frames_lower for word in ['unboxing', 'opening', 'revealing']):
                likely_sound_source = 'packaging/unboxing sounds'
            elif any(word in frames_lower for word in ['skincare', 'makeup', 'routine']):
                likely_sound_source = 'beauty routine sounds'
    
    return {
        'has_meaningful_speech': has_meaningful_speech,
        'transcript_quality': transcript_quality,
        'likely_sound_source': likely_sound_source,
        'audio_visual_correlation': audio_visual_correlation,
        'transcript_text': transcript_text if has_meaningful_speech else None,
        'audio_description': likely_sound_source or transcript_quality,
        'type': 'original_speech' if has_meaningful_speech else 'visual_only',
        'viral_audio_check': transcript_quality == 'viral_audio_possible'
    }


def enhanced_extract_audio_and_frames(tiktok_url, strategy, frames_per_minute, cap, scene_threshold):
    """Enhanced extraction with validation"""
    try:
        print(f"[INFO] Starting enhanced extraction for {tiktok_url}")
        
        audio_path, frames_dir, frame_paths = extract_audio_and_frames(
            tiktok_url, strategy, frames_per_minute, cap, scene_threshold
        )
        
        # Validate audio
        if not audio_path or not os.path.exists(audio_path):
            raise ValueError("Audio extraction failed")
        
        audio_size = os.path.getsize(audio_path)
        if audio_size < 1024:
            raise ValueError(f"Audio file too small ({audio_size} bytes)")
        
        # Validate frames
        if not frame_paths or len(frame_paths) == 0:
            raise ValueError("Frame extraction failed")
        
        valid_frames = []
        for fp in frame_paths:
            if os.path.exists(fp) and os.path.getsize(fp) > 1024:
                valid_frames.append(fp)
            else:
                print(f"[WARNING] Frame file invalid: {fp}")
        
        if len(valid_frames) == 0:
            raise ValueError("No valid frame files found")
        
        print(f"[SUCCESS] Extraction complete: audio + {len(valid_frames)} frames")
        return audio_path, frames_dir, valid_frames
        
    except Exception as e:
        print(f"[ERROR] Enhanced extraction failed: {e}")
        raise e


def enhanced_transcribe_audio_with_context(audio_path, frames_summaries_text):
    """Enhanced transcription that considers visual context for better interpretation"""
    try:
        # Get basic transcription first
        transcript = transcribe_audio(audio_path)
        
        # Analyze with visual context
        audio_analysis = analyze_audio_with_visual_context(transcript, frames_summaries_text)
        
        # Build comprehensive result
        if audio_analysis['has_meaningful_speech']:
            return {
                'transcript': transcript,
                'quality': 'good',
                'quality_reason': f"Clear speech detected ({audio_analysis['transcript_quality']})",
                'is_reliable': True,
                'audio_context': audio_analysis
            }
        else:
            # Provide context-aware description of non-speech audio
            quality_reason = f"Non-speech audio detected: {audio_analysis['likely_sound_source'] or 'ambient sounds'}"
            
            return {
                'transcript': transcript if transcript else "",
                'quality': audio_analysis['transcript_quality'],
                'quality_reason': quality_reason,
                'is_reliable': False,
                'audio_context': audio_analysis,
                'sound_interpretation': audio_analysis['likely_sound_source']
            }
        
    except Exception as e:
        print(f"[ERROR] Transcription error: {e}")
        return {
            'transcript': f"(Transcription error: {str(e)})",
            'quality': 'error',
            'quality_reason': str(e),
            'is_reliable': False,
            'audio_context': {}
        }


def analyze_satisfaction_elements(frames_summaries_text):
    """Detect satisfaction elements in visual content"""
    frames_lower = frames_summaries_text.lower() if frames_summaries_text else ""
    
    # Satisfaction patterns
    satisfaction_patterns = {
        'precision_work': ['coloring within lines', 'precise', 'careful', 'detailed', 'accurate'],
        'transformation': ['filling in', 'covering', 'applying', 'completing', 'transforming'],
        'completion': ['finishing', 'completing', 'final', 'done', 'finished'],
        'rhythmic': ['repetitive', 'rhythmic', 'systematic', 'methodical', 'consistent'],
        'sensory': ['smooth', 'satisfying', 'gentle', 'soft', 'texture']
    }
    
    detected_elements = {}
    for category, patterns in satisfaction_patterns.items():
        detected_elements[category] = any(pattern in frames_lower for pattern in patterns)
    
    satisfaction_score = sum(detected_elements.values())
    
    return {
        'satisfaction_elements': detected_elements,
        'satisfaction_score': satisfaction_score,
        'highly_satisfying': satisfaction_score >= 3,
        'primary_satisfaction': max(detected_elements, key=detected_elements.get) if any(detected_elements.values()) else None
    }


def create_visual_content_description(frames_summaries_text, audio_context=None):
    """Analyze visual content type and satisfaction potential"""
    try:
        frames_lower = frames_summaries_text.lower() if frames_summaries_text else ""
        content_type = 'general'
        
        # Detect content type
        if any(word in frames_lower for word in ['drawing', 'art', 'sketch', 'illustrat', 'coloring']):
            content_type = 'visual_art_process'
        elif any(word in frames_lower for word in ['transform', 'before', 'after', 'change']):
            content_type = 'transformation'
        elif any(word in frames_lower for word in ['routine', 'skincare', 'makeup', 'beauty']):
            content_type = 'beauty_routine'
        elif any(word in frames_lower for word in ['cooking', 'recipe', 'food', 'baking']):
            content_type = 'cooking_tutorial'
        elif any(word in frames_lower for word in ['unbox', 'package', 'reveal', 'opening']):
            content_type = 'unboxing'
        elif any(word in frames_lower for word in ['clean', 'organiz', 'tidy', 'sort']):
            content_type = 'organizing'
        
        # Get satisfaction analysis
        satisfaction_analysis = analyze_satisfaction_elements(frames_summaries_text)
        
        # Check for visual promise/delivery
        has_promise = any(word in frames_lower for word in ['outline', 'sketch', 'empty', 'before', 'start', 'beginning'])
        has_delivery = any(word in frames_lower for word in ['complete', 'finish', 'done', 'final', 'result', 'after'])
        
        return {
            'description': f"Visual analysis: {frames_summaries_text[:200]}...",
            'content_type': content_type,
            'has_strong_visual_narrative': has_promise and has_delivery,
            'satisfaction_analysis': satisfaction_analysis,
            'visual_promise_delivery': {
                'has_promise': has_promise,
                'has_delivery': has_delivery,
                'narrative_strength': 'strong' if (has_promise and has_delivery) else 'weak'
            }
        }
    except Exception as e:
        print(f"[ERROR] Visual content description failed: {e}")
        return {
            'description': "Visual content analysis",
            'content_type': 'general',
            'has_strong_visual_narrative': False,
            'satisfaction_analysis': {'highly_satisfying': False, 'satisfaction_elements': {}},
            'visual_promise_delivery': {'has_promise': False, 'has_delivery': False}
        }


# ==============================
# MAIN ANALYSIS FUNCTION - COMPREHENSIVE & ADAPTIVE
# ==============================

def run_main_analysis(transcript_text, frames_summaries_text, creator_note, platform, target_duration, goal, tone, audience, knowledge_context):
    """Comprehensive analysis that adapts to ALL video types with deep insights"""
    
    # First analyze frames to understand visual content
    visual_content_analysis = create_visual_content_description(frames_summaries_text)
    
    # Then analyze audio with visual context
    audio_analysis = analyze_audio_with_visual_context(transcript_text, frames_summaries_text)
    
    # Use audio analysis results
    has_speech = audio_analysis['has_meaningful_speech']
    audio_type_info = audio_analysis
    
    # Extract actual performance data from creator note
    view_count = None
    performance_level = 'unknown'
    
    if creator_note:
        note_lower = creator_note.lower()
        
        # Extract view count if mentioned
        view_patterns = re.findall(r'(\d+\.?\d*)\s*(k|thousand|m|million|views)', note_lower)
        if view_patterns:
            number, unit = view_patterns[0][:2]
            try:
                num = float(number)
                if unit in ['k', 'thousand']:
                    view_count = f"{num}k"
                    if num >= 100:
                        performance_level = 'moderate'
                    if num >= 500:
                        performance_level = 'good'
                elif unit in ['m', 'million']:
                    view_count = f"{num}M"
                    performance_level = 'viral'
                elif unit == 'views' and num < 1000:
                    view_count = f"{int(num)} views"
                    performance_level = 'low'
            except:
                pass
    
    # Build knowledge section
    knowledge_section = ""
    if knowledge_context and len(knowledge_context.strip()) > 100:
        knowledge_section = f"""
KNOWLEDGE BASE PATTERNS (Apply these insights deeply):
{knowledge_context}

DEEP ANALYSIS REQUIREMENTS:
1. EXPLAIN THE PSYCHOLOGY: Why does each element work or not work?
2. REFERENCE PROVEN PATTERNS: Connect to specific patterns from the knowledge base
3. BE SPECIFIC: Analyze EXACTLY what happens in THIS video, not generic observations
4. DISTINGUISH CONTENT TYPES: Clearly identify what's spoken vs what's shown vs what's written
5. PERFORMANCE REASONING: Explain WHY this got {view_count if view_count else 'its current'} views
6. ACTIONABLE INSIGHTS: Provide specific, implementable improvements
7. TIMING PRECISION: Break down what happens at each second marker
8. HANDLE ALL VIDEO TYPES: Visual-only, speech, viral audio, tutorials, transformations, etc.
"""

    # Adapt prompt based on video type
    video_type_context = ""
    if not has_speech:
        video_type_context = f"""
This is a VISUAL-ONLY or AMBIENT AUDIO video with {audio_type_info.get('likely_sound_source', 'ambient sounds')}.
Focus analysis on:
- Visual hooks and progression
- EXACT On-screen text vs auto-captions
- Visual satisfaction elements: {visual_content_analysis.get('satisfaction_analysis', {}).get('satisfaction_elements', {})}
- Promise/delivery structure: {visual_content_analysis.get('visual_promise_delivery', {})}
- How {audio_type_info.get('likely_sound_source', 'ambient audio')} enhances the visual experience
"""
    else:
        video_type_context = """
This video has VERBAL CONTENT. Analyze:
- How verbal and visual elements work together
- Whether on-screen text reinforces or adds to spoken content
- The relationship between what's said and what's shown
- Speech delivery effectiveness and clarity
"""

    # Build the performance message separately to avoid f-string issues
    if performance_level == 'viral':
        performance_message = f"This video went VIRAL with {view_count} - analyze WHY it succeeded."
    else:
        performance_message = f"This video got {view_count if view_count else 'certain performance'} - analyze what's working and what needs to improve to achieve higher success in relation to the chosen goal."
    
    prompt = f"""
You are a video psychology expert analyzing a {platform} video. {performance_message}

CRITICAL CONTEXT:
- Platform: {platform}
- Performance: {view_count if view_count else 'Not specified'} ({performance_level})
- Creator's note: "{creator_note if creator_note else 'No additional context'}"
- Content type detected: {visual_content_analysis.get('content_type', 'general')}
- Audio type: {audio_type_info.get('audio_description', 'unknown')}
- Goal: {goal}
- Target audience: {audience}
- Duration target: {target_duration}s

AUDIO CONTEXT:
{f"Speech detected: {transcript_text}" if has_speech else f"Non-speech audio: {audio_type_info.get('likely_sound_source', 'ambient sounds')} (based on visual activity)"}

VISUAL CONTENT (frames - what's SHOWN/WRITTEN):
{frames_summaries_text}

VISUAL ANALYSIS:
- Content type: {visual_content_analysis.get('content_type', 'general')}
- Satisfaction score: {visual_content_analysis.get('satisfaction_analysis', {}).get('satisfaction_score', 0)}/5
- Visual narrative: {visual_content_analysis.get('visual_promise_delivery', {}).get('narrative_strength', 'unknown')}

{video_type_context}

{knowledge_section}

COMPREHENSIVE ANALYSIS INSTRUCTIONS:
{f"Since this went VIRAL, identify the EXACT psychological triggers and viral mechanics." if performance_level == 'viral' else "Identify opportunities for this specific video that can help improve based on or inspired by proven patterns. Give 2-3 ideas and examples. Instead of saying something like 'enhance visual storytelling' explain to them how they might do that thing and provide strong examples"}

1. FIRST 3 SECONDS BREAKDOWN:
   - Frame by frame: What EXACTLY appears and why was it successful or unsuccessful?
   - What EXACT texts on the screen are shown (from frames, not transcript)?
   - What's the audio (speech from transcript, or {audio_type_info.get('likely_sound_source', 'sounds')})?
   - Is there a visual hooks grab attention? If so, what?
   - Rate the hook strength and explain WHY while educating on how to improve or how to replicate if its already good.

2. PERFORMANCE MECHANICS:
   {f"- What specific elements made this shareable?{chr(10)}   - What psychological triggers drove the viral spread?{chr(10)}   - How did it tap into platform algorithms?{chr(10)}   - What made people watch to completion?" if performance_level == 'viral' else f"- What's preventing viral growth?{chr(10)}   - Which psychological triggers are missing?{chr(10)}   - How could platform algorithms be better leveraged?{chr(10)}   - Where do viewers likely drop off?"}

3. CONTENT STRUCTURE ANALYSIS:
   - Hook mechanism (0-3s): How does it stop scrolling?
   - Promise delivery (3-10s): What value is promised?
   - Retention mechanics (middle): What keeps viewers?
   - Payoff (end): How does it satisfy or create sharing impulse?

4. AUDIO-VISUAL INTEGRATION:
   - How does {audio_type_info.get('audio_description', 'the audio')} enhance the visual content?
   - Are sounds and visuals synchronized effectively?
   - Does the audio-visual combination create satisfaction?

5. PATTERN MATCHING:
   - Which proven patterns from the knowledge base apply?
   - How well does it execute these patterns?
   - What patterns could be better implemented?
   
MANDATORY: Every formula and hook suggestion MUST include:
1. A specific example INSPIRED BY the knowledge base, no need to adhere too strictly as long as it fits the criteria for a strong hook
2. How to adapt it to either the same niche or 'audience'
3. The exact psychological principles it leverages
4. Expected performance metrics based on similar content

Example output format:
"Hook 1 - Controversial angle: 'Your skincare routine is aging you faster (here's why)' 
- Adaptation: For fitness: 'Your workouts are making you weaker', For cooking: 'Your healthy meals are nutrient-dead'
- Psychology: Challenges existing beliefs, creating cognitive dissonance that demands resolution
- Expected CTR: 2.3x baseline based on controversial hook performance"

Example output format for formulas:
This video would do well being readapted to [recommended formula]. Here is how I'd do it for maximum success in [goal]:
[reformat video into recommended video format based off your experise and the supporting knowledge]

Respond in JSON with DEEP, SPECIFIC insights:

{{
  "analysis": "{f'This video achieved viral status because...' if performance_level == 'viral' else 'This video shows potential but...'} [2-3 paragraphs of DEEP psychological and structural analysis. Explain the WHY behind everything. Reference specific moments. Note that audio is {audio_type_info.get('audio_description', 'ambient')} not random animal sounds.]",
  
  "viral_mechanics": "{f'Here are the specific viral triggers: ' if performance_level == 'viral' else 'To achieve viral potential: '}[Detailed explanation of psychological mechanisms, sharing triggers, algorithm optimization]",
  
  "exact_hook_breakdown": {{
    "first_frame": "0:00 - [EXACTLY what appears in frame 1]",
    "second_moment": "0:01 - [EXACTLY what happens in second 1]",
    "third_second": "0:02 - [EXACTLY what occurs by second 3]",
    "visual_elements": "[Specific visual hooks from frames and how they are or are not effective hooks]",
    "text_overlays": "[EXACT text shown on screen from frames and how they are or are not effective hooks]",
    "audio_element": "[{audio_type_info.get('audio_description', 'Audio type')}]",
    "hook_psychology": "[Deep explanation of why this hook works/doesn't work psychologically]",
    "hook_score": [1-10],
    "hook_reasoning": "[Specific reasoning for the score based on proven patterns]"
  }},
  
  "performance_deep_dive": "{f'With {view_count}, this demonstrates...' if view_count else 'The performance indicates...'}[3-4 sentences explaining the specific reasons for this performance level, referencing actual content elements, video retention principles, and psychological principles. Explain specific examples that can be applied to improve the video's performance that are specific to the video's context itself. Explain why this the examples would help improve the video in terms of psychological principles and viewer retention.]",
  
  "content_type_analysis": {{
    "detected_type": "{visual_content_analysis.get('content_type', 'general')}",
    "audio_type": "{audio_type_info.get('audio_description', 'unknown')}",
    "visual_satisfaction": {visual_content_analysis.get('satisfaction_analysis', {}).get('satisfaction_score', 0)},
    "narrative_structure": "{visual_content_analysis.get('visual_promise_delivery', {}).get('narrative_strength', 'unknown')}",
    "type_specific_insights": "[Insights specific to this content type]"
  }},
  
  "psychological_breakdown": {{
    "emotional_triggers": ["List specific emotions triggered and when"],
    "curiosity_mechanisms": ["How curiosity gaps are created"],
    "satisfaction_points": ["Where viewers get satisfaction"],
    "sharing_psychology": ["Why people would/wouldn't share this"]
  }},
  
  "audio_visual_analysis": {{
    "audio_interpretation": "{audio_type_info.get('audio_description', 'ambient sounds')}",
    "visual_audio_sync": "[How well audio matches visual activity]",
    "enhancement_effect": "[How audio enhances or detracts from visuals]",
    "satisfaction_contribution": "[Does audio add to satisfaction?]"
  }},
  

  "hooks":[
    "[Specific controversial statement that challenges common beliefs, like 'Everyone's doing skincare wrong and here's proof' but adapt for this specific video's context]",
    "[Relatable personal angle like 'The night routine that fixed my skin after trying everything'( but adapt for this specific video's context)]",
    "[Create specific mystery like 'The 3 products dermatologists use but never talk about' ( but adapt for this specific video's context)]",
    "[Unexpected opening like starting mid-action with 'Wait, don't wash your face yet' ( but adapt for this specific video's context)]",
    "[Leverage authority like 'This routine gave me glass skin in 2 weeks (with receipts)'but adapt for this specific video's context] "
    ],


"improvement_opportunities": "[SPECIFIC improvements with examples like: 'Add text overlay at 0:02 saying exactly [suggested text]. Show [suggested object] to the screen in the first 3 seconds. [Anything else that would help the video improve with getting more views and longer watch time]",
  
  "scores": {{
    "hook_strength": [1-10 based on actual effectiveness],
    "promise_clarity": [1-10 based on value proposition],
    "retention_design": [1-10 based on completion likelihood],
    "engagement_potential": [1-10 based on interaction drivers],
    "viral_potential": [1-10 based on sharing likelihood],
    "satisfaction_delivery": [1-10 based on viewer satisfaction],
    "goal_alignment": [1-10 based on achieving {goal}]
  }},
  
  "timing_mastery": {{
    "0-1s": "[Exact content + psychological impact + audio element]",
    "1-3s": "[Exact content + viewer state + audio-visual sync]",
    "3-7s": "[Development + emotional journey + satisfaction building]",
    "7-15s": "[Core value + retention mechanics + audio role]",
    "15s+": "[Resolution + sharing trigger + completion satisfaction]"
  }},
  
  "formulas": {{
    "quick_formula": "[1-sentence replication guide]",
    "detailed_formula": "[Step-by-step breakdown with timing: 0-3s: Do X to create Y effect, 3-7s: Show Z to maintain attention...]",
    "script_template": "[Complete fill-in-the-blank script: 'Start with [your controversial statement about your niche]. At 0:03 show [your transformation moment]. At 0:07 say [your credibility marker]...']",
    "psychology_formula": "[Deep psychological framework: Use curiosity gap by withholding [specific info] until [timestamp], trigger completion desire by showing [incomplete action] at [timestamp]...]",
    "example_adaptation": "[Concrete example with recommended improved formula in same niche. Include example script using improved hook, improved 3-7 seconds, using the improved formula. Explain why this will work better for the given goal.]"
    }},
  
  "improvement_opportunities": "{f'Even this viral video could improve by: ' if performance_level == 'viral' else 'Key improvements: '}[Specific, actionable improvements with psychological reasoning]",
  
  "performance_prediction": "{f'This video succeeded because: ' if performance_level == 'viral' else 'With improvements, this could achieve: '}[Specific prediction with reasoning]",
  
  "knowledge_patterns_applied": [
    "[Pattern 1 from knowledge base]: [How it applies to this video]",
    "[Pattern 2 from knowledge base]: [Specific implementation]",
    "[Pattern 3 from knowledge base]: [Opportunity or execution]"
  ],
  
  "replication_framework": {{
    "core_principles": "[What makes this replicable or how can we adjust to make it better]",
    "adaptation_guide": "[How to apply to different niches]",
    "success_factors": "[Critical elements to maintain or improve for more virality or retention]",
    "common_mistakes": "[What to avoid when replicating or what was in this video that needs to be taken out to improve next time]"
  }}
}}

CRITICAL: 
- Provide DEEP insights, not surface observations
- Explain the WHY behind everything
- Reference specific moments from the video
- Correctly identify audio as {audio_type_info.get('audio_description', 'activity sounds')} not random animal noises
- Make it actionable and educational
- Adapt to the specific content type detected
"""

    try:
        print(f"[INFO] Running COMPREHENSIVE analysis for {performance_level} {visual_content_analysis.get('content_type', 'content')}...")
        print(f"[INFO] View count: {view_count}, Audio type: {audio_type_info.get('audio_description', 'unknown')}")
        print(f"[INFO] Visual satisfaction score: {visual_content_analysis.get('satisfaction_analysis', {}).get('satisfaction_score', 0)}/5")
        print(f"[INFO] Knowledge base: {len(knowledge_context)} chars")
        
        gpt_response = _api_retry(
            client.chat.completions.create,
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in viral psychology and content analysis. Provide DEEP, specific insights about why content succeeds or fails. Always explain the psychological mechanisms. Never give surface-level observations. Correctly interpret audio based on visual context - if someone is drawing, sounds are likely marker/pen sounds, not animal noises."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=6000
        )

        response_text = gpt_response.choices[0].message.content.strip()
        
        # Parse JSON
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        parsed = json.loads(response_text.strip())
        
        # Process scores with performance-based defaults
        scores_raw = parsed.get("scores", {})
        
        # Score defaults based on actual performance
        if performance_level == 'viral':
            score_defaults = {"hook_strength": 9, "promise_clarity": 8, "retention_design": 9, 
                            "engagement_potential": 9, "viral_potential": 10, "satisfaction_delivery": 9, "goal_alignment": 8}
        elif performance_level == 'good':
            score_defaults = {"hook_strength": 7, "promise_clarity": 7, "retention_design": 7, 
                            "engagement_potential": 7, "viral_potential": 6, "satisfaction_delivery": 7, "goal_alignment": 7}
        elif performance_level == 'moderate':
            score_defaults = {"hook_strength": 6, "promise_clarity": 6, "retention_design": 6, 
                            "engagement_potential": 6, "viral_potential": 5, "satisfaction_delivery": 6, "goal_alignment": 6}
        else:
            score_defaults = {"hook_strength": 4, "promise_clarity": 5, "retention_design": 5, 
                            "engagement_potential": 4, "viral_potential": 3, "satisfaction_delivery": 5, "goal_alignment": 5}
        
        scores = {}
        for key, default in score_defaults.items():
            try:
                score_str = str(scores_raw.get(key, default))
                score_match = re.search(r'(\d+)', score_str)
                if score_match:
                    scores[key] = max(1, min(10, int(score_match.group(1))))
                else:
                    scores[key] = default
            except:
                scores[key] = default
        
        # Build comprehensive result with ALL components
        result = {
            # Core analysis
            "analysis": parsed.get("analysis", ""),
            "viral_mechanics": parsed.get("viral_mechanics", ""),
            "psychological_breakdown": parsed.get("psychological_breakdown", {}),
            "performance_deep_dive": parsed.get("performance_deep_dive", ""),
            
            # Content type analysis
            "content_type_analysis": parsed.get("content_type_analysis", {}),
            "audio_visual_analysis": parsed.get("audio_visual_analysis", {}),
            "visual_content_analysis": visual_content_analysis,
            
            # Detailed breakdowns
            "exact_hook_breakdown": parsed.get("exact_hook_breakdown", {}),
            "timing_mastery": parsed.get("timing_mastery", {}),
            
            # Scores and hooks
            "scores": scores,
            "hooks": parsed.get("hooks", []),
            
            # Formulas and frameworks
            "formulas": parsed.get("formulas", {}),
            "replication_framework": parsed.get("replication_framework", {}),
            
            # Improvements and predictions
            "improvement_opportunities": parsed.get("improvement_opportunities", ""),
            "performance_prediction": parsed.get("performance_prediction", ""),
            
            # Knowledge patterns
            "knowledge_patterns_applied": parsed.get("knowledge_patterns_applied", []),
            
            # All timing and structure info
            "timing_breakdown": "\n".join([
                f"{time}: {content}" 
                for time, content in parsed.get("timing_mastery", {}).items()
            ]),
            
            # Individual components for template compatibility
            "content_type_detected": audio_type_info.get('type', 'unknown'),
            "audio_type_detected": audio_type_info.get('audio_description', 'unknown'),
            "visual_hook": parsed.get("exact_hook_breakdown", {}).get("visual_elements", ""),
            "text_hook": parsed.get("exact_hook_breakdown", {}).get("text_overlays", ""),
            "verbal_hook": parsed.get("exact_hook_breakdown", {}).get("audio_element", ""),
            "why_hook_works": parsed.get("exact_hook_breakdown", {}).get("hook_psychology", ""),
            
            # Formula components
            "basic_formula": parsed.get("formulas", {}).get("viral_formula", ""),
            "timing_formula": parsed.get("formulas", {}).get("satisfaction_formula", ""),
            "visual_formula": parsed.get("formulas", {}).get("audio_visual_formula", ""),
            "psychology_formula": parsed.get("formulas", {}).get("platform_formula", ""),
            "hook_formula": parsed.get("formulas", {}).get("hook_formula", ""),
            
            # Performance and quality
            "performance_analysis": parsed.get("performance_deep_dive", ""),
            "video_type_analysis": f"Deep analysis of {visual_content_analysis.get('content_type', 'content')} with {audio_type_info.get('audio_description', 'audio')}",
            "engagement_psychology": parsed.get("psychological_breakdown", {}).get("sharing_psychology", ""),
            "strengths": f"Working elements: {parsed.get('viral_mechanics', '')}",
            "improvement_areas": parsed.get("improvement_opportunities", ""),
            "improvements": parsed.get("improvement_opportunities", ""),
            
            # Viral audio analysis
            "viral_audio_analysis": {
                "is_viral_sound": audio_type_info.get('viral_audio_check', False),
                "audio_type": audio_type_info.get('audio_description', 'unknown'),
                "audio_psychology": parsed.get("audio_visual_analysis", {}).get("enhancement_effect", "")
            },
            
            # Template compatibility fields
            "formula": parsed.get("formulas", {}).get("viral_formula", ""),
            "template_formula": parsed.get("formulas", {}).get("platform_formula", ""),
            "knowledge_insights": " | ".join(parsed.get("knowledge_patterns_applied", [])),
            
            # Meta information
            "knowledge_context_used": bool(knowledge_context.strip()),
            "overall_quality": "strong" if performance_level == 'viral' else "moderate" if performance_level in ['good', 'moderate'] else "needs_work",
            "video_has_speech": has_speech,
            "actual_view_count": view_count,
            "performance_level": performance_level
        }
        
        return result
        
    except Exception as e:
        print(f"[ERROR] Comprehensive analysis failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Return enhanced fallback
        return create_comprehensive_fallback(
            transcript_text, frames_summaries_text, creator_note, 
            platform, goal, audience, has_speech, view_count, performance_level,
            knowledge_context, audio_type_info, visual_content_analysis
        )


def create_comprehensive_fallback(transcript_text, frames_summaries_text, creator_note, platform, goal, audience, has_speech, view_count, performance_level, knowledge_context, audio_type_info, visual_content_analysis):
    """Comprehensive fallback that maintains all functionality even in error cases"""
    
    # Build performance-aware analysis
    if performance_level == 'viral':
        analysis = f"""This video achieved viral success with {view_count}, demonstrating strong psychological triggers and platform optimization.

The viral mechanics include: {visual_content_analysis.get('content_type', 'content')} combined with {audio_type_info.get('audio_description', 'audio elements')}. The satisfaction score of {visual_content_analysis.get('satisfaction_analysis', {}).get('satisfaction_score', 0)}/5 indicates {'high viewer satisfaction' if visual_content_analysis.get('satisfaction_analysis', {}).get('highly_satisfying') else 'room for enhancement'}.

Key success factors: Strong opening hook, clear value delivery, and satisfying payoff. The {audio_type_info.get('audio_description', 'audio')} enhances the visual content effectively."""
    else:
        analysis = f"""This video shows {'strong potential' if performance_level in ['good', 'moderate'] else 'opportunities for growth'} with {view_count if view_count else 'current performance'}.

Content structure: {visual_content_analysis.get('content_type', 'Visual content')} with {audio_type_info.get('audio_description', 'audio elements')}. Satisfaction score: {visual_content_analysis.get('satisfaction_analysis', {}).get('satisfaction_score', 0)}/5.

To improve: Strengthen the opening hook, enhance audio-visual synchronization, and ensure clear value delivery within the first 7 seconds."""
    
    # Dynamic scoring
    base_scores = {
        "hook_strength": 8 if performance_level == 'viral' else 6 if performance_level in ['good', 'moderate'] else 4,
        "promise_clarity": 7 if performance_level == 'viral' else 6 if performance_level in ['good', 'moderate'] else 4,
        "retention_design": 8 if performance_level == 'viral' else 6 if performance_level in ['good', 'moderate'] else 5,
        "engagement_potential": 8 if performance_level == 'viral' else 6 if performance_level in ['good', 'moderate'] else 4,
        "viral_potential": 9 if performance_level == 'viral' else 5 if performance_level in ['good', 'moderate'] else 3,
        "satisfaction_delivery": visual_content_analysis.get('satisfaction_analysis', {}).get('satisfaction_score', 5),
        "goal_alignment": 7 if performance_level == 'viral' else 6 if performance_level in ['good', 'moderate'] else 5
    }
    
    return {
        "analysis": analysis,
        "viral_mechanics": f"{'Success through: ' if performance_level == 'viral' else 'To increase virality: '}Strong hooks, clear value, satisfying payoffs, effective {audio_type_info.get('audio_description', 'audio')}",
        
        "content_type_analysis": {
            "detected_type": visual_content_analysis.get('content_type', 'general'),
            "audio_type": audio_type_info.get('audio_description', 'unknown'),
            "visual_satisfaction": visual_content_analysis.get('satisfaction_analysis', {}).get('satisfaction_score', 0),
            "narrative_structure": visual_content_analysis.get('visual_promise_delivery', {}).get('narrative_strength', 'unknown')
        },
        
        "audio_visual_analysis": {
            "audio_interpretation": audio_type_info.get('audio_description', 'ambient sounds'),
            "visual_audio_sync": "Audio enhances visual content",
            "enhancement_effect": "Creates immersive experience",
            "satisfaction_contribution": "Adds to overall satisfaction"
        },
        
        "psychological_breakdown": {
            "emotional_triggers": ["curiosity", "satisfaction", "surprise", "completion desire"],
            "curiosity_mechanisms": ["Visual progression", "Promise of transformation", "Pattern interrupts"],
            "satisfaction_points": ["Process completion", "Visual payoffs", "Audio-visual harmony"],
            "sharing_psychology": ["Value delivery", "Emotional resonance", "Relatable content"]
        },
        
        "exact_hook_breakdown": {
            "first_frame": "0:00 - Opening visual",
            "second_moment": "0:01 - Development",
            "third_second": "0:02 - Hook establishment",
            "visual_elements": "Visual hooks present",
            "text_overlays": "Text elements if present",
            "audio_element": audio_type_info.get('audio_description', 'Audio element'),
            "hook_psychology": "Creates curiosity and stops scroll",
            "hook_score": base_scores["hook_strength"],
            "hook_reasoning": "Based on performance and content type"
        },
        
        "timing_mastery": {
            "0-1s": f"Opening with {audio_type_info.get('audio_description', 'audio')}",
            "1-3s": "Hook development and curiosity creation",
            "3-7s": "Value reveal and engagement building",
            "7-15s": "Core content delivery",
            "15s+": "Payoff and satisfaction delivery"
        },
        
        "scores": base_scores,
        "hooks": [
            f"Start with strongest visual element from {visual_content_analysis.get('content_type', 'your content')}",
            "Create immediate curiosity with incomplete visual",
            "Use pattern interrupt in first second",
            f"Leverage {audio_type_info.get('audio_description', 'audio')} for immersion",
            "Promise clear transformation or satisfaction"
        ],
        
        "formulas": {
            "viral_formula": "Hook → Curiosity → Development → Payoff → Share trigger",
            "hook_formula": f"Visual interrupt + {audio_type_info.get('audio_description', 'Audio')} + Promise",
            "satisfaction_formula": "Setup → Process → Transformation → Completion",
            "audio_visual_formula": "Sync audio to visual transitions for maximum impact",
            "platform_formula": f"{platform.capitalize()}: Fast pace + Clear value + Shareable moment"
        },
        
        "improvement_opportunities": f"{'Refine' if performance_level == 'viral' else 'Enhance'} hooks, improve audio-visual sync, optimize pacing",
        "performance_prediction": f"{'Continued success with refinements' if performance_level == 'viral' else 'Significant growth potential with optimizations'}",
        
        "knowledge_patterns_applied": ["Hook optimization", "Satisfaction delivery", "Audio-visual integration"],
        "replication_framework": {
            "core_principles": "Strong hooks, satisfaction delivery, audio-visual harmony",
            "adaptation_guide": "Maintain psychological triggers while adapting content",
            "success_factors": "First 3 seconds, satisfaction points, completion",
            "common_mistakes": "Weak hooks, poor audio sync, unclear payoffs"
        },
        
        # Compatibility fields
        "content_type_detected": audio_type_info.get('type', 'unknown'),
        "audio_type_detected": audio_type_info.get('audio_description', 'unknown'),
        "visual_content_analysis": visual_content_analysis,
        "performance_analysis": f"Performance analysis based on {view_count if view_count else 'current metrics'}",
        "video_type_analysis": f"Analysis of {visual_content_analysis.get('content_type', 'content')}",
        "engagement_psychology": "Engagement through curiosity, satisfaction, and value",
        "strengths": "Content creation and platform understanding",
        "improvement_areas": "Hook optimization and pacing refinement",
        "timing_breakdown": "0-1s: Hook\n1-3s: Development\n3-7s: Value\n7-15s: Core\n15s+: Payoff",
        
        # Individual formula components
        "basic_formula": "Hook → Curiosity → Value → Payoff",
        "timing_formula": "0-1s: Stop scroll, 1-3s: Build curiosity, 3-7s: Show value",
        "visual_formula": "Visual hook → Process → Transformation",
        "psychology_formula": "Attention → Interest → Desire → Satisfaction",
        "hook_formula": "Pattern interrupt + Promise + Visual interest",
        
        "viral_audio_analysis": {
            "is_viral_sound": audio_type_info.get('viral_audio_check', False),
            "audio_type": audio_type_info.get('audio_description', 'unknown'),
            "audio_psychology": "Audio enhances viewer experience"
        },
        
        # Template fields
        "formula": "Hook → Development → Payoff",
        "improvements": "Optimize hooks, enhance satisfaction points",
        "template_formula": f"{platform} optimization formula",
        "knowledge_insights": "Apply proven patterns for success",
        
        # Meta
        "knowledge_context_used": bool(knowledge_context and len(knowledge_context) > 100),
        "overall_quality": "strong" if performance_level == 'viral' else "moderate" if performance_level in ['good', 'moderate'] else "needs_work",
        "video_has_speech": has_speech,
        "actual_view_count": view_count,
        "performance_level": performance_level
    }


def prepare_template_variables(gpt_result, transcript_data, frames_summaries_text, form_data, gallery_data_urls, frame_paths, frames_dir, knowledge_citations, knowledge_context):
    """Prepare all template variables with safe defaults"""
    
    template_vars = {
        # Form data
        'tiktok_url': form_data.get('tiktok_url', ''),
        'creator_note': form_data.get('creator_note', ''),
        'platform': form_data.get('platform', 'tiktok'),
        'target_duration': form_data.get('target_duration', '30'),
        'goal': form_data.get('goal', 'follower_growth'),
        'tone': form_data.get('tone', 'confident, friendly'),
        'audience': form_data.get('audience', 'creators and small business owners'),
        'strategy': form_data.get('strategy', 'smart'),
        'frames_per_minute': int(form_data.get('frames_per_minute', 24)),
        'cap': int(form_data.get('cap', 60)),
        'scene_threshold': float(form_data.get('scene_threshold', 0.24)),
        
        # Frame and file data
        'frames_count': len(frame_paths) if frame_paths else 0,
        'frame_gallery': gallery_data_urls if gallery_data_urls else [],
        'frames_dir': frames_dir if frames_dir else "",
        'frame_paths': frame_paths if frame_paths else [],
        
        # Knowledge data
        'knowledge_citations': knowledge_citations if knowledge_citations else [],
        'knowledge_context': knowledge_context if knowledge_context else "",
        
        # Core analysis results
        'analysis': gpt_result.get('analysis', 'Analysis not available'),
        'hooks': gpt_result.get('hooks', []),
        'scores': gpt_result.get('scores', {}),
        'strengths': gpt_result.get('strengths', ''),
        'improvement_areas': gpt_result.get('improvement_areas', ''),
        'improvements': gpt_result.get('improvements', gpt_result.get('improvement_areas', '')),
        'improvement_opportunities': gpt_result.get('improvement_opportunities', ''),
        
        # Timing and formulas
        'timing_breakdown': gpt_result.get('timing_breakdown', ''),
        'timing_mastery': gpt_result.get('timing_mastery', {}),
        'formula': gpt_result.get('formula', gpt_result.get('basic_formula', '')),
        'basic_formula': gpt_result.get('basic_formula', ''),
        'timing_formula': gpt_result.get('timing_formula', ''),
        'template_formula': gpt_result.get('template_formula', gpt_result.get('visual_formula', '')),
        'psychology_formula': gpt_result.get('psychology_formula', ''),
        'hook_formula': gpt_result.get('hook_formula', ''),
        'formulas': gpt_result.get('formulas', {}),
        
        # Transcript data
        'transcript': transcript_data.get('transcript', ''),
        'transcript_quality': transcript_data,
        'transcript_original': transcript_data.get('transcript', ''),
        'transcript_for_analysis': transcript_data.get('transcript', ''),
        'audio_context': transcript_data.get('audio_context', {}),
        'sound_interpretation': transcript_data.get('sound_interpretation', ''),
        
        # Frame analysis
        'frame_summary': frames_summaries_text if frames_summaries_text else "",
        'frame_summaries': [block.strip() for block in frames_summaries_text.split('\n\n') if block.strip()] if frames_summaries_text else [],
        
        # Enhanced analysis fields
        'visual_content_analysis': gpt_result.get('visual_content_analysis', {}),
        'content_type_analysis': gpt_result.get('content_type_analysis', {}),
        'audio_visual_analysis': gpt_result.get('audio_visual_analysis', {}),
        'viral_audio_analysis': gpt_result.get('viral_audio_analysis', {}),
        'content_analysis': gpt_result.get('content_analysis', {}),
        'psychological_breakdown': gpt_result.get('psychological_breakdown', {}),
        'replication_framework': gpt_result.get('replication_framework', {}),
        
        # Performance and predictions
        'performance_prediction': gpt_result.get('performance_prediction', ''),
        'performance_analysis': gpt_result.get('performance_analysis', ''),
        'performance_deep_dive': gpt_result.get('performance_deep_dive', ''),
        'viral_mechanics': gpt_result.get('viral_mechanics', ''),
        
        # Knowledge and insights
        'knowledge_insights': gpt_result.get('knowledge_insights', ''),
        'knowledge_patterns_applied': gpt_result.get('knowledge_patterns_applied', []),
        
        # Video type and hook analysis
        'video_type_analysis': gpt_result.get('video_type_analysis', ''),
        'exact_hook_breakdown': gpt_result.get('exact_hook_breakdown', {}),
        'visual_hook': gpt_result.get('visual_hook', ''),
        'text_hook': gpt_result.get('text_hook', ''),
        'verbal_hook': gpt_result.get('verbal_hook', ''),
        'why_hook_works': gpt_result.get('why_hook_works', ''),
        
        # Compatibility fields
        'gpt_response': gpt_result.get('analysis', ''),
        'engagement_psychology': gpt_result.get('engagement_psychology', ''),
        
        # Meta information
        'knowledge_context_used': gpt_result.get('knowledge_context_used', False),
        'overall_quality': gpt_result.get('overall_quality', 'moderate'),
        'content_type_detected': gpt_result.get('content_type_detected', ''),
        'audio_type_detected': gpt_result.get('audio_type_detected', ''),
        'actual_view_count': gpt_result.get('actual_view_count', ''),
        'performance_level': gpt_result.get('performance_level', 'unknown'),
        'video_has_speech': gpt_result.get('video_has_speech', False),
    }
    
    # Ensure hooks is always a list
    if isinstance(template_vars['hooks'], str):
        template_vars['hooks'] = [template_vars['hooks']]
    elif not template_vars['hooks']:
        template_vars['hooks'] = []
    
    # Ensure scores has all required fields
    required_scores = {
        "hook_strength": 5,
        "promise_clarity": 5,
        "retention_design": 5,
        "engagement_potential": 5,
        "viral_potential": 5,
        "satisfaction_delivery": 5,
        "goal_alignment": 5
    }
    
    scores = template_vars['scores']
    if not scores:
        scores = required_scores
    else:
        for key, default in required_scores.items():
            if key not in scores:
                scores[key] = default
    
    template_vars['scores'] = scores
    
    return template_vars


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
        # Get form data
        form_data = {
            'tiktok_url': request.form.get("tiktok_url", "").strip(),
            'creator_note': request.form.get("creator_note", "").strip(),
            'strategy': request.form.get("strategy", "smart").strip(),
            'frames_per_minute': request.form.get("frames_per_minute", "24"),
            'cap': request.form.get("cap", "60"),
            'scene_threshold': request.form.get("scene_threshold", "0.24"),
            'platform': request.form.get("platform", "tiktok").strip(),
            'target_duration': request.form.get("target_duration", "30").strip(),
            'goal': request.form.get("goal", "follower_growth").strip(),
            'tone': request.form.get("tone", "confident, friendly").strip(),
            'audience': request.form.get("audience", "creators and small business owners").strip(),
        }
        
        # Validate numeric parameters
        try:
            frames_per_minute = int(form_data['frames_per_minute'])
            cap = int(form_data['cap'])
            scene_threshold = float(form_data['scene_threshold'])
        except ValueError as e:
            print(f"[ERROR] Invalid numeric parameter: {e}")
            return "Error: Invalid numeric parameters provided", 400
        
        tiktok_url = form_data['tiktok_url']
        if not tiktok_url:
            return "Error: TikTok URL is required", 400

        print(f"[INFO] Processing: {tiktok_url}")
        print(f"[INFO] Creator note: {form_data['creator_note']}")
        print(f"[INFO] Strategy: {form_data['strategy']}, Goal: {form_data['goal']}")

        # Extract audio and frames
        try:
            audio_path, frames_dir, frame_paths = enhanced_extract_audio_and_frames(
                tiktok_url,
                strategy=form_data['strategy'],
                frames_per_minute=frames_per_minute,
                cap=cap,
                scene_threshold=scene_threshold,
            )
            print(f"[SUCCESS] Extracted {len(frame_paths)} frames")
        except Exception as e:
            print(f"[ERROR] Video processing error: {e}")
            return f"Error processing video: {str(e)}", 500

        # Analyze frames FIRST (needed for audio context)
        try:
            frames_summaries_text, gallery_data_urls = analyze_frames_batch(frame_paths)
            print(f"[SUCCESS] Frame analysis complete")
            print(f"[INFO] Frame analysis preview: {frames_summaries_text[:200]}...")
        except Exception as e:
            print(f"[ERROR] Frame analysis error: {e}")
            frames_summaries_text = ""
            gallery_data_urls = []

        # Transcribe audio WITH visual context
        try:
            transcript_data = enhanced_transcribe_audio_with_context(audio_path, frames_summaries_text)
            print(f"[INFO] Audio interpretation: {transcript_data.get('audio_context', {}).get('audio_description', 'unknown')}")
            print(f"[INFO] Transcript quality: {transcript_data.get('quality', 'unknown')}")
            if transcript_data.get('audio_context', {}).get('likely_sound_source'):
                print(f"[INFO] Likely sound source: {transcript_data['audio_context']['likely_sound_source']}")
        except Exception as e:
            print(f"[ERROR] Transcription error: {e}")
            transcript_data = {
                'transcript': "",
                'quality': 'error',
                'quality_reason': str(e),
                'is_reliable': False,
                'audio_context': {}
            }

        # Get knowledge context using smart RAG retrieval
        try:
            print("[INFO] Loading knowledge using smart RAG retrieval...")
            
            # Try smart context first
            knowledge_context, knowledge_citations = retrieve_smart_context(
                transcript=transcript_data.get('transcript', ''),
                frames=frames_summaries_text[:1000],
                creator_note=form_data['creator_note'],
                goal=form_data['goal'],
                max_chars=75000
            )
            
            if knowledge_context and len(knowledge_context) > 1000:
                print(f"[SUCCESS] Smart RAG retrieved {len(knowledge_context)} chars")
                print(f"[SUCCESS] Citations: {len(knowledge_citations)} relevant chunks")
            else:
                # Fallback to retrieve all
                print("[INFO] Smart retrieval insufficient, loading all context...")
                knowledge_context, knowledge_citations = retrieve_all_context(max_chars=100000)
                print(f"[SUCCESS] Loaded {len(knowledge_context)} chars from knowledge base")
            
        except Exception as e:
            print(f"[ERROR] Knowledge loading error: {e}")
            import traceback
            traceback.print_exc()
            
            # Minimal fallback
            knowledge_context = """
Key patterns for video analysis:
- Strong hooks create curiosity in first 3 seconds
- Front-load value and leave payoff till the end
- Use pattern interrupts to stop scrolls
- Match content to platform expectations
- Visual satisfaction drives completion
- Audio-visual synchronization enhances retention
"""
            knowledge_citations = ["Basic patterns fallback"]

        # Run comprehensive analysis
        try:
            gpt_result = run_main_analysis(
                transcript_data.get('transcript', ''),
                frames_summaries_text,
                form_data['creator_note'],
                form_data['platform'],
                form_data['target_duration'],
                form_data['goal'],
                form_data['tone'],
                form_data['audience'],
                knowledge_context
            )
            
            # Add transcript quality info
            gpt_result['transcript_quality'] = transcript_data
            
            print("[SUCCESS] Analysis complete")
            print(f"[INFO] Content type: {gpt_result.get('content_type_detected', 'unknown')}")
            print(f"[INFO] Audio type: {gpt_result.get('audio_type_detected', 'unknown')}")
            print(f"[INFO] Performance level: {gpt_result.get('performance_level', 'unknown')}")
            
        except Exception as e:
            print(f"[ERROR] Analysis error: {e}")
            import traceback
            traceback.print_exc()
            
            # Use comprehensive fallback
            audio_context = transcript_data.get('audio_context', {})
            visual_analysis = create_visual_content_description(frames_summaries_text, audio_context)
            
            has_speech = audio_context.get('has_meaningful_speech', False)
            view_count = None
            performance_level = 'unknown'
            
            # Try to extract view count for fallback
            if form_data['creator_note']:
                view_patterns = re.findall(r'(\d+\.?\d*)\s*(k|thousand|m|million)', 
                                         form_data['creator_note'].lower())
                if view_patterns:
                    number, unit = view_patterns[0]
                    if unit in ['k', 'thousand']:
                        view_count = f"{number}k"
                        performance_level = 'moderate' if float(number) >= 100 else 'low'
                    elif unit in ['m', 'million']:
                        view_count = f"{number}M"
                        performance_level = 'viral'
            
            gpt_result = create_comprehensive_fallback(
                transcript_data.get('transcript', ''),
                frames_summaries_text,
                form_data['creator_note'],
                form_data['platform'],
                form_data['goal'],
                form_data['audience'],
                has_speech,
                view_count,
                performance_level,
                knowledge_context,
                audio_context,
                visual_analysis
            )

        # Prepare template variables
        try:
            template_vars = prepare_template_variables(
                gpt_result, 
                transcript_data, 
                frames_summaries_text, 
                form_data, 
                gallery_data_urls, 
                frame_paths, 
                frames_dir, 
                knowledge_citations, 
                knowledge_context
            )
            print("[SUCCESS] Template variables prepared")
        except Exception as e:
            print(f"[ERROR] Template preparation error: {e}")
            return f"Error preparing results: {str(e)}", 500

        print("[INFO] Rendering results template")
        return render_template("results.html", **template_vars)

    except Exception as e:
        print(f"[ERROR] Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Unexpected error: {str(e)}", 500


if __name__ == "__main__":
    validate_dependencies()
    app.run(host="0.0.0.0", port=10000, debug=True)