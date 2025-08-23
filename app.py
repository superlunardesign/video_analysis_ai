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

def analyze_audio_type(transcript_text, has_speech):
    """Intelligently detect audio type without over-focusing on viral"""
    
    if not has_speech or not transcript_text or len(transcript_text.strip()) < 20:
        return {
            'type': 'visual_only',
            'viral_audio_check': False,
            'confidence': 'high'
        }
    
    transcript_lower = transcript_text.lower()
    
    # Strong viral audio indicators (high confidence)
    strong_viral_patterns = [
        ('male' in transcript_lower and 'female' in transcript_lower and 'vs' in transcript_lower),
        ('pov:' in transcript_lower and '?' in transcript_text),
        (transcript_text.count('"') >= 4),  # Multiple quoted sections suggest dialogue
    ]
    
    # Moderate viral indicators (medium confidence)
    moderate_viral_patterns = [
        ('?' in transcript_text and len(transcript_text.split('.')) > 2),
        (len([w for w in transcript_text.split() if w.lower() in ['you', 'your']]) > 5),
        any(phrase in transcript_lower for phrase in ['when you', 'tell me', 'wait for it']),
    ]
    
    # Count pattern matches
    strong_matches = sum(strong_viral_patterns)
    moderate_matches = sum(moderate_viral_patterns)
    
    # Only classify as viral if strong evidence
    if strong_matches >= 1 or moderate_matches >= 2:
        return {
            'type': 'potential_viral_audio',
            'viral_audio_check': True,
            'confidence': 'high' if strong_matches >= 1 else 'medium'
        }
    else:
        return {
            'type': 'original_speech',
            'viral_audio_check': False,
            'confidence': 'high'
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


def enhanced_transcribe_audio(audio_path):
    """Enhanced transcription with quality analysis"""
    try:
        transcript = transcribe_audio(audio_path)
        
        if not transcript or len(transcript.strip()) < 10:
            return {
                'transcript': transcript if transcript else "",
                'quality': 'poor',
                'quality_reason': 'Transcript too short or empty',
                'is_reliable': False
            }
        
        words = transcript.lower().split()
        
        # Check for music/ambient indicators
        music_indicators = ['music', 'sound', 'noise', 'audio', 'background']
        if any(indicator in transcript.lower() for indicator in music_indicators):
            return {
                'transcript': transcript,
                'quality': 'ambient',
                'quality_reason': 'Contains music/ambient audio descriptions',
                'is_reliable': False
            }
        
        # Check for repetitive content
        if len(words) > 0 and len(set(words)) < len(words) * 0.3:
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
        print(f"[ERROR] Transcription error: {e}")
        return {
            'transcript': f"(Transcription error: {str(e)})",
            'quality': 'error',
            'quality_reason': str(e),
            'is_reliable': False
        }


def create_visual_content_description(frames_summaries_text, audio_description=None):
    """Analyze visual content type and satisfaction potential"""
    try:
        frames_lower = frames_summaries_text.lower()
        content_type = 'general'
        
        if 'drawing' in frames_lower or 'art' in frames_lower:
            content_type = 'visual_process'
        elif 'transformation' in frames_lower or 'before' in frames_lower:
            content_type = 'transformation'
        elif 'routine' in frames_lower or 'skincare' in frames_lower:
            content_type = 'routine'
        elif 'cooking' in frames_lower or 'recipe' in frames_lower:
            content_type = 'tutorial'
        
        satisfaction_indicators = ['completion', 'finish', 'result', 'final', 'transform', 'reveal']
        highly_satisfying = any(word in frames_lower for word in satisfaction_indicators)
        
        return {
            'description': f"Visual analysis: {frames_summaries_text[:200]}...",
            'content_type': content_type,
            'has_strong_visual_narrative': len(frames_summaries_text) > 200,
            'satisfaction_analysis': {
                'highly_satisfying': highly_satisfying,
                'completion_elements': [word for word in satisfaction_indicators if word in frames_lower]
            }
        }
    except Exception as e:
        print(f"[ERROR] Visual content description failed: {e}")
        return {
            'description': "Visual content analysis",
            'content_type': 'general',
            'has_strong_visual_narrative': False,
            'satisfaction_analysis': {'highly_satisfying': False, 'completion_elements': []}
        }


# ==============================
# MAIN ANALYSIS FUNCTION - COMPREHENSIVE & CONVERSATIONAL
# ==============================

def run_main_analysis(transcript_text, frames_summaries_text, creator_note, platform, target_duration, goal, tone, audience, knowledge_context):
    """Comprehensive analysis with conversational, educational output"""
    
    # Detect content characteristics
    has_speech = transcript_text and len(transcript_text.strip()) > 20 and not any(
        indicator in transcript_text.lower() 
        for indicator in ['music', 'sound', 'noise', 'audio', 'background', 'ambient']
    )
    
    # Smart audio type detection
    audio_type_info = analyze_audio_type(transcript_text, has_speech)
    
    # Performance detection from creator notes
    is_high_performing = False
    performance_indicators = []
    if creator_note:
        note_lower = creator_note.lower()
        if any(word in note_lower for word in ['million', 'viral', 'blew up', 'exploded']):
            is_high_performing = True
            numbers = re.findall(r'(\d+\.?\d*)\s*(million|m|k|thousand)', note_lower)
            if numbers:
                performance_indicators.append(f"{numbers[0][0]}{numbers[0][1]}")
        elif any(word in note_lower for word in ['low', 'poor', "didn't work", 'no views', 'flopped']):
            is_high_performing = False
            performance_indicators.append("underperformed")
    
    # Build knowledge section with specific examples
    knowledge_section = ""
    if knowledge_context and len(knowledge_context.strip()) > 100:
        knowledge_section = f"""
PROVEN PATTERNS FROM YOUR KNOWLEDGE BASE:
{knowledge_context}

INSTRUCTIONS: Use these SPECIFIC examples and patterns in your explanation. 
Reference exact view counts, quote successful hooks, cite timing breakdowns.
Don't give generic advice - compare to the actual examples above.
"""
    
    # Content-specific context
    if audio_type_info['viral_audio_check'] and audio_type_info['confidence'] == 'high':
        audio_analysis_section = """
VIRAL AUDIO ANALYSIS (High confidence viral audio detected):
- Identify the audio source and format
- Analyze audio-visual synchronization
- Explain viral mechanics and shareability
- Assess replication potential
"""
    elif has_speech:
        audio_analysis_section = """
ORIGINAL SPEECH ANALYSIS:
- Message clarity and delivery
- Value proposition effectiveness
- Authority and credibility signals
- Engagement optimization opportunities
"""
    else:
        audio_analysis_section = """
VISUAL-ONLY CONTENT ANALYSIS:
- Visual storytelling effectiveness
- Satisfaction and transformation mechanics
- Pattern interrupts and attention retention
- Rewatch and share factors
"""
    
    # Build conversational prompt
    prompt = f"""
You are a viral content expert having a friendly, educational conversation with a creator about their video.
They told you: "{creator_note if creator_note else 'Help me analyze this video'}"

Your job is to TEACH them the psychology behind what's happening, using specific examples from the knowledge base.
Be conversational but precise. Reference exact moments, compare to successful examples, explain the WHY.

CONTENT TO ANALYZE:
Platform: {platform}
Transcript: {transcript_text if has_speech else "(No speech - visual/ambient only)"}
Visual Content: {frames_summaries_text}
Target: {target_duration}s video for {audience}
Goal: {goal}

{knowledge_section}

CONVERSATIONAL TONE REQUIREMENTS:
- Start responses with "Let me break down..."
- Use specific comparisons: "Your opening vs 'I can't believe they sent this' (5.1M views)"
- Explain psychology: "This works because humans have a 0.3 second decision window..."
- Reference exact timestamps: "At 0:03, you show X, but successful videos do Y"
- Use natural language: "Here's the thing..." not "Additionally, it should be noted..."

ANALYSIS STRUCTURE:

1. OPENING ANALYSIS
Start: "Let me break down exactly {'why you ' + creator_note if creator_note else 'what I see here'}..."
- Compare to specific successful examples from knowledge
- Explain the psychology of what's working/not working
- Reference exact moments and timings

2. HOOK BREAKDOWN
Don't just describe - TEACH:
"Your first second shows [X]. Compare that to [specific example] which got [Y views]. 
The difference? They create mystery in 0.5 seconds, you reveal everything immediately..."

3. ENGAGEMENT PSYCHOLOGY
Explain mechanisms with examples:
"People comment when... Look at how [example] triggered 50K comments by..."
"Shares happen when... [Specific video] got 100K shares because..."

4. ALTERNATIVE HOOKS
For each hook, explain WHY it would work:
"Try 'I can't believe they sent this' - this creates what psychologists call a knowledge gap..."

5. IMPROVEMENTS
Reference specific patterns:
"Instead of [what they did], try [specific technique from knowledge base]. 
[Creator name] used this exact approach and got [X views]..."

{audio_analysis_section}

Respond in JSON but make EVERY field conversational and educational:

{{
  "analysis": "Let me break down exactly what's happening here. [2-3 paragraphs of specific, example-rich analysis]",
  
  "content_type_detected": "{audio_type_info['type']}",
  
  "video_type_analysis": "So you've created a {audio_type_info['type'].replace('_', ' ')} video. Here's how this type actually works: [explain with examples]",
  
  "exact_hook_breakdown": {{
    "first_second": "0:00 - You open with [exact description]. Here's the problem: [compare to successful example]",
    "second_second": "0:01 - [What happens]. Compare this to [viral example] which instead [what they do]",
    "third_second": "0:02 - By now, [explain viewer psychology and decision-making]",
    "visual_hook": "Visually, you show [X]. [Specific successful video] shows [Y] instead, which [psychology explanation]",
    "text_hook": "Your text says [exact quote]. This [works/doesn't work] because [specific reason with example]",
    "audio_hook": "Audio-wise, [what's heard]. The [specific viral video] uses [X sound] to trigger [psychological response]",
    "why_it_works_or_not": "Here's the brutal truth: [specific comparison to successful patterns with numbers]"
  }},
  
  "performance_analysis": "{'Based on what you told me about getting ' + creator_note if creator_note else 'Looking at the structure'}, here's exactly why: [specific reasons with pattern comparisons]",
  
  "hooks": [
    "Hook option 1 - This works because [psychological principle with example]",
    "Hook option 2 - Similar to [viral video example] but adapted for your content",
    "Hook option 3 - Creates curiosity gap like [specific example with views]",
    "Hook option 4 - Uses pattern interrupt technique from [successful creator]",
    "Hook option 5 - Proven format: [specific example] got [X views] with this"
  ],
  
  "scores": {{
    "hook_strength": "[1-10] - [Specific comparison to successful hooks]",
    "promise_clarity": "[1-10] - [Explain what promise exists vs successful examples]",
    "retention_design": "[1-10] - [Reference specific pacing patterns]",
    "engagement_potential": "[1-10] - [Compare triggers to viral examples]",
    "goal_alignment": "[1-10] - [How this serves {goal} based on patterns]"
  }},
  
  "engagement_psychology": "Let's talk about why people actually engage: [specific mechanisms with real examples from knowledge base]",
  
  "strengths": "What you're doing well: [specific elements that align with successful patterns]",
  
  "improvement_areas": "Here's exactly what to fix: [specific changes based on proven patterns with examples]",
  
  "timing_breakdown": "Your video flow: 0-3s: [what happens vs what should], 3-10s: [analysis with pattern comparison]...",
  
  "formulas": {{
    "basic_formula": "Step 1: [Action from successful pattern]\\nStep 2: [Why this works]\\nStep 3: [Expected result based on examples]",
    "timing_formula": "0-1s: [Do X like successful example Y]\\n1-3s: [Create Z like video that got A views]...",
    "visual_formula": "[Element] like in [example] → [Next] as shown in [pattern] → [Result]",
    "psychology_formula": "Create [emotion] (like [example]) → Build [feeling] (using [technique]) → Deliver [satisfaction]"
  }},
  
  "performance_prediction": "Based on these patterns, if you posted this: [specific prediction with reasoning from knowledge base]",
  
  "knowledge_insights": "Comparing to our viral patterns database: [specific numbered comparisons with examples]",
  
  "viral_audio_analysis": {{}} if not audio_type_info['viral_audio_check'] else {{
    "is_viral_sound": true,
    "audio_source": "[Detected source type]",
    "viral_mechanics": "[Why this audio works with examples]",
    "replication_potential": "[How others can use this based on patterns]"
  }},
  
  "content_analysis": {{
    "type": "{audio_type_info['type']}",
    "key_insights": "[Specific insights for this content type with examples]",
    "optimization_opportunities": "[Specific improvements based on successful patterns]"
  }}
}}

Remember: You're TEACHING using specific examples, not just analyzing. Every point should reference patterns from the knowledge base.
"""

    try:
        print(f"[INFO] Sending comprehensive analysis to GPT-4...")
        print(f"[INFO] Using {len(knowledge_context)} chars of knowledge context")
        
        gpt_response = _api_retry(
            client.chat.completions.create,
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a friendly viral content expert who teaches creators by comparing their content to specific successful examples. Always be conversational and educational, explaining the psychology behind everything."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,  # Higher for conversational tone
            max_tokens=4000
        )

        response_text = gpt_response.choices[0].message.content.strip()
        
        # Clean JSON response
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        parsed = json.loads(response_text.strip())
        
        # Process scores with performance awareness
        scores_raw = parsed.get("scores", {})
        if is_high_performing:
            score_defaults = {"hook_strength": 8, "promise_clarity": 8, "retention_design": 8, 
                            "engagement_potential": 8, "goal_alignment": 8}
        elif performance_indicators and "underperformed" in performance_indicators:
            score_defaults = {"hook_strength": 4, "promise_clarity": 4, "retention_design": 5, 
                            "engagement_potential": 4, "goal_alignment": 4}
        else:
            score_defaults = {"hook_strength": 6, "promise_clarity": 6, "retention_design": 6, 
                            "engagement_potential": 5, "goal_alignment": 6}
        
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
        
        # Build comprehensive result
        exact_hook = parsed.get("exact_hook_breakdown", {})
        formulas = parsed.get("formulas", {})
        
        result = {
            # Core analysis
            "analysis": parsed.get("analysis", ""),
            "content_type_detected": parsed.get("content_type_detected", audio_type_info['type']),
            "video_type_analysis": parsed.get("video_type_analysis", ""),
            "performance_analysis": parsed.get("performance_analysis", ""),
            
            # Hooks
            "hooks": parsed.get("hooks", []),
            "scores": scores,
            
            # Hook breakdown
            "exact_hook_breakdown": exact_hook,
            "visual_hook": exact_hook.get("visual_hook", ""),
            "text_hook": exact_hook.get("text_hook", ""),
            "verbal_hook": exact_hook.get("audio_hook", ""),
            "why_hook_works": exact_hook.get("why_it_works_or_not", ""),
            
            # Psychology
            "engagement_psychology": parsed.get("engagement_psychology", ""),
            
            # Content analysis
            "viral_audio_analysis": parsed.get("viral_audio_analysis", {}),
            "content_analysis": parsed.get("content_analysis", {}),
            
            # Improvements
            "strengths": parsed.get("strengths", ""),
            "improvement_areas": parsed.get("improvement_areas", ""),
            
            # Formulas
            "basic_formula": formulas.get("basic_formula", ""),
            "timing_formula": formulas.get("timing_formula", ""),
            "visual_formula": formulas.get("visual_formula", ""),
            "psychology_formula": formulas.get("psychology_formula", ""),
            "timing_breakdown": parsed.get("timing_breakdown", ""),
            
            # Predictions
            "performance_prediction": parsed.get("performance_prediction", ""),
            "knowledge_insights": parsed.get("knowledge_insights", ""),
            
            # Compatibility fields
            "formula": formulas.get("basic_formula", ""),
            "improvements": parsed.get("improvement_areas", ""),
            "template_formula": formulas.get("visual_formula", ""),
            
            # Meta
            "knowledge_context_used": bool(knowledge_context.strip()),
            "overall_quality": "strong" if sum(scores.values())/len(scores) >= 7 else "moderate" if sum(scores.values())/len(scores) >= 5 else "needs_work",
            "video_has_speech": has_speech,
            "audio_type_detected": audio_type_info['type']
        }
        
        return result
        
    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Return conversational fallback
        return create_conversational_fallback(
            transcript_text, frames_summaries_text, creator_note, 
            platform, goal, audience, has_speech, is_high_performing
        )


def create_conversational_fallback(transcript_text, frames_summaries_text, creator_note, platform, goal, audience, has_speech, is_high_performing):
    """Conversational fallback that still provides educational value"""
    
    performance_text = creator_note if creator_note else "your video"
    
    if not has_speech:
        analysis = f"""Let me break down what's happening here. You've created a visual-only video, which can absolutely work - but only if you nail the visual storytelling. 

Looking at your frames, {frames_summaries_text[:200]}... The issue is that without audio, you're fighting for attention with only half your arsenal. 

Successful visual-only content needs what I call 'visual velocity' - something changing every 2-3 seconds to maintain pattern interrupts. Think about those satisfying cake decorating videos that get millions of views - they work because every second reveals progress toward a satisfying conclusion."""
    else:
        analysis = f"""Okay, so here's what I'm seeing with {performance_text}. You've got both visual and verbal elements, which gives you more tools to work with, but you need to synchronize them better.

Your transcript shows: {transcript_text[:150]}... This could work, but the timing feels off. 

Here's the thing about {platform} - viewers make a decision to keep watching in literally 0.3 seconds. Your opening needs to create what psychologists call a 'curiosity gap' - you have to make them NEED to know what happens next."""
    
    base_scores = {
        "hook_strength": 7 if is_high_performing else 5,
        "promise_clarity": 6 if is_high_performing else 4,
        "retention_design": 7 if is_high_performing else 5,
        "engagement_potential": 6 if is_high_performing else 4,
        "goal_alignment": 6 if is_high_performing else 5
    }
    
    return {
        "analysis": analysis,
        "content_type_detected": "visual_only" if not has_speech else "original_speech",
        "video_type_analysis": f"You're working with {'visual-only content' if not has_speech else 'verbal and visual content'}. The key is understanding that {audience} scrolls past 100+ videos per session. You need to pattern-interrupt their scroll.",
        "performance_analysis": f"Based on what you've told me ({creator_note}), this performance makes sense. You're competing against creators who understand the exact psychological triggers that stop scrolls.",
        "hooks": [
            "wait for the transformation - creates curiosity gap",
            "POV: you discover the most satisfying process - relatability + promise",
            "this is oddly satisfying - leverages ASMR psychology",
            "the ending will blow your mind - knowledge gap",
            "you won't believe how this turns out - challenge + curiosity"
        ],
        "scores": base_scores,
        "exact_hook_breakdown": {
            "first_second": "0:00 - Opening establishes context",
            "second_second": "0:01 - Hook develops",
            "third_second": "0:02 - Attention secured",
            "visual_hook": "Visual elements create interest",
            "text_hook": "Text reinforces message",
            "audio_hook": "Audio sets tone",
            "why_it_works_or_not": "Combination creates engagement"
        },
        "engagement_psychology": "Here's what drives engagement: People comment when they disagree or want to add something. They share when it makes them look good. They save when they might need it later.",
        "strengths": "You understand the platform and you're creating content. That puts you ahead of 90% of people who just consume.",
        "improvement_areas": "Focus on your first 3 seconds. Study videos in your niche that get millions of views. What do they do at 0:00 that you don't?",
        "timing_breakdown": "0-3s: Critical hook window, 3-10s: Develop promise, 10-20s: Core value, 20-30s: Payoff",
        "basic_formula": "Step 1: Pattern interrupt\nStep 2: Create curiosity gap\nStep 3: Deliver value",
        "timing_formula": "0-1s: Stop the scroll\n1-3s: Make them curious\n3-10s: Build anticipation",
        "visual_formula": "Unexpected visual → Progressive reveal → Satisfying conclusion",
        "psychology_formula": "Attention → Interest → Desire → Action",
        "performance_prediction": "With these improvements, you could 10x your views. The difference between 300 and 3000 views is usually just the first 3 seconds.",
        "knowledge_insights": "The patterns that work haven't changed - humans still want mystery, satisfaction, and social currency.",
        "knowledge_context_used": False,
        "overall_quality": "moderate",
        "video_has_speech": has_speech,
        "audio_type_detected": "visual_only" if not has_speech else "original_speech"
    }


def run_main_analysis_safe(transcript_text, frames_summaries_text, creator_note, platform, target_duration, goal, tone, audience, knowledge_context):
    """Safe wrapper for main analysis"""
    try:
        result = run_main_analysis(
            transcript_text, frames_summaries_text, creator_note, 
            platform, target_duration, goal, tone, audience, knowledge_context
        )
        
        # Validate result
        if not isinstance(result, dict):
            raise ValueError("Analysis did not return expected format")
        
        required_fields = ['analysis', 'hooks', 'scores']
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing required field: {field}")
        
        return result
        
    except Exception as e:
        print(f"[ERROR] Main analysis failed: {e}")
        
        # Determine basic video characteristics for fallback
        has_speech = transcript_text and len(transcript_text.strip()) > 20
        is_high_performing = creator_note and any(
            word in creator_note.lower() 
            for word in ['million', 'viral', 'blew up']
        )
        
        return create_conversational_fallback(
            transcript_text, frames_summaries_text, creator_note,
            platform, goal, audience, has_speech, is_high_performing
        )


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
        
        # Timing and formulas
        'timing_breakdown': gpt_result.get('timing_breakdown', ''),
        'formula': gpt_result.get('formula', gpt_result.get('basic_formula', '')),
        'basic_formula': gpt_result.get('basic_formula', ''),
        'timing_formula': gpt_result.get('timing_formula', ''),
        'template_formula': gpt_result.get('template_formula', gpt_result.get('visual_formula', '')),
        'psychology_formula': gpt_result.get('psychology_formula', ''),
        
        # Transcript data
        'transcript': transcript_data.get('transcript', ''),
        'transcript_quality': transcript_data,
        'transcript_original': transcript_data.get('transcript', ''),
        'transcript_for_analysis': transcript_data.get('transcript', ''),
        
        # Frame analysis
        'frame_summary': frames_summaries_text if frames_summaries_text else "",
        'frame_summaries': [block.strip() for block in frames_summaries_text.split('\n\n') if block.strip()] if frames_summaries_text else [],
        
        # Enhanced analysis fields
        'visual_content_analysis': gpt_result.get('visual_content_analysis', {}),
        'viral_audio_analysis': gpt_result.get('viral_audio_analysis', {}),
        'content_analysis': gpt_result.get('content_analysis', {}),
        'performance_prediction': gpt_result.get('performance_prediction', ''),
        'knowledge_insights': gpt_result.get('knowledge_insights', ''),
        'performance_analysis': gpt_result.get('performance_analysis', ''),
        'video_type_analysis': gpt_result.get('video_type_analysis', ''),
        'exact_hook_breakdown': gpt_result.get('exact_hook_breakdown', {}),
        
        # Compatibility fields for templates
        'gpt_response': gpt_result.get('analysis', ''),
        'psychological_breakdown': gpt_result.get('analysis', ''),
        'hook_mechanics': gpt_result.get('timing_breakdown', ''),
        'engagement_psychology': gpt_result.get('engagement_psychology', ''),
        'viral_mechanisms': gpt_result.get('viral_mechanisms', ''),
        'audience_psychology': gpt_result.get('audience_psychology', ''),
        
        # Meta information
        'knowledge_context_used': gpt_result.get('knowledge_context_used', False),
        'overall_quality': gpt_result.get('overall_quality', 'moderate'),
        'content_type_detected': gpt_result.get('content_type_detected', ''),
        'audio_type_detected': gpt_result.get('audio_type_detected', ''),
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

        # Transcribe audio
        try:
            transcript_data = enhanced_transcribe_audio(audio_path)
            print(f"[INFO] Transcript quality: {transcript_data['quality']}")
        except Exception as e:
            print(f"[ERROR] Transcription error: {e}")
            transcript_data = {
                'transcript': "",
                'quality': 'error',
                'quality_reason': str(e),
                'is_reliable': False
            }

        # Analyze frames
        try:
            frames_summaries_text, gallery_data_urls = analyze_frames_batch(frame_paths)
            print(f"[SUCCESS] Frame analysis complete")
        except Exception as e:
            print(f"[ERROR] Frame analysis error: {e}")
            frames_summaries_text = ""
            gallery_data_urls = []

        # --- Get knowledge context using SMART RAG retrieval ---
        try:
            print("[INFO] Loading knowledge using smart RAG retrieval...")
            
            # Try smart context first (most relevant)
            knowledge_context, knowledge_citations = retrieve_smart_context(
                transcript=transcript_data.get('transcript', ''),
                frames=frames_summaries_text[:1000],  # First 1000 chars
                creator_note=form_data['creator_note'],
                goal=form_data['goal'],
                max_chars=75000  # Get 75K of RELEVANT content!
            )
            
            if knowledge_context and len(knowledge_context) > 1000:
                print(f"[SUCCESS] Smart RAG retrieved {len(knowledge_context)} chars of relevant knowledge")
                print(f"[SUCCESS] Citations: {len(knowledge_citations)} relevant chunks found")
            else:
                # Fallback to retrieve all if smart retrieval fails
                print("[INFO] Smart retrieval insufficient, loading all context...")
                knowledge_context, knowledge_citations = retrieve_all_context(max_chars=100000)
                print(f"[SUCCESS] Loaded {len(knowledge_context)} chars from knowledge base")
            
        except Exception as e:
            print(f"[ERROR] Knowledge loading error: {e}")
            import traceback
            traceback.print_exc()
            
            # Final fallback
            knowledge_context = """
PROVEN VIRAL CONTENT PATTERNS:

For unboxing videos getting only 300 views:
- Your hook is too generic (showing package = instant scroll)
- No mystery or controversy in first 3 seconds
- Missing ASMR audio elements (knife cutting tape sound)
- Reveal happens too early or too late (optimal: 8-12 seconds)

HOOKS THAT ACTUALLY WORK:
- "I can't believe they sent this..." (5.1M views)
- "The package that got me banned" (3.2M views)
- "Opening what I shouldn't have bought" (2.8M views)

ENGAGEMENT MECHANICS:
- Comments: Create controversy about value/authenticity
- Shares: Make it about a deal others can get
- Saves: Include discount code or limited availability
"""
            knowledge_citations = ["Fallback patterns"]

        # Run main analysis
        try:
            gpt_result = run_main_analysis_safe(
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
            
            # Add visual content analysis if needed
            if not transcript_data.get('is_reliable', False):
                gpt_result['visual_content_analysis'] = create_visual_content_description(
                    frames_summaries_text
                )
            
            gpt_result['transcript_quality'] = transcript_data
            
            print("[SUCCESS] Analysis complete")
            
        except Exception as e:
            print(f"[ERROR] Analysis error: {e}")
            import traceback
            traceback.print_exc()
            
            # Use fallback
            has_speech = transcript_data.get('is_reliable', False)
            is_high_performing = form_data['creator_note'] and 'million' in form_data['creator_note'].lower()
            
            gpt_result = create_conversational_fallback(
                transcript_data.get('transcript', ''),
                frames_summaries_text,
                form_data['creator_note'],
                form_data['platform'],
                form_data['goal'],
                form_data['audience'],
                has_speech,
                is_high_performing
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