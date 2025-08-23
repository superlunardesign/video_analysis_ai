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
    """Conservatively detect audio type - avoid false positives for viral audio"""
    
    if not has_speech or not transcript_text or len(transcript_text.strip()) < 20:
        return {
            'type': 'visual_only',
            'viral_audio_check': False,
            'confidence': 'high'
        }
    
    transcript_lower = transcript_text.lower()
    
    # Very strong indicators only - must have multiple signs
    strong_viral_patterns = 0
    
    # Check for dialogue structure (back and forth conversation)
    if '?' in transcript_text and '"' in transcript_text and transcript_text.count('"') >= 4:
        strong_viral_patterns += 1
    
    # Check for POV format with specific structure
    if 'pov:' in transcript_lower and ('when' in transcript_lower or 'you' in transcript_lower):
        strong_viral_patterns += 1
    
    # Check for clear dialogue markers
    if all(marker in transcript_lower for marker in ['he said', 'she said']):
        strong_viral_patterns += 1
    
    # Only mark as viral if VERY strong evidence
    if strong_viral_patterns >= 2:
        return {
            'type': 'potential_viral_audio',
            'viral_audio_check': True,
            'confidence': 'medium'
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
        elif 'unboxing' in frames_lower or 'package' in frames_lower:
            content_type = 'unboxing'
        
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
# MAIN ANALYSIS FUNCTION - ACCURATE & EDUCATIONAL
# ==============================

def run_main_analysis(transcript_text, frames_summaries_text, creator_note, platform, target_duration, goal, tone, audience, knowledge_context):
    """Analysis based on ACTUAL content without hallucinations"""
    
    # Detect content characteristics
    has_speech = transcript_text and len(transcript_text.strip()) > 20 and not any(
        indicator in transcript_text.lower() 
        for indicator in ['music', 'sound', 'noise', 'audio', 'background', 'ambient']
    )
    
    # Conservative audio type detection
    audio_type_info = analyze_audio_type(transcript_text, has_speech)
    
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
                    view_coundef run_main_analysis(transcript_text, frames_summaries_text, creator_note, platform, target_duration, goal, tone, audience, knowledge_context):
    """Analysis based on ACTUAL content without hallucinations - DEEP & COMPREHENSIVE"""
    
    # Detect content characteristics
    has_speech = transcript_text and len(transcript_text.strip()) > 20 and not any(
        indicator in transcript_text.lower() 
        for indicator in ['music', 'sound', 'noise', 'audio', 'background', 'ambient']
    )
    
    # Conservative audio type detection
    audio_type_info = analyze_audio_type(transcript_text, has_speech)
    
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
    
    # Build knowledge section - ENHANCED
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
8. VIRAL MECHANICS: If viral, explain the specific mechanisms that drove sharing
"""

    prompt = f"""
You are a video psychology expert analyzing a {platform} video. {"This video went VIRAL with " + view_count + " - analyze WHY it succeeded." if performance_level == 'viral' else f"This video got {view_count if view_count else 'certain performance'} - analyze what's working and what could improve."}

CRITICAL CONTEXT:
- Platform: {platform}
- Performance: {view_count if view_count else 'Not specified'} ({performance_level})
- Creator's note: "{creator_note if creator_note else 'No additional context'}"
- Content type detected: {audio_type_info['type']}
- Goal: {goal}
- Target audience: {audience}
- Duration target: {target_duration}s

SPOKEN CONTENT (transcript - what's SAID):
{transcript_text if transcript_text else "(No speech detected - visual/audio only)"}

VISUAL CONTENT (frames - what's SHOWN/WRITTEN):
{frames_summaries_text}

{knowledge_section}

COMPREHENSIVE ANALYSIS INSTRUCTIONS:
{"Since this went VIRAL, identify the EXACT psychological triggers and viral mechanics." if performance_level == 'viral' else "Identify opportunities for improvement based on proven patterns."}

1. FIRST 3 SECONDS BREAKDOWN:
   - Frame by frame: What EXACTLY appears?
   - What text overlays are shown (from frames, not transcript)?
   - What's spoken (from transcript, not frames)?
   - What visual hooks grab attention?
   - Rate the hook strength and explain WHY

2. VIRAL/PERFORMANCE MECHANICS:
   {"- What specific elements made this shareable?\n   - What psychological triggers drove the viral spread?\n   - How did it tap into platform algorithms?\n   - What made people watch to completion?" if performance_level == 'viral' else "- What's preventing viral growth?\n   - Which psychological triggers are missing?\n   - How could platform algorithms be better leveraged?\n   - Where do viewers likely drop off?"}

3. CONTENT STRUCTURE ANALYSIS:
   - Hook mechanism (0-3s): How does it stop scrolling?
   - Promise delivery (3-10s): What value is promised?
   - Retention mechanics (middle): What keeps viewers?
   - Payoff (end): How does it satisfy or create sharing impulse?

4. PSYCHOLOGICAL DEPTH:
   - What emotions does this trigger?
   - What curiosity gaps are created?
   - How does it leverage social dynamics?
   - What makes it memorable or shareable?

5. PATTERN MATCHING:
   - Which proven patterns from the knowledge base apply?
   - How well does it execute these patterns?
   - What patterns could be better implemented?

Respond in JSON with DEEP, SPECIFIC insights:

{{
  "analysis": "{'This video achieved viral status because...' if performance_level == 'viral' else 'This video shows potential but...'} [2-3 paragraphs of DEEP psychological and structural analysis. Explain the WHY behind everything. Reference specific moments and patterns.]",
  
  "viral_mechanics": "{'Here are the specific viral triggers: ' if performance_level == 'viral' else 'To achieve viral potential: '}[Detailed explanation of psychological mechanisms, sharing triggers, algorithm optimization]",
  
  "exact_hook_breakdown": {{
    "first_frame": "0:00 - [EXACTLY what appears in frame 1]",
    "second_moment": "0:01 - [EXACTLY what happens in second 1]",
    "third_second": "0:02 - [EXACTLY what occurs by second 3]",
    "visual_elements": "[Specific visual hooks from frames]",
    "text_overlays": "[EXACT text shown on screen from frame descriptions]",
    "spoken_hook": "[EXACT opening words from transcript]",
    "hook_psychology": "[Deep explanation of why this hook works/doesn't work psychologically]",
    "hook_score": [1-10],
    "hook_reasoning": "[Specific reasoning for the score based on proven patterns]"
  }},
  
  "performance_deep_dive": "{'With ' + view_count + ', this demonstrates...' if view_count else 'The performance indicates...'}[3-4 sentences explaining the specific reasons for this performance level, referencing actual content elements and psychological principles]",
  
  "psychological_breakdown": {{
    "emotional_triggers": ["List specific emotions triggered and when"],
    "curiosity_mechanisms": ["How curiosity gaps are created"],
    "social_dynamics": ["How it leverages social psychology"],
    "cognitive_biases": ["Which biases it exploits"],
    "sharing_psychology": ["Why people would/wouldn't share this"]
  }},
  
  "content_structure_analysis": {{
    "hook_phase": "0-3s: [What happens and WHY it works/doesn't]",
    "development_phase": "3-10s: [How value is established]",
    "retention_phase": "10-20s: [What maintains attention]",
    "payoff_phase": "20s+: [How it delivers satisfaction]",
    "structure_effectiveness": [1-10],
    "structure_insights": "[Why this structure works for this content type]"
  }},
  
  "hooks": [
    "{'Even better hook: ' if performance_level == 'viral' else 'Improved hook 1: '}[Specific, natural language hook with explanation]",
    "Alternative angle: [Different psychological approach with reasoning]",
    "Pattern-based hook: [Hook using proven pattern from knowledge base]",
    "Curiosity-driven: [Hook that creates stronger curiosity gap]",
    "Emotional trigger: [Hook targeting different emotion with explanation]"
  ],
  
  "scores": {{
    "hook_strength": [1-10 based on actual effectiveness],
    "promise_clarity": [1-10 based on value proposition],
    "retention_design": [1-10 based on completion likelihood],
    "engagement_potential": [1-10 based on interaction drivers],
    "viral_potential": [1-10 based on sharing likelihood],
    "goal_alignment": [1-10 based on achieving {goal}]
  }},
  
  "engagement_psychology": "[2-3 sentences explaining the SPECIFIC psychological mechanisms at play in THIS video - not generic theory]",
  
  "strengths": "What's working: [Specific elements with psychological reasoning]",
  
  "improvement_opportunities": "{'Even this viral video could improve by: ' if performance_level == 'viral' else 'Key improvements: '}[Specific, actionable improvements with psychological reasoning]",
  
  "timing_mastery": {{
    "0-1s": "[Exact content + psychological impact]",
    "1-3s": "[Exact content + viewer state]",
    "3-7s": "[Development + emotional journey]",
    "7-15s": "[Core value + retention mechanics]",
    "15s+": "[Resolution + sharing trigger]"
  }},
  
  "formulas": {{
    "viral_formula": "[Step-by-step formula based on this video's success/potential]",
    "hook_formula": "[Specific hook construction method]",
    "retention_formula": "[How to maintain attention throughout]",
    "psychological_formula": "[Emotional journey framework]",
    "platform_formula": "[{platform}-specific optimization formula]"
  }},
  
  "performance_prediction": "{'This video succeeded because: ' if performance_level == 'viral' else 'With improvements, this could achieve: '}[Specific prediction with reasoning]",
  
  "knowledge_patterns_applied": [
    "[Pattern 1 from knowledge base]: [How it applies to this video]",
    "[Pattern 2 from knowledge base]: [Specific implementation]",
    "[Pattern 3 from knowledge base]: [Opportunity or execution]"
  ],
  
  "viral_audio_analysis": {{
    "is_viral_sound": {"true" if audio_type_info['viral_audio_check'] else "false"},
    "audio_psychology": "[How audio enhances or detracts from retention]",
    "audio_visual_sync": "[How audio and visual elements work together]"
  }},
  
  "content_depth_analysis": {{
    "surface_elements": "[What viewers see immediately]",
    "deeper_mechanics": "[Underlying psychological mechanisms]",
    "algorithmic_optimization": "[How it works with {platform} algorithm]",
    "audience_resonance": "[Why {audience} specifically connects]"
  }},
  
  "replication_framework": {{
    "core_principles": "[What makes this replicable]",
    "adaptation_guide": "[How to apply to different niches]",
    "success_factors": "[Critical elements to maintain]",
    "common_mistakes": "[What to avoid when replicating]"
  }}
}}

CRITICAL: Provide DEEP insights, not surface observations. Every point should explain the WHY. Reference specific moments from the video. Use psychological principles. Make it actionable and educational.
"""

    try:
        print(f"[INFO] Running DEEP analysis for {performance_level} content...")
        print(f"[INFO] View count: {view_count}, Knowledge base: {len(knowledge_context)} chars")
        
        gpt_response = _api_retry(
            client.chat.completions.create,
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in viral psychology and content analysis. Provide DEEP, specific insights about why content succeeds or fails. Always explain the psychological mechanisms. Never give surface-level observations. Every analysis should teach the user something about content psychology."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,  # Slightly higher for more insightful analysis
            max_tokens=4500  # More tokens for comprehensive analysis
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
                            "engagement_potential": 9, "viral_potential": 10, "goal_alignment": 8}
        elif performance_level == 'good':
            score_defaults = {"hook_strength": 7, "promise_clarity": 7, "retention_design": 7, 
                            "engagement_potential": 7, "viral_potential": 6, "goal_alignment": 7}
        elif performance_level == 'moderate':
            score_defaults = {"hook_strength": 6, "promise_clarity": 6, "retention_design": 6, 
                            "engagement_potential": 6, "viral_potential": 5, "goal_alignment": 6}
        else:
            score_defaults = {"hook_strength": 4, "promise_clarity": 5, "retention_design": 5, 
                            "engagement_potential": 4, "viral_potential": 3, "goal_alignment": 5}
        
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
        
        # Build comprehensive result with ALL deep insights
        result = {
            # Core deep analysis
            "analysis": parsed.get("analysis", ""),
            "viral_mechanics": parsed.get("viral_mechanics", ""),
            "psychological_breakdown": parsed.get("psychological_breakdown", {}),
            "content_structure_analysis": parsed.get("content_structure_analysis", {}),
            "performance_deep_dive": parsed.get("performance_deep_dive", ""),
            
            # Detailed breakdowns
            "exact_hook_breakdown": parsed.get("exact_hook_breakdown", {}),
            "timing_mastery": parsed.get("timing_mastery", {}),
            "content_depth_analysis": parsed.get("content_depth_analysis", {}),
            
            # Scores and metrics
            "scores": scores,
            "hooks": parsed.get("hooks", []),
            
            # Formulas and frameworks
            "formulas": parsed.get("formulas", {}),
            "replication_framework": parsed.get("replication_framework", {}),
            
            # Patterns and insights
            "knowledge_patterns_applied": parsed.get("knowledge_patterns_applied", []),
            "engagement_psychology": parsed.get("engagement_psychology", ""),
            "strengths": parsed.get("strengths", ""),
            "improvement_opportunities": parsed.get("improvement_opportunities", ""),
            
            # Audio and visual analysis
            "viral_audio_analysis": parsed.get("viral_audio_analysis", {}),
            
            # Predictions and recommendations
            "performance_prediction": parsed.get("performance_prediction", ""),
            
            # All detailed sub-components
            "performance_analysis": parsed.get("performance_deep_dive", ""),
            "video_type_analysis": f"Deep analysis of {audio_type_info['type']} content for {audience}",
            "content_type_detected": audio_type_info['type'],
            
            # Extract individual formula components for compatibility
            "basic_formula": parsed.get("formulas", {}).get("viral_formula", ""),
            "timing_formula": parsed.get("formulas", {}).get("retention_formula", ""),
            "visual_formula": parsed.get("formulas", {}).get("platform_formula", ""),
            "psychology_formula": parsed.get("formulas", {}).get("psychological_formula", ""),
            "hook_formula": parsed.get("formulas", {}).get("hook_formula", ""),
            
            # Timing breakdown for template
            "timing_breakdown": "\n".join([
                f"{time}: {content}" 
                for time, content in parsed.get("timing_mastery", {}).items()
            ]),
            
            # Hook details
            "visual_hook": parsed.get("exact_hook_breakdown", {}).get("visual_elements", ""),
            "text_hook": parsed.get("exact_hook_breakdown", {}).get("text_overlays", ""),
            "verbal_hook": parsed.get("exact_hook_breakdown", {}).get("spoken_hook", ""),
            "why_hook_works": parsed.get("exact_hook_breakdown", {}).get("hook_psychology", ""),
            
            # Compatibility fields
            "improvements": parsed.get("improvement_opportunities", ""),
            "formula": parsed.get("formulas", {}).get("viral_formula", ""),
            "template_formula": parsed.get("formulas", {}).get("platform_formula", ""),
            "knowledge_insights": " | ".join(parsed.get("knowledge_patterns_applied", [])),
            
            # Meta information
            "knowledge_context_used": bool(knowledge_context.strip()),
            "overall_quality": "strong" if performance_level == 'viral' else "moderate" if performance_level in ['good', 'moderate'] else "needs_work",
            "video_has_speech": has_speech,
            "audio_type_detected": audio_type_info['type'],
            "actual_view_count": view_count,
            "performance_level": performance_level
        }
        
        return result
        
    except Exception as e:
        print(f"[ERROR] Deep analysis failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Return enhanced fallback with more depth
        return create_enhanced_fallback(
            transcript_text, frames_summaries_text, creator_note, 
            platform, goal, audience, has_speech, view_count, performance_level,
            knowledge_context
        )


def create_enhanced_fallback(transcript_text, frames_summaries_text, creator_note, platform, goal, audience, has_speech, view_count, performance_level, knowledge_context):
    """Enhanced fallback with deeper insights even in error cases"""
    
    # Extract key insights from available data
    frames_lower = frames_summaries_text.lower() if frames_summaries_text else ""
    transcript_lower = transcript_text.lower() if transcript_text else ""
    
    # Detect content patterns
    is_tutorial = any(word in frames_lower for word in ['step', 'how to', 'tutorial', 'guide'])
    is_transformation = any(word in frames_lower for word in ['before', 'after', 'transform', 'change'])
    is_process = any(word in frames_lower for word in ['making', 'creating', 'building', 'drawing'])
    has_text_overlay = 'text' in frames_lower or 'caption' in frames_lower or 'overlay' in frames_lower
    
    # Build performance-aware analysis
    if performance_level == 'viral':
        analysis = f"""This video achieved viral success with {view_count}, indicating strong psychological triggers and platform optimization.

The viral mechanics likely include: {'transformation/reveal moments' if is_transformation else 'process satisfaction' if is_process else 'educational value delivery' if is_tutorial else 'engaging content structure'} combined with {'clear text overlays for accessibility' if has_text_overlay else 'strong visual storytelling'}.

Key success factors: The first 3 seconds clearly {'promise a transformation' if is_transformation else 'show an intriguing process' if is_process else 'deliver immediate value'}. The content maintains retention through {'visual progression' if not has_speech else 'dual engagement (visual + audio)'} and delivers a satisfying payoff."""
    else:
        analysis = f"""This video {'shows strong potential' if performance_level in ['good', 'moderate'] else 'has opportunities for growth'} with {view_count if view_count else 'current performance'}.

Content structure: {frames_summaries_text[:200] if frames_summaries_text else 'Visual progression'} {'with spoken narration' if has_speech else 'with visual-only storytelling'}.

To improve performance: Focus on strengthening the first 3 seconds with a clearer hook, {'add text overlays for accessibility' if not has_text_overlay else 'optimize text overlay timing'}, and ensure the payoff is visible within the first 7 seconds to improve retention."""
    
    # Dynamic scoring based on performance
    base_scores = {
        "hook_strength": 8 if performance_level == 'viral' else 6 if performance_level in ['good', 'moderate'] else 4,
        "promise_clarity": 7 if performance_level == 'viral' else 6 if performance_level in ['good', 'moderate'] else 4,
        "retention_design": 8 if performance_level == 'viral' else 6 if performance_level in ['good', 'moderate'] else 5,
        "engagement_potential": 8 if performance_level == 'viral' else 6 if performance_level in ['good', 'moderate'] else 4,
        "viral_potential": 9 if performance_level == 'viral' else 5 if performance_level in ['good', 'moderate'] else 3,
        "goal_alignment": 7 if performance_level == 'viral' else 6 if performance_level in ['good', 'moderate'] else 5
    }
    
    # Extract patterns from knowledge if available
    knowledge_patterns = []
    if knowledge_context and len(knowledge_context) > 100:
        if 'hook' in knowledge_context.lower():
            knowledge_patterns.append("Strong hooks stop scrolls in first second")
        if 'curiosity' in knowledge_context.lower():
            knowledge_patterns.append("Curiosity gaps drive watch time")
        if 'payoff' in knowledge_context.lower():
            knowledge_patterns.append("Clear payoffs increase completion rates")
    
    content_type = "visual_only" if not has_speech else "original_speech"
    content_type_display = f"a {content_type.replace('_', ' ')}" if content_type else 'your'
    
    return {
        "analysis": analysis,
        "viral_mechanics": f"{'This achieved virality through: ' if performance_level == 'viral' else 'To increase virality: '}Strong hooks, clear value proposition, and satisfying payoffs. {'The psychological triggers worked effectively.' if performance_level == 'viral' else 'Focus on psychological triggers like curiosity and completion satisfaction.'}",
        "psychological_breakdown": {
            "emotional_triggers": ["curiosity", "satisfaction", "surprise"],
            "curiosity_mechanisms": ["Visual progression", "Incomplete patterns", "Promise of value"],
            "social_dynamics": ["Shareable moments", "Relatable content", "Discussion triggers"],
            "cognitive_biases": ["Completion bias", "Pattern recognition", "Social proof"],
            "sharing_psychology": ["Value delivery", "Emotional resonance", "Social currency"]
        },
        "content_structure_analysis": {
            "hook_phase": "0-3s: Opening creates curiosity",
            "development_phase": "3-10s: Value becomes clear",
            "retention_phase": "10-20s: Content delivers on promise",
            "payoff_phase": "20s+: Satisfying conclusion",
            "structure_effectiveness": base_scores["retention_design"],
            "structure_insights": "Structure aligns with platform expectations"
        },
        "performance_deep_dive": f"With {view_count if view_count else 'current performance'}, this video demonstrates {'successful viral mechanics' if performance_level == 'viral' else 'room for optimization in key areas'}.",
        
        "content_type_detected": content_type,
        "video_type_analysis": f"This is {content_type_display} video optimized for {audience}",
        "performance_analysis": f"{f'With {view_count},' if view_count else 'Your video'} shows {f'successful execution' if performance_level in ['good', 'viral'] else 'opportunity for growth'}",
        
        "hooks": [
            f"{'Refined hook' if performance_level == 'viral' else 'Stronger opening'}: Lead with most intriguing element",
            "Curiosity-driven: Create immediate question in viewer's mind",
            "Value-front: Show the payoff in first 3 seconds",
            "Pattern interrupt: Start with unexpected visual or statement",
            "Emotional hook: Tap into core audience emotion immediately"
        ],
        
        "scores": base_scores,
        
        "exact_hook_breakdown": {
            "first_frame": "0:00 - Opening visual",
            "second_moment": "0:01 - Development begins",
            "third_second": "0:02 - Hook completion",
            "visual_elements": "Opening visual elements",
            "text_overlays": "Text if present",
            "spoken_hook": "Opening audio/speech",
            "hook_psychology": "Psychological impact of opening",
            "hook_score": base_scores["hook_strength"],
            "hook_reasoning": "Based on engagement potential"
        },
        
        "timing_mastery": {
            "0-1s": "Opening: Attention capture",
            "1-3s": "Hook: Curiosity creation",
            "3-7s": "Development: Value reveal",
            "7-15s": "Core: Main content",
            "15s+": "Resolution: Payoff delivery"
        },
        
        "engagement_psychology": f"Engagement {'succeeded through' if performance_level == 'viral' else 'can be improved by leveraging'} curiosity gaps, value delivery, and completion satisfaction.",
        "strengths": f"{'Strong viral execution' if performance_level == 'viral' else 'Creating content and understanding platform dynamics'}",
        "improvement_opportunities": f"{'Even viral content can improve: ' if performance_level == 'viral' else 'Key improvements: '}Hook optimization, faster value delivery, clearer payoffs",
        
        "timing_breakdown": "0-1s: Attention\n1-3s: Curiosity\n3-7s: Value\n7-15s: Content\n15s+: Payoff",
        
        "formulas": {
            "viral_formula": "Hook → Curiosity → Value → Payoff → Share trigger",
            "hook_formula": "Pattern interrupt + Promise + Visual interest",
            "retention_formula": "Continuous value + Visual progression + Completion desire",
            "psychological_formula": "Attention → Interest → Desire → Action → Satisfaction",
            "platform_formula": f"{platform.capitalize()}: Fast pace + Clear value + Shareable moment"
        },
        
        "basic_formula": "Hook → Curiosity → Value → Payoff",
        "timing_formula": "0-1s: Stop scroll\n1-3s: Create curiosity\n3-7s: Show value\n7s+: Deliver payoff",
        "visual_formula": "Visual hook → Progression → Transformation → Result",
        "psychology_formula": "Attention → Interest → Desire → Satisfaction",
        "hook_formula": "Pattern interrupt + Promise + Visual interest",
        
        "performance_prediction": f"{'This succeeded due to strong psychological triggers' if performance_level == 'viral' else 'With optimization, could achieve 10x growth'}",
        "knowledge_patterns_applied": knowledge_patterns if knowledge_patterns else ["Hook optimization", "Value delivery", "Completion satisfaction"],
        "knowledge_insights": " | ".join(knowledge_patterns) if knowledge_patterns else "Strong hooks and clear payoffs drive performance",
        
        "viral_audio_analysis": {
            "is_viral_sound": False,
            "audio_psychology": "Audio enhances retention through engagement",
            "audio_visual_sync": "Audio and visual elements work together"
        },
        
        "content_depth_analysis": {
            "surface_elements": "Immediate visual appeal and clarity",
            "deeper_mechanics": "Psychological triggers and retention mechanics",
            "algorithmic_optimization": f"Optimized for {platform} algorithm signals",
            "audience_resonance": f"Connects with {audience} through relevance"
        },
        
        "replication_framework": {
            "core_principles": "Strong hooks, clear value, satisfying payoffs",
            "adaptation_guide": "Maintain psychological triggers while adapting content",
            "success_factors": "First 3 seconds, value clarity, completion satisfaction",
            "common_mistakes": "Weak hooks, slow value reveal, unclear payoffs"
        },
        
        # Compatibility fields
        "formula": "Hook → Curiosity → Value → Payoff",
        "improvements": f"{'Refine' if performance_level == 'viral' else 'Strengthen'} hooks, optimize pacing, enhance payoffs",
        "template_formula": "Visual hook → Progression → Payoff",
        
        "knowledge_context_used": bool(knowledge_context and len(knowledge_context) > 100),
        "overall_quality": "strong" if performance_level == 'viral' else "moderate" if performance_level in ['good', 'moderate'] else "needs_work",
        "video_has_speech": has_speech,
        "audio_type_detected": content_type,
        "actual_view_count": view_count,
        "performance_level": performance_level
    }
        
        return result
        
    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Return accurate fallback
        return create_accurate_fallback(
            transcript_text, frames_summaries_text, creator_note, 
            platform, goal, audience, has_speech, view_count, performance_level
        )


def create_accurate_fallback(transcript_text, frames_summaries_text, creator_note, platform, goal, audience, has_speech, view_count, performance_level):
    """Fallback that provides accurate analysis without hallucinations"""
    
    # Determine actual content type from frames
    frames_lower = frames_summaries_text.lower() if frames_summaries_text else ""
    
    if not has_speech:
        content_type = "visual_only"
        analysis = f"""Let me analyze your visual-only video. {f'With {view_count},' if view_count else 'Your video'} appears to focus on visual storytelling without narration. 

Based on the frames, {frames_summaries_text[:200] if frames_summaries_text else 'the visual content'}... 

Visual-only content needs strong visual hooks and clear progression to maintain attention without audio support."""
    else:
        content_type = "original_speech"
        analysis = f"""Let me analyze your video. {f'Getting {view_count}' if view_count else 'Your performance'} suggests {f'strong audience connection' if performance_level in ['good', 'viral'] else 'room for optimization'}.

You're speaking: {transcript_text[:150] if transcript_text else 'verbal content'}...

The key is how quickly you deliver value and create curiosity in those first 3 seconds."""
    
    base_scores = {
        "hook_strength": 7 if performance_level in ['good', 'viral'] else 5,
        "promise_clarity": 6 if performance_level in ['good', 'viral'] else 4,
        "retention_design": 7 if performance_level in ['good', 'viral'] else 5,
        "engagement_potential": 7 if performance_level in ['good', 'viral'] else 4,
        "goal_alignment": 6 if performance_level in ['good', 'viral'] else 5
    }
    
    # FIX: Don't nest f-strings, use a conditional expression instead
    content_type_display = f"a {content_type.replace('_', ' ')}" if content_type else 'your'
    
    return {
        "analysis": analysis,
        "content_type_detected": content_type,
        "video_type_analysis": f"This is {content_type_display} video optimized for {audience}",
        "performance_analysis": f"{f'With {view_count},' if view_count else 'Your video'} shows {f'successful execution' if performance_level in ['good', 'viral'] else 'opportunity for growth'}",
        
        "hooks": [
            "Lead with your most intriguing element",
            "Create immediate curiosity or pattern interrupt",
            "Front-load the value or payoff",
            "Start with a question or challenge",
            "Open with unexpected visual or statement"
        ],
        
        "scores": base_scores,
        
        "exact_hook_breakdown": {
            "first_second": "0:00 - Opening moment",
            "second_second": "0:01 - Development",
            "third_second": "0:02 - Hook completion",
            "visual_hook": "Opening visual element",
            "text_hook": "Text overlay if present",
            "audio_hook": "Opening audio/speech",
            "why_it_works_or_not": "Hook effectiveness depends on immediate value/curiosity creation"
        },
        
        "engagement_psychology": "Engagement comes from curiosity gaps, relatable content, and clear value delivery",
        "strengths": "Creating content and understanding your platform",
        "improvement_areas": "Focus on hook optimization and faster value delivery",
        
        "timing_breakdown": "0-3s: Hook, 3-10s: Development, 10-20s: Core content, 20+: Resolution",
        "basic_formula": "1. Strong hook\n2. Quick value\n3. Clear payoff",
        "timing_formula": "0-1s: Stop scroll\n1-3s: Create curiosity\n3-10s: Deliver value",
        "visual_formula": "Visual hook → Development → Payoff",
        "psychology_formula": "Attention → Interest → Desire → Satisfaction",
        
        "performance_prediction": "Optimizing the first 3 seconds could significantly improve performance",
        "knowledge_insights": "Based on patterns: stronger hooks and faster payoffs drive better performance",
        
        "viral_audio_analysis": {
            "is_viral_sound": False,
            "explanation": "Using original audio/speech"
        },
        
        "content_analysis": {
            "type": content_type,
            "key_insights": "Focus on immediate hook optimization",
            "optimization_opportunities": ["Stronger opening", "Clearer value proposition", "Faster payoff"]
        },
        
        # Additional compatibility fields
        "formula": "1. Strong hook\n2. Quick value\n3. Clear payoff",
        "improvements": "Focus on hook optimization and faster value delivery",
        "template_formula": "Visual hook → Development → Payoff",
        
        "knowledge_context_used": False,
        "overall_quality": "moderate",
        "video_has_speech": has_speech,
        "audio_type_detected": content_type,
        "actual_view_count": view_count,
        "performance_level": performance_level
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
        
        # Extract basic info for fallback
        has_speech = transcript_text and len(transcript_text.strip()) > 20
        
        # Try to extract view count from creator note
        view_count = None
        performance_level = 'unknown'
        if creator_note:
            view_patterns = re.findall(r'(\d+\.?\d*)\s*(k|thousand|m|million)', creator_note.lower())
            if view_patterns:
                number, unit = view_patterns[0]
                if unit in ['k', 'thousand']:
                    view_count = f"{number}k"
                    performance_level = 'moderate' if float(number) >= 100 else 'low'
                elif unit in ['m', 'million']:
                    view_count = f"{number}M"
                    performance_level = 'viral'
        
        return create_accurate_fallback(
            transcript_text, frames_summaries_text, creator_note,
            platform, goal, audience, has_speech, view_count, performance_level
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
        
        # Compatibility fields
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
        'actual_view_count': gpt_result.get('actual_view_count', ''),
        'performance_level': gpt_result.get('performance_level', 'unknown'),
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

        # Transcribe audio
        try:
            transcript_data = enhanced_transcribe_audio(audio_path)
            print(f"[INFO] Transcript quality: {transcript_data['quality']}")
            print(f"[INFO] Transcript preview: {transcript_data['transcript'][:100]}...")
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
            print(f"[INFO] Frame analysis preview: {frames_summaries_text[:200]}...")
        except Exception as e:
            print(f"[ERROR] Frame analysis error: {e}")
            frames_summaries_text = ""
            gallery_data_urls = []

        # Get knowledge context using smart RAG retrieval
        try:
            print("[INFO] Loading knowledge using smart RAG retrieval...")
            
            # Try smart context first
            knowledge_context, knowledge_citations = retrieve_smart_context(
                transcript=transcript_data.get('transcript', ''),
                frames=frames_summaries_text[:1000],
                creator_note=form_data['creator_note'],
                goal=form_data['goal'],
                max_chars=75000  # Get 75K of relevant content
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
- Front-load value and payoff
- Use pattern interrupts to stop scrolls
- Match content to platform expectations
"""
            knowledge_citations = ["Basic patterns fallback"]

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
                    elif unit in ['m', 'million']:
                        view_count = f"{number}M"
            
            gpt_result = create_accurate_fallback(
                transcript_data.get('transcript', ''),
                frames_summaries_text,
                form_data['creator_note'],
                form_data['platform'],
                form_data['goal'],
                form_data['audience'],
                has_speech,
                view_count,
                performance_level
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