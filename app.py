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
# KNOWLEDGE MANAGEMENT
# ==============================

def parse_knowledge_folder(knowledge_path="knowledge"):
    """Parse documents directly from knowledge folder"""
    knowledge_content = []
    
    try:
        if not os.path.exists(knowledge_path):
            print(f"[ERROR] Knowledge folder '{knowledge_path}' not found")
            return ""
        
        print(f"[INFO] Scanning knowledge folder: {knowledge_path}")
        
        for filename in os.listdir(knowledge_path):
            file_path = os.path.join(knowledge_path, filename)
            
            if not os.path.abspath(file_path).startswith(os.path.abspath(knowledge_path)):
                print(f"[SECURITY] Skipping file outside knowledge folder: {filename}")
                continue
            
            try:
                if filename.lower().endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        knowledge_content.append(f"=== {filename} ===\n{content}\n")
                        print(f"[LOADED] {filename} ({len(content)} chars)")
                
                elif filename.lower().endswith('.pdf'):
                    try:
                        from pypdf import PdfReader
                        with open(file_path, 'rb') as f:
                            pdf_reader = PdfReader(f)
                            pdf_text = ""
                            for page in pdf_reader.pages:
                                pdf_text += page.extract_text() + "\n"
                            
                            if len(pdf_text.strip()) > 50:
                                knowledge_content.append(f"=== {filename} ===\n{pdf_text}\n")
                                print(f"[LOADED] {filename} ({len(pdf_text)} chars)")
                            else:
                                print(f"[SKIP] {filename} - no readable text found")
                    except ImportError:
                        print(f"[SKIP] {filename} - pypdf not installed")
                    except Exception as pdf_e:
                        print(f"[ERROR] Reading {filename}: {pdf_e}")
                
                elif filename.lower().endswith(('.md', '.markdown')):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        knowledge_content.append(f"=== {filename} ===\n{content}\n")
                        print(f"[LOADED] {filename} ({len(content)} chars)")
                
                elif filename.lower().endswith('.json'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                        content = json.dumps(json_data, indent=2)
                        knowledge_content.append(f"=== {filename} ===\n{content}\n")
                        print(f"[LOADED] {filename} ({len(content)} chars)")
                        
                else:
                    print(f"[SKIP] {filename} - unsupported file type")
                    
            except Exception as e:
                print(f"[ERROR] Processing {filename}: {e}")
        
        total_content = "\n".join(knowledge_content)
        print(f"[SUCCESS] Loaded {len(knowledge_content)} documents, {len(total_content)} total characters")
        
        if len(total_content) > 680000:
            print(f"[INFO] Truncating knowledge content from {len(total_content)} to 700000 chars")
            total_content = total_content[:700000] + "\n\n[Content truncated...]"
        
        return total_content
        
    except Exception as e:
        print(f"[ERROR] Failed to parse knowledge folder: {e}")
        return ""


def get_knowledge_context_robust():
    """More robust knowledge context loading with better fallbacks"""
    knowledge_content = ""
    knowledge_citations = []
    
    # Try direct file parsing first
    try:
        print("[INFO] Attempting direct file parsing...")
        knowledge_content = parse_knowledge_folder("knowledge")
        if len(knowledge_content) > 500:
            knowledge_citations = ["Direct file parsing from /knowledge folder"]
            print(f"[SUCCESS] Direct parsing: {len(knowledge_content)} chars")
            return knowledge_content, knowledge_citations
    except Exception as e:
        print(f"[ERROR] Direct file parsing failed: {e}")
    
    # Try RAG retrieval
    try:
        print("[INFO] Trying RAG retrieval...")
        all_context = retrieve_all_context()
        if all_context and len(str(all_context)) > 500:
            knowledge_content = str(all_context)[:700000]
            knowledge_citations = ["Retrieved from full knowledge base"]
            print(f"[SUCCESS] RAG retrieval: {len(knowledge_content)} chars")
            return knowledge_content, knowledge_citations
    except Exception as e:
        print(f"[ERROR] RAG retrieval failed: {e}")
    
    # Enhanced fallback
    print("[FALLBACK] Using enhanced fallback patterns")
    knowledge_content = """
PROVEN VIRAL CONTENT PATTERNS:

HOOK STRENGTH INDICATORS:
HIGH-PERFORMING (8-10/10):
- Immediate controversy or pattern interrupt
- Personal transformation promise
- Exclusive insight revelation
- Visual or auditory surprise in first 2 seconds

LOW-PERFORMING (1-4/10):
- Generic openings without stakes
- No immediate curiosity gap
- Weak or missing pattern interrupt

RETENTION DESIGN:
- Quick cuts every 2-3 seconds
- Tension building toward payoff
- Progress indicators for structure
- Visual/narrative callbacks to hook

ENGAGEMENT PSYCHOLOGY:
- Controversy drives comments
- Transformation gets saves
- Behind-scenes gets shares
- Educational+personality gets follows
"""
    knowledge_citations = ["Enhanced performance-based fallback patterns"]
    
    return knowledge_content, knowledge_citations


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
# MAIN ANALYSIS FUNCTION - COMPREHENSIVE & BALANCED
# ==============================

def run_main_analysis(transcript_text, frames_summaries_text, creator_note, platform, target_duration, goal, tone, audience, knowledge_context):
    """Comprehensive analysis handling ALL video types with appropriate depth"""
    
    # Detect content characteristics
    has_speech = transcript_text and len(transcript_text.strip()) > 20 and not any(
        indicator in transcript_text.lower() 
        for indicator in ['music', 'sound', 'noise', 'audio', 'background', 'ambient']
    )
    
    # Smart audio type detection (not forcing viral)
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
    
    # Build contextual sections
    knowledge_section = ""
    if knowledge_context.strip():
        knowledge_section = f"""
PROVEN STRATEGIES FROM KNOWLEDGE BASE:
{knowledge_context}

Use these patterns to explain why this video succeeded/failed compared to proven examples.
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
    
    # Performance-aware prompting
    if performance_indicators:
        performance_context = f"This video {performance_indicators[0]}. Explain WHY with specific evidence."
    else:
        performance_context = "Assess likely performance based on structure and patterns."
    
    prompt = f"""
You are analyzing a {platform} video. Creator context: {creator_note if creator_note else "No specific performance data"}.

CONTENT TO ANALYZE:
TRANSCRIPT: {transcript_text if has_speech else "(No speech - visual/ambient only)"}
VISUAL CONTENT: {frames_summaries_text}
TARGET: {target_duration}s video for {audience}
GOAL: {goal}

{knowledge_section}

ANALYSIS REQUIREMENTS:
1. EXACT HOOK BREAKDOWN (0-3 seconds)
   - Describe & explain EXACTLY what happens each second
   - Identify visual, text, and audio hooks and explain why they work or don't work for the goal
   - Distinguish between captions and hook text 
   - Explain in a descriptive and educational way why it works or doesn't

2. PERFORMANCE ANALYSIS
   {performance_context}
   
3. CONTENT-SPECIFIC INSIGHTS
   {audio_analysis_section}

4. ENGAGEMENT PSYCHOLOGY
   - Specific mechanisms driving comments/shares/saves
   - Not generic advice but specific to THIS video

5. ALTERNATIVE HOOKS
   - 5 natural, platform-appropriate alternatives
   - Match the content type and audience
   - No corporate marketing speak

6. IMPROVEMENT OPPORTUNITIES
   - Even successful videos can improve
   - Specific, actionable suggestions
   - Based on proven patterns

7. FORMULAS FOR REPLICATION
   - Second-by-second timing formula
   - Visual progression formula
   - Psychology formula
   - Basic step-by-step process

Provide SPECIFIC, NUANCED analysis thats very explanatory even to someone who doesn't understand short form content or retention. Reference exact moments, quote text, describe visuals precisely and conversationally in a way that explains. Don't be afraid to be verbose. Be sure to line up any transcript with frames to better understand how speech and visuals are working together.

Respond in JSON format:
{{
  "analysis": "Comprehensive breakdown of what's happening and why",
  "content_type_detected": "{audio_type_info['type']}",
  "video_type_analysis": "How this type of video works psychologically",
  
  "exact_hook_breakdown": {{
    "first_second": "0:00 - [EXACT description]",
    "second_second": "0:01 - [EXACT description]",
    "third_second": "0:02 - [EXACT description]",
    "visual_hook": "[What grabs visual attention]",
    "text_hook": "[Exact text and whether hook or caption]",
    "audio_hook": "[Opening audio description]",
    "why_it_works_or_not": "[Specific psychological explanation]"
  }},
  
  "performance_analysis": "[Why this performed as it did]",
  
  "hooks": ["5 natural alternative hooks"],
  
  "scores": {{
    "hook_strength": [1-10 with reasoning],
    "promise_clarity": [1-10 with reasoning],
    "retention_design": [1-10 with reasoning],
    "engagement_potential": [1-10 with reasoning],
    "goal_alignment": [1-10 with reasoning]
  }},
  
  "engagement_psychology": "[Specific mechanisms for THIS video]",
  
  {"viral_audio_analysis" if audio_type_info['viral_audio_check'] else "content_analysis"}: {{
    "type": "{audio_type_info['type']}",
    "key_insights": "[Specific to content type]",
    "optimization_opportunities": "[How to enhance this type]"
  }},
  
  "strengths": "[What genuinely works well]",
  "improvement_areas": "[Specific actionable improvements]",
  
  "timing_breakdown": "[Full video progression analysis]",
  
  "formulas": {{
    "basic_formula": "Step-by-step process",
    "timing_formula": "Second-by-second breakdown",
    "visual_formula": "Visual progression pattern",
    "psychology_formula": "Psychological journey"
  }},
  
  "performance_prediction": "[Likely performance and why]",
  "knowledge_insights": "[Comparison to proven patterns]"
}}
"""

    try:
        print(f"[INFO] Sending analysis to GPT-4...")
        print(f"Content type: {audio_type_info['type']} (confidence: {audio_type_info.get('confidence', 'unknown')})")
        
        gpt_response = _api_retry(
            client.chat.completions.create,
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
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
                # Extract numeric score from string like "7/10 - explanation"
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
            
            # Content-specific analysis (viral audio OR standard)
            "viral_audio_analysis": parsed.get("viral_audio_analysis", {}) if audio_type_info['viral_audio_check'] else {},
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
        
        # Return intelligent fallback
        return create_comprehensive_fallback(
            transcript_text, frames_summaries_text, creator_note, 
            platform, goal, audience, has_speech, is_high_performing
        )


def create_comprehensive_fallback(transcript_text, frames_summaries_text, creator_note, platform, goal, audience, has_speech, is_high_performing):
    """Intelligent fallback that still provides value"""
    
    base_scores = {
        "hook_strength": 7 if is_high_performing else 5,
        "promise_clarity": 6 if is_high_performing else 4,
        "retention_design": 7 if is_high_performing else 5,
        "engagement_potential": 6 if is_high_performing else 4,
        "goal_alignment": 6 if is_high_performing else 5
    }
    
    if not has_speech:
        analysis_type = "visual-focused"
        hooks = [
            "wait for the transformation",
            "POV: you discover the most satisfying process",
            "this is oddly satisfying",
            "the ending will blow your mind",
            "you won't believe how this turns out"
        ]
    else:
        analysis_type = "verbal and visual"
        hooks = [
            "here's what nobody tells you about...",
            "I discovered something that changes everything",
            "stop scrolling - this matters",
            "the secret they don't want you to know",
            "this one thing made all the difference"
        ]
    
    return {
        "analysis": f"This {analysis_type} video uses {platform} best practices for {goal}.",
        "content_type_detected": "visual_only" if not has_speech else "original_speech",
        "video_type_analysis": f"{analysis_type} content optimized for {audience}",
        "performance_analysis": f"Performance analysis based on structure and patterns",
        
        "hooks": hooks,
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
        
        "engagement_psychology": "Engagement driven by curiosity and value delivery",
        "strengths": "Content structure and pacing",
        "improvement_areas": "Optimize hooks and clarity",
        
        "basic_formula": "1. Strong opening 2. Build interest 3. Deliver value 4. Clear CTA",
        "timing_formula": "0-3s: Hook, 3-15s: Setup, 15-25s: Core value, 25-30s: CTA",
        "visual_formula": "Visual hook → Development → Payoff",
        "psychology_formula": "Curiosity → Anticipation → Satisfaction",
        "timing_breakdown": "Progressive value delivery throughout",
        
        "performance_prediction": "Performance depends on execution of fundamentals",
        "knowledge_insights": "Aligns with proven content patterns",
        
        "formula": "Hook → Setup → Value → CTA",
        "improvements": "Focus on hook strength and value clarity",
        "template_formula": "Pattern interrupt → Promise → Proof → Push",
        
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
        
        return create_comprehensive_fallback(
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

        # Get knowledge context
        try:
            knowledge_context, knowledge_citations = get_knowledge_context_robust()
            print(f"[INFO] Knowledge context loaded: {len(knowledge_context)} chars")
        except Exception as e:
            print(f"[ERROR] Knowledge loading error: {e}")
            knowledge_context = ""
            knowledge_citations = []

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
            
            gpt_result = create_comprehensive_fallback(
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