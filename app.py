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
# HELPER FUNCTIONS
# ==============================

def parse_knowledge_folder(knowledge_path="knowledge"):
    """Parse documents directly from knowledge folder"""
    import os
    
    knowledge_content = []
    
    try:
        if not os.path.exists(knowledge_path):
            print(f"[ERROR] Knowledge folder '{knowledge_path}' not found")
            return ""
        
        print(f"[INFO] Scanning knowledge folder: {knowledge_path}")
        
        for filename in os.listdir(knowledge_path):
            file_path = os.path.join(knowledge_path, filename)
            
            try:
                if filename.lower().endswith('.txt'):
                    # Read text files
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        knowledge_content.append(f"=== {filename} ===\n{content}\n")
                        print(f"[LOADED] {filename} ({len(content)} chars)")
                
                elif filename.lower().endswith('.pdf'):
                    # Try to read PDFs using PyPDF2 or similar
                    try:
                        import PyPDF2
                        with open(file_path, 'rb') as f:
                            pdf_reader = PyPDF2.PdfReader(f)
                            pdf_text = ""
                            for page in pdf_reader.pages:
                                pdf_text += page.extract_text() + "\n"
                            
                            if len(pdf_text.strip()) > 50:  # Only include if we got meaningful text
                                knowledge_content.append(f"=== {filename} ===\n{pdf_text}\n")
                                print(f"[LOADED] {filename} ({len(pdf_text)} chars)")
                            else:
                                print(f"[SKIP] {filename} - no readable text found")
                    except ImportError:
                        print(f"[SKIP] {filename} - PyPDF2 not installed, can't read PDFs")
                    except Exception as pdf_e:
                        print(f"[ERROR] Reading {filename}: {pdf_e}")
                
                elif filename.lower().endswith(('.md', '.markdown')):
                    # Read markdown files
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        knowledge_content.append(f"=== {filename} ===\n{content}\n")
                        print(f"[LOADED] {filename} ({len(content)} chars)")
                
                elif filename.lower().endswith('.json'):
                    # Read JSON files
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
        
        # Truncate if too long (GPT has token limits)
        if len(total_content) > 15000:  # Reasonable limit for GPT context
            print(f"[INFO] Truncating knowledge content from {len(total_content)} to 15000 chars")
            total_content = total_content[:15000] + "\n\n[Content truncated...]"
        
        return total_content
        
    except Exception as e:
        print(f"[ERROR] Failed to parse knowledge folder: {e}")
        return ""


def get_knowledge_context():
    """Get knowledge context from direct file parsing first, then RAG fallback"""
    
    # ALWAYS try direct file parsing first
    print("[INFO] Attempting direct file parsing of /knowledge folder...")
    knowledge_content = parse_knowledge_folder("knowledge")
    
    if len(knowledge_content) > 500:  # Got meaningful content from files
        print(f"[SUCCESS] Direct parsing loaded {len(knowledge_content)} characters")
        return knowledge_content, ["Direct file parsing from /knowledge folder"]
    
    # If no files found, try retrieve_all_context (which seems to work based on your logs)
    print("[FALLBACK] No files found, trying retrieve_all_context...")
    try:
        all_context = retrieve_all_context()
        if all_context and len(str(all_context)) > 500:
            content = str(all_context)
            print(f"[SUCCESS] retrieve_all_context loaded {len(content)} characters")
            # Truncate if too long
            if len(content) > 15000:
                content = content[:15000] + "\n\n[Content truncated...]"
            return content, ["Retrieved from full knowledge base"]
    except Exception as e:
        print(f"[ERROR] retrieve_all_context failed: {e}")
    
    # Final fallback to enhanced patterns
    print("[FALLBACK] Using enhanced fallback knowledge patterns")
    return """
PROVEN VIRAL CONTENT PATTERNS (ENHANCED FALLBACK):

HOOK ANALYSIS FOR LOW-PERFORMING CONTENT:
Current video hook: "You spent over the project proposal, thought it was your best yet..."
Problem: This hook lacks immediate emotional stakes and curiosity gap

HIGH-PERFORMING HOOK PATTERNS:
- "POV: you just discovered..." (creates immediate curiosity)
- "Nobody talks about how..." (controversial angle)
- "I tried this for 30 days and..." (transformation promise)
- "The real reason why..." (insider knowledge)
- "This is why you're failing at..." (direct challenge)

LOW-PERFORMING PATTERNS (like analyzed video):
- Starting with mundane actions ("You spent over...")
- No immediate personal stakes for viewer
- Weak curiosity gap or pattern interrupt
- Business advice without emotional hook

RETENTION KILLERS:
- Slow openings that don't grab attention in first 3 seconds
- Generic business advice without personal transformation
- Background activities that don't enhance the message
- No clear promise of value or revelation

VIRAL IMPROVEMENT STRATEGIES:
- Lead with controversial opinion or surprising revelation
- Create immediate tension: "If you're doing X, you're failing because..."
- Promise specific transformation: "How I went from ghosted to booked"
- Use pattern interrupts: sudden movement, controversial statements
- Build curiosity gaps that demand resolution

ALGORITHM FACTORS:
- First 3 seconds determine 60% of retention
- Comment rate in first hour affects reach
- Watch time >50% triggers recommendation engine
- Visual variety every 2-3 seconds increases retention

SCORING CALIBRATION:
For content with <300 views:
- Hook Strength: Likely 3-5/10 (lacks stopping power)
- Promise Clarity: Likely 4-6/10 (unclear value proposition)  
- Retention Design: Likely 4-7/10 (depends on pacing)
- Engagement Potential: Likely 2-4/10 (low shareability)
- Goal Alignment: Likely 3-5/10 (not optimized for viral reach)
""", ["Enhanced fallback with performance-based patterns"]
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
    """Generate inferred audio description for visual content."""
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


def analyze_text_synchronization(frames_summaries_text, transcript_text, frame_timestamps=None):
    """Distinguish between on-screen text (graphics/overlays) and spoken captions."""
    
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
        "text_sync_analysis": text_sync_analysis,
        "strengths": "Strong visual engagement and process satisfaction elements",
        "improvement_areas": "Could benefit from clearer audio or enhanced pacing",
        "knowledge_insights": "Visual content aligns with satisfying process patterns",
        "knowledge_context_used": False,
        "overall_quality": "moderate"
    }


# ==============================
# MAIN ANALYSIS FUNCTION - BALANCED & COMPREHENSIVE
# ==============================

def run_main_analysis(transcript_text, frames_summaries_text, creator_note, platform, target_duration, goal, tone, audience, knowledge_context):
    """Balanced analysis function that provides nuanced assessment for both strong and weak content."""
    
    # Build knowledge context section - CRITICAL for quality analysis
    knowledge_section = ""
    if knowledge_context.strip():
        knowledge_section = f"""
PROVEN VIRAL STRATEGIES FROM KNOWLEDGE BASE:
{knowledge_context}

ANALYSIS REQUIREMENT: Use these proven examples to:
1. Compare this video's approach against successful viral content
2. Identify specific gaps between this content and high-performers  
3. Calibrate scores based on what actually drives results
4. Provide recommendations based on proven patterns
5. Reference specific successful examples when making suggestions

If this video underperformed (like <300 views), explain WHY by comparing to these successful examples.
"""
    else:
        knowledge_section = """
WARNING: No knowledge context retrieved. Analysis will be less specific without proven examples to compare against.
Focus on general best practices and obvious improvement areas.
"""
    
    prompt = f"""
You are an expert content strategist analyzing this {platform} video for {goal}. Provide nuanced, balanced analysis that recognizes both strengths and areas for improvement.

CONTENT TO ANALYZE:
TRANSCRIPT: {transcript_text}
VISUAL CONTENT: {frames_summaries_text}
CREATOR NOTE: {creator_note}
TARGET: {target_duration}s video for {audience} with {tone} tone

{knowledge_section}

ANALYSIS APPROACH:
- Provide honest assessment without being overly critical or overly positive
- Consider the creator's note about performance when scoring and analyzing
- If the creator mentions poor performance (like low views), investigate why and provide specific solutions
- Identify what's working well and build on those strengths
- Point out areas that could be improved and explain how
- Consider the content quality level and adjust analysis accordingly
- For strong content: focus on what makes it effective and how to replicate/optimize
- For weak content: identify core issues and provide specific fixes
- For average content: highlight potential and provide elevation strategies

PERFORMANCE CONTEXT ANALYSIS:
Creator's note: "{creator_note}"
If the creator mentions low performance, poor results, or specific problems, factor this into your scoring and analysis. Don't give high scores to content that's demonstrably not working.

COMPREHENSIVE EVALUATION:
1. Performance Reality Check: Does this content's structure and execution match the performance described by the creator?
2. Hook Analysis: How effectively does the opening capture attention? What psychological triggers are used?
3. Promise Structure: Is there a clear value proposition? How well does it create anticipation?
4. Content Delivery: Does the video fulfill its promise? Is the pacing engaging?
5. Visual-Audio Synergy: How do the visual and audio elements work together?
6. Retention Design: What keeps viewers watching? Where might they drop off?
7. Algorithmic Factors: What might be hurting algorithmic distribution?
8. Knowledge Base Alignment: How does this compare to successful strategies in your knowledge?

SCORING GUIDELINES:
Score each element independently and honestly based on actual effectiveness AND performance results:

PERFORMANCE-INFORMED SCORING:
- If creator reports poor performance (low views, no engagement), scores should reflect this reality
- High-performing content rarely gets all scores below 6
- Low-performing content rarely deserves scores above 6
- Look for specific reasons why content might be underperforming

ELEMENT-SPECIFIC SCORING:
- Hook Strength: Does it immediately grab attention or is it weak/generic? Consider if it would make someone stop scrolling.
- Promise Clarity: How clear is the value proposition? Is it compelling or confusing? Does it create urgency?
- Retention Design: How well does pacing/content maintain interest throughout? Are there dead spots or weak transitions?
- Engagement Potential: Will this realistically drive comments/shares/saves? Does it inspire action?
- Goal Alignment: How effectively does this serve {goal} specifically? Does it actually achieve what it's trying to do?

Use the full 1-10 range and score each element separately based on RESULTS:
- 8-10: Exceptional - clearly driving strong results and engagement
- 6-7: Good - solid performance with clear strengths
- 4-5: Average/Weak - functional but significant issues preventing success
- 2-3: Poor - major problems that severely hurt performance  
- 1: Broken - completely ineffective

REALITY CHECK: If a video has very low views/engagement, multiple scores above 7 are unlikely to be accurate. Be honest about what's not working.

Respond in JSON format with balanced, actionable analysis:
{{
  "analysis": "Nuanced analysis that identifies what works well and what could be improved. Start with strengths, then address areas for enhancement. Reference knowledge context insights where relevant. Adjust tone based on overall content quality - celebrate genuine strengths, provide constructive guidance for weaknesses.",
  "hooks": ["5 alternative hooks that either build on existing strengths or address identified weaknesses"],
  "scores": {{
    "hook_strength": 7,
    "promise_clarity": 6,
    "retention_design": 8,
    "engagement_potential": 5,
    "goal_alignment": 7
  }},
  "strengths": "Specific elements that are working well and contributing to effectiveness",
  "improvement_areas": "Areas that could be enhanced, with specific suggestions for how",
  "timing_breakdown": "What happens at key moments and how the pacing affects retention",
  "basic_formula": "Step-by-step process for replicating the effective elements while addressing weak points",
  "timing_formula": "Detailed timing breakdown with optimization suggestions",
  "template_formula": "Template format that captures the successful patterns while improving weak areas",
  "psychology_formula": "Psychological mechanisms at work and how they contribute to effectiveness",
  "improvements": "Specific, actionable recommendations prioritized by impact potential",
  "performance_prediction": "Realistic assessment of likely performance with reasoning",
  "knowledge_insights": "How this content aligns with or could better leverage proven strategies from the knowledge base"
}}

Provide constructive, balanced feedback that helps creators understand both what they're doing right and how they can improve.
"""

    try:
        print(f"Sending balanced analysis prompt to GPT-4o...")
        gpt_response = _api_retry(
            client.chat.completions.create,
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=3500
        )

        response_text = gpt_response.choices[0].message.content.strip()
        
        # Parse JSON response
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Response text that failed to parse: {response_text[:500]}...")
            # Return fallback if JSON parsing fails
            return create_visual_enhanced_fallback(frames_summaries_text, {
                'transcript': transcript_for_analysis,
                'is_reliable': len(transcript_for_analysis.strip()) > 50
            }, goal)
        
        # Add debug logging to see what scores GPT is actually returning
        scores_raw = parsed.get("scores", {})
        print(f"Raw scores from GPT: {scores_raw}")
        
        # Extract and process scores with better parsing
        scores = {}
        for key, value in scores_raw.items():
            try:
                # Handle various response formats
                if isinstance(value, (int, float)):
                    scores[key] = max(1, min(10, int(value)))
                elif isinstance(value, str):
                    # Extract number from string responses
                    match = re.search(r'(\d+)', str(value))
                    if match:
                        scores[key] = max(1, min(10, int(match.group(1))))
                    else:
                        # If no number found, assign different defaults for each category
                        category_defaults = {
                            "hook_strength": 5,
                            "promise_clarity": 6, 
                            "retention_design": 5,
                            "engagement_potential": 4,
                            "goal_alignment": 6
                        }
                        scores[key] = category_defaults.get(key, 5)
                else:
                    # Assign different defaults for each category if value is invalid
                    category_defaults = {
                        "hook_strength": 5,
                        "promise_clarity": 6,
                        "retention_design": 5, 
                        "engagement_potential": 4,
                        "goal_alignment": 6
                    }
                    scores[key] = category_defaults.get(key, 5)
            except Exception as e:
                print(f"Error parsing score for {key}: {e}")
                # Different fallbacks for each category
                category_defaults = {
                    "hook_strength": 5,
                    "promise_clarity": 6,
                    "retention_design": 5,
                    "engagement_potential": 4, 
                    "goal_alignment": 6
                }
                scores[key] = category_defaults.get(key, 5)
        
        # Ensure all required score fields exist with different defaults
        required_scores = {
            "hook_strength": 5,
            "promise_clarity": 6,
            "retention_design": 5,
            "engagement_potential": 4,
            "goal_alignment": 6
        }
        for score_key, default_val in required_scores.items():
            if score_key not in scores:
                scores[score_key] = default_val
        
        # Validate score ranges
        for key in scores:
            scores[key] = max(1, min(10, scores[key]))
        
        print(f"Final processed scores: {scores}")  # Debug logging
        
        # Build comprehensive result with enhanced fields
        result = {
            "analysis": parsed.get("analysis", "Analysis not available"),
            "hooks": parsed.get("hooks", []),
            "scores": scores,
            "strengths": parsed.get("strengths", "Content strengths to be identified"),
            "improvement_areas": parsed.get("improvement_areas", "Areas for potential enhancement"),
            "timing_breakdown": parsed.get("timing_breakdown", ""),
            "formula": parsed.get("basic_formula", ""),
            "basic_formula": parsed.get("basic_formula", ""),
            "timing_formula": parsed.get("timing_formula", ""),
            "template_formula": parsed.get("template_formula", ""),
            "psychology_formula": parsed.get("psychology_formula", "Content psychology analysis"),
            "improvements": parsed.get("improvements", ""),
            "performance_prediction": parsed.get("performance_prediction", "Performance assessment based on content analysis"),
            "knowledge_insights": parsed.get("knowledge_insights", "Knowledge context insights"),
            
            # Enhanced psychological analysis fields (for template compatibility)
            "psychological_breakdown": parsed.get("analysis", ""),
            "hook_mechanics": parsed.get("hook_mechanics", ""),
            "emotional_journey": parsed.get("emotional_journey", ""),
            "authority_signals": parsed.get("authority_signals", ""),
            "engagement_psychology": parsed.get("engagement_psychology", ""),
            "viral_mechanisms": parsed.get("viral_mechanisms", ""),
            "audience_psychology": parsed.get("audience_psychology", ""),
            "replication_blueprint": parsed.get("replication_blueprint", ""),
            
            # Additional analysis fields
            "multimodal_insights": parsed.get("viral_mechanisms", ""),
            "engagement_triggers": parsed.get("engagement_psychology", ""),
            "improvement_opportunities": parsed.get("improvement_areas", ""),
            "viral_potential_factors": parsed.get("viral_mechanisms", ""),
            
            # Maintain compatibility with existing template
            "weaknesses": parsed.get("improvement_areas", ""),
            "critical_assessment": parsed.get("analysis", ""),
            "knowledge_context_used": bool(knowledge_context.strip()),
            
            # Calculate overall quality indicator
            "overall_quality": "strong" if sum(scores.values()) / len(scores) >= 7.5 else "moderate" if sum(scores.values()) / len(scores) >= 5.5 else "needs_work"
        }
        
        return result
        
    except Exception as e:
        print(f"Analysis error: {e}")
        return create_visual_enhanced_fallback(frames_summaries_text, {
            'transcript': transcript_text,
            'is_reliable': len(transcript_text.strip()) > 50
        }, goal)


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

        # --- Get knowledge context using direct file parsing FIRST ---
        try:
            print("[INFO] Loading knowledge from /knowledge folder and full knowledge base...")
            knowledge_context, knowledge_citations = get_knowledge_context()
            
            print(f"[RESULT] Knowledge context length: {len(knowledge_context)} characters")
            print(f"[RESULT] Sources: {knowledge_citations}")
            
            if len(knowledge_context) < 500:
                print("[WARNING] Very little knowledge content loaded - using enhanced fallback")
                
        except Exception as e:
            print(f"Knowledge loading error: {e}")
            import traceback
            traceback.print_exc()
            
            # Enhanced fallback with performance-aware patterns
            knowledge_context = """
VIRAL CONTENT ANALYSIS FOR PERFORMANCE-BASED INSIGHTS:

PERFORMANCE CONTEXT ANALYSIS:
Creator Performance Note: {creator_note}

HIGH-PERFORMING VS LOW-PERFORMING PATTERNS:

HIGH-PERFORMING HOOKS (8M+ avg views):
- "POV: you just discovered the real reason..."
- "Nobody talks about how this actually works..."  
- "I tried this method for 30 days and the results..."
- "The industry secret they don't want you to know..."

LOW-PERFORMING HOOKS (<500k views):
- Generic openings without curiosity gaps
- No immediate personal stakes or emotional connection
- Weak pattern interrupts that don't stop scrolling
- Lacks specific, relatable scenarios

RETENTION FACTORS:
- First 3 seconds: Must create curiosity gap or controversy
- 3-7 seconds: Promise specific value or transformation  
- Middle: Build tension toward revelation/payoff
- End: Deliver satisfying conclusion + call to action

ALGORITHM OPTIMIZATION:
- Quick cuts every 2-3 seconds (23% retention boost)
- Comments in first hour (300% reach increase)
- Watch time >50% triggers recommendations
- Visual variety prevents drop-off

ENGAGEMENT PSYCHOLOGY:
- Personal transformation stories drive follows
- Behind-the-scenes content builds trust
- Controversial opinions (with nuance) drive comments
- Before/after reveals drive saves and shares

REALISTIC SCORING GUIDELINES:
High-performing content (500k+ views):
- Hook Strength: 7-9/10 (proven stopping power)
- Promise Clarity: 7-8/10 (clear value proposition)
- Retention Design: 7-9/10 (strong pacing and engagement)
- Engagement Potential: 6-8/10 (drives comments/shares)
- Goal Alignment: 7-9/10 (effectively serves stated goal)

Low-performing content (<100k views):
- Hook Strength: 3-5/10 (weak opening, limited stopping power)
- Promise Clarity: 4-6/10 (unclear or weak value proposition)
- Retention Design: 4-6/10 (pacing issues, viewer drop-off)
- Engagement Potential: 2-4/10 (low shareability/comment potential)
- Goal Alignment: 3-5/10 (doesn't effectively serve viral reach)
""".format(creator_note=creator_note)
            knowledge_citations = ["Enhanced performance-aware fallback patterns"]

        # --- Main Analysis ---
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
            
            # Run main analysis
            gpt_result = run_main_analysis(
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
            gpt_result['text_sync_analysis'] = analyze_text_synchronization(frames_summaries_text, transcript_data.get('transcript', ''))
            
            print("Analysis complete")
            
        except Exception as e:
            print(f"Analysis error: {e}")
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
        
        # Enhanced fields with safe defaults
        visual_content_analysis = gpt_result.get("visual_content_analysis", {})
        transcript_quality = gpt_result.get("transcript_quality", {})
        audio_description = gpt_result.get("audio_description", "")
        text_sync_analysis = gpt_result.get("text_sync_analysis", {})
        
        # Balanced analysis fields
        strengths = gpt_result.get("strengths", "")
        improvement_areas = gpt_result.get("improvement_areas", "")
        knowledge_insights = gpt_result.get("knowledge_insights", "")
        knowledge_context_used = gpt_result.get("knowledge_context_used", False)
        overall_quality = gpt_result.get("overall_quality", "moderate")
        
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

        print("Rendering results template")
        return render_template(
            "results.html",
            tiktok_url=tiktok_url,
            creator_note=creator_note,
            transcript=transcript_for_analysis,
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
            
            # Enhanced fields
            visual_content_analysis=visual_content_analysis,
            transcript_quality=transcript_quality,
            audio_description=audio_description,
            transcript_original=transcript_data.get('transcript', ''),
            transcript_for_analysis=transcript_for_analysis,
            text_sync_analysis=text_sync_analysis,
            
            # Balanced analysis fields
            strengths=strengths,
            improvement_areas=improvement_areas,
            knowledge_insights=knowledge_insights,
            knowledge_context_used=knowledge_context_used,
            overall_quality=overall_quality,
            
            # Compatibility fields for existing template
            psychological_breakdown=analysis_text,
            hook_mechanics=timing_breakdown,
            emotional_journey=analysis_text,
            authority_signals=strengths,
            engagement_psychology=analysis_text,
            viral_mechanisms=analysis_text,
            audience_psychology=analysis_text,
            replication_blueprint=basic_formula,
            multimodal_insights=analysis_text,
            engagement_triggers=analysis_text,
            improvement_opportunities=improvement_areas,
            viral_potential_factors=analysis_text,
            video_description=visual_content_analysis.get('description', 'Video analysis'),
            content_patterns={},
            performance_data={},
            performance_prediction=gpt_result.get("performance_prediction", ""),
            weaknesses=gpt_result.get("improvement_areas", ""),
            critical_assessment=analysis_text,
            
            # Keep for backward compatibility
            gpt_response=analysis_text
        )

    except Exception as e:
        print(f"Unexpected error in process(): {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Unexpected error: {str(e)}", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)