import os
import random
import time as _time
import json
import re
import asyncio
import hashlib
from collections import Counter
from datetime import datetime
from flask import Flask, request, render_template, make_response, redirect, url_for, flash, jsonify
from flask_login import login_required, current_user
from openai import OpenAI
from anthropic import Anthropic

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
    _ensure_dirs,
    extract_video_metadata,
    analyze_save_metrics
)
from rag_helper import retrieve_smart_context, retrieve_all_context
from cache_manager import AnalysisCache, PdfCache
from analysis_optimization import intelligent_frame_extraction, parallel_video_processing, optimize_frame_selection
from audio_analysis import enhanced_audio_analysis, ViralSoundDetector
from performance_tracker import AnalysisPerformanceTracker
from models import db, User, Analysis, init_db
from auth import auth_bp, init_auth

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

app = Flask(__name__)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=600.0)
claude_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Initialize database and authentication
init_db(app)
init_auth(app)
app.register_blueprint(auth_bp)

# Initialize optimization components
cache = AnalysisCache()
tracker = AnalysisPerformanceTracker()
pdf_cache = PdfCache()  # Persistent PDF cache that survives server restarts
print("[INFO] Optimization components initialized: cache, tracker, pdf_cache")
print("[INFO] Database and authentication initialized")


# Custom Jinja filter to convert markdown-style bold to HTML
@app.template_filter('markdown_bold')
def markdown_bold_filter(text, color='#fff'):
    """Convert **text** to <strong>text</strong>"""
    if not text:
        return text
    # Use regex to properly match **text** patterns
    import re
    pattern = r'\*\*([^*]+)\*\*'
    replacement = f'<strong style="color: {color};">\\1</strong>'
    return re.sub(pattern, replacement, str(text))


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

    try:
        from anthropic import Anthropic
    except ImportError:
        missing_deps.append("anthropic")

    if not os.getenv("OPENAI_API_KEY"):
        missing_deps.append("OPENAI_API_KEY environment variable")

    if not os.getenv("ANTHROPIC_API_KEY"):
        missing_deps.append("ANTHROPIC_API_KEY environment variable")

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
            print(f"[retry] API call failed ({attempt}/{max_tries}): {e}. Retrying in {sleep_s:.1f}s")
            _time.sleep(sleep_s)


def parse_delimited_response(response_text):
    """
    Parse delimiter-based response format into structured data.
    This replaces JSON parsing with 100% reliable delimiter parsing.

    Format:
    ===SECTION_NAME===
    content here

    ===NEXT_SECTION===
    more content
    """
    sections = {}
    current_section = None
    current_content = []

    lines = response_text.split('\n')

    for line in lines:
        # Check if this is a section delimiter
        if line.strip().startswith('===') and line.strip().endswith('==='):
            # Save previous section if exists
            if current_section and current_section != 'END':
                sections[current_section] = '\n'.join(current_content).strip()

            # Start new section
            section_name = line.strip().strip('=').strip()
            if section_name != 'END':
                current_section = section_name
                current_content = []
        else:
            # Add line to current section
            if current_section:
                current_content.append(line)

    # Save last section
    if current_section and current_section != 'END':
        sections[current_section] = '\n'.join(current_content).strip()

    # Parse SCORES section specially (convert to dict with integers)
    scores = {}
    if 'SCORES' in sections:
        print(f"[SCORES] Raw SCORES section content:\n{sections['SCORES']}")
        for line in sections['SCORES'].split('\n'):
            if ':' in line and not line.strip().startswith('IMPORTANT') and not line.strip().startswith('Example') and not line.strip().startswith('Use the'):
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                # Extract number from value
                import re
                match = re.search(r'(\d+)', value)
                if match:
                    scores[key] = int(match.group(1))
                    print(f"[SCORES] Parsed {key}: {scores[key]}")
                else:
                    scores[key] = 5  # Default score
                    print(f"[SCORES] Using default for {key}: 5 (couldn't extract number from '{value}')")

    # Parse EXACT_HOOK_BREAKDOWN into dict
    exact_hook = {}
    if 'EXACT_HOOK_BREAKDOWN' in sections:
        for line in sections['EXACT_HOOK_BREAKDOWN'].split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                exact_hook[key.strip()] = value.strip()

    # Parse TIMING_MASTERY into dict
    timing = {}
    if 'TIMING_MASTERY' in sections:
        for line in sections['TIMING_MASTERY'].split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                timing[key.strip()] = value.strip()

    # Parse REPLICATION_FORMULA into structured dict with multi-line support
    replication = {}
    if 'REPLICATION_FORMULA' in sections:
        content = sections['REPLICATION_FORMULA']
        lines = content.split('\n')

        current_key = None
        current_value = []

        for line in lines:
            # Check if this is a key line (has colon and doesn't start with **)
            if ':' in line and not line.strip().startswith('**'):
                # Save previous key-value if exists
                if current_key:
                    value_text = '\n'.join(current_value).strip()
                    if current_key == 'scenarios_for_same_niche':
                        # Parse scenarios by splitting on **Scenario markers
                        scenarios = []
                        scenario_blocks = value_text.split('**Scenario')
                        for block in scenario_blocks:
                            if block.strip():
                                scenarios.append('**Scenario' + block.strip())
                        replication[current_key] = scenarios if scenarios else [value_text]
                    elif current_key == 'text_template':
                        # Keep text template as-is for proper formatting
                        replication[current_key] = value_text
                    else:
                        replication[current_key] = value_text

                # Start new key
                key, value = line.split(':', 1)
                current_key = key.strip()
                current_value = [value.strip()] if value.strip() else []
            else:
                # Continuation of current value
                if current_key and line.strip():
                    current_value.append(line)

        # Don't forget the last key-value pair
        if current_key:
            value_text = '\n'.join(current_value).strip()
            if current_key == 'scenarios_for_same_niche':
                scenarios = []
                scenario_blocks = value_text.split('**Scenario')
                for block in scenario_blocks:
                    if block.strip():
                        scenarios.append('**Scenario' + block.strip())
                replication[current_key] = scenarios if scenarios else [value_text]
            elif current_key == 'text_template':
                replication[current_key] = value_text
            else:
                replication[current_key] = value_text

    # Parse ALL_HOOKS sections into a combined dict
    all_hooks = {}
    if 'ALL_HOOKS_TEXT' in sections:
        hooks = [line.strip('- ').strip() for line in sections['ALL_HOOKS_TEXT'].split('\n') if line.strip().startswith('-')]
        all_hooks['text_hooks'] = hooks

    if 'ALL_HOOKS_VISUAL' in sections:
        hooks = [line.strip('- ').strip() for line in sections['ALL_HOOKS_VISUAL'].split('\n') if line.strip().startswith('-')]
        all_hooks['visual_hooks'] = hooks

    if 'ALL_HOOKS_VERBAL' in sections:
        hooks = [line.strip('- ').strip() for line in sections['ALL_HOOKS_VERBAL'].split('\n') if line.strip().startswith('-')]
        all_hooks['verbal_hooks'] = hooks

    if 'ALL_HOOKS_PSYCHOLOGICAL' in sections:
        hooks = [line.strip('- ').strip() for line in sections['ALL_HOOKS_PSYCHOLOGICAL'].split('\n') if line.strip().startswith('-')]
        all_hooks['psychological_hooks'] = hooks

    # Parse KNOWLEDGE_PATTERNS_APPLIED into list
    knowledge_patterns = []
    if 'KNOWLEDGE_PATTERNS_APPLIED' in sections:
        knowledge_patterns = [line.strip('- ').strip() for line in sections['KNOWLEDGE_PATTERNS_APPLIED'].split('\n') if line.strip().startswith('-')]

    # Return structured data in the same format template expects
    return {
        'what_this_video_is': sections.get('WHAT_THIS_VIDEO_IS', ''),
        'why_it_performed': sections.get('WHY_IT_PERFORMED', ''),
        'all_hooks_identified': all_hooks,
        'exact_hook_breakdown': exact_hook,
        'replication_formula': replication,
        'improvements': sections.get('IMPROVEMENTS', ''),
        'viral_mechanics': sections.get('VIRAL_MECHANICS', ''),
        'scores': scores,
        'timing_mastery': timing,
        'performance_prediction': sections.get('PERFORMANCE_PREDICTION', ''),
        'knowledge_patterns_applied': knowledge_patterns,

        # Also include as single field for compatibility
        'analysis': sections.get('WHAT_THIS_VIDEO_IS', ''),
        'performance_deep_dive': sections.get('WHY_IT_PERFORMED', ''),
    }


def escape_unescaped_quotes_in_json(json_str):
    """
    Aggressively escape unescaped quotes within JSON string values.

    This handles cases where Claude includes quotes like:
    "text": "She said "hello" to me"

    And converts it to:
    "text": "She said \"hello\" to me"
    """
    result = []
    in_string = False
    in_value = False  # Track if we're in a value vs a key
    escape_next = False
    i = 0

    while i < len(json_str):
        char = json_str[i]

        # Handle already-escaped characters
        if escape_next:
            result.append(char)
            escape_next = False
            i += 1
            continue

        if char == '\\':
            escape_next = True
            result.append(char)
            i += 1
            continue

        # Handle quotes
        if char == '"':
            if not in_string:
                # Starting a string (either key or value)
                in_string = True
                result.append(char)
            else:
                # Ending a string - but is it really the end?
                # Look ahead to see what's next
                next_chars = json_str[i+1:i+10].lstrip()

                # If next char is : then we just finished a key
                if next_chars.startswith(':'):
                    in_string = False
                    in_value = False  # About to start a value
                    result.append(char)
                # If next char is , or } or ] then we finished a value
                elif next_chars.startswith(',') or next_chars.startswith('}') or next_chars.startswith(']'):
                    in_string = False
                    in_value = False
                    result.append(char)
                # Otherwise, this might be an unescaped quote WITHIN a value
                elif in_value:
                    # This is likely an unescaped quote within the value
                    print(f"[REPAIR] Found unescaped quote at position {i}, escaping...")
                    result.append('\\')
                    result.append(char)
                else:
                    # Not sure, assume it's the end of string
                    in_string = False
                    result.append(char)
        else:
            result.append(char)
            # Track when we're in a value (after : and inside string)
            if char == ':' and not in_string:
                in_value = True

        i += 1

    return ''.join(result)


def repair_unterminated_strings(json_str):
    """
    Detect and repair unterminated strings in JSON.
    Common issue: string values that span multiple lines without closing quotes.

    Strategy:
    1. Find unterminated strings (quote followed by newline without closing quote)
    2. Close the string at the end of the line (before newline)
    3. Continue looking for the next field
    """
    lines = json_str.split('\n')
    repaired_lines = []
    in_string = False
    escape_next = False

    for line_idx, line in enumerate(lines):
        if not line.strip():
            repaired_lines.append(line)
            continue

        repaired_line = []
        i = 0

        while i < len(line):
            char = line[i]

            # Handle escape sequences
            if escape_next:
                repaired_line.append(char)
                escape_next = False
                i += 1
                continue

            if char == '\\':
                escape_next = True
                repaired_line.append(char)
                i += 1
                continue

            # Track string state
            if char == '"':
                if in_string:
                    in_string = False
                else:
                    in_string = True
                repaired_line.append(char)
            else:
                repaired_line.append(char)

            i += 1

        # If we reach end of line and still in a string, close it
        if in_string and line_idx < len(lines) - 1:
            # Check if next line looks like a new JSON field or continuation
            next_line = lines[line_idx + 1].strip()

            # If next line starts with a quote or looks like a new field, close this string
            if (next_line.startswith('"') or
                next_line.startswith('}') or
                next_line.startswith(']') or
                re.match(r'^\s*"[\w_]+"\s*:', next_line)):

                print(f"[REPAIR] Closing unterminated string at line {line_idx + 1}")
                repaired_line.append('"')
                in_string = False

        repaired_lines.append(''.join(repaired_line))

    return '\n'.join(repaired_lines)


def _repair_json(json_str):
    """Attempt to repair common JSON issues from LLM responses."""
    # Remove code block markers if present
    if json_str.startswith("```json"):
        json_str = json_str[7:]
    if json_str.startswith("```"):
        json_str = json_str[3:]
    if json_str.endswith("```"):
        json_str = json_str[:-3]

    json_str = json_str.strip()

    # Try to parse as-is first
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"[WARN] Initial JSON parsing failed: {e}")

        # PRIORITY: Use JSONDecoder.raw_decode() - this handles most cases
        try:
            from json import JSONDecoder
            decoder = JSONDecoder()
            result, end_idx = decoder.raw_decode(json_str)
            extra_content = json_str[end_idx:].strip()
            if extra_content:
                print(f"[INFO] Extracted valid JSON, ignoring {len(extra_content)} extra chars: '{extra_content[:100]}'")
            else:
                print(f"[INFO] Successfully parsed JSON with JSONDecoder")
            return result
        except json.JSONDecodeError as e2:
            print(f"[WARN] JSONDecoder.raw_decode failed: {e2}")
        except Exception as e2:
            print(f"[WARN] Unexpected error in JSONDecoder: {e2}")

    # Try removing trailing commas
    try:
        fixed = re.sub(r',(\s*[}\]])', r'\1', json_str)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Try to repair unterminated strings (common error from line breaks)
    try:
        print("[INFO] Attempting to repair unterminated strings...")
        repaired = repair_unterminated_strings(json_str)
        if repaired != json_str:
            print(f"[INFO] String repair made changes, attempting parse...")
            return json.loads(repaired)
    except json.JSONDecodeError as e:
        print(f"[WARN] String repair didn't fix the issue: {e}")
    except Exception as e:
        print(f"[WARN] String repair error: {e}")

    # Try aggressive quote escaping within string values
    try:
        print("[INFO] Attempting aggressive quote escaping...")
        repaired = escape_unescaped_quotes_in_json(json_str)
        if repaired != json_str:
            print(f"[INFO] Quote escaping made changes, attempting parse...")
            return json.loads(repaired)
    except json.JSONDecodeError as e:
        print(f"[WARN] Quote escaping didn't fix the issue: {e}")
    except Exception as e:
        print(f"[WARN] Quote escaping error: {e}")

    # Try to extract JSON by finding balanced braces
    try:
        first_brace = json_str.find('{')
        if first_brace != -1:
            # Count braces to find the matching closing brace
            brace_count = 0
            in_string = False
            escape_next = False

            for i in range(first_brace, len(json_str)):
                char = json_str[i]

                # Handle escape sequences in strings
                if escape_next:
                    escape_next = False
                    continue

                if char == '\\':
                    escape_next = True
                    continue

                # Track whether we're inside a string
                if char == '"' and not in_string:
                    in_string = True
                elif char == '"' and in_string:
                    in_string = False

                # Only count braces outside of strings
                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            # Found the matching closing brace
                            extracted = json_str[first_brace:i+1]
                            print(f"[INFO] Extracted JSON using balanced brace counting ({len(extracted)} chars)")
                            return json.loads(extracted)
    except json.JSONDecodeError as e:
        print(f"[WARN] Balanced brace extraction failed: {e}")
    except Exception as e:
        print(f"[WARN] Unexpected error in brace counting: {e}")

    # If all repairs fail, raise the original error
    print(f"[ERROR] All JSON repair attempts failed")
    print(f"[ERROR] First 500 chars: {json_str[:500]}")
    print(f"[ERROR] Last 200 chars: {json_str[-200:]}")
    raise json.JSONDecodeError("Failed to parse JSON after all repair attempts", json_str, 0)


# ==============================
# AUDIO/VIDEO ANALYSIS HELPERS
# ==============================

def analyze_audio_with_visual_context(transcript_text, frames_summaries_text):
    """
    Intelligently analyze audio by considering visual context
    Don't assume random sounds - correlate with what's happening visually. some transcripts may actually be songs or trending quotes from movies, tv shows, or other viral videos. determine what type of sound this is.
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
    """Enhanced extraction with validation and fallback"""
    try:
        print(f"[INFO] Starting enhanced extraction for {tiktok_url}")
        
        # First attempt with requested strategy
        audio_path, frames_dir, frame_paths = extract_audio_and_frames(
            tiktok_url, strategy, frames_per_minute, cap, scene_threshold
        )
        
        # If smart strategy yields too few frames, fall back to uniform
        if strategy == 'smart' and len(frame_paths) < 5:
            print(f"[WARNING] Smart extraction only got {len(frame_paths)} frames, falling back to uniform")
            
            try:
                # Re-extract with uniform strategy
                audio_path, frames_dir, frame_paths = extract_audio_and_frames(
                    tiktok_url, 
                    strategy='uniform',
                    frames_per_minute=30,  # Every 2 seconds
                    cap=60,
                    scene_threshold=scene_threshold
                )
                    
            except Exception as e:
                print(f"[ERROR] Fallback extraction failed: {e}")
                # Continue with whatever frames we have
                if len(frame_paths) == 0:
                    raise ValueError("No frames could be extracted")
        
        # Validate audio
        if not audio_path or not os.path.exists(audio_path):
            print("[WARNING] Audio extraction failed, continuing without audio")
            audio_path = None
        else:
            audio_size = os.path.getsize(audio_path)
            if audio_size < 1024:
                print(f"[WARNING] Audio file too small ({audio_size} bytes)")
        
        # Validate frames
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


def generate_timing_breakdown(duration_seconds):
    """
    Generate dynamic timing breakdown based on video duration.
    Returns a formatted string for the prompt.
    """
    try:
        duration = int(duration_seconds)
    except (ValueError, TypeError):
        duration = 30  # Default fallback

    # Generate timing intervals based on video length
    if duration <= 15:
        # Very short videos
        intervals = [
            "0-1s: [Hook and opening]",
            "1-5s: [Core content/reveal]",
            f"5-{duration}s: [Payoff and close]"
        ]
    elif duration <= 30:
        # Short videos (15-30s)
        mid = duration // 2
        end = duration
        intervals = [
            "0-1s: [Hook]",
            "1-3s: [Promise/setup]",
            f"3-{mid}s: [Tension/context building]",
            f"{mid}-{end-3}s: [Building to payoff]",
            f"{end-3}-{end}s: [Payoff and close]"
        ]
    elif duration <= 60:
        # Medium videos (30-60s)
        q1 = duration // 4
        mid = duration // 2
        q3 = (duration * 3) // 4
        end = duration
        intervals = [
            "0-1s: [Hook]",
            "1-3s: [Promise]",
            f"3-{q1}s: [Initial context/stakes]",
            f"{q1}-{mid}s: [Tension building/examples]",
            f"{mid}-{q3}s: [Further development]",
            f"{q3}-{end-3}s: [Final build to payoff]",
            f"{end-3}-{end}s: [Payoff delivery and close]"
        ]
    else:
        # Long videos (60s+)
        q1 = duration // 4
        mid = duration // 2
        q3 = (duration * 3) // 4
        end = duration
        intervals = [
            "0-1s: [Hook]",
            "1-3s: [Promise]",
            f"3-{q1}s: [Context and stakes]",
            f"{q1}-{mid}s: [Tension and examples]",
            f"{mid}-{q3}s: [Secondary loops/development]",
            f"{q3}-{end-5}s: [Building to climax]",
            f"{end-5}-{end}s: [Payoff and resolution]"
        ]

    return "\n".join(intervals)


# ==============================
# PDF GENERATION - SERVER-SIDE WITH PLAYWRIGHT
# ==============================

async def generate_pdf_from_html(html_content, output_path=None):
    """
    Generate PDF from HTML using playwright (headless Chrome).
    Returns PDF bytes if output_path is None, otherwise saves to file.
    """
    from playwright.async_api import async_playwright

    try:
        print("[PDF] Launching headless browser...")
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage']
            )

            page = await browser.new_page()

            # Set viewport for consistent rendering
            await page.set_viewport_size({'width': 1200, 'height': 1600})

            print("[PDF] Loading HTML content...")
            await page.set_content(html_content, wait_until='networkidle', timeout=30000)

            # Wait a bit for any dynamic content to render
            await asyncio.sleep(1)

            print("[PDF] Generating PDF...")
            pdf_options = {
                'format': 'A4',
                'print_background': False,  # Plain text on white background
                'scale': 0.5,  # Shrink content to fit more on page
                'margin': {
                    'top': '10mm',
                    'right': '10mm',
                    'bottom': '10mm',
                    'left': '10mm'
                }
            }

            if output_path:
                pdf_options['path'] = output_path
                await page.pdf(**pdf_options)
                print(f"[PDF] Saved to {output_path}")
                await browser.close()
                print("[PDF] Browser closed")
                return output_path
            else:
                pdf_bytes = await page.pdf(**pdf_options)
                print(f"[PDF] Generated {len(pdf_bytes)} bytes")
                await browser.close()
                print("[PDF] Browser closed")
                return pdf_bytes

    except Exception as e:
        print(f"[PDF ERROR] Failed to generate PDF: {e}")
        raise


def generate_pdf_sync(html_content, output_path=None):
    """Synchronous wrapper for async PDF generation"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(generate_pdf_from_html(html_content, output_path))
        loop.close()
        return result
    except Exception as e:
        print(f"[PDF ERROR] Sync wrapper failed: {e}")
        raise


# ==============================
# MAIN ANALYSIS FUNCTION - COMPREHENSIVE & ADAPTIVE
# ==============================

def _build_comment_section(comment_insights):
    """Build comment analysis section for GPT prompt"""
    if not comment_insights or not comment_insights.get('total_comments'):
        return ""

    total = comment_insights.get('total_comments', 0)
    categorization = comment_insights.get('categorization', {})
    consensus = comment_insights.get('consensus_patterns', [])
    ai_insights = comment_insights.get('ai_insights', '')

    section = f"""
VIEWER COMMENT ANALYSIS ({total} comments analyzed):

ENGAGEMENT TYPE BREAKDOWN:
- Substantive comments: {categorization.get('percentages', {}).get('substantive', 0)}% (viewers articulating specific moments)
- Emoji-only reactions: {categorization.get('percentages', {}).get('emoji_only', 0)}% (emotional but non-verbal)
- Generic reactions: {categorization.get('percentages', {}).get('generic', 0)}%
→ {categorization.get('insight', '')}

CONSENSUS PATTERNS (what viewers collectively agreed on - ranked by total likes):
"""

    if consensus:
        for i, pattern in enumerate(consensus[:5], 1):
            section += f"\n{i}. {pattern['theme']} - {pattern['total_likes']} total likes across {pattern['mentions']} comments"
            if pattern.get('examples'):
                section += f"\n   Example: \"{pattern['examples'][0]['text']}\" ({pattern['examples'][0]['likes']} likes)"
    else:
        section += "\nNo strong consensus patterns detected."

    section += f"""

VIEWER INSIGHTS (what comments reveal):
{ai_insights}

CRITICAL ANALYSIS REQUIREMENTS FOR COMMENTS:
1. WHY PEOPLE WATCHED: What do comments reveal drew them in? (intended hook or accidental element?)
2. WHAT INTRIGUED THEM: What created curiosity? What kept them engaged?
3. WHAT IRRITATED THEM: Did anything frustrate viewers? (Note: irritation can drive engagement)
4. QUESTIONS ASKED:
   - Unanswered curiosity gaps (video opened question but didn't close it)
   - Content ideas for future videos
   - Confusion that needs clarity

Use these insights to identify:
- Gap between INTENT (what video tried to do) vs REALITY (what actually engaged viewers)
- Hidden hooks (unintentional elements that drove more engagement than main content)
- Specific moments that resonated (timestamp mentions, quoted phrases)
- Opportunities to replicate what worked or close gaps that frustrated viewers
"""

    return section


def run_main_analysis(transcript_text, frames_summaries_text, creator_note, platform, target_duration, goal, tone, audience, knowledge_context, view_count=None, performance_level='unknown', metadata=None, audio_insights=None, analysis_depth='standard', comment_insights=None):
    """Comprehensive analysis that adapts to ALL video types with deep insights"""
    
    # First analyze frames to understand visual content
    visual_content_analysis = create_visual_content_description(frames_summaries_text)
    
    # Then analyze audio with visual context
    audio_analysis = analyze_audio_with_visual_context(transcript_text, frames_summaries_text)
    
    # Use audio analysis results
    has_speech = audio_analysis['has_meaningful_speech']
    audio_type_info = audio_analysis
    
    # USE PASSED-IN VIEW COUNT (don't re-parse from creator_note unless not provided)
    if not view_count and creator_note:
        # Only parse from creator_note if view_count wasn't provided
        note_lower = creator_note.lower().replace(',', '').strip()
        view_patterns = re.findall(r'(\d+\.?\d*)\s*(k|thousand|m|million|views)?', note_lower)
        if view_patterns:
            for pattern in view_patterns:
                try:
                    num = float(pattern[0])
                    unit = pattern[1] if len(pattern) > 1 else ''
                    if unit in ['k', 'thousand']:
                        view_count = f"{num}k"
                        performance_level = 'good' if num >= 500 else 'moderate' if num >= 100 else 'low'
                        break
                    elif unit in ['m', 'million']:
                        view_count = f"{num}M"
                        performance_level = 'viral'
                        break
                    elif num >= 1000000:
                        view_count = f"{num/1000000:.1f}M"
                        performance_level = 'viral'
                        break
                    elif num >= 1000:
                        view_count = f"{num/1000:.0f}k"
                        performance_level = 'good' if num >= 500000 else 'moderate' if num >= 100000 else 'low'
                        break
                except ValueError:
                    continue
    
    # Build knowledge section
    knowledge_section = ""
    if knowledge_context and len(knowledge_context.strip()) > 100:
        knowledge_section = f"""
KNOWLEDGE BASE PATTERNS (Apply these insights deeply):
{knowledge_context}

DEEP ANALYSIS REQUIREMENTS:
1. EXPLAIN THE PSYCHOLOGY: Why does this video keep people watching until the end. explain it plainly but in an explanatory manner. how did the hook or hooks work to grab attention? what curiosity gap was created and what were the questions the viewer's mind that kept them watching? when was the curiosity gap closed and did the video end there or continue to create a new hook, promise, payoff. Also explain this in a way something similar to "the hook(s): [hooks here] made people pay attention because → the promise/curiosity gap created the question of [insert question or questions created by curioisty gap]. What they expect to see/hear/understand. → Tension is built by [explain what builds tension that keeps people wanting to stay to close the gap] → Gap is closed when results/ending/answer is shown" if it didn't follow the formulas explain why. what elements work or don't work?
2. REFERENCE PROVEN PATTERNS: Connect to specific patterns from the knowledge base
3. BE SPECIFIC: Analyze EXACTLY what happens in THIS video, not generic observations
4. DISTINGUISH CONTENT TYPES: Clearly differentiate and identify what's spoken vs what's shown vs what's written. if text on screen correlates with script then it may be captions, sometimes though people do not add captions and add text on screen that matches what they said. 
5. PERFORMANCE REASONING: Explain WHY this got {view_count if view_count else 'its current'} views
6. ACTIONABLE INSIGHTS: Provide specific improvements to this exact video like explaining how a stronger curiosity gap could have been created or delivery could have been saved until after saying something.
7. TIMING PRECISION: Break down what happens at each second marker and watch for lulls in conversation to understand where pacing could have been improved if applicable
8. HANDLE ALL VIDEO TYPES: Visual-only, speech, viral audio, tutorials, transformations, etc.
9. CONSIDER CREATOR NOTES. IF QUESTIONS OR REQUESTS ARE MADE, answer THEM. IF CONTEXT IS PROVIDED, CONSIDER IT WITH YOUR ANALYSIS.
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
- How verbal (froms transcript) and visual elements work together
- Whether on-screen text reinforces or adds to spoken content
- The relationship between what's said and what's shown
- Speech delivery effectiveness and clarity
- ensure to compare transcript with text on screen to determine if its added text or captions. if transcript does not match up with text on screen, it is likely added text. if they match up, it is captions.
"""

    # Build the performance message separately to avoid f-string issues
    if performance_level == 'viral':
        performance_message = f"This video went VIRAL with {view_count} - analyze WHY it succeeded."
    else:
        performance_message = f"This video got {view_count if view_count else 'certain performance'} - analyze what's working and what needs to improve to achieve higher success in relation to the chosen goal."
    
    prompt = f"""
You are a video psychology expert analyzing transcribed and extracted data from a {platform} video. 
You have been provided with frame-by-frame descriptions and audio transcription below.
You are NOT being asked to view a video directly - you are analyzing the provided text data. {performance_message}

CRITICAL CONTEXT:
- Platform: {platform}
- Performance: {view_count if view_count else 'Not specified'} ({performance_level})
- Creator's note: "{creator_note if creator_note else 'No additional context'}"
- Content type detected: {visual_content_analysis.get('content_type', 'general')}
- Audio type: {audio_type_info.get('audio_description', 'unknown')}
- Goal: {goal}
- Target audience: {audience}
- Duration target: {target_duration}s
- Views: {view_count}

AUDIO CONTEXT:
{f"Speech detected: {transcript_text}" if has_speech else f"Non-speech audio: {audio_type_info.get('likely_sound_source', 'ambient sounds')} (based on visual activity)"}

VISUAL CONTENT (frames - what's SHOWN/WRITTEN):
{frames_summaries_text}

VISUAL ANALYSIS:
- Content type: {visual_content_analysis.get('content_type', 'general')}
- Satisfaction score: {visual_content_analysis.get('satisfaction_analysis', {}).get('satisfaction_score', 0)}/5
- Visual narrative: {visual_content_analysis.get('visual_promise_delivery', {}).get('narrative_strength', 'unknown')}

{video_type_context}

{_build_comment_section(comment_insights) if comment_insights else ""}

{knowledge_section}

COMPREHENSIVE ANALYSIS INSTRUCTIONS:
{f"Since this went VIRAL, identify the EXACT psychological triggers and viral mechanics." if performance_level == 'viral' else "Identify opportunities for this specific video that can help improve based on or inspired by proven patterns. Give 2-3 ideas and examples. Instead of saying something like 'enhance visual storytelling' explain to them how they might do that thing and provide strong examples"}


#### View Count Context Matters
**DO NOT judge sales or conversion videos by entertainment metrics!**

**Entertainment Content:**
- 10K views = Low
- 100K views = Moderate  
- 500K+ views = Good
- 1M+ views = Viral

**B2B/Service Provider Content:**
- 10K views = Solid
- 50K views = Excellent
- 90K+ views = HIGHLY SUCCESSFUL
- 200K+ views = Exceptional

**WHY THE DIFFERENCE:**
B2B content with 90K views generating multiple inquiries is worth more than entertainment content with 1M views generating nothing. ALWAYS consider:
- Niche size (designers vs general public)
- Conversion value (one client = $5-50K)
- Business impact (inquiries, not just views)

BUT there is also a world where a video MEANT to sell only amassed many views and didn't generate sales. We need to explain in the output why this happened and what sales psychology tactics are missing and then HOW to implement them.

**IF A VIDEO SEEMS TO BE ABOUT ANY TYPE OF SELLING/SERVICE/PRODUCT**
START WITH: "This video achieved strong performance because..."
NOT: "This video shows potential but..."
IDENTIFY: What psychological triggers drove success

For videos with 50K+ views in B2C/B2B/service niches:
Trust building
Objection handling
Authority demonstration
Problem-solving display

THEN SUGGEST: Amplifications to go even bigger

"To reach 200K+, consider..."
"To maximize inquiries, add..."
"To increase conversion, include..."

### NICHE-SPECIFIC VIEW EXPECTATIONS

**Design/Creative Services:**
- 20K views = Good
- 50K views = Excellent
- 90K+ views = Outstanding

**Consulting/Coaching:**
- 10K views = Good
- 30K views = Excellent
- 50K+ views = Outstanding

**Real Estate/Local Services:**
- 5K views = Good
- 20K views = Excellent
- 40K+ views = Outstanding

**E-commerce/Products:**
- 50K views = Good
- 200K views = Excellent
- 500K+ views = Outstanding

### CRITICAL REMINDERS

1. **90K views generating business = MASSIVE SUCCESS**
2. **One inquiry worth $10K > 100K views worth $0**
3. **Trust-building > Entertainment for B2B**
4. **Process content > Portfolio content for conversion**
5. **Vulnerability = Relatability = Inquiries**


1. FIRST 3 SECONDS BREAKDOWN:
   - Frame by frame: What EXACTLY appears and why was it successful or unsuccessful?
   - What EXACT texts on the screen are shown (from frames, not transcript)? DO NOT MAKE UP TEXT ELEMENTS THAT ARE NOT IN ANY FRAMES.
   - What's the audio (speech from transcript, or {audio_type_info.get('likely_sound_source', 'sounds')})?
   - Is there a visual hooks grab attention? If so, what? Why did it work?
   - Rate the hook strength and explain WHY while educating on how to improve or how to replicate if its already good.

2. PERFORMANCE MECHANICS:
   {f"- What specific elements made this shareable?{chr(10)}   - What psychological triggers drove the viral spread?{chr(10)}   - How did it tap into platform algorithms?{chr(10)}   - What made people watch to completion?" if performance_level == 'viral' else f"- What's preventing viral growth?{chr(10)}   - Which psychological triggers are missing?{chr(10)}   - How could platform algorithms be better leveraged?{chr(10)}   - Where do viewers likely drop off?"}

3. CONTENT STRUCTURE ANALYSIS:
   - Hook mechanisms (0-3s): How does it stop scrolling?
   - Promise delivery (3-10s): What, is anything, is promised or an implied promise? What curiosity gap is created?
   - Retention mechanics (middle): What keeps viewers? What builds tension while they wait for the promise delivery.
   - Payoff (end): How does it close the curiosity gap, satisfy, or create sharing impulse?

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
[reformat video into recommended video format with an exact script based off your experise and the supporting knowledge. give exact script example and include where to place pattern interrupts and other opportunities to layers hooks, reintroduce hooks, and continually leave the curiosity gap open until the end of the video. Explain in an educational and descriptive way without too much jargon.]

CRITICAL CONTEXT:
- Platform: {platform}
- Performance: {view_count if view_count else 'Not specified'} ({performance_level})
- Creator's note: "{creator_note if creator_note else 'No additional context'}"
- Content type detected: {visual_content_analysis.get('content_type', 'general')}
- Audio type: {audio_type_info.get('audio_description', 'unknown')}
- Goal: {goal}
- Target audience: {audience}
- Duration target: {target_duration}s
- Niche context: {' (mentions inquiries/sales)' if creator_note and ('inquir' in creator_note.lower() or 'sale' in creator_note.lower()) else ''}

PERFORMANCE CALIBRATION:
- For B2B/professional content: 90k views = HIGHLY SUCCESSFUL
- For entertainment: 90k views = moderate
- If creator mentions "inquiries" or "sales" = SUCCESS regardless of views

AUDIO CONTEXT:
{f"Speech detected: {transcript_text}" if has_speech else f"Non-speech audio: {audio_type_info.get('likely_sound_source', 'ambient sounds')} (based on visual activity)"}

VISUAL CONTENT (frames - what's SHOWN/WRITTEN):
{frames_summaries_text}

{knowledge_section}

{video_type_context}

{performance_message}


{"Respond using SECTION DELIMITERS with DEEP, COMPREHENSIVE insights. Provide detailed, educational explanations with full psychological breakdowns. Don't hold back on details and examples." if analysis_depth == 'deep' else "Respond using SECTION DELIMITERS with CONCISE, ACTIONABLE insights. Keep each section brief (2-4 sentences). Focus on the most important takeaways only."}

You can use quotes "like this" freely, write multiple paragraphs, and format naturally without worrying about escaping.

Use this EXACT format with === delimiters:

===WHAT_THIS_VIDEO_IS===
{"This is a [specific formula/pattern] video that [explain the core idea/hook/appeal in educational, explanatory language]. It works because [detailed psychological reason with moment-by-moment breakdown explaining hook effectiveness, promise clarity, tension building, and satisfaction delivery. Point out specific seconds where retention tactics work and explain why]." if analysis_depth == 'deep' else "This is a [specific formula/pattern] video. Core hook: [what grabbed attention]. Why it works: [1-2 sentence psychological explanation]. Result: [did it satisfy or not, briefly]."}

===WHY_IT_PERFORMED===
{"This video got {view_count if view_count else 'these views'} because [comprehensive analysis: list each performance driver, explain why it generated views, break down hook/promise/retention/satisfaction, identify what worked or didn't work, provide specific improvement suggestions]. Main psychological trigger: [detailed explanation]. Viewers stayed because: [comprehensive reason]. It attracted [specific audience] who [detailed engagement explanation]." if analysis_depth == 'deep' else "This video got {view_count if view_count else 'these views'} because: [1-2 main reasons]. Key trigger: [one psychological driver]. Audience: [who watched and why in 1 sentence]."}

===ALL_HOOKS_TEXT===
ONLY list text that appears in the FIRST 3 SECONDS that STOPS the scroll.
Not all text is a hook - only attention-grabbing opening text.
- [Text from first 3 seconds only]

===ALL_HOOKS_VISUAL===
ONLY list visual elements in FIRST 3 SECONDS that grabbed attention.
- [Opening visual hooks only, not all visuals]

===ALL_HOOKS_VERBAL===
ONLY list what's SAID in FIRST 3 SECONDS that grabbed attention.
Promises, explanations, and body content are NOT hooks - only opening attention-grabbers.
- [Opening verbal hooks only]

===ALL_HOOKS_PSYCHOLOGICAL===
What mental trigger(s) in the OPENING caused someone to stop scrolling?
- [Primary psychological hook trigger only]

===EXACT_HOOK_BREAKDOWN===
first_frame: 0:00 - [EXACTLY what appears]
second_moment: 0:01 - [What happens]
third_second: 0:02 - [What occurs]
visual_elements: [Visual hooks and their effectiveness]
text_overlays: [Text on screen and impact]
audio_element: {audio_type_info.get('audio_description', 'Audio type')}
hook_psychology: [Why this hook works or doesn't psychologically]
hook_score: [1-10]
hook_reasoning: [Score reasoning]

===REPLICATION_FORMULA===
formula_name: The [Name] Formula
structure: 0-Xs: [what to do], X-Ys: [next step], Y-Zs: [final step]

scenarios_for_same_niche:
IMPORTANT: Format each scenario with a clear title and spacing for readability.

**Scenario 1: [Descriptive Title]**
[Full script + scenes to capture and hold attention]

**Scenario 2: [Descriptive Title]**
[Full script + scenes to capture and hold attention]

why_it_works: This formula works because [psychological explanation in educational, explanatory terms, referring to specific moments, scenes, and promises that make it work]

text_template:
IMPORTANT: Break down into individual timestamped sections for easy reading. Format like this:

**Added Text:** "[TEXT OVERLAY 1]"

**0-3s:** "[Hook text]"

**3-8s:** "[Promise/setup text]"

**8-12s:** "[Context/credibility text]"

[Continue with all remaining timestamped sections based on video length]

**[Final seconds]:** "[Payoff/CTA text]"

visual_requirements: Show [specific visuals needed. Give 2-3 ideas that could help them capture attention and/or increase viewer retention and explain why they would work.]

===IMPROVEMENTS===
REMEMBER: NEVER suggest revealing payoff earlier than 75% through video. Focus on hook strength, visual variety, tension building, and engagement hooks instead.
{"Improvement 1: [Detailed specific actionable change with exact word-for-word script suggestions, pattern interrupts, and retention tactics. Explain why this would work psychologically]. Improvement 2: [Another detailed improvement with scripts and explanations]. Improvement 3: [Third improvement]. This could push views to [realistic projection with detailed reasoning]." if analysis_depth == 'deep' else "1. [Most important improvement in 1 sentence]. 2. [Second improvement in 1 sentence]. 3. [Third improvement in 1 sentence]. Potential: [projected views with brief reason]."}

===VIRAL_MECHANICS===
CRITICAL: NEVER suggest "front-loading the reveal" or "satisfying curiosity faster" - this destroys retention. Suggest adding tension, stakes, visual variety, engagement hooks BEFORE the payoff instead.
{'"This went viral because: [List 3-5 specific viral triggers with detailed psychological explanations, referring to exact moments and mechanics]" if performance_level == "viral" else "To go viral, this video needs: [List 3-5 specific viral triggers to add/strengthen with explanations - focus on hook power, tension building, visual pattern breaks, NOT moving the payoff earlier]"' if analysis_depth == 'deep' else '"Viral because: [2 main reasons]" if performance_level == "viral" else "To go viral: [2 key additions needed - NOT earlier payoff]"'}

===SCORES===
IMPORTANT: Provide actual numeric scores based on THIS video's performance, NOT placeholders.
Use the performance level ({performance_level}) and view count ({view_count if view_count else 'unknown'}) to calibrate scores.

hook_strength: [Number 1-10]
promise_clarity: [Number 1-10]
retention_design: [Number 1-10]
engagement_potential: [Number 1-10]
viral_potential: [Number 1-10]
satisfaction_delivery: [Number 1-10]
goal_alignment: [Number 1-10]

Example format:
hook_strength: 8
promise_clarity: 7
(Just the number, no extra text)

===TIMING_MASTERY===
Analyze what happens at each key moment through the FULL {target_duration}s video.
Provide specific observations for each time interval below:

{generate_timing_breakdown(target_duration)}

For each interval, describe:
- What content/actions occur
- Viewer psychological state
- Retention tactics used (or missing)
- How it builds toward payoff

===PERFORMANCE_PREDICTION===
{'"This succeeded because: [Detailed analysis of why this hit ' + str(view_count) + ' - break down each success factor]. Future potential: [What it could achieve with tweaks]." if performance_level == "viral" else "With the improvements above, this could achieve: [Specific view target with detailed reasoning about what would drive that growth]."' if analysis_depth == 'deep' else '"Success factors: [2 key reasons]. Potential: [projected views with 1 sentence why]." if performance_level == "viral" else "Could reach: [view target] if [1-2 key improvements made]."'}

===KNOWLEDGE_PATTERNS_APPLIED===
- [Pattern from knowledge base]: [How it applies to this video]
- [Another pattern]: [Implementation example]

===END===

CRITICAL INSTRUCTIONS:

🚨 PAYOFF TIMING RULE (MOST IMPORTANT):
NEVER suggest revealing the answer/payoff/promise early in the video.
- The payoff should be delivered at 75-90% through the video (last 10-20%)
- For a 45s video: reveal at 34-41 seconds (NOT at 8-10 seconds!)
- For a 60s video: reveal at 45-54 seconds
- For a 30s video: reveal at 23-27 seconds

WRONG ADVICE EXAMPLES (NEVER SAY THIS):
❌ "Reveal the answer within the first 10 seconds"
❌ "Front-load the reveal in the first 8 seconds"
❌ "Satisfy curiosity faster to prevent drop-off"
❌ "Close the curiosity gap earlier"

CORRECT ADVICE:
✅ "Current reveal timing at 30s in 45s video (67%) is good - maintain this structure"
✅ "Push the reveal slightly later to 36-38s (80-84%) for maximum retention"
✅ "Build more tension before the payoff"

WHY EARLY PAYOFF KILLS RETENTION:
- If you reveal at 10s in a 45s video, viewers have 35s of post-satisfaction drop-off
- Algorithm sees retention crash after payoff = lower distribution
- The HOOK creates curiosity gap → TENSION builds it → PAYOFF at END satisfies it
- Short videos (15s) work because they're SHORT, not because of early reveals

THE "15-SECOND RULE" DOES NOT MEAN REVEAL THE ANSWER:
"Give viewers a reason to stay every 15 seconds" means:
✅ Add new context/information that builds toward the answer
✅ Escalate stakes ("this can cost you $50k")
✅ Add visual pattern breaks (B-roll, examples, text overlays)
✅ Create secondary curiosity loops ("but here's what most don't realize...")
✅ Tease what's coming ("and the worst part is...")
✅ Mini-reveals (NOT the main payoff)

❌ NEVER interpret this as "reveal the main answer/payoff early"
❌ NEVER interpret this as "satisfy the primary curiosity gap"
❌ NEVER interpret this as "front-load the reveal"

Example for "Never hire designer who asks..." video:
- 0-3s: Hook (NEVER HIRE text)
- 3-7s: Promise + context ("the question is about aesthetics, but here's the problem...")
- 7-15s: Stakes ("this thinking cost businesses $50k in rebranding")
- 15-25s: Tension building (examples, contrast, why it matters)
- 30-35s: THE REVEAL (what the question actually is)
- 35-45s: Why it's dangerous + CTA

ONLY suggest earlier reveals for:
- Tutorial/educational content where process IS the value
- Transformation videos where before/after is shown throughout
- "Watch me do X" format where the doing is the entertainment
"""

    # Add depth-specific instructions
    if analysis_depth == 'deep':
        prompt += """- Write comprehensive, educational explanations without holding back on details
- Provide moment-by-moment psychological breakdowns
- Include specific examples and scenarios
- Give exact scripts and word-for-word suggestions
- Explain the "why" behind every observation
"""
    else:
        prompt += """- Keep each section concise (2-4 sentences max)
- Focus on most important insights only
- Prioritize actionable takeaways over theory
- Use simple, direct language
"""

    prompt += """- Use EXACT section names with === delimiters as shown above
- FILL EVERY SECTION - do not leave any section empty
- All hooks sections should focus on FIRST 3 SECONDS only
- You can use quotes, line breaks, and write naturally within each section
- End with ===END=== marker
"""

    # Store raw response for fallback
    raw_response_text = None

    try:
        print(f"[INFO] Running COMPREHENSIVE analysis for {performance_level} {visual_content_analysis.get('content_type', 'content')}...")
        print(f"[INFO] View count: {view_count}, Audio type: {audio_type_info.get('audio_description', 'unknown')}")
        print(f"[INFO] Visual satisfaction score: {visual_content_analysis.get('satisfaction_analysis', {}).get('satisfaction_score', 0)}/5")
        print(f"[INFO] Knowledge base: {len(knowledge_context)} chars")

        gpt_response = _api_retry(
            claude_client.messages.create,
            model="claude-sonnet-4-5-20250929",
            max_tokens=6000,
            temperature=0.7,
            system="You are an expert in viral psychology and content analysis. Provide DEEP, specific insights about why content succeeds or fails. Always explain the psychological mechanisms. Never give surface-level observations. Correctly interpret audio based on visual context - if someone is drawing, sounds are likely marker/pen sounds, not animal noises.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        response_text = gpt_response.content[0].text.strip()
        raw_response_text = response_text  # Save for fallback

        # Parse delimiter-based response (no JSON issues!)
        print("[INFO] Parsing delimited response...")
        parsed = parse_delimited_response(response_text)
        print(f"[INFO] Successfully parsed {len(parsed)} fields from response")
        
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
        
        # Build comprehensive result with ALL new conversational fields
        result = {
            # New conversational analysis fields
            "what_this_video_is": parsed.get("what_this_video_is", ""),
            "why_it_performed": parsed.get("why_it_performed", ""),
            "all_hooks_identified": parsed.get("all_hooks_identified", {}),
            "replication_formula": parsed.get("replication_formula", {}),
            
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
            "improvements": parsed.get("improvements", parsed.get("improvement_opportunities", "")),
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

        # If we have Claude's raw response, use it
        if raw_response_text:
            print(f"[FALLBACK] Using raw text response after error ({len(raw_response_text)} chars)")
            return create_raw_text_fallback(
                raw_response_text, view_count, performance_level,
                audio_type_info, visual_content_analysis, has_speech
            )
        else:
            # Return enhanced fallback
            return create_comprehensive_fallback(
                transcript_text, frames_summaries_text, creator_note,
                platform, goal, audience, has_speech, view_count, performance_level,
                knowledge_context, audio_type_info, visual_content_analysis
            )


def create_raw_text_fallback(raw_response_text, view_count, performance_level, audio_type_info, visual_content_analysis, has_speech):
    """
    Create a fallback result using Claude's raw text response when JSON parsing fails.
    This ensures users still get the intelligent analysis, just without the structured formatting.
    """
    # Create basic scores based on performance
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
        # Special flag to indicate this is a raw text fallback
        "is_raw_text_fallback": True,
        "raw_analysis_text": raw_response_text,

        # Basic metadata for template compatibility
        "scores": base_scores,
        "actual_view_count": view_count,
        "performance_level": performance_level,
        "video_has_speech": has_speech,

        # Provide minimal structure to prevent template errors
        "what_this_video_is": "Analysis available in raw text format below",
        "why_it_performed": f"Performance level: {performance_level}",
        "analysis": raw_response_text,
        "viral_mechanics": "See raw analysis below",
        "hooks": [],
        "formulas": {},
        "improvements": "See raw analysis below",

        # Audio/visual metadata
        "content_type_detected": audio_type_info.get('type', 'unknown'),
        "audio_type_detected": audio_type_info.get('audio_description', 'unknown'),
        "visual_content_analysis": visual_content_analysis,
        "viral_audio_analysis": {
            "is_viral_sound": audio_type_info.get('viral_audio_check', False),
            "audio_type": audio_type_info.get('audio_description', 'unknown')
        },

        # Template compatibility
        "overall_quality": "strong" if performance_level == 'viral' else "moderate" if performance_level in ['good', 'moderate'] else "needs_work",
        "knowledge_context_used": False
    }


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
        # New conversational fields with fallback content
        "what_this_video_is": f"This is a {visual_content_analysis.get('content_type', 'content')} video that {'went viral' if performance_level == 'viral' else 'performed well' if performance_level in ['good', 'moderate'] else 'has growth potential'}.",
        "why_it_performed": f"This video got {view_count if view_count else 'its views'} because of its {audio_type_info.get('audio_description', 'audio')} combined with {visual_content_analysis.get('content_type', 'visual content')}.",
        "all_hooks_identified": {
            "text_hooks": ["Text elements from video"],
            "visual_hooks": ["Visual hooks identified"],
            "verbal_hooks": ["Verbal content if present"],
            "psychological_hooks": ["Psychological triggers used"]
        },
        "replication_formula": {
            "formula_name": "The Success Formula",
            "structure": "0-3s: Hook, 3-7s: Value, 7-15s: Payoff",
            "scenarios_for_same_niche": ["Scenario 1", "Scenario 2"],
            "why_it_works": "Creates curiosity and delivers satisfaction",
            "text_template": "Template for text overlays",
            "visual_requirements": "Visual elements needed"
        },
        
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
        "improvements": f"To improve: {'Refine' if performance_level == 'viral' else 'Strengthen'} the opening hook, enhance satisfaction points, optimize for {platform}",
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
        'view_count': form_data.get('view_count', ''),
        'platform': form_data.get('platform', 'tiktok'),
        'target_duration': form_data.get('target_duration', '30'),
        'goal': form_data.get('goal', 'follower_growth'),
        'tone': form_data.get('tone', 'confident, friendly'),
        'audience': form_data.get('audience', 'creators and small business owners'),
        'strategy': form_data.get('strategy', 'smart'),
        'frames_per_minute': int(form_data.get('frames_per_minute', 24)),
        'cap': int(form_data.get('cap', 60)),
        'scene_threshold': float(form_data.get('scene_threshold', 0.24)),
        'niche': form_data.get('niche', 'general'),  # For pattern matching

        # New conversational fields
        'what_this_video_is': gpt_result.get('what_this_video_is', ''),
        'why_it_performed': gpt_result.get('why_it_performed', ''),
        'all_hooks_identified': gpt_result.get('all_hooks_identified', {}),
        'replication_formula': gpt_result.get('replication_formula', {}),
        'improvements': gpt_result.get('improvements', gpt_result.get('improvement_opportunities', '')),
        
        # Frame and file data
        'frames_count': len(frame_paths) if frame_paths else 0,
        'frame_gallery': gallery_data_urls if gallery_data_urls else [],
        'frames_dir': frames_dir if frames_dir else "",
        'frame_paths': frame_paths if frame_paths else [],
        'video_thumbnail': gallery_data_urls[0] if gallery_data_urls else None,  # First frame as thumbnail
        
        # Knowledge data
        'knowledge_citations': knowledge_citations if knowledge_citations else [],
        'knowledge_context': knowledge_context if knowledge_context else "",
        
        # Core analysis results
        'analysis': gpt_result.get('analysis', 'Analysis not available'),
        'hooks': gpt_result.get('hooks', []),
        'scores': gpt_result.get('scores', {}),
        'strengths': gpt_result.get('strengths', ''),
        'improvement_areas': gpt_result.get('improvement_areas', ''),
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

        # Raw text fallback support
        'is_raw_text_fallback': gpt_result.get('is_raw_text_fallback', False),
        'raw_analysis_text': gpt_result.get('raw_analysis_text', ''),
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
    # Track analysis ID for this request (used for updating status)
    current_analysis_id = None

    try:
        start_time = _time.time()

        # Get form data with improved view count handling
        form_data = {
            'tiktok_url': request.form.get("tiktok_url", "").strip(),
            'view_count': request.form.get("view_count", "").strip(),  # Capture view count
            'creator_note': request.form.get("creator_note", "").strip(),
            'strategy': request.form.get("strategy", "smart").strip(),
            'frames_per_minute': request.form.get("frames_per_minute", "24"),
            'cap': request.form.get("cap", "40"),  # Reduced from 60 to 40 for faster analysis
            'scene_threshold': request.form.get("scene_threshold", "0.24"),
            'platform': request.form.get("platform", "tiktok").strip(),
            'target_duration': request.form.get("target_duration", "30").strip(),
            'goal': request.form.get("goal", "follower_growth").strip(),
            'tone': request.form.get("tone", "confident, friendly").strip(),
            'audience': request.form.get("audience", "creators and small business owners").strip(),
            'analysis_depth': request.form.get("analysis_depth", "standard").strip(),  # Get analysis depth selection
        }

        # CHECK FOR IN-PROGRESS ANALYSIS (only for authenticated users)
        if current_user and current_user.is_authenticated:
            processing_analysis = get_user_processing_analysis(current_user.id)
            if processing_analysis:
                # User already has an analysis in progress - redirect to processing page
                print(f"[INFO] User {current_user.id} has analysis {processing_analysis.id} in progress")
                return render_template("processing.html",
                    analysis_id=processing_analysis.id,
                    video_url=processing_analysis.video_url
                )

            # Create a new processing analysis record
            processing_record = create_processing_analysis(current_user.id, form_data['tiktok_url'])
            if processing_record:
                current_analysis_id = processing_record.id

        # 1. CHECK CACHE FIRST (unless force_refresh is requested)
        force_refresh = request.form.get('force_refresh', 'false').lower() == 'true'
        if not force_refresh:
            cached_result = cache.get_cached_analysis(form_data['tiktok_url'])
            if cached_result:
                print(f"[CACHE HIT] Returning cached analysis")

                # Generate cache key for PDF even for cached results
                try:
                    cache_key = hashlib.md5(
                        f"{form_data['tiktok_url']}{_time.time()}".encode()
                    ).hexdigest()[:16]

                    metadata = cached_result.get('metadata', {})
                    video_title = metadata.get('title', 'analysis') if metadata else 'analysis'

                    pdf_cache[cache_key] = {
                        'template_vars': cached_result,
                        'video_title': video_title,
                        'timestamp': _time.time()
                    }

                    cached_result['pdf_cache_key'] = cache_key
                    print(f"[PDF CACHE] Cached result stored for PDF with key: {cache_key}")
                except Exception as e:
                    print(f"[WARNING] Failed to cache PDF data for cached result: {e}")

                return render_template("results.html", **cached_result)

        # 2. EXTRACT METADATA AUTOMATICALLY (includes saves!)
        print("[INFO] Auto-extracting video metadata with save counts...")
        metadata = extract_video_metadata(form_data['tiktok_url'])

        # Initialize view_count and performance_level from metadata
        view_count = None
        performance_level = 'unknown'

        if metadata.get('view_count') and metadata['view_count'] > 0:
            # Use auto-detected metadata
            view_count_raw = metadata['view_count']
            if view_count_raw >= 1000000:
                view_count = f"{view_count_raw/1000000:.1f}M"
                performance_level = 'viral'
            elif view_count_raw >= 1000:
                view_count = f"{view_count_raw/1000:.0f}k"
                performance_level = metadata.get('performance_level', 'moderate')
            else:
                view_count = f"{view_count_raw} views"
                performance_level = metadata.get('performance_level', 'low')

            print(f"[AUTO-DETECTED] Uploader: {metadata.get('uploader', 'Unknown')}")
            if metadata.get('track'):
                print(f"[AUTO-DETECTED] Track: {metadata.get('track')} by {metadata.get('artist', 'Unknown')}")
            print(f"[AUTO-DETECTED] Views: {view_count}")
            print(f"[AUTO-DETECTED] Likes: {metadata.get('like_count', 0):,}")
            print(f"[AUTO-DETECTED] Reposts: {metadata.get('repost_count', 0):,} ({metadata.get('engagement_metrics', {}).get('repost_rate', 0)}%)")
            print(f"[AUTO-DETECTED] Engagement: {metadata.get('engagement_metrics', {}).get('total_engagement_rate', 0)}%")

            # Add repost insights to creator note (high reposts = viral potential)
            repost_rate = metadata.get('engagement_metrics', {}).get('repost_rate', 0)
            if repost_rate > 2.0:
                form_data['creator_note'] += f" | HIGH REPOST RATE: {repost_rate:.1f}% - Strong shareability"
        else:
            # Fallback to manual parsing if auto-extraction fails
            print("[WARNING] Auto-extraction failed, trying manual parsing")

        # If auto-detection failed, try manual parsing as fallback
        view_count_input = form_data.get('view_count', '') or form_data.get('creator_note', '')

        if not view_count and view_count_input:
            # Extract numbers with units (fixed regex to handle commas)
            import re
            
            # First, clean the input and look for patterns
            clean_input = view_count_input.lower().replace(',', '').strip()
            
            # Updated regex to handle various formats
            patterns = re.findall(r'(\d+\.?\d*)\s*(k|m|thousand|million|views)?', clean_input)
            
            if patterns:
                for pattern in patterns:
                    try:
                        number = float(pattern[0])
                        unit = pattern[1] if len(pattern) > 1 else ''
                        
                        print(f"[DEBUG] Parsed: number={number}, unit='{unit}' from input '{view_count_input}'")
                        
                        if unit in ['k', 'thousand']:
                            view_count = f"{number}k"
                            if number >= 500:
                                performance_level = 'good'
                            elif number >= 100:
                                performance_level = 'moderate'
                            else:
                                performance_level = 'low'
                            break
                        elif unit in ['m', 'million']:
                            view_count = f"{number}M"
                            performance_level = 'viral'
                            break
                        else:
                            # Plain number - handle both with and without 'views'
                            if number >= 1000000:
                                view_count = f"{number/1000000:.1f}M"
                                performance_level = 'viral'
                            elif number >= 1000:
                                view_count = f"{number/1000:.0f}k"
                                if number >= 500000:
                                    performance_level = 'good'
                                elif number >= 100000:
                                    performance_level = 'moderate'
                                else:
                                    performance_level = 'low'
                            else:
                                view_count = f"{int(number)} views"
                                performance_level = 'low'
                            break
                    except ValueError:
                        continue
            
        print(f"[INFO] Final parsed view count: {view_count} (Performance: {performance_level})")
        
        # Validate numeric parameters
        try:
            frames_per_minute = int(form_data['frames_per_minute'])
            cap = int(form_data['cap'])
            scene_threshold = float(form_data['scene_threshold'])
        except ValueError as e:
            print(f"[ERROR] Invalid numeric parameter: {e}")
            return "Error: Invalid numeric parameters provided", 400

        # Apply Results Speed tier settings
        results_speed = form_data.get('results_speed', 'standard')
        if results_speed == 'ultra_fast':
            cap = 1  # Only first frame for hook analysis
            scene_threshold = 0.9  # Very selective
            print("[SPEED] Ultra Fast mode: 1 frame only")
        elif results_speed == 'fast':
            cap = 20
            scene_threshold = 0.45  # More selective
            print("[SPEED] Fast mode: 20 frames max")
        elif results_speed == 'thorough':
            cap = 70
            scene_threshold = 0.24  # More sensitive motion detection
            print("[SPEED] Thorough mode: 70 frames with enhanced motion detection")
        else:  # standard
            cap = 40
            scene_threshold = 0.35
            print("[SPEED] Standard mode: 40 frames")

        tiktok_url = form_data['tiktok_url']
        if not tiktok_url:
            return "Error: TikTok URL is required", 400

        print(f"[INFO] Processing: {tiktok_url}")
        print(f"[INFO] Creator note: {form_data['creator_note']}")
        print(f"[INFO] Strategy: {form_data['strategy']}, Goal: {form_data['goal']}")
        print(f"[INFO] Results Speed: {results_speed} (cap={cap}, threshold={scene_threshold})")

        # 3. VIDEO EXTRACTION (sequential is more reliable than parallel for video processing)
        print("[INFO] Extracting audio and frames...")
        audio_path, frames_dir, frame_paths = enhanced_extract_audio_and_frames(
            tiktok_url,
            strategy=form_data['strategy'],
            frames_per_minute=frames_per_minute,
            cap=cap,
            scene_threshold=scene_threshold,
        )
        print(f"[SUCCESS] Extracted {len(frame_paths)} frames")

        # 4. PARALLEL ANALYSIS of independent components (REAL speedup!)
        print("[INFO] Starting parallel analysis of audio and frames...")
        from concurrent.futures import ThreadPoolExecutor, as_completed

        analysis_results = {
            'transcript': '',
            'frames_summaries': '',
            'gallery_urls': [],
            'audio_analysis': {'viral_sound': {'is_viral': False}}
        }

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}

            # Submit independent tasks that use different APIs/services
            futures['transcription'] = executor.submit(transcribe_audio, audio_path)
            futures['audio_analysis'] = executor.submit(enhanced_audio_analysis, audio_path)
            # Frame analysis needs to wait for transcription for context, so we'll do it after

            # Wait for transcription first (needed for frame analysis context)
            try:
                basic_transcript = futures['transcription'].result(timeout=60)
                analysis_results['transcript'] = basic_transcript
                print(f"[PARALLEL] Transcription complete: {len(basic_transcript)} chars")
            except Exception as e:
                print(f"[WARNING] Transcription failed: {e}")
                basic_transcript = ""

            # Now start frame analysis with transcript context
            futures['frames'] = executor.submit(analyze_frames_batch, frame_paths, basic_transcript)

            # Collect remaining results
            for name, future in futures.items():
                if name == 'transcription':  # Already handled
                    continue
                try:
                    if name == 'frames':
                        frames_summaries_text, gallery_data_urls = future.result(timeout=480)
                        analysis_results['frames_summaries'] = frames_summaries_text
                        analysis_results['gallery_urls'] = gallery_data_urls
                        print(f"[PARALLEL] Frame analysis complete: {len(frames_summaries_text)} chars")
                    elif name == 'audio_analysis':
                        audio_analysis = future.result(timeout=45)  # Increased for ACRCloud API
                        analysis_results['audio_analysis'] = audio_analysis
                        print(f"[PARALLEL] Audio analysis complete")

                        if audio_analysis.get('viral_sound', {}).get('is_viral'):
                            sound_info = audio_analysis['viral_sound']
                            print(f"[VIRAL SOUND] Detected: {sound_info.get('sound_name')} by {sound_info.get('artist')}")
                            form_data['creator_note'] += f" | Viral Sound: {sound_info.get('sound_name')} by {sound_info.get('artist')}"
                except Exception as e:
                    print(f"[WARNING] {name} failed: {type(e).__name__}: {e}")
                    if name == 'frames':
                        analysis_results['frames_summaries'] = ""
                        analysis_results['gallery_urls'] = []
                    elif name == 'audio_analysis':
                        # Provide safe fallback for audio analysis (timeout or ACRCloud slow)
                        print("[FALLBACK] Using default audio analysis (ACRCloud timeout is normal for slow networks)")
                        analysis_results['audio_analysis'] = {
                            'type': 'unknown',
                            'audio_description': 'audio elements',
                            'viral_audio_check': False,
                            'viral_sound': {'is_viral': False}
                        }

        # Extract results
        basic_transcript = analysis_results['transcript']
        frames_summaries_text = analysis_results['frames_summaries']
        gallery_data_urls = analysis_results['gallery_urls']
        audio_analysis = analysis_results['audio_analysis']

        # Enhanced audio transcription WITH visual context
        try:
            transcript_data = enhanced_transcribe_audio_with_context(audio_path, frames_summaries_text)
            print(f"[INFO] Audio interpretation: {transcript_data.get('audio_context', {}).get('audio_description', 'unknown')}")
            print(f"[INFO] Transcript quality: {transcript_data.get('quality', 'unknown')}")
            if transcript_data.get('audio_context', {}).get('likely_sound_source'):
                print(f"[INFO] Likely sound source: {transcript_data['audio_context']['likely_sound_source']}")
        except Exception as e:
            print(f"[ERROR] Enhanced transcription error: {e}")
            # Fallback to basic transcript if enhanced analysis fails
            transcript_data = {
                'transcript': basic_transcript,
                'quality': 'basic',
                'quality_reason': str(e),
                'is_reliable': bool(basic_transcript),
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

        # Fetch and analyze comments (optional, don't fail if unavailable)
        comment_insights = None
        try:
            print("[COMMENTS] Attempting to fetch comments...")
            from comment_fetcher import CommentFetcher
            from comment_analyzer import CommentAnalyzer

            fetcher = CommentFetcher()
            comments = fetcher.fetch_comments(
                video_url=form_data['tiktok_url'],
                max_comments=100,
                platform=form_data['platform']
            )

            if comments:
                print(f"[COMMENTS] Analyzing {len(comments)} comments...")
                analyzer = CommentAnalyzer()
                comment_analysis = analyzer.analyze_comments(
                    comments=comments,
                    video_transcript=transcript_data.get('transcript', ''),
                    frame_analysis=frames_summaries_text
                )

                # Extract key insights for main analysis
                comment_insights = {
                    'total_comments': comment_analysis.get('total_comments', 0),
                    'categorization': comment_analysis.get('categorization', {}),
                    'consensus_patterns': comment_analysis.get('consensus_patterns', []),
                    'ai_insights': comment_analysis.get('ai_insights', ''),
                    'timestamp_mentions': comment_analysis.get('timestamp_mentions', []),
                    'emotion_analysis': comment_analysis.get('emotion_analysis', {})
                }

                print(f"[COMMENTS] Analysis complete: {comment_insights['total_comments']} comments")
                print(f"[COMMENTS] Consensus patterns found: {len(comment_insights['consensus_patterns'])}")
            else:
                print("[COMMENTS] No comments fetched (API may be unavailable)")

        except Exception as e:
            print(f"[COMMENTS] Comment fetching failed (non-critical): {e}")
            # Don't fail the whole analysis if comments fail
            comment_insights = None

        # Check if user only wants extraction (no AI analysis)
        if form_data.get('analysis_depth') == 'extraction_only':
            print("[INFO] Extraction-only mode: Skipping AI analysis")
            # Create minimal result with just extraction data
            gpt_result = {
                'extraction_only': True,
                'transcript_quality': transcript_data,
                'actual_view_count': view_count,
                'performance_level': performance_level,
                'metadata': metadata,
                'audio_analysis': audio_analysis,
            }
        else:
            # Run comprehensive analysis with metadata and audio insights
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
                    knowledge_context,
                    view_count,
                    performance_level,
                    metadata=metadata,  # Pass metadata with saves
                    audio_insights=audio_analysis,  # Pass audio analysis
                    analysis_depth=form_data.get('analysis_depth', 'standard'),  # Pass analysis depth
                    comment_insights=comment_insights  # Pass comment analysis
                )

                # Add transcript quality info, view data, and new metrics
                gpt_result['transcript_quality'] = transcript_data
                gpt_result['actual_view_count'] = view_count
                gpt_result['performance_level'] = performance_level
                gpt_result['metadata'] = metadata
                gpt_result['audio_analysis'] = audio_analysis
                gpt_result['save_insights'] = analyze_save_metrics(metadata) if metadata else {}

                print("[SUCCESS] Analysis complete")
                print(f"[INFO] Content type: {gpt_result.get('content_type_detected', 'unknown')}")
                print(f"[INFO] Audio type: {gpt_result.get('audio_type_detected', 'unknown')}")
                print(f"[INFO] Performance level: {gpt_result.get('performance_level', 'unknown')}")

            except Exception as e:
                print(f"[ERROR] Analysis error: {e}")
                import traceback
                traceback.print_exc()

                # IMPORTANT: Try to salvage the analysis even if JSON parsing failed
                # Claude might have returned valid analysis text that we can still use
                print("[RECOVERY] Attempting to salvage analysis from error...")

                # Use comprehensive fallback - ensures user ALWAYS gets results
                audio_context = transcript_data.get('audio_context', {})
                visual_analysis = create_visual_content_description(frames_summaries_text, audio_context)

                has_speech = audio_context.get('has_meaningful_speech', False)

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

                # Add a warning flag so template can show this was a fallback
                gpt_result['is_fallback'] = True
                gpt_result['fallback_reason'] = str(e)[:200]
                print("[RECOVERY] Fallback analysis generated successfully")

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

        # 5. TRACK PERFORMANCE for continuous improvement
        try:
            tracker.record_prediction(form_data['tiktok_url'], metadata, gpt_result)
            print("[TRACKER] Performance data recorded")
        except Exception as e:
            print(f"[WARNING] Failed to track performance: {e}")

        # 6. CACHE RESULTS for 24-hour reuse
        try:
            cache.save_analysis(form_data['tiktok_url'], 'full', template_vars)
            print("[CACHE] Analysis cached for future requests")
        except Exception as e:
            print(f"[WARNING] Failed to cache results: {e}")

        elapsed = _time.time() - start_time
        print(f"[SUCCESS] Analysis completed in {elapsed:.1f}s")

        # 7. STORE DATA FOR PDF GENERATION
        try:
            # Generate unique cache key for PDF
            cache_key = hashlib.md5(
                f"{form_data['tiktok_url']}{_time.time()}".encode()
            ).hexdigest()[:16]

            # Get video title for PDF filename
            video_title = metadata.get('title', 'analysis') if metadata else 'analysis'

            # Store in pdf_cache
            pdf_cache[cache_key] = {
                'template_vars': template_vars,
                'video_title': video_title,
                'timestamp': _time.time()
            }

            # Add cache_key to template_vars so it can be used in template
            template_vars['pdf_cache_key'] = cache_key

            print(f"[PDF CACHE] Stored data with key: {cache_key}")
        except Exception as e:
            print(f"[WARNING] Failed to cache PDF data: {e}")
            cache_key = None
            # Continue anyway - PDF generation is optional

        # 8. SAVE TO USER'S HISTORY (if logged in)
        if current_user.is_authenticated:
            try:
                # Get thumbnail from first frame in gallery
                thumbnail_url = None
                if template_vars.get('frame_gallery'):
                    thumbnail_url = template_vars['frame_gallery'][0]

                if current_analysis_id:
                    # Update the existing processing analysis to completed
                    complete_analysis(
                        analysis_id=current_analysis_id,
                        video_title=video_title,
                        thumbnail_url=thumbnail_url,
                        template_vars=template_vars,
                        pdf_cache_key=cache_key
                    )
                else:
                    # Legacy: Create new analysis record
                    save_analysis_to_db(
                        user_id=current_user.id,
                        video_url=form_data['tiktok_url'],
                        video_title=video_title,
                        thumbnail_url=thumbnail_url,
                        template_vars=template_vars,
                        pdf_cache_key=cache_key
                    )
            except Exception as e:
                print(f"[WARNING] Failed to save to user history: {e}")
                # Continue anyway - this shouldn't block the response

        print("[INFO] Rendering results template")
        return render_template("results.html", **template_vars)

    except ValueError as e:
        # Mark analysis as failed if we had one in progress
        if current_analysis_id:
            fail_analysis(current_analysis_id, str(e))

        # User-friendly errors (video access issues, etc.)
        print(f"[USER ERROR] {str(e)}")
        error_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Video Access Error</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    padding: 20px;
                }}
                .error-container {{
                    background: white;
                    border-radius: 16px;
                    padding: 40px;
                    max-width: 600px;
                    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                }}
                h1 {{ color: #e74c3c; margin-bottom: 20px; }}
                pre {{
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 8px;
                    border-left: 4px solid #e74c3c;
                    white-space: pre-wrap;
                    line-height: 1.6;
                }}
                .back-link {{
                    display: inline-block;
                    background: #667eea;
                    color: white;
                    text-decoration: none;
                    padding: 12px 24px;
                    border-radius: 8px;
                    margin-top: 20px;
                }}
                .back-link:hover {{ background: #5568d3; }}
            </style>
        </head>
        <body>
            <div class="error-container">
                <h1>⚠️ Video Access Error</h1>
                <pre>{str(e)}</pre>
                <a href="/" class="back-link">← Try Another Video</a>
            </div>
        </body>
        </html>
        """
        return error_html, 400

    except Exception as e:
        # Mark analysis as failed if we had one in progress
        if current_analysis_id:
            fail_analysis(current_analysis_id, str(e))

        print(f"[ERROR] Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Unexpected error: {str(e)}", 500


@app.route("/download_pdf/<cache_key>", methods=["GET"])
def download_pdf(cache_key):
    """
    Generate and download PDF from cached analysis results.
    Uses server-side rendering with Playwright for reliable PDF generation.
    """
    print(f"[PDF] Download request for cache_key: {cache_key}")

    # Retrieve cached data (checks both memory and disk cache)
    cached_data = pdf_cache.get(cache_key)
    if cached_data is None:
        print(f"[PDF ERROR] Cache key not found: {cache_key}")
        return "PDF data not found. The analysis may have expired. Please re-run the analysis.", 404

    try:
        template_vars = cached_data['template_vars']
        video_title = cached_data.get('video_title', 'analysis')

        print(f"[PDF] Rendering HTML template for: {video_title}")

        # Render the HTML template with all the data
        html_content = render_template("results.html", **template_vars)

        # Generate filename with timestamp
        now = datetime.now()
        date_str = now.strftime('%Y-%m-%d')
        time_str = now.strftime('%H%M')

        # Sanitize video title for filename
        safe_title = re.sub(r'[^\w\s-]', '', video_title)[:50]
        safe_title = re.sub(r'[-\s]+', '_', safe_title)
        filename = f"tiktok_{safe_title}_{date_str}_{time_str}.pdf"

        print(f"[PDF] Generating PDF: {filename}")

        # Generate PDF using playwright
        pdf_bytes = generate_pdf_sync(html_content)

        # Create response
        response = make_response(pdf_bytes)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'

        print(f"[PDF SUCCESS] Generated {len(pdf_bytes)} bytes")
        return response

    except Exception as e:
        print(f"[PDF ERROR] Failed to generate PDF: {e}")
        import traceback
        traceback.print_exc()
        return f"Failed to generate PDF: {str(e)}", 500


@app.route("/clear_cache", methods=["POST"])
def clear_cache():
    """
    Clear all cached analyses to force fresh analysis on next request.
    Useful when fixes are deployed that affect analysis quality.
    """
    try:
        cache.clear_cache()
        return {"status": "success", "message": "Cache cleared successfully"}, 200
    except Exception as e:
        print(f"[CACHE ERROR] Failed to clear cache: {e}")
        return {"status": "error", "message": str(e)}, 500


@app.route("/preview_pattern", methods=["POST"])
@login_required
def preview_pattern():
    """
    Preview what will be stored before submitting pattern for learning.
    Shows pattern, context, constraints, and applicability.
    """
    try:
        # Admin check
        if not current_user.is_authenticated or current_user.email.lower() != 'christina@superlunardesign.com':
            return {
                "status": "error",
                "message": "This feature is only available for admin users"
            }, 403

        data = request.json

        # Get data
        cache_key = data.get('cache_key')
        video_url = data.get('video_url')
        curator_notes = data.get('notes', '').strip()
        niche = data.get('niche', 'general')
        platform = data.get('platform', 'tiktok')
        audience = data.get('audience', '')
        video_type = data.get('video_type', '')

        print(f"[PREVIEW] cache_key received: '{cache_key}'")

        if not curator_notes:
            return {
                "status": "error",
                "message": "Please add notes about what patterns to learn from this video"
            }, 400

        # Get cached analysis
        if not cache_key:
            return {
                "status": "error",
                "message": "No cache key provided. Please re-analyze the video first."
            }, 400

        # Try to get from cache, or fall back to reconstructing from saved analysis
        if cache_key in pdf_cache:
            cached_data = pdf_cache[cache_key]
            template_vars = cached_data.get('template_vars', {})
        else:
            # Cache expired - try to get from database
            print(f"[PREVIEW] Cache expired for key {cache_key}, attempting to load from DB")

            # Find the analysis by video_url for current user
            from models import Analysis
            analysis = Analysis.query.filter_by(
                video_url=video_url,
                user_id=current_user.id
            ).order_by(Analysis.created_at.desc()).first()

            if not analysis or not analysis.analysis_data:
                return {
                    "status": "error",
                    "message": "Analysis cache expired and no saved data found. Please re-analyze the video."
                }, 404

            template_vars = analysis.analysis_data
            print(f"[PREVIEW] Loaded analysis from DB: {analysis.id}")

        analysis_text = template_vars.get('analysis', '') or \
                       template_vars.get('what_this_video_is', '') + '\n' + \
                       template_vars.get('why_it_performed', '')

        # Get metrics
        metrics = {
            'views': data.get('views', 0),
            'engagement_rate': data.get('engagement_rate', 0),
            'watch_time': data.get('watch_time', '')
        }

        # Initialize context-aware pattern store
        from success_patterns_improved import ContextAwarePatternStore
        pattern_store = ContextAwarePatternStore()

        # Generate preview
        preview = pattern_store.preview_pattern(
            analysis_text=analysis_text,
            video_url=video_url,
            metrics=metrics,
            curator_notes=curator_notes,
            niche=niche,
            platform=platform,
            audience=audience,
            video_type=video_type
        )

        print(f"[PREVIEW] Generated pattern preview for {current_user.email}")

        return {
            "status": "success",
            "preview": preview
        }, 200

    except Exception as e:
        print(f"[ERROR] Failed to generate preview: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}, 500


@app.route("/submit_for_learning", methods=["POST"])
@login_required
def submit_for_learning():
    """
    Submit a video for learning (Admin only: christina@superlunardesign.com).
    Stores format patterns with context, constraints, and applicability rules.
    """
    try:
        # Admin check - must be logged in as admin
        if not current_user.is_authenticated or current_user.email.lower() != 'christina@superlunardesign.com':
            return {
                "status": "error",
                "message": "This feature is only available for admin users"
            }, 403

        data = request.json

        # Get data
        cache_key = data.get('cache_key')
        video_url = data.get('video_url')
        curator_notes = data.get('notes', '').strip()
        pattern_data = data.get('pattern_data')  # From preview

        if not curator_notes:
            return {
                "status": "error",
                "message": "Please add notes about what patterns to learn from this video"
            }, 400

        # Get cached analysis
        if cache_key not in pdf_cache:
            return {"status": "error", "message": "Analysis not found"}, 404

        cached_data = pdf_cache[cache_key]
        template_vars = cached_data.get('template_vars', {})
        analysis_text = template_vars.get('analysis', '')

        # Get metrics
        metrics = {
            'views': data.get('views', 0),
            'engagement_rate': data.get('engagement_rate', 0),
            'watch_time': data.get('watch_time', ''),
            'curator_notes': curator_notes,
            'submitted_by': current_user.email,
            'submission_date': datetime.now().isoformat()
        }

        # Initialize context-aware pattern store
        from success_patterns_improved import ContextAwarePatternStore
        pattern_store = ContextAwarePatternStore()

        # If pattern_data provided (from preview), use it; otherwise generate new
        if not pattern_data:
            # Generate pattern if not previewed
            preview = pattern_store.preview_pattern(
                analysis_text=analysis_text,
                video_url=video_url,
                metrics=metrics,
                curator_notes=curator_notes,
                niche=data.get('niche', 'general'),
                platform=data.get('platform', 'tiktok'),
                audience=data.get('audience', ''),
                video_type=data.get('video_type', '')
            )
            pattern_data = preview['preview']

        # Store pattern with context
        pattern_store.store_pattern(
            pattern_data=pattern_data,
            video_url=video_url,
            metrics=metrics,
            curator_notes=curator_notes
        )

        print(f"[LEARNING] Context-aware pattern submitted by {current_user.email}")
        print(f"  Pattern: {pattern_data.get('pattern_summary', 'N/A')}")
        print(f"  Context: {pattern_data.get('context', {}).get('niche', 'N/A')}")

        return {
            "status": "success",
            "message": "Video submitted for learning! Pattern stored with context, constraints, and applicability rules."
        }, 200

    except Exception as e:
        print(f"[ERROR] Failed to submit for learning: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}, 500


# ==============================
# USER HISTORY AND ANALYSIS MANAGEMENT
# ==============================

@app.route("/history")
@login_required
def history():
    """Display user's analysis history."""
    page = request.args.get('page', 1, type=int)
    per_page = 12

    # Check for any in-progress analysis
    processing = Analysis.query.filter_by(
        user_id=current_user.id,
        status='processing'
    ).first()

    # Show completed analyses (and legacy ones without status)
    pagination = Analysis.query.filter_by(user_id=current_user.id)\
        .filter(Analysis.status.in_(['completed', None]))\
        .order_by(Analysis.created_at.desc())\
        .paginate(page=page, per_page=per_page, error_out=False)

    return render_template(
        'history.html',
        analyses=pagination.items,
        pagination=pagination,
        processing=processing
    )


@app.route("/analysis/<int:analysis_id>")
@login_required
def view_analysis(analysis_id):
    """View a specific past analysis."""
    analysis = Analysis.query.filter_by(id=analysis_id, user_id=current_user.id).first()

    if not analysis:
        flash('Analysis not found.', 'error')
        return redirect(url_for('history'))

    template_vars = None

    # First, try to get full data from PDF cache (has complete template_vars)
    if analysis.pdf_cache_key and analysis.pdf_cache_key in pdf_cache:
        cached_data = pdf_cache[analysis.pdf_cache_key]
        template_vars = cached_data.get('template_vars', {}).copy()
        print(f"[VIEW] Using full cached data for analysis {analysis_id}")

    # Fall back to lightweight database data
    if not template_vars:
        template_vars = analysis.analysis_data or {}
        print(f"[VIEW] Using lightweight DB data for analysis {analysis_id}")

    # Add context from database record
    template_vars['video_url'] = analysis.video_url
    template_vars['video_title'] = analysis.video_title
    template_vars['thumbnail_url'] = analysis.thumbnail_url
    template_vars['created_at'] = analysis.created_at
    template_vars['analysis_id'] = analysis.id
    template_vars['pdf_cache_key'] = analysis.pdf_cache_key
    template_vars['is_saved_analysis'] = True  # Flag for template to know this is saved

    # Ensure all required fields have safe defaults for results.html
    template_vars.setdefault('tiktok_url', analysis.video_url)
    template_vars.setdefault('platform', 'tiktok')
    template_vars.setdefault('target_duration', '30')
    template_vars.setdefault('goal', 'engagement')
    template_vars.setdefault('niche', 'general')
    template_vars.setdefault('scores', {})
    template_vars.setdefault('metadata', {})
    template_vars.setdefault('transcript_quality', {})
    template_vars.setdefault('replication_formula', {})
    template_vars.setdefault('exact_hook_breakdown', {})
    template_vars.setdefault('all_hooks_identified', {})

    # Try to use full results template, fall back to summary if it fails
    try:
        return render_template("results.html", **template_vars)
    except Exception as e:
        print(f"[WARNING] Could not render results.html for saved analysis: {e}")
        import traceback
        traceback.print_exc()
        return render_template("analysis_summary.html", **template_vars)


@app.route("/analysis/<int:analysis_id>", methods=["DELETE"])
@login_required
def delete_analysis(analysis_id):
    """Delete a specific analysis."""
    analysis = Analysis.query.filter_by(id=analysis_id, user_id=current_user.id).first()

    if not analysis:
        return jsonify({'error': 'Analysis not found'}), 404

    try:
        db.session.delete(analysis)
        db.session.commit()
        return jsonify({'status': 'success'}), 200
    except Exception as e:
        db.session.rollback()
        print(f"[ERROR] Failed to delete analysis: {e}")
        return jsonify({'error': str(e)}), 500


@app.route("/analysis/<int:analysis_id>/status")
@login_required
def analysis_status(analysis_id):
    """Check the status of an analysis."""
    analysis = Analysis.query.filter_by(id=analysis_id, user_id=current_user.id).first()

    if not analysis:
        return jsonify({'error': 'Analysis not found'}), 404

    return jsonify({
        'status': analysis.status or 'completed',  # Default to completed for older records
        'video_title': analysis.video_title,
        'created_at': analysis.created_at.strftime('%B %d, %Y at %I:%M %p') if analysis.created_at else None
    })


@app.route("/check_existing", methods=["POST"])
@login_required
def check_existing():
    """Check if user has already analyzed this video URL."""
    data = request.get_json()
    video_url = data.get('url', '').strip()

    if not video_url:
        return jsonify({'exists': False})

    # Normalize URL for comparison (remove query params, etc.)
    normalized_url = normalize_video_url(video_url)

    # Check for existing analysis
    existing = Analysis.query.filter_by(
        user_id=current_user.id,
        video_url=normalized_url
    ).order_by(Analysis.created_at.desc()).first()

    if existing:
        return jsonify({
            'exists': True,
            'analysis_id': existing.id,
            'video_title': existing.video_title,
            'created_at': existing.created_at.strftime('%B %d, %Y')
        })

    return jsonify({'exists': False})


def normalize_video_url(url):
    """Normalize TikTok URL for comparison."""
    # Extract video ID from various TikTok URL formats
    import re

    # Remove query parameters
    url = url.split('?')[0]

    # Handle various TikTok URL formats
    patterns = [
        r'tiktok\.com/@[\w.]+/video/(\d+)',
        r'tiktok\.com/t/(\w+)',
        r'vm\.tiktok\.com/(\w+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return url  # Return cleaned URL

    return url


def get_user_processing_analysis(user_id):
    """Check if user has an analysis currently in progress."""
    try:
        # Find any analysis with status='processing' for this user
        processing = Analysis.query.filter_by(
            user_id=user_id,
            status='processing'
        ).first()
        return processing
    except Exception as e:
        print(f"[DB ERROR] Failed to check processing analysis: {e}")
        return None


def create_processing_analysis(user_id, video_url):
    """Create a new analysis record with status='processing'."""
    try:
        normalized_url = normalize_video_url(video_url)

        analysis = Analysis(
            user_id=user_id,
            video_url=normalized_url,
            status='processing'
        )

        db.session.add(analysis)
        db.session.commit()

        print(f"[DB] Created processing analysis {analysis.id} for user {user_id}")
        return analysis
    except Exception as e:
        db.session.rollback()
        print(f"[DB ERROR] Failed to create processing analysis: {e}")
        return None


def complete_analysis(analysis_id, video_title, thumbnail_url, template_vars, pdf_cache_key):
    """Update a processing analysis to completed with results."""
    try:
        analysis = Analysis.query.get(analysis_id)
        if not analysis:
            print(f"[DB ERROR] Analysis {analysis_id} not found")
            return False

        # Create lightweight analysis data (exclude large base64 images)
        # Include all fields needed by results.html template
        lightweight_data = {
            # Basic info
            'video_title': template_vars.get('video_title'),
            'creator': template_vars.get('creator'),
            'platform': template_vars.get('platform', 'tiktok'),
            'target_duration': template_vars.get('target_duration'),
            'goal': template_vars.get('goal'),
            'niche': template_vars.get('niche', 'general'),

            # Metrics
            'view_count': template_vars.get('view_count'),
            'like_count': template_vars.get('like_count'),
            'comment_count': template_vars.get('comment_count'),
            'share_count': template_vars.get('share_count'),

            # Content
            'video_description': template_vars.get('video_description'),
            'hashtags': template_vars.get('hashtags'),
            'audio_type': template_vars.get('audio_type'),
            'music_info': template_vars.get('music_info'),

            # Analysis results
            'what_this_video_is': template_vars.get('what_this_video_is'),
            'why_it_performed': template_vars.get('why_it_performed'),
            'replication_formula': template_vars.get('replication_formula', {}),
            'improvements': template_vars.get('improvements'),
            'viral_mechanics': template_vars.get('viral_mechanics'),
            'performance_prediction': template_vars.get('performance_prediction'),
            'scores': template_vars.get('scores', {}),
            'exact_hook_breakdown': template_vars.get('exact_hook_breakdown', {}),
            'all_hooks_identified': template_vars.get('all_hooks_identified', {}),

            # Legacy fields
            'goal_analysis': template_vars.get('goal_analysis'),
            'overall_assessment': template_vars.get('overall_assessment'),
            'primary_strengths': template_vars.get('primary_strengths'),
            'areas_for_improvement': template_vars.get('areas_for_improvement'),
        }

        analysis.video_title = video_title
        analysis.thumbnail_url = thumbnail_url
        analysis.analysis_data = lightweight_data
        analysis.pdf_cache_key = pdf_cache_key
        analysis.status = 'completed'
        analysis.completed_at = datetime.utcnow()

        db.session.commit()

        print(f"[DB] Completed analysis {analysis_id}: {video_title}")
        return True
    except Exception as e:
        db.session.rollback()
        print(f"[DB ERROR] Failed to complete analysis: {e}")
        return False


def fail_analysis(analysis_id, error_message=None):
    """Mark an analysis as failed."""
    try:
        analysis = Analysis.query.get(analysis_id)
        if analysis:
            analysis.status = 'failed'
            analysis.analysis_data = {'error': error_message} if error_message else None
            db.session.commit()
            print(f"[DB] Marked analysis {analysis_id} as failed")
    except Exception as e:
        db.session.rollback()
        print(f"[DB ERROR] Failed to mark analysis as failed: {e}")


def save_analysis_to_db(user_id, video_url, video_title, thumbnail_url, template_vars, pdf_cache_key):
    """Save analysis results to database for the user (legacy function for backward compatibility)."""
    try:
        # Normalize URL
        normalized_url = normalize_video_url(video_url)

        # Create lightweight analysis data (exclude large base64 images)
        # The full data is preserved in pdf_cache for PDF downloads
        lightweight_data = {
            'video_title': template_vars.get('video_title'),
            'creator': template_vars.get('creator'),
            'goal_analysis': template_vars.get('goal_analysis'),
            'view_count': template_vars.get('view_count'),
            'like_count': template_vars.get('like_count'),
            'comment_count': template_vars.get('comment_count'),
            'share_count': template_vars.get('share_count'),
            'video_description': template_vars.get('video_description'),
            'hashtags': template_vars.get('hashtags'),
            'overall_assessment': template_vars.get('overall_assessment'),
            'primary_strengths': template_vars.get('primary_strengths'),
            'areas_for_improvement': template_vars.get('areas_for_improvement'),
            'audio_type': template_vars.get('audio_type'),
            'music_info': template_vars.get('music_info'),
            # Exclude: frame_gallery (base64 images ~200KB)
            # Exclude: frame_analyses (detailed per-frame data)
        }

        # Create analysis record
        analysis = Analysis(
            user_id=user_id,
            video_url=normalized_url,
            video_title=video_title,
            thumbnail_url=thumbnail_url,
            analysis_data=lightweight_data,
            pdf_cache_key=pdf_cache_key,
            status='completed',
            completed_at=datetime.utcnow()
        )

        db.session.add(analysis)
        db.session.commit()

        print(f"[DB] Saved analysis for user {user_id}: {video_title}")
        return analysis.id
    except Exception as e:
        db.session.rollback()
        print(f"[DB ERROR] Failed to save analysis: {e}")
        return None


if __name__ == "__main__":
    validate_dependencies()
    app.run(host="0.0.0.0", port=10000, debug=True)
