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


def analyze_transcript_quality(transcript_text):
    """Detect when transcript is likely misinterpreted sounds vs actual speech."""
    
    # Clean transcript
    words = transcript_text.lower().split()
    
    # Check for repetitive non-words
    word_counts = Counter(words)
    total_words = len(words)
    
    if total_words == 0:
        return "silent", "No audio detected"
    
    # Most common word analysis
    most_common_word, most_common_count = word_counts.most_common(1)[0]
    repetition_ratio = most_common_count / total_words
    
    # Patterns that suggest misinterpreted sounds
    sound_patterns = [
        'roar', 'tap', 'click', 'scratch', 'scrape', 'brush', 'rub',
        'swoosh', 'pop', 'snap', 'crackle', 'fizz', 'sizzle', 'hum',
        'buzz', 'beep', 'ding', 'ping', 'ring', 'whistle', 'blow'
    ]
    
    # Check for excessive repetition (>50% same word)
    if repetition_ratio > 0.5:
        return "likely_misinterpreted", f"Excessive repetition of '{most_common_word}' ({repetition_ratio:.1%})"
    
    # Check if transcript is mostly sound words
    sound_word_count = sum(1 for word in words if any(sound in word for sound in sound_patterns))
    sound_ratio = sound_word_count / total_words
    
    if sound_ratio > 0.7:
        return "likely_misinterpreted", f"High ratio of sound words ({sound_ratio:.1%})"
    
    # Check for very short repeated words (often artifacts)
    short_repeated = [word for word, count in word_counts.items() 
                     if len(word) <= 3 and count > total_words * 0.3]
    
    if short_repeated:
        return "likely_misinterpreted", f"Short repeated words: {short_repeated}"
    
    # Check for non-sensical patterns
    unique_words = len(word_counts)
    if total_words > 20 and unique_words < 5:
        return "likely_misinterpreted", f"Very low vocabulary diversity ({unique_words} unique words)"
    
    return "valid_speech", "Appears to be actual speech"


def generate_inferred_audio_description(frames_summaries_text, transcript_quality_info):
    """Generate audio description based on visual analysis when transcript is unreliable."""
    
    quality, reason = transcript_quality_info
    
    if quality == "silent":
        return "No audio - silent video"
    
    if quality == "likely_misinterpreted":
        # Analyze visual content to infer what sounds might be present
        frames_lower = frames_summaries_text.lower()
        
        # Map visual activities to likely sounds
        sound_mapping = {
            'drawing': 'sound of marker/pen on paper',
            'coloring': 'sound of marker/crayon on textured surface', 
            'painting': 'brush strokes and paint application',
            'writing': 'pen/pencil on paper',
            'typing': 'keyboard typing sounds',
            'cooking': 'chopping, sizzling, mixing sounds',
            'folding': 'paper/fabric rustling',
            'organizing': 'objects being moved and arranged',
            'makeup': 'brush application, product opening',
            'skincare': 'product application, gentle rubbing',
            'hair': 'brush strokes, styling tool sounds',
            'cleaning': 'wiping, scrubbing sounds',
            'crafting': 'cutting, gluing, material manipulation'
        }
        
        detected_sounds = []
        for activity, sound_desc in sound_mapping.items():
            if activity in frames_lower:
                detected_sounds.append(sound_desc)
        
        if detected_sounds:
            inferred_audio = f"Audio appears to be ambient sounds: {', '.join(detected_sounds[:2])}"
        else:
            inferred_audio = "Audio consists of ambient activity sounds"
        
        return f"{inferred_audio} (transcript misinterpreted as repeated words)"
    
    return None  # Use original transcript


def enhanced_transcribe_audio(audio_path):
    """Enhanced transcription with quality analysis."""
    
    # Your existing transcription code
    def _remote_transcribe():
        with open(audio_path, "rb") as f:
            return client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=f
            )
    
    try:
        tr = _api_retry(_remote_transcribe)
        transcript = tr.text
        
        # Analyze transcript quality
        quality_info = analyze_transcript_quality(transcript)
        
        return {
            'transcript': transcript,
            'quality': quality_info[0],
            'quality_reason': quality_info[1],
            'is_reliable': quality_info[0] == "valid_speech"
        }
        
    except Exception as e:
        print(f"[warn] Remote transcription failed: {e}")
        return {
            'transcript': "(transcription unavailable due to connection error)",
            'quality': "error",
            'quality_reason': str(e),
            'is_reliable': False
        }


def build_distributed_sampling_times(duration, change_times, max_frames):
    """
    Build sampling times that are distributed across the entire video duration,
    not just clustered at the beginning.
    """
    
    # Always include key anchor points distributed across video
    anchors = [
        0.0,  # Very start
        0.3,  # After initial hook
        duration * 0.15,  # Early section
        duration * 0.35,  # Early-mid 
        duration * 0.5,   # Middle
        duration * 0.65,  # Mid-late
        duration * 0.85,  # Near end
        max(0.0, duration - 1.0),  # Close to end
        max(0.0, duration - 0.3)   # Very end
    ]
    
    # Remove anchors that are too close to duration
    anchors = [a for a in anchors if a < duration - 0.05]
    
    # Settle offsets for natural moments after changes
    settle_offsets = [0.3, 0.6, 1.0]
    
    # Combine all potential timestamps
    all_timestamps = []
    
    # Add anchors with settle offsets
    for anchor in anchors:
        all_timestamps.append(anchor)
        for offset in settle_offsets:
            if anchor + offset < duration - 0.05:
                all_timestamps.append(anchor + offset)
    
    # Add change times with settle offsets
    for change_time in change_times:
        all_timestamps.append(change_time)
        for offset in settle_offsets:
            if change_time + offset < duration - 0.05:
                all_timestamps.append(change_time + offset)
    
    # Remove duplicates and sort
    all_timestamps = sorted(set(all_timestamps))
    
    if len(all_timestamps) <= max_frames:
        return all_timestamps
    
    # If we have too many, distribute them evenly across the video
    return distribute_frames_evenly(all_timestamps, duration, max_frames)


def distribute_frames_evenly(timestamps, duration, max_frames):
    """
    Distribute frames evenly across video duration while preserving important moments.
    """
    
    # Divide video into segments
    num_segments = max(3, max_frames // 3)  # At least 3 segments
    segment_duration = duration / num_segments
    
    distributed_frames = []
    
    for i in range(num_segments):
        segment_start = i * segment_duration
        segment_end = (i + 1) * segment_duration
        
        # Find timestamps in this segment
        segment_timestamps = [t for t in timestamps if segment_start <= t < segment_end]
        
        if not segment_timestamps:
            # If no interesting moments in segment, add middle of segment
            mid_point = segment_start + (segment_duration / 2)
            if mid_point < duration - 0.05:
                distributed_frames.append(mid_point)
        else:
            # Take best timestamps from this segment
            frames_per_segment = max(1, max_frames // num_segments)
            
            # Prioritize timestamps that are well-spaced within segment
            if len(segment_timestamps) <= frames_per_segment:
                distributed_frames.extend(segment_timestamps)
            else:
                # Select evenly spaced frames within segment
                step = len(segment_timestamps) // frames_per_segment
                selected = [segment_timestamps[i * step] for i in range(frames_per_segment)]
                distributed_frames.extend(selected)
    
    # Ensure we don't exceed max_frames
    if len(distributed_frames) > max_frames:
        # Take evenly spaced frames from our distributed set
        step = len(distributed_frames) / max_frames
        distributed_frames = [distributed_frames[int(i * step)] for i in range(max_frames)]
    
    return sorted(distributed_frames)


def enhanced_extract_audio_and_frames(
    tiktok_url,
    strategy="smart",
    frames_per_minute=24,
    cap=60,
    scene_threshold=0.24
):
    """
    Enhanced version with better frame distribution.
    """
    _ensure_dirs()
    video_path = download_video(tiktok_url)
    audio_path = extract_audio(video_path)
    dur = probe_duration(video_path)

    frames_dir = os.path.join("frames", f"set_{int(_time.time())}")
    os.makedirs(frames_dir, exist_ok=True)

    if strategy == "uniform":
        paths = extract_frames_uniform(video_path, frames_dir, frames_per_minute, cap)
    else:
        # SMART strategy with improved distribution
        print(f"[smart] Video duration: {dur:.1f}s, target frames: {cap}")
        
        sc_times = scene_change_times(video_path, threshold=scene_threshold)
        mo_times = motion_event_times(video_path, window_sec=0.30, mag_thresh=12.0)

        merged_changes = sorted(set(sc_times + mo_times))
        print(f"[smart] Found {len(merged_changes)} change points across video")
        
        # Use improved distribution algorithm
        ts_list = build_distributed_sampling_times(dur, merged_changes, max_frames=cap)
        print(f"[smart] Selected {len(ts_list)} timestamps distributed across {dur:.1f}s")
        
        # Show distribution
        if ts_list:
            segments = [0, dur/4, dur/2, 3*dur/4, dur]
            for i in range(len(segments)-1):
                start, end = segments[i], segments[i+1]
                count = len([t for t in ts_list if start <= t < end])
                print(f"[smart] Segment {i+1} ({start:.1f}s-{end:.1f}s): {count} frames")

        paths = extract_frames_at_times(video_path, frames_dir, ts_list)
        
        # Apply quality filters
        print(f"[smart] Extracted {len(paths)} frames, applying quality filters...")
        paths = [p for p in paths if not is_blurry(p)]
        print(f"[smart] After blur filter: {len(paths)} frames")
        
        paths = dedupe_frames_by_phash(paths, dist=4)
        print(f"[smart] After deduplication: {len(paths)} frames")
        
        paths = keep_text_heavy_frames(paths, min_chars=0)
        paths = sorted(paths)[:cap]
        
        print(f"[smart] Final frame count: {len(paths)}")

        if not paths:
            print("[smart] No frames after filtering → fallback to uniform")
            paths = extract_frames_uniform(video_path, frames_dir, frames_per_minute=18, cap=min(cap, 20))

    return audio_path, frames_dir, paths


def analyze_visual_promise_delivery(frames_summaries_text):
    """
    Analyze visual content for promise/delivery patterns in non-verbal videos.
    """
    
    frames_lower = frames_summaries_text.lower()
    
    # Visual promise indicators (setup, before state, raw materials)
    promise_patterns = {
        'outline': ['outline', 'sketch', 'template', 'guide', 'lines', 'shape'],
        'raw_materials': ['ingredients', 'supplies', 'tools', 'products', 'materials'],
        'before_state': ['messy', 'unorganized', 'plain', 'empty', 'bare', 'clean face'],
        'process_start': ['beginning', 'starting', 'first step', 'initial', 'prep'],
        'incomplete': ['partial', 'halfway', 'in progress', 'working on']
    }
    
    # Visual delivery indicators (completion, transformation, final result)
    delivery_patterns = {
        'completion': ['finished', 'complete', 'done', 'final', 'completed'],
        'transformation': ['transformed', 'changed', 'different', 'improved', 'enhanced'],
        'filled_in': ['colored', 'filled', 'painted', 'covered', 'applied'],
        'organized': ['organized', 'arranged', 'neat', 'tidy', 'sorted'],
        'final_result': ['result', 'outcome', 'end', 'finished product', 'final look']
    }
    
    promise_score = 0
    delivery_score = 0
    
    found_promises = []
    found_deliveries = []
    
    # Check for promise patterns
    for category, patterns in promise_patterns.items():
        for pattern in patterns:
            if pattern in frames_lower:
                promise_score += 1
                found_promises.append(f"{category}: {pattern}")
                break
    
    # Check for delivery patterns  
    for category, patterns in delivery_patterns.items():
        for pattern in patterns:
            if pattern in frames_lower:
                delivery_score += 1
                found_deliveries.append(f"{category}: {pattern}")
                break
    
    # Analyze progression
    has_clear_progression = promise_score >= 2 and delivery_score >= 2
    
    return {
        'promise_score': promise_score,
        'delivery_score': delivery_score,
        'has_progression': has_clear_progression,
        'found_promises': found_promises,
        'found_deliveries': found_deliveries,
        'visual_narrative': 'strong' if has_clear_progression else 'weak'
    }


def detect_satisfying_processes_enhanced(transcript_text, frames_summaries_text):
    """Enhanced detection focusing on visual satisfaction markers."""
    
    frames_lower = frames_summaries_text.lower()
    
    # Precision/skill-based activities
    precision_activities = [
        'coloring within lines', 'precise application', 'careful placement',
        'detailed work', 'fine motor', 'delicate handling', 'steady hand',
        'controlled movement', 'accurate coloring', 'neat application'
    ]
    
    # Transformation activities  
    transformation_activities = [
        'filling in', 'covering', 'applying', 'spreading', 'blending',
        'layering', 'building up', 'adding color', 'completing'
    ]
    
    # Completion satisfaction markers
    completion_markers = [
        'finishing', 'completing section', 'final touches', 'last step',
        'nearly done', 'almost finished', 'wrapping up'
    ]
    
    # Repetitive/rhythmic markers
    rhythmic_markers = [
        'repetitive', 'rhythmic', 'systematic', 'methodical', 'step by step',
        'consistent', 'regular pattern', 'steady rhythm'
    ]
    
    # ASMR/sensory markers
    sensory_markers = [
        'smooth', 'satisfying', 'gentle', 'soft', 'texture', 'tactile',
        'calming', 'soothing', 'relaxing', 'meditative'
    ]
    
    detected_elements = {
        'precision_work': any(marker in frames_lower for marker in precision_activities),
        'transformation': any(marker in frames_lower for marker in transformation_activities), 
        'completion_moments': any(marker in frames_lower for marker in completion_markers),
        'rhythmic_action': any(marker in frames_lower for marker in rhythmic_markers),
        'sensory_appeal': any(marker in frames_lower for marker in sensory_markers)
    }
    
    # Calculate satisfaction score
    satisfaction_score = sum(detected_elements.values())
    
    return {
        'satisfaction_elements': detected_elements,
        'satisfaction_score': satisfaction_score,
        'highly_satisfying': satisfaction_score >= 3,
        'primary_satisfaction': max(detected_elements, key=detected_elements.get) if detected_elements else None
    }


def create_visual_content_description(frames_summaries_text, audio_description):
    """Create comprehensive description for visual-heavy content."""
    
    # Analyze visual promise/delivery
    progression_analysis = analyze_visual_promise_delivery(frames_summaries_text)
    
    # Analyze satisfying processes
    satisfaction_analysis = detect_satisfying_processes_enhanced("", frames_summaries_text)
    
    # Build description
    description_parts = []
    
    # Main activity
    frames_lower = frames_summaries_text.lower()
    if 'coloring' in frames_lower or 'drawing' in frames_lower:
        description_parts.append("Creative drawing/coloring process")
    elif 'skincare' in frames_lower or 'face' in frames_lower:
        description_parts.append("Skincare routine application")
    elif 'hair' in frames_lower:
        description_parts.append("Hair styling/care routine")
    elif 'makeup' in frames_lower:
        description_parts.append("Makeup application process")
    else:
        description_parts.append("Visual process/activity")
    
    # Add satisfaction elements
    if satisfaction_analysis['highly_satisfying']:
        primary = satisfaction_analysis['primary_satisfaction']
        if primary == 'precision_work':
            description_parts.append("with precise, skillful execution")
        elif primary == 'transformation':
            description_parts.append("showing clear transformation")
        elif primary == 'completion_moments':
            description_parts.append("with satisfying completion moments")
        elif primary == 'rhythmic_action':
            description_parts.append("using rhythmic, methodical movements")
        elif primary == 'sensory_appeal':
            description_parts.append("with strong sensory/ASMR appeal")
    
    # Add progression info
    if progression_analysis['has_progression']:
        description_parts.append("featuring clear promise-to-delivery progression")
    
    # Add audio context
    if audio_description and "ambient sounds" in audio_description:
        description_parts.append("accompanied by satisfying ambient sounds")
    
    base_description = " ".join(description_parts)
    
    return {
        'description': base_description,
        'progression_analysis': progression_analysis,
        'satisfaction_analysis': satisfaction_analysis,
        'content_type': 'visual_process',
        'has_strong_visual_narrative': progression_analysis['has_progression']
    }


def detect_content_patterns(transcript_text, frames_summaries_text):
    """Enhanced detection for content patterns including satisfying background processes."""
    
    # Satisfying process keywords
    satisfying_processes = [
        'folding', 'organizing', 'makeup', 'skincare', 'cooking', 'baking', 'eating',
        'painting', 'drawing', 'crafting', 'cleaning', 'tidying', 'styling', 'braiding',
        'gaming', 'typing', 'building', 'assembling', 'decorating', 'planting',
        'chopping', 'mixing', 'blending', 'brushing', 'arranging', 'sorting', 'laundry'
    ]
    
    # Content type keywords
    controversial_indicators = [
        'unpopular opinion', 'controversial', 'hot take', 'nobody talks about',
        'people hate when', 'this will upset', 'i don\'t care if', 'fight me',
        'wrong', 'bad', 'terrible', 'hate', 'annoying', 'overrated', 'shit'
    ]
    
    educational_indicators = [
        'tutorial', 'how to', 'lesson', 'teach', 'learn', 'explain', 'guide',
        'tips', 'tricks', 'advice', 'steps', 'method', 'technique'
    ]
    
    storytelling_indicators = [
        'story time', 'let me tell you', 'this happened', 'experience',
        'journey', 'day i', 'time when', 'remember when'
    ]
    
    # Check for patterns
    text_combined = f"{transcript_text} {frames_summaries_text}".lower()
    
    satisfying_count = sum(1 for keyword in satisfying_processes if keyword in text_combined)
    controversial_count = sum(1 for keyword in controversial_indicators if keyword in text_combined)
    educational_count = sum(1 for keyword in educational_indicators if keyword in text_combined)
    story_count = sum(1 for keyword in storytelling_indicators if keyword in text_combined)
    
    # Determine patterns
    patterns = {
        'has_satisfying_process': satisfying_count >= 1,
        'is_controversial': controversial_count >= 1,
        'is_educational': educational_count >= 2,
        'is_storytelling': story_count >= 1,
        'dual_engagement': satisfying_count >= 1 and len(transcript_text.strip()) > 50
    }
    
    return patterns


def extract_content_themes(transcript_text):
    """Extract actual themes from any transcript without hardcoded categories."""
    
    # Look for common topic indicators in any niche
    words = transcript_text.lower().split()
    
    # Common patterns that indicate main topics
    topic_indicators = []
    
    # Look for repeated important nouns (likely main topics)
    word_count = {}
    important_words = []
    
    # Filter for meaningful words (nouns, important concepts)
    for word in words:
        # Clean word
        clean_word = word.strip('.,!?":;()[]{}')
        if len(clean_word) > 3 and clean_word not in ['that', 'this', 'with', 'have', 'will', 'they', 'them', 'were', 'been', 'said', 'what', 'when', 'where', 'your', 'like', 'just', 'know', 'think', 'people', 'going', 'really', 'because']:
            word_count[clean_word] = word_count.get(clean_word, 0) + 1
    
    # Get most frequently mentioned meaningful words
    frequent_topics = [word for word, count in word_count.items() if count >= 2][:5]
    
    return frequent_topics if frequent_topics else ['general content']


def create_universal_video_description(transcript_text, frames_summaries_text):
    """Create description for any video content without hardcoded activities."""
    
    frames_lower = frames_summaries_text.lower()
    transcript_lower = transcript_text.lower()
    
    # Detect any visual activity mentioned in frame analysis
    visual_activities = []
    
    # Look for action words in frame descriptions
    action_patterns = [
        'applying', 'making', 'cooking', 'preparing', 'folding', 'organizing', 
        'cleaning', 'sorting', 'building', 'creating', 'drawing', 'writing',
        'demonstrating', 'showing', 'presenting', 'explaining', 'teaching',
        'working', 'using', 'holding', 'moving', 'performing'
    ]
    
    detected_activity = None
    for action in action_patterns:
        if action in frames_lower:
            # Try to get the object of the action
            if action == 'applying' and 'makeup' in frames_lower:
                detected_activity = 'applying makeup'
                break
            elif action == 'cooking' or 'kitchen' in frames_lower:
                detected_activity = 'cooking'
                break
            elif action == 'folding' and ('clothes' in frames_lower or 'laundry' in frames_lower):
                detected_activity = 'folding laundry'
                break
            elif action in ['demonstrating', 'showing', 'presenting']:
                detected_activity = 'presenting content'
                break
            elif action in ['explaining', 'teaching']:
                detected_activity = 'teaching'
                break
    
    # Determine main content topic from frequent words
    content_themes = extract_content_themes(transcript_text)
    main_topic = f"discussing {', '.join(content_themes[:2])}"
    
    # Create description
    if detected_activity and detected_activity != 'presenting content':
        return f"Subject is {detected_activity} while {main_topic}"
    else:
        return f"Subject is {main_topic}"


def analyze_performance_indicators(creator_note, transcript_text, frames_summaries_text):
    """Extract performance metrics and determine if video was successful."""
    
    performance_data = {
        'views': None,
        'engagement': None,
        'followers_gained': None,
        'watch_time': None,
        'saves': None,
        'shares': None,
        'success_indicators': []
    }
    
    if creator_note:
        note_lower = creator_note.lower()
        
        # Extract numerical metrics
        # Views (k, m patterns)
        view_match = re.search(r'(\d+(?:\.\d+)?)\s*([km])?\s*views?', note_lower)
        if view_match:
            num, unit = view_match.groups()
            multiplier = 1000 if unit == 'k' else 1000000 if unit == 'm' else 1
            performance_data['views'] = int(float(num) * multiplier)
        
        # Followers gained
        follower_match = re.search(r'(\d+)\s*(?:new\s+)?followers?', note_lower)
        if follower_match:
            performance_data['followers_gained'] = int(follower_match.group(1))
        
        # Watch time (hours)
        watch_match = re.search(r'(\d+)\s*h.*?watch.*?time', note_lower)
        if watch_match:
            performance_data['watch_time'] = int(watch_match.group(1))
        
        # Saves
        save_match = re.search(r'(\d+)\s*saves?', note_lower)
        if save_match:
            performance_data['saves'] = int(save_match.group(1))
        
        # Shares  
        share_match = re.search(r'(\d+)\s*shares?', note_lower)
        if share_match:
            performance_data['shares'] = int(share_match.group(1))
    
    # Determine success level
    success_level = "unknown"
    success_reasons = []
    
    if performance_data['views']:
        if performance_data['views'] >= 100000:
            success_level = "highly_successful"
            success_reasons.append(f"{performance_data['views']:,} views indicates viral reach")
        elif performance_data['views'] >= 10000:
            success_level = "successful"
            success_reasons.append(f"{performance_data['views']:,} views shows good performance")
        else:
            success_level = "moderate"
            success_reasons.append(f"{performance_data['views']:,} views is moderate performance")
    
    if performance_data['followers_gained'] and performance_data['followers_gained'] >= 100:
        success_reasons.append(f"{performance_data['followers_gained']} new followers shows strong audience building")
    
    if performance_data['saves'] and performance_data['saves'] >= 100:
        success_reasons.append(f"{performance_data['saves']} saves indicates valuable content")
    
    performance_data['success_level'] = success_level
    performance_data['success_reasons'] = success_reasons
    
    return performance_data


def run_enhanced_gpt_analysis(transcript_text, frames_summaries_text, creator_note, platform, target_duration, goal, tone, audience, knowledge_context=""):
    """Enhanced analysis that works for any content topic and handles unreliable transcripts."""
    
    # Get patterns and description without hardcoded assumptions
    patterns = detect_content_patterns(transcript_text, frames_summaries_text)
    video_description = create_universal_video_description(transcript_text, frames_summaries_text)
    performance_data = analyze_performance_indicators(creator_note, transcript_text, frames_summaries_text)
    content_themes = extract_content_themes(transcript_text)
    
    print(f"Video description: {video_description}")
    print(f"Content themes: {content_themes}")
    print(f"Performance data: {performance_data}")
    
    # Build performance context
    performance_context = ""
    if performance_data['success_level'] != "unknown":
        performance_context = f"""
PERFORMANCE CONTEXT:
This video achieved: {', '.join(performance_data['success_reasons'])}
Success Level: {performance_data['success_level']}
Use this performance data to validate your analysis - explain WHY this video achieved this level of success.
        """
    
    # Enhanced prompt with rich multimodal analysis framework
    dual_engagement_note = ""
    if patterns.get('dual_engagement', False):
        dual_engagement_note = "\n🎯 DUAL ENGAGEMENT DETECTED: This video combines satisfying visual processes with verbal content delivery - analyze how both channels work together for retention."

    prompt = f"""
You are an expert TikTok/short-form content strategist analyzing videos for retention psychology and engagement mechanics.

TRANSCRIPT (What they're saying): {transcript_text}

VISUAL FRAMES (What viewers see): {frames_summaries_text}

CREATOR NOTE: {creator_note}
VIDEO DESCRIPTION: {video_description}
MAIN TOPICS: {', '.join(content_themes)}
PLATFORM: {platform} | GOAL: {goal} | DURATION: {target_duration}s
{dual_engagement_note}

{performance_context}

COMPREHENSIVE MULTIMODAL ANALYSIS FRAMEWORK:

Analyze this video by combining both the spoken content (transcript) and visual elements (frames) to understand the full retention strategy:

1. HOOK ANALYSIS (0-3 seconds):
- How do the opening words work with the visual presentation?
- Does the on-screen text reinforce or contradict the verbal hook?
- Are there visual pattern interrupts (gestures, movements, graphics)?
- Combined hook effectiveness: Does audio + visual create stronger curiosity?
- What specific psychological trigger is used in the opening moment?

2. PROMISE IDENTIFICATION (3-7 seconds):
- What promise is made verbally vs. visually?
- Do the frames show setup for what's promised in speech?
- Is there visual foreshadowing of the payoff?
- How well aligned are the words and visuals in setting expectations?
- Is the promise specific and compelling enough to retain viewers?

3. RETENTION MECHANICS (Mid-content):
- Story progression: How do visuals support the narrative flow?
- Engagement elements: Eye contact, expressions, gestures that drive comments
- Visual variety: Do frame changes maintain interest during speech?
- Pacing alignment: Do visual cuts match verbal rhythm and emphasis?
- Satisfying processes: Are there repetitive, satisfying activities that retain attention?
- Authority building: How does the creator establish credibility through words and actions?

4. PAYOFF DELIVERY (Final moments):
- Does the visual reveal align with the verbal conclusion?
- Are key moments emphasized both verbally and visually?
- Is the satisfaction delivered through words, visuals, or both?
- How well does the ending fulfill the opening promise?
- Is there a strong engagement trigger at the end?

5. MULTIMODAL SYNCHRONIZATION:
- Text overlays + speech content alignment
- Facial expressions + verbal tone consistency
- Visual demonstrations + explanations coordination
- Environmental changes + narrative progression harmony
- Moments where audio and visual elements reinforce vs. conflict

6. ENGAGEMENT PSYCHOLOGY:
- What specific emotions does this content trigger?
- Why would someone comment on THIS particular approach?
- What makes this shareable beyond just the topic?
- How does the creator's positioning affect viewer perception?
- What social triggers are embedded in the content?

HOOK GENERATION REQUIREMENTS:
Generate 5 alternative hooks that sound natural and platform-native:

TONE REQUIREMENTS:
- Use conversational, casual language (not marketing speak)
- Match the energy and vocabulary of the original video
- Sound like something a real person would actually say on TikTok
- Be specific to the actual topic/niche, not generic

AVOID THESE AI-SOUNDING PHRASES:
- "Discover the secret to..."
- "Unlock your potential..."
- "Transform your life with..."
- "The one trick that..."
- "You won't believe what happens when..."
- "Game-changing technique"
- "Revolutionary method"

INSTEAD USE NATURAL LANGUAGE PATTERNS:
- "wait this actually works"
- "nobody talks about this but..."
- "I tried this for [timeframe] and..."
- "my [relationship/job/etc] changed when I..."
- "this sounds fake but..."
- "POV: you just found out..."
- "telling my [person] that I..."
- "the day I accidentally..."
- "why [common thing] is actually..."

HOOK TYPES TO CONSIDER:
- Personal story openings with unexpected twists
- Controversial opinions about common beliefs
- Behind-the-scenes revelations
- Mistake/failure stories with lessons
- Comparison setups that subvert expectations

Respond in JSON format:
{{
  "analysis": "Comprehensive multimodal analysis explaining how visual and verbal elements work together (or against each other) to create retention. Include specific moments where synchronization enhances or detracts from the experience. Minimum 300 words covering hook mechanics, promise delivery, retention tactics, and engagement psychology.",
  "hooks": [
    "Natural-sounding hook 1 using personal story pattern",
    "Conversational hook 2 with controversial opinion setup", 
    "Relatable hook 3 using mistake/learning angle",
    "Behind-scenes hook 4 revealing unexpected insight",
    "POV-style hook 5 creating immediate relatability"
  ],
  "scores": {{
    "hook_strength": "Rate 1-10: How compelling is the audio+visual opening combination?",
    "promise_clarity": "Rate 1-10: How clear is the expected payoff across both channels?",
    "retention_design": "Rate 1-10: How well do visuals and audio work together for watch-through?",
    "engagement_potential": "Rate 1-10: Will the combination drive comments/shares?",
    "goal_alignment": "Rate 1-10: How well does the full experience serve {goal}?"
  }},
  "timing_breakdown": "Second-by-second breakdown of how audio and visual elements build retention throughout the video",
  "multimodal_insights": "Specific analysis of how visual and verbal elements reinforce or conflict with each other",
  "engagement_triggers": "Detailed breakdown of psychological triggers that drive comments, shares, and rewatches",
  "authority_signals": "How the creator builds credibility and trust through both words and visual presentation",
  "improvement_opportunities": "Specific ways to better synchronize audio and visual retention tactics",
  "viral_potential_factors": "What elements make this content likely to spread beyond the initial audience",
  "basic_formula": "Step-by-step process for creating similar multimodal content",
  "timing_formula": "Timing strategy for coordinating visual and verbal elements",
  "template_formula": "Template format showing how to structure audio-visual retention",
  "psychology_formula": "Psychology framework explaining why this multimodal approach works",
  "performance_prediction": "Detailed prediction based on retention analysis and engagement mechanics"
}}

Focus on the interplay between what viewers see and hear, identifying moments of synchronization and missed opportunities for enhanced retention.
    """

    try:
        print(f"Sending enhanced analysis prompt to GPT-4o...")
        gpt_response = _api_retry(
            client.chat.completions.create,
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=3500
        )

        response_text = gpt_response.choices[0].message.content.strip()
        
        # Parse JSON
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        try:
            parsed = json.loads(response_text)
            print("Successfully parsed JSON response")
            
            result = {
                "analysis": parsed.get("analysis", "Analysis not available").strip(),
                "hooks": parsed.get("hooks", []),
                "scores": parsed.get("scores", {}),
                "timing_breakdown": parsed.get("timing_breakdown", "").strip(),
                "formula": parsed.get("basic_formula", "").strip(),
                "basic_formula": parsed.get("basic_formula", "").strip(),
                "timing_formula": parsed.get("timing_formula", "").strip(),
                "template_formula": parsed.get("template_formula", "").strip(),
                "psychology_formula": parsed.get("psychology_formula", "").strip(),
                "improvements": parsed.get("improvement_opportunities", "").strip(),
                "performance_prediction": parsed.get("performance_prediction", "").strip(),
                "video_description": video_description,
                "content_patterns": patterns,
                "performance_data": performance_data,
                
                # New rich analysis fields
                "multimodal_insights": parsed.get("multimodal_insights", "").strip(),
                "engagement_triggers": parsed.get("engagement_triggers", "").strip(),
                "authority_signals": parsed.get("authority_signals", "").strip(),
                "improvement_opportunities": parsed.get("improvement_opportunities", "").strip(),
                "viral_potential_factors": parsed.get("viral_potential_factors", "").strip()
            }
            
            # Generic fallback hooks if needed (based on detected patterns)
            if not result["hooks"] or len(result["hooks"]) == 0:
                print("Generating content-adaptive fallback hooks")
                
                if patterns.get('is_controversial', False):
                    result["hooks"] = [
                        "this opinion is going to upset people but it's true",
                        "everyone's wrong about this and I can prove it",
                        "this harsh truth will change how you think",
                        "nobody wants to admit this but here's reality",
                        "this controversial take will make you rethink everything"
                    ]
                elif patterns.get('is_educational', False):
                    result["hooks"] = [
                        "this changed everything I thought I knew",
                        "nobody taught me this and I wish they had",
                        "here's what I wish someone told me earlier",
                        "this method actually works and here's why",
                        "the mistake everyone makes that you can avoid"
                    ]
                else:
                    result["hooks"] = [
                        "wait this actually works",
                        "nobody talks about this but they should",
                        "POV: you just found out the truth",
                        "this sounds fake but I promise it's not",
                        "the day I realized most people are wrong"
                    ]
            
            # Adjust scores based on performance
            if performance_data['success_level'] == 'highly_successful':
                for key in result["scores"]:
                    result["scores"][key] = min(10, result["scores"][key] + 1)
            
            print(f"Enhanced analysis complete - Topics: {content_themes}")
            return result
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            return create_enhanced_fallback(video_description, patterns, content_themes, goal, performance_data)
            
    except Exception as e:
        print(f"GPT analysis error: {e}")
        return create_enhanced_fallback(video_description, patterns, content_themes, goal, performance_data)


def create_enhanced_fallback(video_description, patterns, content_themes, goal, performance_data):
    """Enhanced fallback with richer analysis framework."""
    
    # Build comprehensive analysis
    analysis = f"{video_description}. "
    
    if performance_data['success_level'] != 'unknown':
        analysis += f"This video achieved {performance_data['success_level']} performance with {', '.join(performance_data['success_reasons'])}. "
    
    # Multimodal insights
    multimodal_analysis = "The content uses a combination of verbal delivery and visual presentation to maintain viewer attention. "
    if patterns.get('dual_engagement', False):
        multimodal_analysis += "Visual and auditory elements work together to create dual engagement, with satisfying processes complementing the spoken content. "
    
    analysis += f"The content focuses on {', '.join(content_themes[:3]) if content_themes else 'the main topic'} using "
    
    if patterns.get('is_controversial', False):
        analysis += "controversial positioning and expert authority to challenge common beliefs. The creator builds credibility through confident delivery and specific examples, creating engagement through disagreement and validation."
        engagement_triggers = "Controversial positioning triggers strong reactions - agreement from those who relate, disagreement from those who don't. The specific examples make it feel personal and authentic."
        authority_signals = "Direct, confident language and specific examples establish expertise. The willingness to take a controversial stance signals authority and authenticity."
    elif patterns.get('is_educational', False):
        analysis += "educational delivery and valuable insights to teach viewers. The creator positions themselves as a helpful expert, building trust through clear explanations and actionable advice."
        engagement_triggers = "Educational value drives saves and shares. Viewers comment with questions, their own experiences, and gratitude for the insights shared."
        authority_signals = "Clear, structured explanations and specific techniques demonstrate expertise. The teaching approach builds trust and positions the creator as knowledgeable."
    else:
        analysis += "engaging storytelling and relatable content to connect with viewers. Personal anecdotes and relatable situations create emotional connection and authenticity."
        engagement_triggers = "Relatability drives comments as viewers share their own similar experiences. Personal storytelling creates emotional connection leading to shares."
        authority_signals = "Authentic personal experience and honest vulnerability establish trust. Real stories and genuine emotion signal credibility."
    
    analysis += " The retention strategy combines direct communication with specific insights to maintain viewer engagement throughout."
    
    # Generate natural hooks based on patterns
    if patterns.get('is_controversial', False):
        hooks = [
            "nobody wants to admit this but it's true",
            "this opinion is going to upset people",
            "I don't care if this is unpopular but...",
            "everyone's doing this wrong and here's why",
            "this harsh truth changed everything for me"
        ]
    elif patterns.get('is_educational', False):
        hooks = [
            "I wish someone told me this sooner",
            "this method actually works and here's proof", 
            "the mistake I was making for years",
            "nobody teaches you this but they should",
            "this changed my entire approach"
        ]
    else:
        hooks = [
            "this happened to me and I learned...",
            "the day I realized I was doing it wrong",
            "my biggest mistake was thinking...",
            "POV: you finally understand why...",
            "telling my story because maybe it helps"
        ]
    
    return {
        "analysis": analysis,
        "hooks": hooks,
        "scores": {
            "hook_strength": 8 if performance_data['success_level'] == 'highly_successful' else 7,
            "promise_clarity": 7,
            "retention_design": 8 if performance_data['success_level'] != 'unknown' else 7,
            "engagement_potential": 8,
            "goal_alignment": 7
        },
        "timing_breakdown": "Content builds from engaging opening through valuable insights to satisfying conclusion",
        "formula": "Strong opener → Value delivery → Authority building → Clear conclusion",
        "basic_formula": "1. Create immediate interest 2. Deliver specific value 3. Build credibility 4. End with clear takeaway",
        "timing_formula": "0-3s: Hook creation, 3-7s: Value setup, Middle: Content delivery, End: Strong conclusion",
        "template_formula": "[Engaging Hook] → [Value Promise] → [Content Delivery] → [Clear Conclusion]",
        "psychology_formula": "Attention → Interest → Value → Satisfaction",
        "improvements": f"Enhance opening hook, add more specific examples, optimize pacing for {goal}",
        "performance_prediction": f"Based on content analysis: {performance_data['success_level'] if performance_data['success_level'] != 'unknown' else 'moderate to strong performance expected'}",
        "video_description": video_description,
        "content_patterns": patterns,
        "performance_data": performance_data,
        
        # Rich analysis fields
        "multimodal_insights": multimodal_analysis,
        "engagement_triggers": engagement_triggers,
        "authority_signals": authority_signals,
        "improvement_opportunities": f"Strengthen visual-verbal synchronization, enhance credibility signals, optimize for {goal} with more specific calls-to-action",
        "viral_potential_factors": "Authentic delivery, relatable content, and specific insights create shareability. The natural conversational tone makes it feel genuine rather than promotional."
    }


def create_visual_enhanced_fallback(frames_summaries_text, transcript_data, goal):
    """Enhanced fallback for when GPT analysis fails on visual content."""
    
    visual_analysis = create_visual_content_description(frames_summaries_text, None)
    
    analysis = f"Visual content analysis: {visual_analysis['description']}. "
    
    if not transcript_data['is_reliable']:
        analysis += f"Audio consists of ambient activity sounds rather than speech. "
    
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
        "transcript_quality": transcript_data
    }


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
            
            # Run enhanced analysis
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
        
        # New enhanced fields
        visual_content_analysis = gpt_result.get("visual_content_analysis", {})
        transcript_quality = gpt_result.get("transcript_quality", {})
        audio_description = gpt_result.get("audio_description", "")
        
        # New rich analysis fields
        multimodal_insights = gpt_result.get("multimodal_insights", "")
        engagement_triggers = gpt_result.get("engagement_triggers", "")
        authority_signals = gpt_result.get("authority_signals", "")
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
            
            # New rich analysis fields
            multimodal_insights=multimodal_insights,
            engagement_triggers=engagement_triggers,
            authority_signals=authority_signals,
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