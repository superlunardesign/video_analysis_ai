# processing.py — lean version (no custom GPT)
# Download, audio extraction, smart/uniform frame sampling,
# quality filtering, robust transcription, and vision batch analysis.

import os
import re
import cv2
import math
import time              # for time.time()
import time as _time     # for retry helper
import base64
import ffmpeg
import subprocess
from pathlib import Path
from typing import List, Tuple

from PIL import Image
from yt_dlp import YoutubeDL
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# OPTIONAL local whisper fallback (pip install openai-whisper)
try:
    import whisper as local_whisper
except Exception:
    local_whisper = None


# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------

def _ensure_dirs():
    Path("downloads").mkdir(exist_ok=True)
    Path("audio").mkdir(exist_ok=True)
    Path("frames").mkdir(exist_ok=True)


def _api_retry(callable_fn, *args, **kwargs):
    """Retry wrapper for OpenAI calls: 4 tries, exponential backoff with jitter."""
    max_tries = 4
    base = 1.25
    for attempt in range(1, max_tries + 1):
        try:
            return callable_fn(*args, **kwargs)
        except Exception as e:
            if attempt == max_tries:
                raise
            sleep_s = (base ** attempt) + 0.5 * (os.urandom(1)[0] / 255.0)
            print(f"[retry] OpenAI call failed ({attempt}/{max_tries}): {e}. Retrying in {sleep_s:.1f}s")
            _time.sleep(sleep_s)


# ------------------------------------------------------------------------------
# Download + audio
# ------------------------------------------------------------------------------

def download_video(tiktok_url: str) -> str:
    """
    Download social video using yt-dlp. Returns absolute mp4 path.
    """
    _ensure_dirs()
    stamp = str(int(time.time()))
    out_tmpl = os.path.abspath(os.path.join("downloads", f"vid_{stamp}.%(ext)s"))

    ydl_opts = {
        "outtmpl": out_tmpl,
        "format": "mp4/bestvideo+bestaudio/best",
        "quiet": True,
        "retries": 5,
        "noprogress": True,
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([tiktok_url])
    except Exception as e:
        error_msg = str(e)

        # Handle specific TikTok errors with user-friendly messages
        if "status code 10204" in error_msg or "Video not available" in error_msg:
            raise ValueError(
                "❌ This TikTok video is not accessible. Common causes:\n"
                "• Video is private or deleted\n"
                "• Video requires login to view\n"
                "• Video is geo-restricted\n"
                "• Creator has restricted sharing\n\n"
                "Try a different public video or check if the link is correct."
            )
        elif "Sign in to confirm you're not a bot" in error_msg:
            raise ValueError(
                "❌ TikTok is blocking automated access. This can happen when:\n"
                "• Too many requests in short time\n"
                "• TikTok's anti-bot protection is active\n\n"
                "Wait a few minutes and try again with a different video."
            )
        else:
            # Re-raise original error if not a known case
            raise

    mp4s = sorted(Path("downloads").glob(f"vid_{stamp}*.mp4"))
    if mp4s:
        return str(mp4s[-1])
    # fallback: accept any container if mp4 not produced
    candidates = sorted(Path("downloads").glob(f"vid_{stamp}*.*"))
    if not candidates:
        raise FileNotFoundError("Video download failed.")
    return str(candidates[-1])


def probe_duration(video_path: str) -> float:
    info = ffmpeg.probe(video_path)
    return float(info["format"]["duration"])


def extract_audio(video_path: str) -> str:
    _ensure_dirs()
    audio_path = os.path.abspath(os.path.join("audio", f"aud_{int(time.time())}.mp3"))
    (
        ffmpeg
        .input(video_path)
        .output(audio_path, format="mp3", acodec="mp3", ar=44100, ac=2)
        .overwrite_output()
        .run(quiet=True)
    )
    return audio_path


# ------------------------------------------------------------------------------
# Scene & motion detection
# ------------------------------------------------------------------------------

def scene_change_times(video_path: str, threshold: float = 0.25) -> List[float]:
    """
    Use FFmpeg select scene filter and parse precise pts_time values from stderr.
    Lower threshold (0.20–0.30) is more sensitive to subtle UI/graphics changes.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats",
        "-i", video_path,
        "-filter_complex", f"select='gt(scene,{threshold})',metadata=print",
        "-f", "null", "-"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, err = proc.communicate()

    times = []
    for line in err.decode("utf-8", errors="ignore").splitlines():
        if "pts_time" in line and "lavfi.scene_score" in line:
            m = re.search(r"pts_time[:=]([0-9\.]+)", line)
            if m:
                times.append(round(float(m.group(1)), 3))

    times = sorted(set(times))
    print(f"[smart] scene hits: {len(times)} at {times[:6]}{'...' if len(times) > 6 else ''}")
    return times


def motion_event_times(video_path: str, window_sec: float = 0.3,
                       mag_thresh: float = 12.0, max_events: int = 60) -> List[float]:
    """
    Lightweight motion detector to catch slide-ins/animations that per-frame scene
    miss. Samples ~15 fps; flags windows whose mean absdiff exceeds mag_thresh.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap or not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(fps / 15))        # ~15 fps sampling
    win  = max(2, int(window_sec * (fps / step)))

    ret, prev = cap.read()
    if not ret:
        cap.release()
        return []
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    diffs, idx = [], 0
    while True:
        for _ in range(step - 1):
            cap.grab()
            idx += 1
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mag  = float(cv2.absdiff(gray, prev_gray).mean())
        t    = idx / fps
        diffs.append((t, mag))
        prev_gray = gray

    cap.release()

    # rolling mean
    hits, buf = [], []
    last_t = -999
    for t, mag in diffs:
        buf.append(mag)
        if len(buf) > win:
            buf.pop(0)
        if len(buf) == win and sum(buf)/len(buf) >= mag_thresh:
            if t - last_t > 0.5:
                hits.append(round(t, 3)); last_t = t
            if len(hits) >= max_events:
                break

    print(f"[smart] motion hits: {len(hits)} at {hits[:6]}{'...' if len(hits) > 6 else ''}")
    return hits


# ------------------------------------------------------------------------------
# Timestamp building
# ------------------------------------------------------------------------------

def build_sampling_times(duration: float, change_times: List[float], max_frames: int) -> List[float]:
    """
    Merge anchors + settle offsets with detected change times.
    Clip to duration and dedupe. Allow ~3x cap for downstream pruning.
    """
    anchors = [0.0, 0.3, 0.8, 1.0, 1.5, 3.0]  # extra 1.0s helps mid-intro slide-ins
    outro = max(0.0, duration - 0.8)
    anchors.append(outro)

    settle_offsets = [0.3, 0.6]

    ts = []
    for t in change_times:
        ts.append(t)
        for off in settle_offsets: ts.append(t + off)
    for a in anchors:
        ts.append(a)
        for off in settle_offsets: ts.append(a + off)

    ts = [round(max(0.0, min(duration - 0.05, x)), 3) for x in ts]
    ts = sorted(set(ts))

    if not ts:
        # fallback to uniform N points if nothing detected
        N = max_frames
        step = duration / (N + 1)
        ts = [round(step * i, 3) for i in range(1, N + 1)]

    return ts[: max_frames * 3]


# ------------------------------------------------------------------------------
# Frame extraction
# ------------------------------------------------------------------------------

def extract_frames_at_times(video_path: str, out_dir: str, timestamps: List[float]) -> List[str]:
    """
    Seek on INPUT at each timestamp so FFmpeg returns distinct, correct frames.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, ts in enumerate(timestamps, start=1):
        out_path = os.path.join(out_dir, f"ts_{i:04d}.jpg")
        (
            ffmpeg
            .input(video_path, ss=float(ts))  # input-side seek
            .output(out_path, vframes=1, vf="scale=480:-1", vsync="vfr")
            .overwrite_output()
            .run(quiet=True)
        )
        paths.append(out_path)
    return paths


def extract_frames_uniform(video_path: str, frames_dir: str,
                           frames_per_minute: int, cap: int) -> List[str]:
    dur = probe_duration(video_path)
    total = min(cap, max(1, int(frames_per_minute * (dur / 60.0))))
    if total <= 0: total = min(cap, 10)
    step = dur / (total + 1)
    timestamps = [round(step * i, 3) for i in range(1, total + 1)]
    return extract_frames_at_times(video_path, frames_dir, timestamps)


# ------------------------------------------------------------------------------
# Quality filtering
# ------------------------------------------------------------------------------

def is_blurry(img_path: str, thr: float = 80.0) -> bool:
    """Reject frames with low sharpness (variance of Laplacian)."""
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return True
        fm = cv2.Laplacian(img, cv2.CV_64F).var()
        return fm < thr
    except Exception:
        return False


def _ahash(img: Image.Image, hash_size: int = 8) -> int:
    """Average hash; simple near-duplicate detector without extra deps."""
    img = img.convert("L").resize((hash_size, hash_size), Image.BILINEAR)
    px = list(img.getdata())
    avg = sum(px) / len(px)
    bits = 0
    for i, p in enumerate(px):
        bits |= (1 if p > avg else 0) << i
    return bits


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def dedupe_frames_by_phash(paths: List[str], dist: int = 4) -> List[str]:
    kept, hashes = [], []
    for p in paths:
        try:
            with Image.open(p) as im:
                h = _ahash(im)
        except Exception:
            continue
        if all(_hamming(h, hh) > dist for hh in hashes):
            kept.append(p); hashes.append(h)
    return kept


def keep_text_heavy_frames(paths: List[str], min_chars: int = 0) -> List[str]:
    """
    Placeholder to prefer text-on-screen frames (wire OCR here if desired).
    Currently returns unchanged.
    """
    return paths


# ------------------------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------------------------

def extract_audio_and_frames(
    tiktok_url: str,
    strategy: str = "smart",            # "smart" or "uniform"
    frames_per_minute: int = 24,        # used if uniform
    cap: int = 60,                      # max frames returned
    scene_threshold: float = 0.24       # lower = more sensitive
) -> Tuple[str, str, List[str]]:
    """
    Download video, extract audio, pick frames by strategy, apply quality filters.
    Returns (audio_path, frames_dir, [frame_paths]).
    """
    _ensure_dirs()
    video_path = download_video(tiktok_url)
    audio_path = extract_audio(video_path)
    dur = probe_duration(video_path)

    frames_dir = os.path.join("frames", f"set_{int(time.time())}")
    os.makedirs(frames_dir, exist_ok=True)

    if strategy == "uniform":
        paths = extract_frames_uniform(video_path, frames_dir, frames_per_minute, cap)
    else:
        # SMART: scene + motion + anchors + settle, then quality filters
        sc_times = scene_change_times(video_path, threshold=scene_threshold)
        mo_times = motion_event_times(video_path, window_sec=0.30, mag_thresh=12.0)

        merged = sorted(set(sc_times + mo_times))
        ts_list = build_sampling_times(dur, merged, max_frames=cap)
        print(f"[smart] timestamps after merge+anchors: {len(ts_list)}")

        paths = extract_frames_at_times(video_path, frames_dir, ts_list)
        paths = [p for p in paths if not is_blurry(p)]
        paths = dedupe_frames_by_phash(paths, dist=4)
        paths = keep_text_heavy_frames(paths, min_chars=0)
        paths = sorted(paths)[:cap]

        if not paths:
            print("[smart] empty after filters → fallback to uniform")
            paths = extract_frames_uniform(video_path, frames_dir, frames_per_minute=18, cap=min(cap, 20))

    return audio_path, frames_dir, paths


# ------------------------------------------------------------------------------
# Transcription (OpenAI with retry; local Whisper fallback)
# ------------------------------------------------------------------------------

def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe with OpenAI (gpt-4o-mini-transcribe).
    On connection failure, retry; if still failing and local Whisper is present, use it.
    """
    def _remote_transcribe():
        with open(audio_path, "rb") as f:
            return client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=f
            )

    try:
        tr = _api_retry(_remote_transcribe)
        return tr.text
    except Exception as e:
        print(f"[warn] Remote transcription failed: {e}")
        if local_whisper is not None:
            try:
                print("[info] Falling back to local Whisper (base). This is slower on CPU.")
                w = local_whisper.load_model("base")
                out = w.transcribe(audio_path)
                return out.get("text", "").strip() or "(transcription empty)"
            except Exception as e2:
                print(f"[error] Local Whisper failed: {e2}")
        return "(transcription unavailable due to connection error)"


# ------------------------------------------------------------------------------
# Batch frame analysis (vision) — always gpt-4o
# ------------------------------------------------------------------------------

def analyze_frames_batch(frame_paths, transcript=None):
    """
    Enhanced frame analysis with better text distinction and context awareness.

    Args:
        frame_paths: List of paths to frame images
        transcript: Optional transcript text to help identify captions vs overlays

    Returns:
        Tuple of (detailed_analysis_text, gallery_urls)
    """

    if not frame_paths:
        return "", []

    print(f"[INFO] Analyzing {len(frame_paths)} frames with enhanced text detection")

    # Prepare frame descriptions with structured analysis
    frame_analyses = []
    gallery_urls = []

    # Group frames for context (analyze 10 at a time for efficiency)
    # Larger groups = fewer API calls = faster analysis
    frame_groups = []
    group_size = 10  # Increased from 3 to reduce API calls (60 frames = 6 calls instead of 20)
    for i in range(0, len(frame_paths), group_size):
        frame_groups.append(frame_paths[i:i+group_size])

    # Process each group of frames
    total_groups = len(frame_groups)
    print(f"[INFO] Frame analysis: {len(frame_paths)} frames in {total_groups} groups (group size: {group_size})")

    for group_idx, frame_group in enumerate(frame_groups):
        print(f"[INFO] Analyzing frame group {group_idx + 1}/{total_groups} ({len(frame_group)} frames)...")

        # Convert frames to base64 for GPT-4 Vision
        base64_images = []
        for frame_path in frame_group:
            try:
                with open(frame_path, 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode()
                    base64_images.append(img_data)
                    gallery_urls.append(f"data:image/jpeg;base64,{img_data}")
            except Exception as e:
                print(f"[ERROR] Failed to process frame {frame_path}: {e}")
                continue

        if not base64_images:
            continue

        # Build the analysis prompt with transcript context
        transcript_context = ""
        if transcript:
            # Estimate which part of transcript might correspond to these frames
            # (This is approximate - you might want to use timing info if available)
            transcript_words = transcript.split()
            total_groups = len(frame_groups)
            words_per_group = len(transcript_words) // total_groups if total_groups > 0 else len(transcript_words)
            start_idx = group_idx * words_per_group
            end_idx = min(start_idx + words_per_group, len(transcript_words))
            transcript_segment = ' '.join(transcript_words[start_idx:end_idx])
            transcript_context = f"LIKELY TRANSCRIPT SEGMENT: '{transcript_segment}'"

        # Create structured prompt for GPT-4o
        prompt = f"""Analyze these {len(frame_group)} sequential video frames in detail.

{transcript_context}

For EACH frame, provide a structured analysis:

1. TEXT DETECTION AND CLASSIFICATION:
   Identify ALL visible text and classify each element:

   a) CAPTIONS (auto-generated subtitles):
      - Usually white/black text with outline
      - Bottom center position
      - Matches transcript words
      - Consistent font across frames
      - Example: "this is what I said in the video"

   b) OVERLAY TEXT (added for emphasis/hooks):
      - Stylized, colorful, or animated
      - Can appear anywhere on screen
      - Doesn't match transcript exactly
      - Used for hooks, CTAs, or key points
      - Example: "WAIT FOR IT..." or "3 TIPS!" or "Link in bio"

   c) UI TEXT (platform interface):
      - Username, likes, comments, share buttons
      - Part of TikTok/Instagram/YouTube interface

2. VISUAL CONTENT:
   Describe what's happening visually (actions, objects, people, etc.)

3. FRAME PURPOSE:
   What is this frame's role in the video? (establishing shot, reveal, demonstration, etc.)

4. ATTENTION ELEMENTS:
   What grabs attention? (motion, colors, text placement, visual hooks)

FORMAT YOUR RESPONSE EXACTLY LIKE THIS for each frame:

FRAME [NUMBER]:
TEXT FOUND:
- [CAPTION] "exact text here" | Position: bottom-center | Style: white with black outline
- [OVERLAY] "exact text here" | Position: top-left | Style: bold yellow, animated entrance
- [UI] "@username" | Position: bottom-left | Style: platform default

VISUAL CONTENT:
[Describe what's shown in the frame]

PURPOSE: [Frame's role in video narrative]

ATTENTION GRABBERS: [What catches the eye]

---

CRITICAL:
- Quote text EXACTLY as it appears
- If text appears to match the transcript segment = likely CAPTION
- If text is stylized/emphatic and doesn't match transcript = likely OVERLAY
- If no text visible, write "No text in this frame"
- Pay attention to text positioning and styling for classification"""

        # Build the messages for GPT-4o Vision
        messages = [
            {
                "role": "system",
                "content": "You are an expert video analyst specializing in distinguishing between different types of on-screen text (captions vs overlays vs UI). You have excellent OCR capabilities and understand video editing conventions."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Add each image to the message
        for idx, base64_img in enumerate(base64_images):
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_img}",
                    "detail": "high"  # Use high detail for better text recognition
                }
            })

        try:
            # Call GPT-4o for analysis
            response = client.chat.completions.create(
                model="gpt-4o",  # Latest and best for vision + OCR
                messages=messages,
                max_tokens=2000,
                temperature=0.3  # Lower temperature for more consistent text recognition
            )

            frame_analyses.append(response.choices[0].message.content)

        except Exception as e:
            print(f"[ERROR] GPT-4 Vision API call failed for group {group_idx}: {e}")
            # Fallback analysis
            frame_analyses.append(f"Frame group {group_idx + 1}: Unable to analyze frames - {str(e)}")

    # Combine all analyses into a comprehensive summary
    full_analysis = "\n\n".join(frame_analyses)

    # Post-process to create a summary section
    summary = create_analysis_summary(full_analysis, transcript)

    # Add summary to the beginning of the analysis
    final_output = f"""VIDEO FRAME ANALYSIS SUMMARY:
{summary}

DETAILED FRAME-BY-FRAME ANALYSIS:
{full_analysis}"""

    print(f"[SUCCESS] Frame analysis complete")
    return final_output, gallery_urls


def create_analysis_summary(full_analysis, transcript=None):
    """
    Create a summary of the frame analysis highlighting key findings.
    """
    summary_lines = []

    # Extract key patterns from the analysis
    caption_count = full_analysis.lower().count('[caption]')
    overlay_count = full_analysis.lower().count('[overlay]')

    summary_lines.append(f"Total text elements detected: {caption_count + overlay_count}")
    summary_lines.append(f"- Captions (matching speech): {caption_count}")
    summary_lines.append(f"- Overlay text (added hooks/emphasis): {overlay_count}")

    # Identify common overlay patterns
    if '[overlay]' in full_analysis.lower():
        summary_lines.append("\nKey overlay text identified (hooks and CTAs):")
        # Extract overlay texts (this is a simple extraction, could be enhanced)
        import re
        overlay_pattern = r'\[OVERLAY\] "(.*?)"'
        overlays = re.findall(overlay_pattern, full_analysis, re.IGNORECASE)
        for overlay in overlays[:5]:  # Show first 5 overlays
            summary_lines.append(f"  • \"{overlay}\"")

    # Add transcript matching note
    if transcript:
        summary_lines.append(f"\nTranscript provided for caption matching: Yes ({len(transcript.split())} words)")
    else:
        summary_lines.append("\nTranscript provided for caption matching: No (classification based on visual cues only)")

    # Check for visual patterns
    if 'hook' in full_analysis.lower() or 'attention' in full_analysis.lower():
        summary_lines.append("\nVisual hooks detected - see detailed analysis below")

    return "\n".join(summary_lines)


def extract_text_with_ocr(frame_path):
    """
    Optional: Pre-process frames with dedicated OCR for better text extraction.
    This can be called before GPT-4 analysis for higher accuracy.

    Requires: pip install easyocr
    """
    try:
        import easyocr
        reader = easyocr.Reader(['en'])

        results = reader.readtext(frame_path)

        extracted_texts = []
        for (bbox, text, confidence) in results:
            if confidence > 0.5:  # Only include high-confidence text
                # Determine position based on bbox
                top_y = bbox[0][1]
                center_x = (bbox[0][0] + bbox[1][0]) / 2

                # Simple position classification
                img_height = 1920  # Assume standard dimensions, adjust as needed
                img_width = 1080

                if top_y < img_height * 0.3:
                    v_position = "top"
                elif top_y > img_height * 0.7:
                    v_position = "bottom"
                else:
                    v_position = "middle"

                if center_x < img_width * 0.3:
                    h_position = "left"
                elif center_x > img_width * 0.7:
                    h_position = "right"
                else:
                    h_position = "center"

                extracted_texts.append({
                    'text': text,
                    'position': f"{v_position}-{h_position}",
                    'confidence': confidence,
                    'bbox': bbox
                })

        return extracted_texts

    except ImportError:
        print("[WARNING] easyocr not installed. Skipping OCR pre-processing.")
        return []
    except Exception as e:
        print(f"[ERROR] OCR extraction failed: {e}")
        return []


# ------------------------------------------------------------------------------
# AUTO METADATA EXTRACTION WITH SAVE COUNTS
# ------------------------------------------------------------------------------

def extract_video_metadata(tiktok_url):
    """
    Extract comprehensive video metadata including SAVES/BOOKMARKS.
    NO API NEEDED - yt-dlp handles everything.
    """
    import yt_dlp
    import re

    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(tiktok_url, download=False)

            # Debug available fields - print all fields and their values for debugging
            print(f"[DEBUG] Available fields in yt-dlp: {', '.join(info.keys())}")

            # Show all numeric fields (likely to contain counts)
            numeric_fields = {k: v for k, v in info.items() if isinstance(v, (int, float)) and v > 0}
            print(f"[DEBUG] All numeric fields with values > 0: {numeric_fields}")

            # Look for save-related fields
            save_related_fields = [
                'save_count', 'bookmark_count', 'favorite_count', 'collect_count',
                'collection_count', 'saved_count', 'bookmarks', 'favorites',
                'saves', 'collect', 'collected'
            ]

            print("[DEBUG] Checking for save count fields:")
            for field in save_related_fields:
                value = info.get(field)
                if value is not None:
                    print(f"  {field}: {value}")

            # Extract saves (field name varies) - try all possible names
            save_count = (
                info.get('collect_count') or  # TikTok often uses 'collect_count' for saves
                info.get('save_count') or
                info.get('bookmark_count') or
                info.get('favorite_count') or
                info.get('collection_count') or
                info.get('saved_count') or
                info.get('collect') or
                0
            )

            # If still 0, check statistics dict if it exists
            if save_count == 0 and 'statistics' in info:
                stats = info['statistics']
                print(f"[DEBUG] Found statistics dict: {stats.keys() if isinstance(stats, dict) else stats}")
                if isinstance(stats, dict):
                    save_count = (
                        stats.get('collect_count') or
                        stats.get('save_count') or
                        stats.get('bookmark_count') or
                        stats.get('favorite_count') or
                        0
                    )

            metadata = {
                'url': tiktok_url,
                'title': info.get('title', ''),
                'description': info.get('description', ''),

                # Engagement metrics
                'view_count': info.get('view_count', 0),
                'like_count': info.get('like_count', 0),
                'comment_count': info.get('comment_count', 0),
                'repost_count': info.get('repost_count', 0),  # Changed from share_count
                'save_count': save_count,  # SAVES/BOOKMARKS (keeping for backwards compatibility)

                # Video details
                'duration': info.get('duration', 0),
                'upload_date': info.get('upload_date', ''),
                'timestamp': info.get('timestamp', ''),

                # Creator info
                'uploader': info.get('uploader', 'Unknown'),  # Video creator/uploader
                'uploader_id': info.get('uploader_id', ''),

                # Audio/Music info
                'track': info.get('track', ''),  # Song/audio track name
                'artist': info.get('artist', ''),  # Artist name

                # Content
                'hashtags': extract_hashtags(info.get('description', '')),

                # Technical
                'thumbnail': info.get('thumbnail', ''),
                'video_id': info.get('id', ''),
            }

            # Calculate engagement rates
            if metadata['view_count'] > 0:
                views = metadata['view_count']
                metadata['engagement_metrics'] = {
                    'like_rate': round((metadata['like_count'] / views) * 100, 2),
                    'comment_rate': round((metadata['comment_count'] / views) * 100, 2),
                    'repost_rate': round((metadata['repost_count'] / views) * 100, 2),
                    'save_rate': round((metadata['save_count'] / views) * 100, 2),
                    'total_engagement_rate': round(
                        ((metadata['like_count'] + metadata['comment_count'] +
                          metadata['repost_count'] + metadata['save_count']) / views) * 100, 2
                    ),
                    'save_to_like_ratio': round(
                        metadata['save_count'] / metadata['like_count'], 3
                    ) if metadata['like_count'] > 0 else 0,
                    'repost_to_like_ratio': round(
                        metadata['repost_count'] / metadata['like_count'], 3
                    ) if metadata['like_count'] > 0 else 0,
                }

                # High-value content flag (reposts indicate shareability)
                metadata['high_value_content'] = (
                    metadata['engagement_metrics']['save_rate'] > 1.5 or
                    metadata['engagement_metrics']['repost_rate'] > 2.0
                )

            # Performance level with reposts considered
            views = metadata['view_count']
            repost_rate = metadata['engagement_metrics'].get('repost_rate', 0) if metadata.get('engagement_metrics') else 0

            if views >= 1000000:
                metadata['performance_level'] = 'viral'
            elif views >= 100000 or (views >= 50000 and repost_rate > 2):
                metadata['performance_level'] = 'good'
            elif views >= 10000:
                metadata['performance_level'] = 'moderate'
            else:
                metadata['performance_level'] = 'low'

            print(f"[SUCCESS] Metadata extracted:")
            print(f"  Uploader: {metadata['uploader']}")
            print(f"  Track: {metadata['track'] or 'None'}")
            print(f"  Views: {metadata['view_count']:,}")
            print(f"  Likes: {metadata['like_count']:,}")
            print(f"  Reposts: {metadata['repost_count']:,} ({metadata['engagement_metrics'].get('repost_rate', 0)}%)")
            print(f"  Comments: {metadata['comment_count']:,}")

            return metadata

    except Exception as e:
        print(f"[ERROR] Metadata extraction failed: {e}")
        return {'view_count': 0, 'error': str(e)}


def extract_hashtags(description):
    import re
    return re.findall(r'#(\w+)', description)


def analyze_save_metrics(metadata):
    """Analyze what save patterns indicate about content value."""
    save_count = metadata.get('save_count', 0)
    view_count = metadata.get('view_count', 0)

    if view_count == 0:
        return {}

    save_rate = (save_count / view_count) * 100

    analysis = {
        'save_rate': save_rate,
        'value_indicators': []
    }

    if save_rate > 2:
        analysis['value_indicators'].append("EXCEPTIONAL VALUE - Tutorial/reference content")
    elif save_rate > 1:
        analysis['value_indicators'].append("HIGH VALUE - Worth keeping for later")
    elif save_rate > 0.5:
        analysis['value_indicators'].append("MODERATE VALUE - Some reference value")

    return analysis
