"""
Gemini video analysis — sends the full video to Gemini for native visual understanding.
Returns the same (frames_summaries_text, gallery_data_urls) format as the frame-based pipeline.
"""

import os
import time
import base64
from pathlib import Path

def is_gemini_available():
    return bool(os.environ.get('GOOGLE_API_KEY'))


def analyze_video_with_gemini(video_path, frame_paths=None, transcript=None):
    """
    Upload the full video to Gemini and get a comprehensive visual analysis.

    Args:
        video_path: Path to the video file
        frame_paths: Optional list of extracted frame paths (used for gallery thumbnails only)
        transcript: Optional transcript for context

    Returns:
        Tuple of (detailed_analysis_text, gallery_data_urls)
    """
    import google.generativeai as genai

    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set")

    genai.configure(api_key=api_key)

    # Upload video file
    print(f"[GEMINI] Uploading video: {video_path} ({os.path.getsize(video_path) / 1024 / 1024:.1f} MB)")
    video_file = genai.upload_file(path=video_path)

    # Wait for Gemini to process the video
    max_wait = 120
    waited = 0
    while video_file.state.name == "PROCESSING" and waited < max_wait:
        time.sleep(3)
        waited += 3
        video_file = genai.get_file(video_file.name)
        if waited % 15 == 0:
            print(f"[GEMINI] Still processing video... ({waited}s)")

    if video_file.state.name == "FAILED":
        raise RuntimeError(f"Gemini video processing failed: {video_file.state.name}")

    if video_file.state.name != "ACTIVE":
        raise RuntimeError(f"Gemini video not ready after {max_wait}s: {video_file.state.name}")

    print(f"[GEMINI] Video processed, running analysis...")

    transcript_context = ""
    if transcript:
        transcript_context = f"""
TRANSCRIPT (what is being said in the video):
{transcript[:3000]}
"""

    prompt = f"""You are an expert video analyst. Watch this entire video carefully and provide a comprehensive visual analysis.

{transcript_context}

Analyze the video from start to finish. For every distinct moment or scene change, describe:

1. **TIMESTAMP & SCENE**: What's happening at this point in the video (approximate seconds)

2. **TEXT ON SCREEN**: Quote ALL visible text exactly as it appears and classify each:
   - [CAPTION]: Auto-generated subtitles (usually white text at bottom)
   - [OVERLAY]: Added text for emphasis/hooks (stylized, colorful, positioned creatively)
   - [UI]: Platform interface elements (username, likes, buttons)

3. **VISUAL CONTENT**: What's shown — actions, objects, people, settings, products

4. **MOTION & PHYSICAL CUES** (especially important in the first 3 seconds):
   - Hand gestures, camera movement, body language
   - Object interactions, props, staging choices
   - Pattern interrupts or attention-grabbing movements

5. **SCENE PURPOSE**: What role this moment plays in the video narrative
   (hook, setup, demonstration, reveal, transition, CTA, etc.)

6. **ATTENTION ELEMENTS**: What grabs the viewer's eye at this moment

IMPORTANT GUIDELINES:
- Cover the ENTIRE video from first frame to last — do not skip middle sections
- Note transitions between scenes
- Pay special attention to the first 3 seconds (the hook) and the ending
- Identify the overall narrative arc and pacing
- Note any visual techniques (jump cuts, slow motion, split screen, zooms)
- Describe the visual quality, lighting, and production style

FORMAT: Organize chronologically with clear timestamps. Be thorough — this analysis
replaces frame-by-frame extraction, so every visual detail matters."""

    model = genai.GenerativeModel('gemini-2.0-flash')

    response = model.generate_content(
        [video_file, prompt],
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=4000,
            temperature=0.3
        )
    )

    analysis_text = response.text
    print(f"[GEMINI] Analysis complete: {len(analysis_text)} chars")

    # Clean up the uploaded file
    try:
        genai.delete_file(video_file.name)
        print(f"[GEMINI] Cleaned up uploaded video")
    except Exception as e:
        print(f"[GEMINI] Warning: could not delete uploaded file: {e}")

    # Build summary header
    final_output = f"""VIDEO VISUAL ANALYSIS (Gemini Full-Video):
{analysis_text}"""

    # Build gallery URLs from extracted frames (for thumbnail display)
    gallery_urls = []
    if frame_paths:
        for frame_path in frame_paths:
            try:
                with open(frame_path, 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode()
                    gallery_urls.append(f"data:image/jpeg;base64,{img_data}")
            except Exception:
                continue

    return final_output, gallery_urls
