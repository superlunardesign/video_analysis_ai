"""
Gemini video analysis — sends the full video to Gemini for native visual understanding.
Returns the same (frames_summaries_text, gallery_data_urls) format as the frame-based pipeline.

Uses the google-genai package (successor to the deprecated google-generativeai).
"""

import os
import time
import base64


def is_gemini_available():
    return bool(os.environ.get('GOOGLE_API_KEY'))


def analyze_video_with_gemini(video_path, frame_paths=None, transcript=None):
    """
    Upload the full video to Gemini and get a comprehensive visual analysis.

    Returns:
        Tuple of (detailed_analysis_text, gallery_data_urls)
    """
    from google import genai
    from google.genai import types

    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set")

    client = genai.Client(api_key=api_key)

    # Upload video file
    file_size_mb = os.path.getsize(video_path) / 1024 / 1024
    print(f"[GEMINI] Uploading video: {video_path} ({file_size_mb:.1f} MB)")

    video_file = client.files.upload(file=video_path)

    # Wait for Gemini to process the video
    max_wait = 120
    waited = 0
    while video_file.state == "PROCESSING" and waited < max_wait:
        time.sleep(3)
        waited += 3
        video_file = client.files.get(name=video_file.name)
        if waited % 15 == 0:
            print(f"[GEMINI] Still processing video... ({waited}s)")

    if video_file.state == "FAILED":
        raise RuntimeError(f"Gemini video processing failed")

    if video_file.state not in ("ACTIVE", "active"):
        # Also check string representation for newer SDK versions
        state_str = str(video_file.state)
        if "ACTIVE" not in state_str.upper():
            raise RuntimeError(f"Gemini video not ready after {max_wait}s: {video_file.state}")

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

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=[
            types.Content(
                parts=[
                    types.Part.from_uri(
                        file_uri=video_file.uri,
                        mime_type=video_file.mime_type,
                    ),
                    types.Part.from_text(text=prompt),
                ]
            )
        ],
        config=types.GenerateContentConfig(
            max_output_tokens=4000,
            temperature=0.3,
        ),
    )

    analysis_text = response.text
    print(f"[GEMINI] Analysis complete: {len(analysis_text)} chars")

    # Clean up the uploaded file
    try:
        client.files.delete(name=video_file.name)
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
