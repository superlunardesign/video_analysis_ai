import os
import cv2
import tempfile
import uuid
import openai
import subprocess
from pytube import YouTube
from moviepy.editor import VideoFileClip

# Ensure frames output folder exists
FRAMES_DIR = os.path.join("static", "frames")
os.makedirs(FRAMES_DIR, exist_ok=True)

openai.api_key = os.getenv("OPENAI_API_KEY")

def download_tiktok_or_youtube(url, output_path):
    """Downloads TikTok/YouTube video to output_path."""
    if "tiktok.com" in url or "youtube.com" in url or "youtu.be" in url:
        yt = YouTube(url)
        stream = yt.streams.filter(file_extension="mp4").order_by("resolution").desc().first()
        stream.download(filename=output_path)
    else:
        raise ValueError("Unsupported URL provided.")

def extract_frames(video_path, frames_per_minute=6):
    """Extract frames from video at a set interval."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * (60 / frames_per_minute))

    frame_paths = []
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            frame_filename = f"{uuid.uuid4().hex}.jpg"
            frame_path = os.path.join(FRAMES_DIR, frame_filename)
            cv2.imwrite(frame_path, frame)
            frame_paths.append(os.path.join("static", "frames", frame_filename))
            saved_count += 1
        frame_count += 1

    cap.release()
    return frame_paths

def extract_audio_transcript(video_path):
    """Extract audio from video and get transcript from OpenAI."""
    temp_audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path, "-q:a", "0", "-map", "a", temp_audio
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    with open(temp_audio, "rb") as audio_file:
        transcript = openai.Audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=audio_file
        )
    os.remove(temp_audio)
    return transcript.text.strip()

def analyze_content_with_gpt(transcript, goal):
    """Analyze transcript + provide hooks, formulas, and holistic score."""
    prompt = f"""
You are an expert in TikTok/Reels/Shorts video performance.
Analyze the following transcript and content for:
1. Hooks detected
2. Retention patterns/formulas
3. Holistic performance score (0-100)
4. Concise written analysis (max 200 words)

Transcript:
{transcript}

Goal of video: {goal}

    Please:
    1. Describe clearly what happens in the video from start to finish.
    2. Identify and categorize hooks: Text, Visual, Verbal.
    3. Break down the content structure into a repeatable 'formula'.
    4. Explain why this formula might work for {goal} (viral reach, follower growth, or sales).
    5. Give actionable insights for someone adapting this style for their own niche.
    6. Assign a score (0–100) based on hook strength, pacing, structure, and clarity.
    7. Output in JSON with keys:
       description, hooks, formula, success_reasoning, adaptation_tips, score
"""
    response = openai.Chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={ "type": "json_object" }
    )

    data = response.choices[0].message.parsed
    return data

def analyze_tiktok_video(input_source, local_file=False, goal="General Analysis"):
    """Main function to analyze TikTok/YouTube video or local file."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
        if local_file:
            temp_video_path = input_source
        else:
            temp_video_path = temp_video.name
            download_tiktok_or_youtube(input_source, temp_video_path)

        # Extract transcript
        transcript = extract_audio_transcript(temp_video_path)

        # Extract frames
        frames = extract_frames(temp_video_path, frames_per_minute=6)

        # Run GPT analysis
        gpt_results = analyze_content_with_gpt(transcript, goal)

        return {
            "video_url": None if local_file else input_source,
            "transcript": transcript,
            "overall_score": gpt_results.get("score"),
            "hooks": gpt_results.get("hooks", []),
            "formulas": gpt_results.get("formulas", []),
            "frames": frames,
            "analysis": gpt_results.get("analysis")
        }
