import os
import subprocess
import base64
import ffmpeg
import whisper
from openai import OpenAI
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def run_full_analysis(tiktok_url, video_id):
    # Create paths
    video_output = f"videos/{video_id}.mp4"
    audio_output = f"audio/{video_id}.wav"
    frame_output_dir = f"static/frames/{video_id}/"
    os.makedirs(frame_output_dir, exist_ok=True)

    # Step 1: Download video using yt-dlp
    subprocess.run(["yt-dlp", "-o", video_output, tiktok_url])

    # Step 2: Extract audio from video
    ffmpeg.input(video_output).output(audio_output).run()

    # Step 3: Transcribe using Whisper
    model = whisper.load_model("base")
    transcript = model.transcribe(audio_output)["text"]

    # Step 4: Extract frames (1 every 2 seconds)
    ffmpeg.input(video_output).output(f"{frame_output_dir}/frame_%03d.jpg", vf="fps=0.5").run()
    frame_files = sorted(os.listdir(frame_output_dir))
    frame_paths = [os.path.join(frame_output_dir, f) for f in frame_files if f.endswith(".jpg")]

    # Step 5: Analyze each frame with GPT-4o
    descriptions = []
    for f in frame_paths:
        encoded = encode_image(f)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this video frame's visual hook, mood, or vibe."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}
                ]
            }],
            max_tokens=300
        )
        descriptions.append(response.choices[0].message.content)

    # Step 6: Generate full AI summary
    frame_text = "\n\n".join([f"Frame {i+1}: {desc}" for i, desc in enumerate(descriptions)])
    summary_prompt = f"Transcript:\n{transcript}\n\nVisuals:\n{frame_text}\n\nSummarize the video's hook, topic, structure, and what could improve engagement."

    summary_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": summary_prompt}],
        max_tokens=800,
    )

    return {
        "summary": summary_response.choices[0].message.content,
        "transcript": transcript,
        "frames": frame_paths
    }
