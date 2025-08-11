import os
import subprocess
import base64
import ffmpeg
import whisper
from config import OPENAI_API_KEY
from openai import OpenAI
from PIL import Image

client = OpenAI(api_key=OPENAI_API_KEY)

def download_tiktok(url, output_dir="videos"):
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "yt-dlp",
        "-o", f"{output_dir}/%(id)s.%(ext)s",
        url
    ]
    subprocess.run(cmd)
    # Get the filename of the most recently downloaded video
    files = sorted(os.listdir(output_dir), key=lambda x: os.path.getmtime(os.path.join(output_dir, x)))
    return os.path.join(output_dir, files[-1])

def extract_audio(video_path, audio_path="audio/audio.wav"):
    os.makedirs("audio", exist_ok=True)
    ffmpeg.input(video_path).output(audio_path).run()
    return audio_path

def extract_frames(video_path, frame_dir="frames", fps=0.5):
    os.makedirs(frame_dir, exist_ok=True)
    ffmpeg.input(video_path).output(f"{frame_dir}/frame_%03d.jpg", vf=f"fps={fps}").run()
    return sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith(".jpg")])

def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result['text']

def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def analyze_frame(image_path):
    image_base64 = encode_image(image_path)
    vision_prompt = [
        {"type": "text", "text": "Analyze this video frame. What’s visually happening? Is there text? What vibe or mood does it give off?"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
    ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": vision_prompt}],
        max_tokens=500,
    )
    return response.choices[0].message.content

def summarize_video(transcript, frame_summaries):
    all_frames_summary = "\n\n".join([f"Frame {i+1}: {desc}" for i, desc in enumerate(frame_summaries)])
    prompt = f"""
You are analyzing a TikTok video based on its transcript and visuals. Summarize:
- The hook in the first few seconds
- The main topic
- The video structure
- Visual techniques used (text, expressions, movement, etc.)
- Strengths and weaknesses
- What could improve engagement?

Transcript:
{transcript}

Visuals:
{all_frames_summary}
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
    )
    return response.choices[0].message.content

# 🟢 Main Run
if __name__ == "__main__":
    tiktok_url = input("Paste TikTok video link: ").strip()
    print("\n📥 Downloading video...")
    video_path = download_tiktok(tiktok_url)

    print("🔊 Extracting audio...")
    audio_path = extract_audio(video_path)

    print("📝 Transcribing audio...")
    transcript = transcribe_audio(audio_path)
    print("\n--- Transcript ---\n", transcript)

    print("🎞️ Extracting frames (1 every 2 seconds)...")
    frame_paths = extract_frames(video_path, fps=0.5)

    print(f"🧠 Analyzing {len(frame_paths)} frames with GPT-4o...")
    frame_descriptions = []
    for i, path in enumerate(frame_paths):
        print(f"Analyzing Frame {i+1}/{len(frame_paths)}...")
        desc = analyze_frame(path)
        frame_descriptions.append(desc)

    print("🧾 Generating full summary...")
    full_summary = summarize_video(transcript, frame_descriptions)

    print("\n--- FINAL SUMMARY ---\n")
    print(full_summary)
