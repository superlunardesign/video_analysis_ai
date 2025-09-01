import os
import ffmpeg
import whisper
import base64
from PIL import Image
from config import OPENAI_API_KEY
from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY)

video_path = "videos/your_video.mp4"
audio_path = "audio/audio.wav"
frame_path = "frames/frame_001.jpg"

# Make folders
os.makedirs("audio", exist_ok=True)
os.makedirs("frames", exist_ok=True)

# Step 1: Extract audio
ffmpeg.input(video_path).output(audio_path).run()

# Step 2: Extract 1 frame at 3 seconds
ffmpeg.input(video_path, ss=3).output(frame_path, vframes=1).run()

# Step 3: Transcribe audio with Whisper
model = whisper.load_model("base")
result = model.transcribe(audio_path)
transcript = result['text']
print("\n--- Transcript ---")
print(transcript)

# Step 4: Analyze frame with GPT-4o (Vision)
def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

image_base64 = encode_image(frame_path)

vision_prompt = [
    {"type": "text", "text": "Analyze this frame from a TikTok-style video. What do you notice? Is there text on screen? What mood or vibe does it give off?"},
    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": vision_prompt}],
    max_tokens=500,
)

visual_description = response.choices[0].message.content
print("\n--- Visual Analysis ---")
print(visual_description)

# Step 5: Summarize full content using GPT-4o
summary_prompt = f"""
You’re analyzing a short-form social video. Based on the transcript and this visual analysis, tell me:
- What are the hooks in the first 3 seconds?
- What’s the topic?
- What’s the content structure?
- Is there a CTA or pattern worth replicating?
- If the text on screen differs from the script, it is text hooks or added text. if it is the same as the transcript, it is captions.
- Does the transcript compare to songs, viral speech/sounds, or does it seem like original speech?

Transcript:
{transcript}

Visual Analysis:
{visual_description}
"""

summary_response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": summary_prompt}],
    max_tokens=500,
)

summary = summary_response.choices[0].message.content
print("\n--- AI Summary ---")
print(summary)
