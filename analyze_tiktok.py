import os
import subprocess
import base64
import random  # Added missing import
import ffmpeg
import whisper
from openai import OpenAI
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

def run_analysis(filepath, goal):  # Fixed indentation
    # Pretend we read the transcript / visuals from the KB or ML model
    sample_hooks = [
        "how i accidentally became a 12yo scammer",
        "good morning, we're making sourdough",
        "I quit my job with zero backup plan",
    ]

    chosen_hook = random.choice(sample_hooks)

    hooks_output = {
        "text_hooks": [chosen_hook],
        "visual_hooks": ["Prolonged eye contact", "Unusual action (sliding sourdough into frame)"],
        "verbal_hooks": ["Story-opening with curiosity ('I've been coding since I was 12')"]
    }

    # Why it worked / how it can improve
    analysis_text = f"""
    **Hook Analysis**
    - **Text Hook:** "{chosen_hook}" — Captures curiosity by combining something unusual with a personal twist.
    - **Promise:** The video sets up a clear expectation early — the viewer knows they'll hear the full story or see the end result.
    - **Delivery:** The payoff is delayed until the end, keeping watch time high.
    - **Retention Drivers:** Storytelling, escalating intrigue, and engagement bait (eye contact, humor).

    **Improvement Suggestion:**
    - For stronger {goal.lower()}, integrate micro-promises mid-video (small reveals) while still saving the final payoff for the end.
    - Consider adding a visual/onscreen text cue halfway to re-hook skimmers.
    """

    # Goal-specific formula
    if goal == "viral_reach":  # Updated to match form values
        formula = """
        1. Open with a curiosity-driven hook.
        2. Set the "promise" in first 3 seconds.
        3. Deliver story/visuals with mini-escalations.
        4. Save final reveal/payoff for last 10%.
        5. End with share/comment trigger.
        """
    elif goal == "follower_growth":  # Updated to match form values
        formula = """
        1. Hook with something your niche audience instantly relates to.
        2. State the promise of value they'll get from following you.
        3. Deliver a short, engaging story or tip.
        4. End with clear call-to-follow for more content like this.
        """
    else:  # sales_conversions
        formula = """
        1. Hook with a problem your product/service solves.
        2. Promise the solution or transformation.
        3. Show proof/results.
        4. End with call-to-action to DM/click/buy.
        """

    return {
        "hooks": hooks_output,
        "analysis": analysis_text.strip(),
        "formula": formula.strip()
    }

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