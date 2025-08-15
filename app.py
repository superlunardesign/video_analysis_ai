import os
import random
import time as _time
from flask import Flask, request, render_template, jsonify
from openai import OpenAI

from processing import (
    extract_audio_and_frames,
    transcribe_audio,
    analyze_frames_batch,
)
from rag_helper import retrieve_context, retrieve_all_context

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=60.0)


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


def run_gpt_analysis(transcript, frames_summaries_text, creator_note="", knowledge_context=""):
    note = f"\n\nCreator’s Note:\n{creator_note}" if creator_note else ""
    kc = f"\n\nKnowledge Context:\n{knowledge_context}" if knowledge_context else ""

    prompt = f"""
Transcript:
{transcript}

Frame-by-frame visual notes:
{frames_summaries_text}
{note}{kc}

Give an in-depth analysis of the video’s performance, sales psychology, retention, and a reusable formula.
"""

    def _call():
        return client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are TikTok Analyzer, a GPT trained to analyze short-form video."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=900,
        )

    resp = _api_retry(_call)
    return resp.choices[0].message.content


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze_async", methods=["POST"])
def analyze_async():
    # This now just shows the progress bar page
    return render_template("progress.html", form_data=request.form.to_dict())


@app.route("/process", methods=["POST"])
def process():
    form = request.form
    tiktok_url = form.get("tiktok_url", "").strip()
    creator_note = form.get("creator_note", "").strip()
    strategy = form.get("strategy", "smart").strip().lower()
    frames_per_min = int(form.get("frames_per_minute", 24))
    cap = int(form.get("cap", 60))
    scene_threshold = float(form.get("scene_threshold", 0.24))
    platform = form.get("platform", "tiktok").strip()
    target_duration = int(form.get("target_duration", 30))
    goal = form.get("goal", "follows").strip()
    tone = form.get("tone", "confident, friendly").strip()
    audience = form.get("audience", "creators and small business owners").strip()

    # --- Video Processing ---
    audio_path, frames_dir, frame_paths = extract_audio_and_frames(
        tiktok_url,
        strategy=strategy,
        frames_per_minute=frames_per_min,
        cap=cap,
        scene_threshold=scene_threshold,
    )

    transcript = transcribe_audio(audio_path)
    frames_summaries_text, gallery_data_urls = analyze_frames_batch(frame_paths)

    rag_query = f"Short-form content strategy.\nTranscript:\n{transcript}\n\nVisual notes:\n{frames_summaries_text}"
    knowledge_context, knowledge_citations = retrieve_all_context(max_chars=16000)
    if not knowledge_context:
        knowledge_context, knowledge_citations = retrieve_context(rag_query, top_k=12, max_chars=4000)

    gpt_response = run_gpt_analysis(transcript, frames_summaries_text, creator_note, knowledge_context)

    return render_template(
        "results.html",
        tiktok_url=tiktok_url,
        creator_note=creator_note,
        transcript=transcript,
        frame_summary=frames_summaries_text,
        frame_gallery=gallery_data_urls,
        gpt_response=gpt_response,
        strategy=strategy,
        frames_per_minute=frames_per_min,
        cap=cap,
        scene_threshold=scene_threshold,
        platform=platform,
        target_duration=target_duration,
        goal=goal,
        tone=tone,
        audience=audience,
        knowledge_citations=knowledge_citations,
        knowledge_context=knowledge_context,
        frames_dir=frames_dir,
        frame_paths=frame_paths,
        video_title=video_title,
    analysis=analysis_data.get("analysis", ""),
    hooks=analysis_data.get("hooks", []),  # ✅ Never None
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
