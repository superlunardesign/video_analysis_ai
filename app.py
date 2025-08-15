# app.py — lean version: always uses local RAG + gpt-4o (no custom GPT)
import uuid
import threading
from flask import jsonify, redirect, url_for, render_template
from progress import (
    start as prog_start, set_progress, set_error, set_result,
    get as prog_get, pop_result
)

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid OpenMP dup warnings on macOS

from flask import Flask, request, render_template
import random
import time as _time

from openai import OpenAI
from config import OPENAI_API_KEY

# Video/audio/frames pipeline you already have
from processing import (
    extract_audio_and_frames,     # -> (audio_path, frames_dir, [frame_paths])
    transcribe_audio,             # -> transcript string (with retry/local fallback)
    analyze_frames_batch,         # -> (frames_summaries_text, [data:image/jpeg;base64,...])
)

# Local RAG helpers (vector top-k + all-knowledge mode)
from rag_helper import retrieve_context, retrieve_all_context

app = Flask(__name__)

# One client; longer timeout helps on heavier calls
client = OpenAI(api_key=OPENAI_API_KEY, timeout=60.0)


# -------------------------
# Small retry for hiccups
# -------------------------
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


# -------------------------
# GPT helpers (always gpt-4o + knowledge context)
# -------------------------
def run_gpt_analysis(transcript: str, frames_summaries_text: str, creator_note: str = "", knowledge_context: str = "") -> str:
    note = f"\n\nCreator’s Note:\n{creator_note}" if creator_note else ""
    kc   = f"\n\nKnowledge Context (authoritative; prefer these facts; cite bracket numbers like [1] when used):\n{knowledge_context}" if knowledge_context else ""

    prompt = f"""
You are analyzing a short-form video using transcript + multiple visual snapshots.

Transcript:
{transcript}

Frame-by-frame visual notes:
{frames_summaries_text}
{note}{kc}

Return an in depth, conversational, and explanation driven analysis covering:
Why this generated the results shared in the content notes. What about it likely is the reason for any noted views, shares, website visits, project inquiries, sales, or virality. Use Sales psychology and everything you know about viral videos 

An analysis of the video overall and why it was or wasn't successful depending on what is revealed in the creator notes. Then share an in-depth formula for the video that could be used for any topic. Include tips that the video uses to generate engagement, viewer retention, and conversion to website visits or purchases or simply why its viral.

Suggest what this formula might be most useful for. Views, engagement, follows, sales, etc. 
"""

    def _call():
        return client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are TikTok Analyzer, a GPT trained to analyze short-form video and give strategic feedback."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=900,
        )

    resp = _api_retry(_call)
    return resp.choices[0].message.content


def generate_improved_script(
    transcript: str,
    frames_summaries_text: str,
    platform: str = "tiktok",
    target_duration: int = 30,
    goal: str = "follows",
    tone: str = "same as video",
    audience: str = "same as video",
    knowledge_context: str = ""
) -> str:
    kc = f"\n\nKnowledge Context (authoritative; prefer these facts; cite bracket numbers like [1] when used):\n{knowledge_context}" if knowledge_context else ""

    prompt = f"""

"""

    def _call():
        return client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a high-performing short-form video scriptwriter and strategist."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1200,
        )

    resp = _api_retry(_call)
    return resp.choices[0].message.content

def _run_pipeline_job(job_id, tiktok_url, creator_note, strategy, frames_per_min, cap, scene_threshold,
                      platform, target_duration, goal, tone, audience):
    try:
        set_progress(job_id, "Downloading video…", 8)
        audio_path, frames_dir, frame_paths = extract_audio_and_frames(
            tiktok_url,
            strategy=strategy,
            frames_per_minute=frames_per_min,
            cap=cap,
            scene_threshold=scene_threshold,
        )

        set_progress(job_id, "Transcribing audio…", 30)
        transcript = transcribe_audio(audio_path)

        set_progress(job_id, "Analyzing frames…", 55)
        frames_summaries_text, gallery_data_urls = analyze_frames_batch(frame_paths)

        set_progress(job_id, "Loading knowledge…", 68)
        rag_query = f"Short-form content strategy for TikTok/Reels.\nTranscript:\n{transcript}\n\nVisual notes:\n{frames_summaries_text}"
        from rag_helper import retrieve_context, retrieve_all_context
        knowledge_context, knowledge_citations = retrieve_all_context(max_chars=16000)
        if not knowledge_context:
            knowledge_context, knowledge_citations = retrieve_context(rag_query, top_k=12, max_chars=4000)

        set_progress(job_id, "Generating analysis…", 78)
        gpt_response = run_gpt_analysis(
            transcript,
            frames_summaries_text,
            creator_note=creator_note,
            knowledge_context=knowledge_context,
        )

        set_progress(job_id, "Drafting improved script…", 90)
        improved_script = generate_improved_script(
            transcript,
            frames_summaries_text,
            platform=platform,
            target_duration=target_duration,
            goal=goal,
            tone=tone,
            audience=audience,
            knowledge_context=knowledge_context,
        )

        # Pack the same payload your results.html expects
        result_payload = dict(
            tiktok_url=tiktok_url,
            creator_note=creator_note,
            transcript=transcript,
            frame_summary=frames_summaries_text,
            frame_gallery=gallery_data_urls,
            gpt_response=gpt_response,
            improved_script=improved_script,
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
        )

        set_progress(job_id, "Finalizing…", 98)
        set_result(job_id, result_payload)

    except Exception as e:
        set_error(job_id, str(e))

# -------------------------
# Routes
# -------------------------
@app.route("/", methods=["GET"])
def index():
    """
    Your index.html should have fields:
      - tiktok_url
      - creator_note
      - strategy ['smart'|'uniform']
      - frames_per_minute (uniform)
      - cap (max frames)
      - scene_threshold (smart)
      - platform, target_duration, goal, tone, audience (for script suggestions)
    """
    return render_template("index.html")

@app.route("/analyze_async", methods=["POST"])
def analyze_async():
    # Collect form values
    tiktok_url = request.form.get("tiktok_url", "").strip()
    if not tiktok_url:
        return "Please provide a TikTok/Reels/Shorts URL.", 400

    # Optional fields you already use
    creator_note     = request.form.get("creator_note", "").strip()
    strategy         = (request.form.get("strategy", "smart") or "smart").strip().lower()
    frames_per_min   = int(request.form.get("frames_per_minute", "24") or 24)
    cap              = int(request.form.get("cap", "60") or 60)
    scene_threshold  = float(request.form.get("scene_threshold", "0.24") or 0.24)
    platform         = (request.form.get("platform", "tiktok") or "tiktok").strip()
    target_duration  = int(request.form.get("target_duration", "30") or 30)
    goal             = (request.form.get("goal", "follows") or "follows").strip()
    tone             = (request.form.get("tone", "confident, friendly") or "confident, friendly").strip()
    audience         = (request.form.get("audience", "creators and small business owners") or "creators and small business owners").strip()

    job_id = uuid.uuid4().hex
    prog_start(job_id)
    set_progress(job_id, "Starting…", 2)

    # Kick off a background thread
    args = (job_id, tiktok_url, creator_note, strategy, frames_per_min, cap, scene_threshold,
            platform, target_duration, goal, tone, audience)
    threading.Thread(target=_run_pipeline_job, args=args, daemon=True).start()

    # Show progress page
    return redirect(url_for("progress_page", job_id=job_id))

@app.route("/status/<job_id>")
def status(job_id):
    data = prog_get(job_id)
    if not data:
        return jsonify({"error": "unknown job"}), 404
    return jsonify({
        "stage": data["stage"],
        "percent": data["percent"],
        "done": data["done"],
        "error": data["error"],
    })

@app.route("/progress/<job_id>")
def progress_page(job_id):
    return render_template("progress.html", job_id=job_id)

@app.route("/analyze", methods=["POST"])
def analyze():
    # Required
    tiktok_url = request.form.get("tiktok_url", "").strip()
    if not tiktok_url:
        return "Please provide a TikTok/Reels/Shorts URL.", 400
    
    # add this route in app.py (anywhere after your other @app.route defs)
@app.route("/results_async/<job_id>")
def results_async(job_id):
    data = prog_get(job_id)
    if not data:
        return "Unknown job", 404
    if data.get("error"):
        return f"Job failed: {data['error']}", 500
    if not data.get("done"):
        # Not finished yet -> send user back to the progress page
        return redirect(url_for("progress_page", job_id=job_id))

    # Get the saved result payload (what results.html expects)
    result = pop_result(job_id) or data.get("result")
    if not result:
        return "No result found for this job.", 500

    return render_template("results.html", **result)


    # Optional UI parameters
    creator_note     = request.form.get("creator_note", "").strip()
    strategy         = (request.form.get("strategy", "smart") or "smart").strip().lower()
    frames_per_min   = int(request.form.get("frames_per_minute", "24") or 24)
    cap              = int(request.form.get("cap", "60") or 60)
    scene_threshold  = float(request.form.get("scene_threshold", "0.24") or 0.24)

    platform         = (request.form.get("platform", "tiktok") or "tiktok").strip()
    target_duration  = int(request.form.get("target_duration", "30") or 30)
    goal             = (request.form.get("goal", "follows") or "follows").strip()
    tone             = (request.form.get("tone", "confident, friendly") or "confident, friendly").strip()
    audience         = (request.form.get("audience", "creators and small business owners") or "creators and small business owners").strip()

    # 1) Download + extract audio + frames (smart/uniform handled in processing.py)
    audio_path, frames_dir, frame_paths = extract_audio_and_frames(
        tiktok_url,
        strategy=strategy,
        frames_per_minute=frames_per_min,
        cap=cap,
        scene_threshold=scene_threshold,
    )

    # 2) Transcribe audio (robust)
    transcript = transcribe_audio(audio_path)

    # 3) Analyze frames in a single vision pass
    frames_summaries_text, gallery_data_urls = analyze_frames_batch(frame_paths)

    # 4) Retrieve ALL local knowledge (concat) with fallback to top-k if index missing
    rag_query = f"Short-form content strategy for TikTok/Reels.\nTranscript:\n{transcript}\n\nVisual notes:\n{frames_summaries_text}"
    knowledge_context, knowledge_citations = retrieve_all_context(max_chars=16000)
    if not knowledge_context:
        knowledge_context, knowledge_citations = retrieve_context(rag_query, top_k=12, max_chars=4000)

    # 5) GPT: analysis + improved script (always gpt-4o + knowledge context)
    gpt_response = run_gpt_analysis(
        transcript,
        frames_summaries_text,
        creator_note=creator_note,
        knowledge_context=knowledge_context,
    )

    improved_script = generate_improved_script(
        transcript,
        frames_summaries_text,
        platform=platform,
        target_duration=target_duration,
        goal=goal,
        tone=tone,
        audience=audience,
        knowledge_context=knowledge_context,
    )

    # 6) Render
    return render_template(
        "results.html",
        tiktok_url=tiktok_url,
        creator_note=creator_note,

        transcript=transcript,
        frame_summary=frames_summaries_text,
        frame_gallery=gallery_data_urls,    # list of data:image/jpeg;base64,...

        gpt_response=gpt_response,
        improved_script=improved_script,

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
        knowledge_context=knowledge_context,  # optional to display in <details>
        frames_dir=frames_dir,                # optional debug
        frame_paths=frame_paths,              # optional debug
    )


if __name__ == "__main__":
    app.run(host="10.0.0.145", port=5000, debug=True)
