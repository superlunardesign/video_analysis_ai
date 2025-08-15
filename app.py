import os
import random
import time as _time
import json
from flask import Flask, request, render_template
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


def run_gpt_analysis(transcript_text, frames_summaries_text, creator_note, platform, target_duration, goal, tone, audience, knowledge_context=""):
    """
    Sends transcript_text and visuals to GPT and returns a dict with:
      - "analysis": full analysis text from GPT
      - "hooks": list of extracted hooks
    """

    prompt = f"""
You are analyzing a social media video for performance improvement.

Transcript:
{transcript_text}

Frame-by-frame visual notes:
{frames_summaries_text}

Creator note: {creator_note}
Platform: {platform}
Target duration: {target_duration}
Goal: {goal}
Tone: {tone}
Audience: {audience}

Knowledge context for reference:
{knowledge_context}

Provide a detailed analysis including:
- Strengths in hook, delivery, structure
- Weaknesses and missed opportunities
- Suggestions for improvement
- Timing notes for pacing/retention
- Any examples of how to reword/improve specific lines

Then separately provide 5 strong alternative hooks for the same content.

Respond in **valid JSON** with exactly two keys:
- "analysis": string
- "hooks": array of strings
    """

    gpt_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    try:
        parsed = json.loads(gpt_response.choices[0].message.content)
        return {
            "analysis": parsed.get("analysis", ""),
            "hooks": parsed.get("hooks", [])
        }
    except (json.JSONDecodeError, KeyError):
        gpt_text = gpt_response.choices[0].message.content
        return {
            "analysis": gpt_text,
            "hooks": []
        }


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze_async", methods=["POST"])
def analyze_async():
    return render_template("progress.html", form_data=request.form.to_dict())


@app.route("/process", methods=["POST"])
def process():
    try:
        # --- Form data ---
        tiktok_url = request.form.get("tiktok_url", "").strip()
        creator_note = request.form.get("creator_note", "").strip()
        strategy = request.form.get("strategy", "smart").strip()
        frames_per_minute = int(request.form.get("frames_per_minute", 24))
        cap = int(request.form.get("cap", 60))
        scene_threshold = float(request.form.get("scene_threshold", 0.24))
        platform = request.form.get("platform", "tiktok").strip()
        target_duration = request.form.get("target_duration", "30").strip()
        goal = request.form.get("goal", "follows").strip()
        tone = request.form.get("tone", "confident, friendly").strip()
        audience = request.form.get("audience", "creators and small business owners").strip()

        # --- Video processing ---
        audio_path, frames_dir, frame_paths = extract_audio_and_frames(
            tiktok_url,
            strategy=strategy,
            frames_per_minute=frames_per_minute,
            cap=cap,
            scene_threshold=scene_threshold,
        )

        transcript = transcribe_audio(audio_path)
        frames_summaries_text, gallery_data_urls = analyze_frames_batch(frame_paths)

        # --- Retrieve knowledge base context ---
        rag_query = f"Short-form content strategy.\nTranscript:\n{transcript}\n\nVisual notes:\n{frames_summaries_text}"
        knowledge_context, knowledge_citations = retrieve_all_context(max_chars=16000)
        if not knowledge_context:
            knowledge_context, knowledge_citations = retrieve_context(rag_query, top_k=12, max_chars=4000)

        # --- AI Analysis ---
        gpt_result = run_gpt_analysis(
            transcript,
            frames_summaries_text,
            creator_note,
            platform,
            target_duration,
            goal,
            tone,
            audience,
            knowledge_context
        )

        # --- Ensure safe types ---
        analysis_text = gpt_result.get("analysis", "")
        hooks_list = gpt_result.get("hooks", [])
        if isinstance(hooks_list, str):
            hooks_list = [hooks_list]

        return render_template(
            "results.html",
            tiktok_url=tiktok_url,
            creator_note=creator_note,
            transcript=transcript,
            frame_summary=frames_summaries_text,
            frame_gallery=gallery_data_urls,
            strategy=strategy,
            frames_per_minute=frames_per_minute,
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
            analysis=analysis_text,
            hooks=hooks_list
        )

    except Exception as e:
        return f"Error in process(): {str(e)}", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
