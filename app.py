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
        goal = request.form.get("goal", "follower_growth").strip()
        tone = request.form.get("tone", "confident, friendly").strip()
        audience = request.form.get("audience", "creators and small business owners").strip()

        if not tiktok_url:
            return "Error: TikTok URL is required", 400

        print(f"Processing: {tiktok_url}")
        print(f"Strategy: {strategy}, Goal: {goal}")

        # --- Video processing ---
        try:
            audio_path, frames_dir, frame_paths = extract_audio_and_frames(
                tiktok_url,
                strategy=strategy,
                frames_per_minute=frames_per_minute,
                cap=cap,
                scene_threshold=scene_threshold,
            )
            print(f"Extracted {len(frame_paths)} frames")
        except Exception as e:
            print(f"Video processing error: {e}")
            return f"Error processing video: {str(e)}", 500

        # --- Transcription ---
        try:
            transcript = transcribe_audio(audio_path)
            print(f"Transcript length: {len(transcript)} chars")
        except Exception as e:
            print(f"Transcription error: {e}")
            transcript = "(Transcription failed)"

        # --- Frame analysis ---
        try:
            frames_summaries_text, gallery_data_urls = analyze_frames_batch(frame_paths)
            print(f"Frame analysis complete, gallery has {len(gallery_data_urls)} images")
        except Exception as e:
            print(f"Frame analysis error: {e}")
            frames_summaries_text = "(Frame analysis failed)"
            gallery_data_urls = []

        # --- Retrieve knowledge base context ---
        try:
            rag_query = f"Short-form content strategy.\nTranscript:\n{transcript}\n\nVisual notes:\n{frames_summaries_text}"
            knowledge_context, knowledge_citations = retrieve_all_context(max_chars=16000)
            if not knowledge_context:
                knowledge_context, knowledge_citations = retrieve_context(rag_query, top_k=12, max_chars=4000)
        except Exception as e:
            print(f"RAG retrieval error: {e}")
            knowledge_context, knowledge_citations = "", []

        # --- AI Analysis ---
        try:
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
            print("GPT analysis complete")
        except Exception as e:
            print(f"GPT analysis error: {e}")
            gpt_result = {"analysis": f"Analysis failed: {str(e)}", "hooks": []}

        # --- Ensure safe types ---
        analysis_text = gpt_result.get("analysis", "Analysis not available")
        hooks_list = gpt_result.get("hooks", [])
        if isinstance(hooks_list, str):
            hooks_list = [hooks_list]

        # --- Generate formula using analyze_tiktok.py logic ---
        try:
            from analyze_tiktok import run_analysis
            additional_analysis = run_analysis("", goal)
            formula = additional_analysis.get("formula", "")
        except Exception as e:
            print(f"Formula generation error: {e}")
            formula = "Formula generation failed"

        # --- Prepare frame summaries for template ---
        frame_summaries = []
        if frames_summaries_text:
            # Split by double newlines or by numbered frames
            blocks = frames_summaries_text.split('\n\n')
            frame_summaries = [block.strip() for block in blocks if block.strip()]
        
        # If no blocks found, try splitting by frame numbers
        if not frame_summaries and frames_summaries_text:
            frame_summaries = [frames_summaries_text]

        print("Rendering results template")
        return render_template(
            "results.html",
            tiktok_url=tiktok_url,
            creator_note=creator_note,
            transcript=transcript,
            frame_summary=frames_summaries_text,
            frame_summaries=frame_summaries,
            frame_gallery=gallery_data_urls,
            strategy=strategy,
            frames_per_minute=frames_per_minute,
            cap=cap,
            scene_threshold=scene_threshold,
            frames_count=len(frame_paths) if frame_paths else 0,
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
            hooks=hooks_list,
            gpt_response=analysis_text,
            formula=formula
        )

    except Exception as e:
        print(f"Unexpected error in process(): {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Unexpected error: {str(e)}", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
