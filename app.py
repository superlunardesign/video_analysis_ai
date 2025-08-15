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
    Analyze video using retention psychology framework focusing on hooks, promises, and payoff timing.
    """

    prompt = f"""
You are an expert TikTok/short-form content strategist analyzing videos for retention psychology and engagement mechanics.

TRANSCRIPT:
{transcript_text}

VISUAL FRAMES:
{frames_summaries_text}

CREATOR NOTE: {creator_note}
PLATFORM: {platform} | GOAL: {goal} | DURATION: {target_duration}s

ANALYSIS FRAMEWORK:
Use this retention psychology framework to analyze the video:

1. HOOK ANALYSIS (0-3 seconds):
   - Text hooks (on-screen text, captions)
   - Verbal hooks (opening words/statements)  
   - Visual hooks (eye contact, unusual actions, pattern interrupts)
   - Rate hook strength: Does it create curiosity/intrigue?

2. PROMISE IDENTIFICATION (3-7 seconds):
   - What does the video promise to deliver?
   - How quickly does it reinforce the hook?
   - Does it create expectation for a payoff?

3. RETENTION MECHANICS:
   - Story structure: Does it delay gratification?
   - Engagement bait: Eye contact, comments-driving elements
   - Pacing: Are there mini-revelations to maintain interest?
   - Pattern interrupts: Unexpected elements that reset attention

4. PAYOFF TIMING:
   - When is the main promise delivered?
   - Does it save the best for last?
   - Is there a satisfying conclusion?

5. PSYCHOLOGICAL HOOKS EXAMPLES:
   - Curiosity gaps ("how I accidentally became...")
   - Social proof/status ("12yo scammer")
   - Process reveal (sourdough making)
   - Personal story openings
   - Controversial/shocking statements

SCORING (1-10 scale):
- Hook Strength: How compelling is the opening?
- Promise Clarity: How clear is the expected payoff?
- Retention Design: How well structured for full watch-through?
- Engagement Potential: Will it drive comments/shares?
- Goal Alignment: How well does it serve the stated goal ({goal})?

DELIVERABLES:
1. Detailed breakdown of what makes this video work/not work
2. Specific timing analysis (what happens when)
3. 5 alternative hooks following the same psychological principles
4. A reusable formula this creator can apply to future content
5. Goal-specific improvements for {goal}

Focus on actionable insights, not generic advice. Reference the sourdough creator example (promise → process → payoff) and phishing story structure (hook → story building → delayed revelation).

Respond in valid JSON:
{{
  "analysis": "detailed analysis string",
  "hooks": ["hook1", "hook2", "hook3", "hook4", "hook5"],
  "scores": {{
    "hook_strength": 0-10,
    "promise_clarity": 0-10, 
    "retention_design": 0-10,
    "engagement_potential": 0-10,
    "goal_alignment": 0-10
  }},
  "timing_breakdown": "when key moments happen",
  "formula": "reusable step-by-step formula",
  "improvements": "specific suggestions for this video"
}}
    """

    try:
        gpt_response = _api_retry(
            client.chat.completions.create,
            model="gpt-4o",  # Use GPT-4o for better analysis
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # Lower temperature for more consistent analysis
            max_tokens=1500
        )

        response_text = gpt_response.choices[0].message.content
        
        # Try to parse JSON response
        try:
            parsed = json.loads(response_text)
            return {
                "analysis": parsed.get("analysis", ""),
                "hooks": parsed.get("hooks", []),
                "scores": parsed.get("scores", {}),
                "timing_breakdown": parsed.get("timing_breakdown", ""),
                "formula": parsed.get("formula", ""),
                "improvements": parsed.get("improvements", "")
            }
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "analysis": response_text,
                "hooks": [],
                "scores": {},
                "timing_breakdown": "",
                "formula": "",
                "improvements": ""
            }
            
    except Exception as e:
        print(f"GPT analysis error: {e}")
        return {
            "analysis": f"Analysis failed: {str(e)}",
            "hooks": [],
            "scores": {},
            "timing_breakdown": "",
            "formula": "",
            "improvements": ""
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

        # --- Skip RAG for now to focus on retention analysis ---
        # We can add back knowledge context later if needed
        knowledge_context = ""
        knowledge_citations = []

        # --- Retention-focused AI Analysis ---
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
            print("Retention analysis complete")
        except Exception as e:
            print(f"GPT analysis error: {e}")
            gpt_result = {
                "analysis": f"Analysis failed: {str(e)}", 
                "hooks": [],
                "scores": {},
                "timing_breakdown": "",
                "formula": "",
                "improvements": ""
            }

        # --- Extract results ---
        analysis_text = gpt_result.get("analysis", "Analysis not available")
        hooks_list = gpt_result.get("hooks", [])
        scores = gpt_result.get("scores", {})
        timing_breakdown = gpt_result.get("timing_breakdown", "")
        formula = gpt_result.get("formula", "")
        improvements = gpt_result.get("improvements", "")
        
        if isinstance(hooks_list, str):
            hooks_list = [hooks_list]

        # --- Prepare frame summaries for template ---
        frame_summaries = []
        if frames_summaries_text:
            blocks = frames_summaries_text.split('\n\n')
            frame_summaries = [block.strip() for block in blocks if block.strip()]
        
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
            scores=scores,
            timing_breakdown=timing_breakdown,
            formula=formula,
            improvements=improvements,
            # Keep these for backward compatibility
            gpt_response=analysis_text
        )

    except Exception as e:
        print(f"Unexpected error in process(): {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Unexpected error: {str(e)}", 500
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
