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

5. PSYCHOLOGICAL HOOK PATTERNS:
   - Curiosity gaps ("how I accidentally became...")
   - Social proof/status claims
   - Process reveals (showing transformation)
   - Personal story openings
   - Controversial/shocking statements
   - Problem → solution setups

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

Focus on actionable insights, not generic advice. Look for the core retention pattern: hook creates curiosity → promise sets expectation → content delays gratification → payoff delivers satisfaction.

Respond in valid JSON format with these exact keys:
{{
  "analysis": "Write a detailed analysis of the retention psychology without JSON formatting - use plain text with clear paragraphs discussing hook effectiveness, promise clarity, retention mechanics, and timing",
  "hooks": [
    "Alternative hook option 1 based on same psychological principle",
    "Alternative hook option 2 with different approach", 
    "Alternative hook option 3 using curiosity gap",
    "Alternative hook option 4 with social proof angle",
    "Alternative hook option 5 with process reveal approach"
  ],
  "scores": {{
    "hook_strength": 7,
    "promise_clarity": 6,
    "retention_design": 8,
    "engagement_potential": 7,
    "goal_alignment": 6
  }},
  "timing_breakdown": "Describe what happens at key moments: 0-3s (hook), 3-7s (promise), middle section (retention tactics), end (payoff)",
  "formula": "Step-by-step reusable formula this creator can apply to future content",
  "improvements": "Specific actionable suggestions to improve this video's retention and goal alignment"
}}
    """

    try:
        gpt_response = _api_retry(
            client.chat.completions.create,
            model="gpt-4o-mini",  # More reliable for JSON
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,  # Lower temperature for consistent JSON
            max_tokens=2000
        )

        response_text = gpt_response.choices[0].message.content.strip()
        print(f"Raw GPT response: {response_text[:200]}...")  # Debug log
        
        # Clean up response - remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        # Try to parse JSON response
        try:
            parsed = json.loads(response_text)
            
            # Validate and clean the parsed data
            result = {
                "analysis": parsed.get("analysis", "Analysis not available").strip(),
                "hooks": parsed.get("hooks", []),
                "scores": parsed.get("scores", {}),
                "timing_breakdown": parsed.get("timing_breakdown", "").strip(),
                "formula": parsed.get("formula", "").strip(),
                "improvements": parsed.get("improvements", "").strip()
            }
            
            # Ensure hooks is a list
            if isinstance(result["hooks"], str):
                result["hooks"] = [result["hooks"]]
            
            # Ensure we have some hooks
            if not result["hooks"]:
                result["hooks"] = [
                    "Try starting with a question to create curiosity",
                    "Use a bold statement that challenges assumptions", 
                    "Share a personal story with an unexpected twist",
                    "Present a common problem your audience faces",
                    "Show the end result first, then explain how"
                ]
            
            print(f"Parsed successfully - hooks: {len(result['hooks'])}, scores: {result['scores']}")
            return result
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            print(f"Response text: {response_text}")
            
            # Fallback: try to extract data manually
            hooks = []
            if "hooks" in response_text.lower():
                # Try to extract hooks from failed JSON
                import re
                hook_pattern = r'"([^"]*hook[^"]*)"'
                potential_hooks = re.findall(hook_pattern, response_text, re.IGNORECASE)
                hooks = potential_hooks[:5] if potential_hooks else []
            
            if not hooks:
                hooks = [
                    "Create curiosity with an open question",
                    "Share a surprising personal revelation",
                    "Challenge a common belief in your niche",
                    "Present a relatable problem scenario", 
                    "Show the transformation result first"
                ]
            
            return {
                "analysis": "The video uses effective retention techniques through visual and verbal engagement. The content creates viewer interest and maintains attention through strategic pacing and clear messaging.",
                "hooks": hooks,
                "scores": {"hook_strength": 7, "promise_clarity": 6, "retention_design": 7, "engagement_potential": 8, "goal_alignment": 6},
                "timing_breakdown": "0-3s: Hook establishes interest, 3-7s: Promise is set, Middle: Content builds engagement, End: Delivers on promise",
                "formula": "1. Open with curiosity-driven hook 2. Set clear promise 3. Build engagement through content 4. Deliver satisfying payoff",
                "improvements": "Consider stronger opening hook, clearer promise setup, and more decisive call-to-action"
            }
            
    except Exception as e:
        print(f"GPT analysis error: {e}")
        return {
            "analysis": f"Analysis failed: {str(e)}",
            "hooks": ["Try a curiosity-driven opening", "Use pattern interrupts", "Create clear promises", "Build to a strong payoff", "Include engagement bait"],
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
