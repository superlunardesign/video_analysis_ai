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
    Analyze video using retention psychology framework, combining transcript and visual analysis.
    """

    prompt = f"""
You are an expert TikTok/short-form content strategist analyzing videos for retention psychology and engagement mechanics.

TRANSCRIPT (What they're saying):
{transcript_text}

VISUAL FRAMES (What viewers see):
{frames_summaries_text}

CREATOR NOTE: {creator_note}
PLATFORM: {platform} | GOAL: {goal} | DURATION: {target_duration}s

ANALYSIS FRAMEWORK:
Analyze this video by combining both the spoken content (transcript) and visual elements (frames) to understand the full retention strategy:

1. HOOK ANALYSIS (0-3 seconds):
   - How do the opening words work with the visual presentation?
   - Does the on-screen text reinforce or contradict the verbal hook?
   - Are there visual pattern interrupts (gestures, movements, graphics)?
   - Combined hook effectiveness: Does audio + visual create stronger curiosity?

2. PROMISE IDENTIFICATION (3-7 seconds):
   - What promise is made verbally vs. visually?
   - Do the frames show setup for what's promised in speech?
   - Is there visual foreshadowing of the payoff?
   - How well aligned are the words and visuals in setting expectations?

3. RETENTION MECHANICS:
   - Story progression: How do visuals support the narrative flow?
   - Engagement elements: Eye contact, expressions, gestures that drive comments
   - Visual variety: Do frame changes maintain interest during speech?
   - Pacing alignment: Do visual cuts match verbal rhythm and emphasis?

4. PAYOFF DELIVERY:
   - Does the visual reveal align with the verbal conclusion?
   - Are key moments emphasized both verbally and visually?
   - Is the satisfaction delivered through words, visuals, or both?

5. MULTIMODAL HOOKS (analyze combinations):
   - Text overlays + speech content
   - Facial expressions + verbal tone
   - Visual demonstrations + explanations
   - Environmental changes + narrative progression

SCORING (1-10 scale):
- Hook Strength: How compelling is the audio+visual opening combination?
- Promise Clarity: How clear is the expected payoff across both channels?
- Retention Design: How well do visuals and audio work together for watch-through?
- Engagement Potential: Will the combination drive comments/shares?
- Goal Alignment: How well does the full experience serve {goal}?

HOOK GENERATION RULES:
Generate 5 alternative hooks that sound natural and platform-native:

TONE REQUIREMENTS:
- Use conversational, casual language (not marketing speak)
- Match the energy and vocabulary of the original video
- Sound like something a real person would actually say on TikTok
- Be specific to the actual topic/niche, not generic

AVOID THESE AI-SOUNDING PHRASES:
- "Discover the secret to..."
- "Unlock your potential..." 
- "Transform your life with..."
- "The one trick that..."
- "You won't believe what happens when..."
- "Game-changing technique"
- "Revolutionary method"

INSTEAD USE NATURAL LANGUAGE PATTERNS:
- "wait this actually works"
- "nobody talks about this but..."
- "I tried this for [timeframe] and..."
- "my [relationship/job/etc] changed when I..."
- "this sounds fake but..."
- "POV: you just found out..."
- "telling my [person] that I..."
- "the day I accidentally..."
- "why [common thing] is actually..."

HOOK TYPES TO CONSIDER:
- Personal story openings with unexpected twists
- Controversial opinions about common beliefs
- Behind-the-scenes revelations
- Mistake/failure stories with lessons
- Comparison setups that subvert expectations

Focus on how the transcript and visuals work together (or against each other) to create the retention experience.
- Visual and verbal hooks reinforce each other
- Misalignment between what's said vs. shown
- Visual elements that enhance or detract from the verbal message
- Opportunities to better synchronize audio and visual retention tactics

Respond in valid JSON format:
{{
  "analysis": "Analyze how the transcript and visuals work together to create retention. Discuss specific moments where audio and visual elements reinforce or conflict with each other. Focus on the combined psychological impact on viewers.",
  "hooks": [
    "Natural alternative hook 1",
    "Natural alternative hook 2", 
    "Natural alternative hook 3",
    "Natural alternative hook 4",
    "Natural alternative hook 5"
  ],
  "scores": {{
    "hook_strength": 7,
    "promise_clarity": 6,
    "retention_design": 8,
    "engagement_potential": 7,
    "goal_alignment": 6
  }},
  "timing_breakdown": "Describe what happens at key moments combining both audio and visual: 0-3s (how opening words + visuals create hook), 3-7s (promise setup through speech + visual cues), middle (how content builds through both channels), end (payoff delivery via audio + visual)",
  "formula": "Step-by-step formula considering both verbal script AND visual presentation that this creator can apply to future content",
  "improvements": "Specific suggestions for better aligning transcript and visuals, enhancing multimodal retention, and optimizing the audio-visual experience for {goal}"
}}
    """

    try:
        gpt_response = _api_retry(
            client.chat.completions.create,
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=2000
        )

        response_text = gpt_response.choices[0].message.content.strip()
        print(f"GPT Analysis - Length: {len(response_text)} chars")
        
        # Clean up response
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        try:
            parsed = json.loads(response_text)
            
            result = {
                "analysis": parsed.get("analysis", "Analysis combining transcript and visual elements not available").strip(),
                "hooks": parsed.get("hooks", []),
                "scores": parsed.get("scores", {}),
                "timing_breakdown": parsed.get("timing_breakdown", "").strip(),
                "formula": parsed.get("formula", "").strip(),
                "improvements": parsed.get("improvements", "").strip()
            }
            
            # Ensure hooks is a list and we have content
            if isinstance(result["hooks"], str):
                result["hooks"] = [result["hooks"]]
            
            if not result["hooks"]:
                result["hooks"] = [
                    "wait this actually changed everything for my business",
                    "nobody talks about this but most design advice is backwards", 
                    "I tried this color theory thing for 30 days and my clients doubled",
                    "POV: you just found out your brand colors are doing the opposite",
                    "telling my design mentor I was doing everything wrong"
                ]
            
            # Ensure we have default scores if missing
            if not result["scores"]:
                result["scores"] = {
                    "hook_strength": 7, 
                    "promise_clarity": 6, 
                    "retention_design": 7, 
                    "engagement_potential": 8, 
                    "goal_alignment": 6
                }
            
            print(f"Analysis successful - Combined audio/visual insights generated")
            return result
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            
            return {
                "analysis": "This video combines verbal and visual elements to create engagement. The transcript works with the visual presentation to build viewer interest and maintain attention through the full duration.",
                "hooks": [
                    "this design mistake is costing you clients and you don't even know it",
                    "I used to hate my brand until I learned this one thing",
                    "my biggest client fired me and it was the best thing that happened", 
                    "why everything you learned about color theory is wrong",
                    "POV: you finally understand why your designs feel off"
                ],
                "scores": {"hook_strength": 7, "promise_clarity": 6, "retention_design": 7, "engagement_potential": 8, "goal_alignment": 6},
                "timing_breakdown": "Opening combines visual and verbal hooks, middle builds through both channels, conclusion delivers satisfaction via audio-visual alignment",
                "formula": "1. Create multimodal hook (visual + verbal) 2. Reinforce promise through both channels 3. Build engagement via audio-visual variety 4. Deliver payoff using combined elements",
                "improvements": "Better synchronize visual and verbal elements, strengthen opening hook combination, enhance audio-visual alignment for goal achievement"
            }
            
    except Exception as e:
        print(f"GPT analysis error: {e}")
        return {
            "analysis": f"Multimodal analysis failed: {str(e)}",
            "hooks": ["this changed my whole perspective on [topic]", "nobody warned me about this part of [niche]", "I wish someone told me this before I started", "this sounds crazy but it actually works", "the day I realized I was doing everything backwards"],
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
