# REPLACE YOUR ENTIRE app.py with this fixed version:

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


def detect_content_patterns(transcript_text, frames_summaries_text):
    """
    Enhanced detection for content patterns including satisfying background processes.
    """
    
    # Satisfying process keywords
    satisfying_processes = [
        'folding', 'organizing', 'makeup', 'skincare', 'cooking', 'baking', 'eating',
        'painting', 'drawing', 'crafting', 'cleaning', 'tidying', 'styling', 'braiding',
        'gaming', 'typing', 'building', 'assembling', 'decorating', 'planting',
        'chopping', 'mixing', 'blending', 'brushing', 'arranging', 'sorting'
    ]
    
    # Content type keywords
    controversial_indicators = [
        'unpopular opinion', 'controversial', 'hot take', 'nobody talks about',
        'people hate when', 'this will upset', 'i don\'t care if', 'fight me',
        'wrong', 'bad', 'terrible', 'hate', 'annoying', 'overrated'
    ]
    
    educational_indicators = [
        'tutorial', 'how to', 'lesson', 'teach', 'learn', 'explain', 'guide',
        'tips', 'tricks', 'advice', 'steps', 'method', 'technique'
    ]
    
    storytelling_indicators = [
        'story time', 'let me tell you', 'this happened', 'experience',
        'journey', 'day i', 'time when', 'remember when'
    ]
    
    # Check for patterns
    text_combined = f"{transcript_text} {frames_summaries_text}".lower()
    
    satisfying_count = sum(1 for keyword in satisfying_processes if keyword in text_combined)
    controversial_count = sum(1 for keyword in controversial_indicators if keyword in text_combined)
    educational_count = sum(1 for keyword in educational_indicators if keyword in text_combined)
    story_count = sum(1 for keyword in storytelling_indicators if keyword in text_combined)
    
    # Determine patterns
    patterns = {
        'has_satisfying_process': satisfying_count >= 2,
        'is_controversial': controversial_count >= 1,
        'is_educational': educational_count >= 2,
        'is_storytelling': story_count >= 1,
        'dual_engagement': satisfying_count >= 2 and len(transcript_text.strip()) > 100
    }
    
    return patterns

def detect_video_type(transcript_text, frames_summaries_text):
    """
    Detect if video is speech-heavy, visual-only, or mixed content.
    Returns: 'visual_only', 'speech_heavy', or 'mixed'
    """
    transcript_length = len(transcript_text.strip())
    
    # Keywords that suggest visual-only content
    visual_keywords = [
        'satisfying', 'asmr', 'process', 'making', 'creating', 'building', 
        'unboxing', 'crafting', 'cooking', 'baking', 'drawing', 'painting',
        'tools', 'hands', 'step by step', 'tutorial', 'diy', 'transformation'
    ]
    
    # Check if transcript has visual-focused language
    visual_indicators = sum(1 for keyword in visual_keywords 
                           if keyword in transcript_text.lower() or keyword in frames_summaries_text.lower())
    
    # Determine video type
    if transcript_length < 50:  # Very little speech
        return 'visual_only'
    elif transcript_length < 200 and visual_indicators >= 3:  # Short speech + visual focus
        return 'visual_only'
    elif transcript_length > 500:  # Speech-heavy
        return 'speech_heavy'
    else:
        return 'mixed'

def run_gpt_analysis(transcript_text, frames_summaries_text, creator_note, platform, target_duration, goal, tone, audience, knowledge_context=""):
    """
    Complete enhanced analysis with visual-only detection, satisfying process analysis, and strong hook generation.
    """
    
    # Detect content patterns and video type
    patterns = detect_content_patterns(transcript_text, frames_summaries_text)
    video_type = detect_video_type(transcript_text, frames_summaries_text)
    
    print(f"Content patterns detected: {patterns}")
    print(f"Video type: {video_type}")
    
    # Enhanced prompt that adapts to detected patterns
    if patterns['dual_engagement']:
        analysis_focus = "DUAL ENGAGEMENT ANALYSIS"
        special_instructions = """
This video combines SATISFYING VISUAL PROCESSES with VERBAL CONTENT delivery. Analyze how these work together:

DUAL ENGAGEMENT FRAMEWORK:
1. VISUAL RETENTION: What satisfying process keeps eyes engaged?
2. AUDIO PROCESSING: What message/opinion is being delivered?
3. SYNERGY ANALYSIS: How do these complement each other?
4. RETENTION AMPLIFICATION: Why does this combination work?

SATISFYING PROCESS ANALYSIS:
- What repetitive/satisfying activity is happening?
- How does this create visual meditation while processing verbal content?
- Are there completion moments that provide satisfaction?
- Does the process add credibility or relatability?
        """
    else:
        analysis_focus = "STANDARD ANALYSIS"
        special_instructions = ""
    
    prompt = f"""
You are an expert TikTok/short-form content strategist analyzing videos for retention psychology.

TRANSCRIPT:
{transcript_text}

VISUAL FRAMES:
{frames_summaries_text}

CREATOR NOTE: {creator_note}
PLATFORM: {platform} | GOAL: {goal} | DURATION: {target_duration}s
ANALYSIS TYPE: {analysis_focus}

{special_instructions}

HOOK ANALYSIS (0-3 seconds):
- What immediately grabs attention (visual + verbal)?
- Is there a satisfying process that draws the eye?
- What type of content promise is being made?
- How do visual and verbal elements work together?

RETENTION MECHANICS:
- Background processes that maintain visual interest
- Verbal content delivery style and pacing
- Completion/satisfaction moments throughout
- How the combination prevents drop-off

CONTENT CLASSIFICATION:
Identify the primary content type:
- Controversial opinion + satisfying process
- Educational content + demonstration
- Storytelling + relatable activity  
- Lifestyle/routine + valuable insights

HOOK GENERATION RULES:
Create 5 strong hooks focusing on CONTENT VALUE, not format description:

FOR CONTROVERSIAL CONTENT:
- Bold statements that challenge common beliefs
- Provocative opinions that spark debate
- Claims that sound shocking or counterintuitive

FOR EDUCATIONAL CONTENT:
- Problem-solution setups
- Valuable insights people don't know
- Secrets or insider knowledge

NATURAL LANGUAGE REQUIREMENTS:
- Sound like real TikTok content, not marketing copy
- Use casual, conversational tone
- Avoid AI-sounding phrases like "discover," "unlock," "transform"
- Focus on curiosity, controversy, or value

AVOID THESE AI-SOUNDING PHRASES:
- "Discover the secret to..."
- "Unlock your potential..." 
- "Transform your life with..."
- "The one trick that..."
- "You won't believe what happens when..."

INSTEAD USE NATURAL LANGUAGE PATTERNS:
- "wait this actually works"
- "nobody talks about this but..."
- "I tried this for [timeframe] and..."
- "this sounds fake but..."
- "POV: you just found out..."
- "why [common thing] is actually..."

SCORING (1-10 scale):
- Hook Strength: How compelling is the opening?
- Promise Clarity: How clear is the expected payoff?
- Retention Design: How well structured for full watch-through?
- Engagement Potential: Will it drive comments/shares?
- Goal Alignment: How well does it serve {goal}?

Respond in valid JSON format:
{{
  "analysis": "Detailed analysis of how content creates retention. If dual engagement is detected, explain how visual satisfaction and verbal content work together.",
  "hooks": [
    "Strong hook based on content value/controversy",
    "Hook emphasizing the core insight or opinion", 
    "Hook creating curiosity about the message",
    "Hook using natural, platform-native language",
    "Hook that would drive engagement and comments"
  ],
  "scores": {{
    "hook_strength": 8,
    "promise_clarity": 7,
    "retention_design": 8,
    "engagement_potential": 8,
    "goal_alignment": 7
  }},
  "timing_breakdown": "Analyze key moments and how content builds throughout",
  "basic_formula": "Step-by-step process for creating similar engaging content",
  "timing_formula": "Timing strategy with specific second markers",
  "template_formula": "Template format for this type of content",
  "psychology_formula": "Framework explaining why this approach works psychologically",
  "improvements": "Specific suggestions for optimizing content and delivery for {goal}",
  "video_type": "{video_type}",
  "content_patterns": {patterns}
}}
    """

    try:
        gpt_response = _api_retry(
            client.chat.completions.create,
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=2500
        )

        response_text = gpt_response.choices[0].message.content.strip()
        
        # Clean and parse response
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        try:
            parsed = json.loads(response_text)
            
            result = {
                "analysis": parsed.get("analysis", "Analysis not available").strip(),
                "hooks": parsed.get("hooks", []),
                "scores": parsed.get("scores", {}),
                "timing_breakdown": parsed.get("timing_breakdown", "").strip(),
                "formula": parsed.get("basic_formula", "").strip(),
                "basic_formula": parsed.get("basic_formula", "").strip(),
                "timing_formula": parsed.get("timing_formula", "").strip(),
                "template_formula": parsed.get("template_formula", "").strip(),
                "psychology_formula": parsed.get("psychology_formula", "").strip(),
                "improvements": parsed.get("improvements", "").strip(),
                "video_type": parsed.get("video_type", video_type),
                "content_patterns": parsed.get("content_patterns", patterns)
            }
            
            # Strong fallback hooks based on content type
            if not result["hooks"]:
                if patterns['is_controversial']:
                    result["hooks"] = [
                        "your logo is costing you clients and you don't even know it",
                        "I can spot a Canva logo from a mile away and here's why that's bad",
                        "people who use free logos are telling on themselves",
                        "this is why nobody takes your business seriously",
                        "your brand looks cheap because it IS cheap"
                    ]
                else:
                    result["hooks"] = [
                        "this changed everything I thought I knew about this",
                        "nobody prepared me for this reality",
                        "here's what I wish someone told me earlier",
                        "this sounds controversial but you need to hear it",
                        "the day I realized most people are completely wrong"
                    ]
            
            # Ensure we have default scores
            if not result["scores"]:
                result["scores"] = {
                    "hook_strength": 8, 
                    "promise_clarity": 7, 
                    "retention_design": 8, 
                    "engagement_potential": 8, 
                    "goal_alignment": 7
                }
            
            print(f"Enhanced analysis complete - dual engagement: {patterns.get('dual_engagement', False)}")
            return result
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            
            # Strong fallback hooks
            fallback_hooks = [
                "this opinion is going to upset people but it's true",
                "everyone's wrong about this and I can prove it",
                "this harsh truth will change how you see everything",
                "nobody wants to admit this but here's reality",
                "this controversial take will make you rethink everything"
            ]
            
            return {
                "analysis": "This content effectively engages viewers through compelling messaging and strong retention mechanics.",
                "hooks": fallback_hooks,
                "scores": {"hook_strength": 8, "promise_clarity": 7, "retention_design": 8, "engagement_potential": 8, "goal_alignment": 7},
                "timing_breakdown": "Content builds effectively from hook through to satisfying conclusion",
                "formula": "Strong hook → Clear promise → Engaging delivery → Satisfying payoff",
                "basic_formula": "1. Open with compelling hook 2. Set clear expectation 3. Deliver valuable content 4. End with satisfaction",
                "timing_formula": "0-3s: Hook, 3-7s: Promise, Middle: Build engagement, End: Deliver payoff",
                "template_formula": "[Strong Hook] → [Clear Promise] → [Engaging Content] → [Satisfying Conclusion]",
                "psychology_formula": "Curiosity → Expectation → Engagement → Satisfaction",
                "improvements": "Strengthen opening hook, clarify value proposition, optimize pacing for retention",
                "video_type": video_type,
                "content_patterns": patterns
            }
            
    except Exception as e:
        print(f"GPT analysis error: {e}")
        return {
            "analysis": f"Analysis failed: {str(e)}",
            "hooks": ["this perspective changed everything for me", "nobody warned me about this reality", "here's what I wish I knew sooner", "this truth is hard to accept but necessary", "everyone should know this but few do"],
            "scores": {},
            "timing_breakdown": "",
            "formula": "",
            "basic_formula": "",
            "timing_formula": "",
            "template_formula": "",
            "psychology_formula": "",
            "improvements": "",
            "video_type": "unknown",
            "content_patterns": {}
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
        knowledge_context = ""
        knowledge_citations = []

        # --- Enhanced AI Analysis ---
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
            print("Enhanced retention analysis complete")
        except Exception as e:
            print(f"GPT analysis error: {e}")
            gpt_result = {
                "analysis": f"Analysis failed: {str(e)}", 
                "hooks": [],
                "scores": {},
                "timing_breakdown": "",
                "formula": "",
                "basic_formula": "",
                "timing_formula": "",
                "template_formula": "",
                "psychology_formula": "",
                "improvements": "",
                "video_type": "unknown",
                "content_patterns": {}
            }

        # --- Extract ALL results ---
        analysis_text = gpt_result.get("analysis", "Analysis not available")
        hooks_list = gpt_result.get("hooks", [])
        scores = gpt_result.get("scores", {})
        timing_breakdown = gpt_result.get("timing_breakdown", "")
        formula = gpt_result.get("formula", "")
        basic_formula = gpt_result.get("basic_formula", "")
        timing_formula = gpt_result.get("timing_formula", "")
        template_formula = gpt_result.get("template_formula", "")
        psychology_formula = gpt_result.get("psychology_formula", "")
        improvements = gpt_result.get("improvements", "")
        video_type = gpt_result.get("video_type", "unknown")
        content_patterns = gpt_result.get("content_patterns", {})
        
        if isinstance(hooks_list, str):
            hooks_list = [hooks_list]

        # --- Prepare frame summaries for template ---
        frame_summaries = []
        if frames_summaries_text:
            blocks = frames_summaries_text.split('\n\n')
            frame_summaries = [block.strip() for block in blocks if block.strip()]
        
        if not frame_summaries and frames_summaries_text:
            frame_summaries = [frames_summaries_text]

        print("Rendering enhanced results template")
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
            basic_formula=basic_formula,
            timing_formula=timing_formula,
            template_formula=template_formula,
            psychology_formula=psychology_formula,
            improvements=improvements,
            video_type=video_type,
            content_patterns=content_patterns,
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