import os
import random
import time as _time
import json
import re
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
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=600.0)


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
    """Enhanced detection for content patterns including satisfying background processes."""
    
    # Satisfying process keywords
    satisfying_processes = [
        'folding', 'organizing', 'makeup', 'skincare', 'cooking', 'baking', 'eating',
        'painting', 'drawing', 'crafting', 'cleaning', 'tidying', 'styling', 'braiding',
        'gaming', 'typing', 'building', 'assembling', 'decorating', 'planting',
        'chopping', 'mixing', 'blending', 'brushing', 'arranging', 'sorting', 'laundry'
    ]
    
    # Content type keywords
    controversial_indicators = [
        'unpopular opinion', 'controversial', 'hot take', 'nobody talks about',
        'people hate when', 'this will upset', 'i don\'t care if', 'fight me',
        'wrong', 'bad', 'terrible', 'hate', 'annoying', 'overrated', 'shit'
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
        'has_satisfying_process': satisfying_count >= 1,
        'is_controversial': controversial_count >= 1,
        'is_educational': educational_count >= 2,
        'is_storytelling': story_count >= 1,
        'dual_engagement': satisfying_count >= 1 and len(transcript_text.strip()) > 50
    }
    
    return patterns

def extract_content_themes(transcript_text):
    """Extract actual themes from any transcript without hardcoded categories."""
    
    # Look for common topic indicators in any niche
    words = transcript_text.lower().split()
    
    # Common patterns that indicate main topics
    topic_indicators = []
    
    # Look for repeated important nouns (likely main topics)
    word_count = {}
    important_words = []
    
    # Filter for meaningful words (nouns, important concepts)
    for word in words:
        # Clean word
        clean_word = word.strip('.,!?":;()[]{}')
        if len(clean_word) > 3 and clean_word not in ['that', 'this', 'with', 'have', 'will', 'they', 'them', 'were', 'been', 'said', 'what', 'when', 'where', 'your', 'like', 'just', 'know', 'think', 'people', 'going', 'really', 'because']:
            word_count[clean_word] = word_count.get(clean_word, 0) + 1
    
    # Get most frequently mentioned meaningful words
    frequent_topics = [word for word, count in word_count.items() if count >= 2][:5]
    
    return frequent_topics if frequent_topics else ['general content']

def create_universal_video_description(transcript_text, frames_summaries_text):
    """Create description for any video content without hardcoded activities."""
    
    frames_lower = frames_summaries_text.lower()
    transcript_lower = transcript_text.lower()
    
    # Detect any visual activity mentioned in frame analysis
    visual_activities = []
    
    # Look for action words in frame descriptions
    action_patterns = [
        'applying', 'making', 'cooking', 'preparing', 'folding', 'organizing', 
        'cleaning', 'sorting', 'building', 'creating', 'drawing', 'writing',
        'demonstrating', 'showing', 'presenting', 'explaining', 'teaching',
        'working', 'using', 'holding', 'moving', 'performing'
    ]
    
    detected_activity = None
    for action in action_patterns:
        if action in frames_lower:
            # Try to get the object of the action
            if action == 'applying' and 'makeup' in frames_lower:
                detected_activity = 'applying makeup'
                break
            elif action == 'cooking' or 'kitchen' in frames_lower:
                detected_activity = 'cooking'
                break
            elif action == 'folding' and ('clothes' in frames_lower or 'laundry' in frames_lower):
                detected_activity = 'folding laundry'
                break
            elif action in ['demonstrating', 'showing', 'presenting']:
                detected_activity = 'presenting content'
                break
            elif action in ['explaining', 'teaching']:
                detected_activity = 'teaching'
                break
    
    # Determine main content topic from frequent words
    content_themes = extract_content_themes(transcript_text)
    main_topic = f"discussing {', '.join(content_themes[:2])}"
    
    # Create description
    if detected_activity and detected_activity != 'presenting content':
        return f"Subject is {detected_activity} while {main_topic}"
    else:
        return f"Subject is {main_topic}"

def analyze_performance_indicators(creator_note, transcript_text, frames_summaries_text):
    """Extract performance metrics and determine if video was successful."""
    
    performance_data = {
        'views': None,
        'engagement': None,
        'followers_gained': None,
        'watch_time': None,
        'saves': None,
        'shares': None,
        'success_indicators': []
    }
    
    if creator_note:
        note_lower = creator_note.lower()
        
        # Extract numerical metrics
        # Views (k, m patterns)
        view_match = re.search(r'(\d+(?:\.\d+)?)\s*([km])?\s*views?', note_lower)
        if view_match:
            num, unit = view_match.groups()
            multiplier = 1000 if unit == 'k' else 1000000 if unit == 'm' else 1
            performance_data['views'] = int(float(num) * multiplier)
        
        # Followers gained
        follower_match = re.search(r'(\d+)\s*(?:new\s+)?followers?', note_lower)
        if follower_match:
            performance_data['followers_gained'] = int(follower_match.group(1))
        
        # Watch time (hours)
        watch_match = re.search(r'(\d+)\s*h.*?watch.*?time', note_lower)
        if watch_match:
            performance_data['watch_time'] = int(watch_match.group(1))
        
        # Saves
        save_match = re.search(r'(\d+)\s*saves?', note_lower)
        if save_match:
            performance_data['saves'] = int(save_match.group(1))
        
        # Shares  
        share_match = re.search(r'(\d+)\s*shares?', note_lower)
        if share_match:
            performance_data['shares'] = int(share_match.group(1))
    
    # Determine success level
    success_level = "unknown"
    success_reasons = []
    
    if performance_data['views']:
        if performance_data['views'] >= 100000:
            success_level = "highly_successful"
            success_reasons.append(f"{performance_data['views']:,} views indicates viral reach")
        elif performance_data['views'] >= 10000:
            success_level = "successful"
            success_reasons.append(f"{performance_data['views']:,} views shows good performance")
        else:
            success_level = "moderate"
            success_reasons.append(f"{performance_data['views']:,} views is moderate performance")
    
    if performance_data['followers_gained'] and performance_data['followers_gained'] >= 100:
        success_reasons.append(f"{performance_data['followers_gained']} new followers shows strong audience building")
    
    if performance_data['saves'] and performance_data['saves'] >= 100:
        success_reasons.append(f"{performance_data['saves']} saves indicates valuable content")
    
    performance_data['success_level'] = success_level
    performance_data['success_reasons'] = success_reasons
    
    return performance_data

def run_universal_gpt_analysis(transcript_text, frames_summaries_text, creator_note, platform, target_duration, goal, tone, audience, knowledge_context=""):
    """Universal analysis that works for any content topic."""
    
    # Get patterns and description without hardcoded assumptions
    patterns = detect_content_patterns(transcript_text, frames_summaries_text)
    video_description = create_universal_video_description(transcript_text, frames_summaries_text)
    performance_data = analyze_performance_indicators(creator_note, transcript_text, frames_summaries_text)
    content_themes = extract_content_themes(transcript_text)
    
    print(f"Video description: {video_description}")
    print(f"Content themes: {content_themes}")
    print(f"Performance data: {performance_data}")
    
    # Build performance context
    performance_context = ""
    if performance_data['success_level'] != "unknown":
        performance_context = f"""
PERFORMANCE CONTEXT:
This video achieved: {', '.join(performance_data['success_reasons'])}
Success Level: {performance_data['success_level']}
Use this performance data to validate your analysis - explain WHY this video achieved this level of success.
        """
    
    # Universal prompt that adapts to any content
    prompt = f"""
You are a retention psychology expert analyzing short-form content. Provide deep, specific insights about what makes THIS video addictive.

TRANSCRIPT: {transcript_text}

VISUAL FRAMES: {frames_summaries_text}

CREATOR NOTE: {creator_note}
VIDEO DESCRIPTION: {video_description}
MAIN TOPICS: {', '.join(content_themes)}
GOAL: {goal}

{performance_context}

ANALYSIS REQUIREMENTS:

1. CONTENT ANALYSIS:
- What is specifically happening in this video?
- What topics/themes are being discussed?
- How does the creator position themselves?
- What makes this content unique or valuable?

2. RETENTION PSYCHOLOGY:
- How does the opening create curiosity or pattern interrupt?
- What authority signals or credibility markers are present?
- How do visual and verbal elements work together?
- What psychological triggers keep people watching?

3. ENGAGEMENT MECHANICS:
- What would make someone comment on THIS specific content?
- Why would someone share this particular message?
- What emotions or reactions does this trigger?
- How does it make viewers feel about the topic?

4. HOOK GENERATION:
Generate 5 hooks based on THIS video's actual content and approach. 
- Use the same topic/niche as the original
- Match the creator's energy, tone, and vocabulary  
- Focus on the specific value or controversy presented
- Sound natural and platform-native

CRITICAL: Base everything on the actual content. Don't use generic templates.

5. PERFORMANCE ANALYSIS:
{f"Explain why this video achieved {performance_data['success_level']} results based on the retention mechanisms you identified." if performance_data['success_level'] != 'unknown' else "Predict likely performance based on retention psychology."}

Respond in JSON format:
{{
  "analysis": "Detailed analysis of THIS specific video's retention psychology. Explain what's happening, the psychological mechanisms at work, how visual and verbal elements interact, and why this approach works for this topic. Minimum 200 words of specific insights.",
  "hooks": [
    "Hook 1 specific to this video's topic and approach",
    "Hook 2 matching the content and energy",
    "Hook 3 using the same niche concepts and language", 
    "Hook 4 based on the actual value or controversy",
    "Hook 5 that captures the specific psychological trigger"
  ],
  "scores": {{
    "hook_strength": 8,
    "promise_clarity": 7,
    "retention_design": 9,
    "engagement_potential": 8,
    "goal_alignment": 7
  }},
  "timing_breakdown": "Second-by-second analysis of how retention builds in THIS video",
  "basic_formula": "Step-by-step process for creating similar content in this topic/approach",
  "timing_formula": "Timing strategy specific to this content style",
  "template_formula": "Template format for this type of content",
  "psychology_formula": "Psychology framework explaining why this approach works",
  "improvements": "Specific suggestions for optimizing THIS video for {goal}",
  "performance_prediction": "Performance prediction based on retention analysis"
}}

Focus on the actual content and psychological mechanisms, not generic advice.
    """

    try:
        print(f"Sending universal analysis prompt to GPT-4o...")
        gpt_response = _api_retry(
            client.chat.completions.create,
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=3500
        )

        response_text = gpt_response.choices[0].message.content.strip()
        
        # Parse JSON
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        try:
            parsed = json.loads(response_text)
            print("Successfully parsed JSON response")
            
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
                "performance_prediction": parsed.get("performance_prediction", "").strip(),
                "video_description": video_description,
                "content_patterns": patterns,
                "performance_data": performance_data
            }
            
            # Generic fallback hooks if needed (based on detected patterns)
            if not result["hooks"] or len(result["hooks"]) == 0:
                print("Generating content-adaptive fallback hooks")
                
                if patterns.get('is_controversial', False):
                    result["hooks"] = [
                        "this opinion is going to upset people but it's true",
                        "everyone's wrong about this and I can prove it",
                        "this harsh truth will change how you think",
                        "nobody wants to admit this but here's reality",
                        "this controversial take will make you rethink everything"
                    ]
                elif patterns.get('is_educational', False):
                    result["hooks"] = [
                        "this changed everything I thought I knew",
                        "nobody taught me this and I wish they had",
                        "here's what I wish someone told me earlier",
                        "this method actually works and here's why",
                        "the mistake everyone makes that you can avoid"
                    ]
                else:
                    result["hooks"] = [
                        "wait this actually works",
                        "nobody talks about this but they should",
                        "POV: you just found out the truth",
                        "this sounds fake but I promise it's not",
                        "the day I realized most people are wrong"
                    ]
            
            # Adjust scores based on performance
            if performance_data['success_level'] == 'highly_successful':
                for key in result["scores"]:
                    result["scores"][key] = min(10, result["scores"][key] + 1)
            
            print(f"Universal analysis complete - Topics: {content_themes}")
            return result
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            return create_universal_fallback(video_description, patterns, content_themes, goal, performance_data)
            
    except Exception as e:
        print(f"GPT analysis error: {e}")
        return create_universal_fallback(video_description, patterns, content_themes, goal, performance_data)

def create_universal_fallback(video_description, patterns, content_themes, goal, performance_data):
    """Create fallback result that adapts to any content topic."""
    
    analysis = f"{video_description}. "
    
    if performance_data['success_level'] != 'unknown':
        analysis += f"This video achieved {performance_data['success_level']} performance with {', '.join(performance_data['success_reasons'])}. "
    
    analysis += f"The content focuses on {', '.join(content_themes[:3]) if content_themes else 'the main topic'} using "
    
    if patterns.get('is_controversial', False):
        analysis += "controversial positioning and expert authority to challenge common beliefs. "
    elif patterns.get('is_educational', False):
        analysis += "educational delivery and valuable insights to teach viewers. "
    else:
        analysis += "engaging storytelling and relatable content to connect with viewers. "
    
    analysis += "The retention strategy combines direct communication with specific insights to maintain viewer engagement throughout."
    
    # Adaptive hooks based on content patterns
    if patterns.get('is_controversial', False):
        hooks = [
            "this opinion is going to upset people but it's true",
            "everyone's wrong about this and I can prove it",
            "this harsh truth will change how you think",
            "nobody wants to admit this but here's reality",
            "this controversial take will make you rethink everything"
        ]
    elif patterns.get('is_educational', False):
        hooks = [
            "this changed everything I thought I knew",
            "nobody taught me this and I wish they had", 
            "here's what I wish someone told me earlier",
            "this method actually works and here's why",
            "the mistake everyone makes that you can avoid"
        ]
    else:
        hooks = [
            "wait this actually works",
            "nobody talks about this but they should",
            "POV: you just found out the truth",
            "this sounds fake but I promise it's not",
            "the day I realized most people are wrong"
        ]
    
    return {
        "analysis": analysis,
        "hooks": hooks,
        "scores": {
            "hook_strength": 8 if performance_data['success_level'] == 'highly_successful' else 7,
            "promise_clarity": 7,
            "retention_design": 8 if performance_data['success_level'] != 'unknown' else 7,
            "engagement_potential": 8,
            "goal_alignment": 7
        },
        "timing_breakdown": "Content builds from engaging opening through valuable insights to satisfying conclusion",
        "formula": "Strong opener → Value delivery → Authority building → Clear conclusion",
        "basic_formula": "1. Create immediate interest 2. Deliver specific value 3. Build credibility 4. End with clear takeaway",
        "timing_formula": "0-3s: Hook creation, 3-7s: Value setup, Middle: Content delivery, End: Strong conclusion",
        "template_formula": "[Engaging Hook] → [Value Promise] → [Content Delivery] → [Clear Conclusion]",
        "psychology_formula": "Attention → Interest → Value → Satisfaction",
        "improvements": f"Enhance opening hook, add more specific examples, optimize pacing for {goal}",
        "performance_prediction": f"Based on content analysis: {performance_data['success_level'] if performance_data['success_level'] != 'unknown' else 'moderate to strong performance expected'}",
        "video_description": video_description,
        "content_patterns": patterns,
        "performance_data": performance_data
    }

def run_gpt_analysis(transcript_text, frames_summaries_text, creator_note, platform, target_duration, goal, tone, audience, knowledge_context=""):
    """Universal analysis that works for any content topic."""
    return run_universal_gpt_analysis(transcript_text, frames_summaries_text, creator_note, platform, target_duration, goal, tone, audience, knowledge_context)

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
            print(f"Transcript preview: {transcript[:200]}...")
        except Exception as e:
            print(f"Transcription error: {e}")
            transcript = "(Transcription failed)"

        # --- Frame analysis ---
        try:
            frames_summaries_text, gallery_data_urls = analyze_frames_batch(frame_paths)
            print(f"Frame analysis complete, gallery has {len(gallery_data_urls)} images")
            print(f"Frame analysis preview: {frames_summaries_text[:200]}...")
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
            gpt_result = create_universal_fallback(
                "Video analysis", 
                {"is_controversial": False, "dual_engagement": False}, 
                ["general content"],
                goal,
                {"success_level": "unknown", "success_reasons": []}
            )

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
        video_description = gpt_result.get("video_description", "Video analysis")
        content_patterns = gpt_result.get("content_patterns", {})
        performance_data = gpt_result.get("performance_data", {})
        
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
            video_description=video_description,
            content_patterns=content_patterns,
            performance_data=performance_data,
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