# processing.py — lean version (no custom GPT)
# Download, audio extraction, smart/uniform frame sampling,
# quality filtering, robust transcription, and vision batch analysis.

import os
import re
import cv2
import math
import time              # for time.time()
import time as _time     # for retry helper
import base64
import ffmpeg
import subprocess
from pathlib import Path
from typing import List, Tuple

from PIL import Image
from yt_dlp import YoutubeDL
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# OPTIONAL local whisper fallback (pip install openai-whisper)
try:
    import whisper as local_whisper
except Exception:
    local_whisper = None


# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------

def _ensure_dirs():
    Path("downloads").mkdir(exist_ok=True)
    Path("audio").mkdir(exist_ok=True)
    Path("frames").mkdir(exist_ok=True)


def _api_retry(callable_fn, *args, **kwargs):
    """Retry wrapper for OpenAI calls: 4 tries, exponential backoff with jitter."""
    max_tries = 4
    base = 1.25
    for attempt in range(1, max_tries + 1):
        try:
            return callable_fn(*args, **kwargs)
        except Exception as e:
            if attempt == max_tries:
                raise
            sleep_s = (base ** attempt) + 0.5 * (os.urandom(1)[0] / 255.0)
            print(f"[retry] OpenAI call failed ({attempt}/{max_tries}): {e}. Retrying in {sleep_s:.1f}s")
            _time.sleep(sleep_s)


# ------------------------------------------------------------------------------
# Download + audio
# ------------------------------------------------------------------------------

def download_video(tiktok_url: str) -> str:
    """
    Download social video using yt-dlp. Returns absolute mp4 path.
    """
    _ensure_dirs()
    stamp = str(int(time.time()))
    out_tmpl = os.path.abspath(os.path.join("downloads", f"vid_{stamp}.%(ext)s"))

    ydl_opts = {
        "outtmpl": out_tmpl,
        "format": "mp4/bestvideo+bestaudio/best",
        "quiet": True,
        "retries": 5,
        "noprogress": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([tiktok_url])

    mp4s = sorted(Path("downloads").glob(f"vid_{stamp}*.mp4"))
    if mp4s:
        return str(mp4s[-1])
    # fallback: accept any container if mp4 not produced
    candidates = sorted(Path("downloads").glob(f"vid_{stamp}*.*"))
    if not candidates:
        raise FileNotFoundError("Video download failed.")
    return str(candidates[-1])


def probe_duration(video_path: str) -> float:
    info = ffmpeg.probe(video_path)
    return float(info["format"]["duration"])


def extract_audio(video_path: str) -> str:
    _ensure_dirs()
    audio_path = os.path.abspath(os.path.join("audio", f"aud_{int(time.time())}.mp3"))
    (
        ffmpeg
        .input(video_path)
        .output(audio_path, format="mp3", acodec="mp3", ar=44100, ac=2)
        .overwrite_output()
        .run(quiet=True)
    )
    return audio_path


# ------------------------------------------------------------------------------
# Scene & motion detection
# ------------------------------------------------------------------------------

def scene_change_times(video_path: str, threshold: float = 0.25) -> List[float]:
    """
    Use FFmpeg select scene filter and parse precise pts_time values from stderr.
    Lower threshold (0.20–0.30) is more sensitive to subtle UI/graphics changes.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats",
        "-i", video_path,
        "-filter_complex", f"select='gt(scene,{threshold})',metadata=print",
        "-f", "null", "-"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, err = proc.communicate()

    times = []
    for line in err.decode("utf-8", errors="ignore").splitlines():
        if "pts_time" in line and "lavfi.scene_score" in line:
            m = re.search(r"pts_time[:=]([0-9\.]+)", line)
            if m:
                times.append(round(float(m.group(1)), 3))

    times = sorted(set(times))
    print(f"[smart] scene hits: {len(times)} at {times[:6]}{'...' if len(times) > 6 else ''}")
    return times


def motion_event_times(video_path: str, window_sec: float = 0.3,
                       mag_thresh: float = 12.0, max_events: int = 60) -> List[float]:
    """
    Lightweight motion detector to catch slide-ins/animations that per-frame scene
    miss. Samples ~15 fps; flags windows whose mean absdiff exceeds mag_thresh.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap or not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(fps / 15))        # ~15 fps sampling
    win  = max(2, int(window_sec * (fps / step)))

    ret, prev = cap.read()
    if not ret:
        cap.release()
        return []
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    diffs, idx = [], 0
    while True:
        for _ in range(step - 1):
            cap.grab()
            idx += 1
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mag  = float(cv2.absdiff(gray, prev_gray).mean())
        t    = idx / fps
        diffs.append((t, mag))
        prev_gray = gray

    cap.release()

    # rolling mean
    hits, buf = [], []
    last_t = -999
    for t, mag in diffs:
        buf.append(mag)
        if len(buf) > win:
            buf.pop(0)
        if len(buf) == win and sum(buf)/len(buf) >= mag_thresh:
            if t - last_t > 0.5:
                hits.append(round(t, 3)); last_t = t
            if len(hits) >= max_events:
                break

    print(f"[smart] motion hits: {len(hits)} at {hits[:6]}{'...' if len(hits) > 6 else ''}")
    return hits


# ------------------------------------------------------------------------------
# Timestamp building
# ------------------------------------------------------------------------------

def build_sampling_times(duration: float, change_times: List[float], max_frames: int) -> List[float]:
    """
    Merge anchors + settle offsets with detected change times.
    Clip to duration and dedupe. Allow ~3x cap for downstream pruning.
    """
    anchors = [0.0, 0.3, 0.8, 1.0, 1.5, 3.0]  # extra 1.0s helps mid-intro slide-ins
    outro = max(0.0, duration - 0.8)
    anchors.append(outro)

    settle_offsets = [0.3, 0.6]

    ts = []
    for t in change_times:
        ts.append(t)
        for off in settle_offsets: ts.append(t + off)
    for a in anchors:
        ts.append(a)
        for off in settle_offsets: ts.append(a + off)

    ts = [round(max(0.0, min(duration - 0.05, x)), 3) for x in ts]
    ts = sorted(set(ts))

    if not ts:
        # fallback to uniform N points if nothing detected
        N = max_frames
        step = duration / (N + 1)
        ts = [round(step * i, 3) for i in range(1, N + 1)]

    return ts[: max_frames * 3]


# ------------------------------------------------------------------------------
# Frame extraction
# ------------------------------------------------------------------------------

def extract_frames_at_times(video_path: str, out_dir: str, timestamps: List[float]) -> List[str]:
    """
    Seek on INPUT at each timestamp so FFmpeg returns distinct, correct frames.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, ts in enumerate(timestamps, start=1):
        out_path = os.path.join(out_dir, f"ts_{i:04d}.jpg")
        (
            ffmpeg
            .input(video_path, ss=float(ts))  # input-side seek
            .output(out_path, vframes=1, vf="scale=480:-1", vsync="vfr")
            .overwrite_output()
            .run(quiet=True)
        )
        paths.append(out_path)
    return paths


def extract_frames_uniform(video_path: str, frames_dir: str,
                           frames_per_minute: int, cap: int) -> List[str]:
    dur = probe_duration(video_path)
    total = min(cap, max(1, int(frames_per_minute * (dur / 60.0))))
    if total <= 0: total = min(cap, 10)
    step = dur / (total + 1)
    timestamps = [round(step * i, 3) for i in range(1, total + 1)]
    return extract_frames_at_times(video_path, frames_dir, timestamps)


# ------------------------------------------------------------------------------
# Quality filtering
# ------------------------------------------------------------------------------

def is_blurry(img_path: str, thr: float = 80.0) -> bool:
    """Reject frames with low sharpness (variance of Laplacian)."""
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return True
        fm = cv2.Laplacian(img, cv2.CV_64F).var()
        return fm < thr
    except Exception:
        return False


def _ahash(img: Image.Image, hash_size: int = 8) -> int:
    """Average hash; simple near-duplicate detector without extra deps."""
    img = img.convert("L").resize((hash_size, hash_size), Image.BILINEAR)
    px = list(img.getdata())
    avg = sum(px) / len(px)
    bits = 0
    for i, p in enumerate(px):
        bits |= (1 if p > avg else 0) << i
    return bits


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def dedupe_frames_by_phash(paths: List[str], dist: int = 4) -> List[str]:
    kept, hashes = [], []
    for p in paths:
        try:
            with Image.open(p) as im:
                h = _ahash(im)
        except Exception:
            continue
        if all(_hamming(h, hh) > dist for hh in hashes):
            kept.append(p); hashes.append(h)
    return kept


def keep_text_heavy_frames(paths: List[str], min_chars: int = 0) -> List[str]:
    """
    Placeholder to prefer text-on-screen frames (wire OCR here if desired).
    Currently returns unchanged.
    """
    return paths


# ------------------------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------------------------

def extract_audio_and_frames(
    tiktok_url: str,
    strategy: str = "smart",            # "smart" or "uniform"
    frames_per_minute: int = 24,        # used if uniform
    cap: int = 60,                      # max frames returned
    scene_threshold: float = 0.24       # lower = more sensitive
) -> Tuple[str, str, List[str]]:
    """
    Download video, extract audio, pick frames by strategy, apply quality filters.
    Returns (audio_path, frames_dir, [frame_paths]).
    """
    _ensure_dirs()
    video_path = download_video(tiktok_url)
    audio_path = extract_audio(video_path)
    dur = probe_duration(video_path)

    frames_dir = os.path.join("frames", f"set_{int(time.time())}")
    os.makedirs(frames_dir, exist_ok=True)

    if strategy == "uniform":
        paths = extract_frames_uniform(video_path, frames_dir, frames_per_minute, cap)
    else:
        # SMART: scene + motion + anchors + settle, then quality filters
        sc_times = scene_change_times(video_path, threshold=scene_threshold)
        mo_times = motion_event_times(video_path, window_sec=0.30, mag_thresh=12.0)

        merged = sorted(set(sc_times + mo_times))
        ts_list = build_sampling_times(dur, merged, max_frames=cap)
        print(f"[smart] timestamps after merge+anchors: {len(ts_list)}")

        paths = extract_frames_at_times(video_path, frames_dir, ts_list)
        paths = [p for p in paths if not is_blurry(p)]
        paths = dedupe_frames_by_phash(paths, dist=4)
        paths = keep_text_heavy_frames(paths, min_chars=0)
        paths = sorted(paths)[:cap]

        if not paths:
            print("[smart] empty after filters → fallback to uniform")
            paths = extract_frames_uniform(video_path, frames_dir, frames_per_minute=18, cap=min(cap, 20))

    return audio_path, frames_dir, paths


# ------------------------------------------------------------------------------
# Transcription (OpenAI with retry; local Whisper fallback)
# ------------------------------------------------------------------------------

def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe with OpenAI (gpt-4o-mini-transcribe).
    On connection failure, retry; if still failing and local Whisper is present, use it.
    """
    def _remote_transcribe():
        with open(audio_path, "rb") as f:
            return client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=f
            )

    try:
        tr = _api_retry(_remote_transcribe)
        return tr.text
    except Exception as e:
        print(f"[warn] Remote transcription failed: {e}")
        if local_whisper is not None:
            try:
                print("[info] Falling back to local Whisper (base). This is slower on CPU.")
                w = local_whisper.load_model("base")
                out = w.transcribe(audio_path)
                return out.get("text", "").strip() or "(transcription empty)"
            except Exception as e2:
                print(f"[error] Local Whisper failed: {e2}")
        return "(transcription unavailable due to connection error)"


# ------------------------------------------------------------------------------
# Batch frame analysis (vision) — always gpt-4o
# ------------------------------------------------------------------------------

# REPLACE the analyze_frames_batch function in your processing.py with this:

def analyze_frames_batch(image_paths: List[str]) -> Tuple[str, List[str]]:
    """
    Enhanced frame analysis that identifies satisfying background processes and retention drivers.
    """
    
    analysis_prompt = """Analyze these video frames for TikTok retention psychology. Focus on identifying DUAL ENGAGEMENT MECHANISMS:

PRIMARY CONTENT ANALYSIS:
- What is the main message/topic being delivered?
- Is this educational, controversial, storytelling, or opinion-based?
- What verbal hooks or claims are being made?

SATISFYING BACKGROUND PROCESSES:
Look for activities that provide visual satisfaction while the main message is delivered:

REPETITIVE/RHYTHMIC ACTIONS:
- Folding, organizing, sorting items
- Makeup application (blending, brushing, precise movements)
- Food preparation (chopping, mixing, plating)
- Art creation (painting, drawing, crafting)
- Cleaning/tidying (wiping, arranging, decluttering)
- Hair styling/braiding
- Gaming/typing with satisfying precision

PROCESS SATISFACTION ELEMENTS:
- Transformation moments (before/after states)
- Completion satisfaction (finishing a task)
- Precision movements (careful, skilled actions)
- Texture interactions (smooth, satisfying materials)
- Organization/symmetry creation
- Problem-solving in real-time

MULTITASKING APPEAL:
- Productive activities (cleaning while teaching)
- Self-care routines (skincare while storytelling)
- Creative processes (art while explaining)
- Skill demonstrations (cooking while sharing tips)

RETENTION PSYCHOLOGY:
- Does the visual process keep eyes engaged during slower verbal moments?
- Are there satisfying completion moments throughout?
- Does the background activity create a meditative, watchable quality?
- Is there anticipation for the finished result?

ENGAGEMENT AMPLIFIERS:
- Skilled/expert-level execution that's impressive to watch
- Relatable daily activities viewers connect with
- Aspirational lifestyle elements (organized spaces, skills)
- ASMR-like visual satisfaction

For each frame, describe:
1. What background process/activity is happening
2. How satisfying/engaging this process appears
3. How it supports or distracts from the main message
4. What completion/transformation moments are visible
5. Why this combination would retain viewer attention

Focus on the DUAL ENGAGEMENT: eyes watching satisfying process + ears processing valuable content."""

    contents = [{
        "type": "text", 
        "text": analysis_prompt
    }]

    gallery = []
    for p in image_paths:
        with open(p, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        contents.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        gallery.append(f"data:image/jpeg;base64,{b64}")

    def _call():
        return client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": contents}],
            max_tokens=1500,  # Increased for detailed process analysis
        )

    resp = _api_retry(_call)
    return resp.choices[0].message.content, gallery
