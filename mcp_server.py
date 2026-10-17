"""
MCP Server for Video Analysis AI.

Exposes video analysis capabilities as tools that Claude can call
through the Model Context Protocol. Shares the same database and
models as the Flask web app.

Run as a separate service: python mcp_server.py
"""
import os
import sys
import json
import logging
import base64
from datetime import datetime

try:
    from mcp.server.fastmcp import FastMCP as MCPServer
except (ImportError, ModuleNotFoundError):
    from mcp.server.mcpserver import MCPServer

try:
    from mcp.server.mcpserver import Image as MCPImage
except (ImportError, ModuleNotFoundError):
    from mcp.server.mcpserver.utilities.types import Image as MCPImage

logging.basicConfig(level=logging.INFO, format="[MCP] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Database setup (shares models with Flask app)
# ---------------------------------------------------------------------------
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker, scoped_session


def _get_database_url():
    raw = os.environ.get("DATABASE_URL", "").strip().strip("'\"")
    if raw.startswith("postgres://"):
        raw = raw.replace("postgres://", "postgresql+psycopg2://", 1)
    elif raw.startswith("postgresql://") and "+" not in raw.split("://")[0]:
        raw = raw.replace("postgresql://", "postgresql+psycopg2://", 1)
    if not raw:
        raw = "sqlite:///video_analysis.db"
    return raw


engine = create_engine(_get_database_url(), pool_pre_ping=True, pool_recycle=300)
Session = scoped_session(sessionmaker(bind=engine))

from models import Analysis, ReviewFeedback, KnowledgePattern

# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------
MCP_AUTH_TOKEN = os.environ.get("MCP_AUTH_TOKEN", "")

# ---------------------------------------------------------------------------
# MCP Server (FastMCP high-level API)
# ---------------------------------------------------------------------------
mcp = MCPServer("video-analysis")


def _analysis_to_dict(row):
    """Convert a raw DB row to a clean dict for Claude."""
    data = row.analysis_data or {}
    result = {
        "id": row.id,
        "video_url": row.video_url,
        "video_title": row.video_title,
        "status": row.status,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "completed_at": row.completed_at.isoformat() if row.completed_at else None,
    }

    for key in [
        "what_this_video_is", "why_it_performed", "viral_mechanics",
        "performance_prediction", "improvements", "transcript",
        "scores", "all_hooks_identified", "exact_hook_breakdown",
        "replication_formula", "timing_mastery", "format_fingerprint",
        "format_match", "comment_insights", "hook_mechanisms", "hook_intelligence",
        "follow_up_ideas",
    ]:
        if data.get(key):
            result[key] = data[key]

    for key in ["view_count", "like_count", "comment_count", "share_count"]:
        if data.get(key):
            result[key] = data[key]

    for key in ["content_type", "niche", "platform", "audio_type",
                 "music_info", "video_description", "hashtags",
                 "content_goal", "performance_classification",
                 "performance_section_type", "replication_section_type"]:
        if data.get(key):
            result[key] = data[key]

    return result


def _analysis_summary(row):
    """Compact summary for list views."""
    data = row.analysis_data or {}
    return {
        "id": row.id,
        "video_url": row.video_url,
        "video_title": row.video_title,
        "status": row.status,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "view_count": data.get("view_count"),
        "scores": data.get("scores"),
        "hook": data.get("hook", ""),
    }


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def get_analysis(analysis_id: int, include_frames: bool = False) -> str:
    """Get the full analysis for a specific video by analysis ID. Returns all sections: what it is, why it performed, hooks, timing, replication formula, transcript, scores, comments, and frame descriptions."""
    session = Session()
    try:
        analysis = session.query(Analysis).get(analysis_id)
        if not analysis:
            return "Analysis not found."

        result = _analysis_to_dict(analysis)

        if include_frames:
            data = analysis.analysis_data or {}
            frames = data.get("frame_gallery", [])
            if frames:
                result["frame_count"] = len(frames)
                result["frames_preview"] = f"{len(frames)} frames available (base64 images omitted for size)"

        return json.dumps(result, indent=2, default=str)
    finally:
        Session.remove()


@mcp.tool()
def list_analyses(limit: int = 10, status: str = "completed") -> str:
    """List recent video analyses. Returns summaries with IDs, titles, scores, and URLs. Use get_analysis for full details on a specific one."""
    session = Session()
    try:
        q = session.query(Analysis).order_by(Analysis.created_at.desc())
        if status != "all":
            q = q.filter(Analysis.status == status)
        rows = q.limit(limit).all()
        results = [_analysis_summary(r) for r in rows]
        return json.dumps(results, indent=2, default=str)
    finally:
        Session.remove()


@mcp.tool()
def search_analyses(query: str, limit: int = 10) -> str:
    """Search analyses by video title, URL, or content. Returns matching analysis summaries."""
    session = Session()
    try:
        pattern = f"%{query}%"
        rows = session.query(Analysis).filter(
            (Analysis.video_title.ilike(pattern)) |
            (Analysis.video_url.ilike(pattern))
        ).order_by(Analysis.created_at.desc()).limit(limit).all()
        results = [_analysis_summary(r) for r in rows]
        return json.dumps(results, indent=2, default=str)
    finally:
        Session.remove()


@mcp.tool()
def compare_analyses(analysis_id_1: int, analysis_id_2: int) -> str:
    """Compare two video analyses side by side. Returns both analyses with key differences highlighted: hook structure, scores, timing, format fingerprint."""
    session = Session()
    try:
        a1 = session.query(Analysis).get(analysis_id_1)
        a2 = session.query(Analysis).get(analysis_id_2)
        if not a1 or not a2:
            return "One or both analyses not found."

        d1 = _analysis_to_dict(a1)
        d2 = _analysis_to_dict(a2)
        comparison = {
            "analysis_1": d1,
            "analysis_2": d2,
            "score_comparison": {
                "analysis_1_scores": d1.get("scores", {}),
                "analysis_2_scores": d2.get("scores", {}),
            },
            "format_comparison": {
                "analysis_1_fingerprint": d1.get("format_fingerprint", {}),
                "analysis_2_fingerprint": d2.get("format_fingerprint", {}),
            },
        }
        return json.dumps(comparison, indent=2, default=str)
    finally:
        Session.remove()


@mcp.tool()
def submit_review(analysis_id: int, section_name: str, rating: str, correction: str = "") -> str:
    """Submit a review/correction for a specific section of an analysis. Corrections build fine-tuning training data. Only rate sections you have a correction for — unrated sections are left as-is. section_name must be one of: what_this_video_is, why_it_performed, viral_mechanics, hook_breakdown, timing_mastery, replication_formula, improvements, comment_ai_insights. rating must be: accurate, partial, or wrong."""
    session = Session()
    try:
        valid_sections = [
            "what_this_video_is", "why_it_performed", "viral_mechanics",
            "hook_breakdown", "timing_mastery", "replication_formula",
            "improvements", "comment_ai_insights",
        ]
        if section_name not in valid_sections:
            return f"Invalid section. Must be one of: {', '.join(valid_sections)}"
        if rating not in ("accurate", "partial", "wrong"):
            return "Invalid rating. Must be: accurate, partial, or wrong."
        if rating in ("partial", "wrong") and not correction:
            return "Correction text is required for partial/wrong ratings."

        analysis = session.query(Analysis).get(analysis_id)
        if not analysis:
            return "Analysis not found."

        data = analysis.analysis_data or {}
        section_map = {
            "what_this_video_is": data.get("what_this_video_is", ""),
            "why_it_performed": data.get("why_it_performed", ""),
            "viral_mechanics": data.get("viral_mechanics", ""),
            "hook_breakdown": json.dumps(data.get("all_hooks_identified", {})),
            "timing_mastery": json.dumps(data.get("timing_mastery", {})),
            "replication_formula": json.dumps(data.get("replication_formula", {})),
            "improvements": data.get("improvements", ""),
            "comment_ai_insights": (data.get("comment_insights") or {}).get("ai_insights", ""),
        }
        original_text = section_map.get(section_name, "")

        video_context = {
            "video_url": analysis.video_url,
            "content_type": data.get("content_type", ""),
            "transcript_snippet": (data.get("transcript") or "")[:500],
            "view_count": data.get("view_count", ""),
            "niche": data.get("niche", ""),
        }

        fb = ReviewFeedback(
            analysis_id=analysis_id,
            section_name=section_name,
            rating=rating,
            original_text=original_text,
            correction=correction if rating != "accurate" else None,
            video_context=video_context,
        )
        session.add(fb)
        session.commit()

        return f"Review saved: {section_name} rated as '{rating}'."
    finally:
        Session.remove()


@mcp.tool()
def get_review_stats() -> str:
    """Get statistics on accumulated review/training data: total reviews, ratings breakdown, number of analyses reviewed."""
    session = Session()
    try:
        total = session.query(ReviewFeedback).count()
        accurate = session.query(ReviewFeedback).filter_by(rating="accurate").count()
        partial = session.query(ReviewFeedback).filter_by(rating="partial").count()
        wrong = session.query(ReviewFeedback).filter_by(rating="wrong").count()
        analyses_reviewed = session.query(
            func.count(func.distinct(ReviewFeedback.analysis_id))
        ).scalar()

        stats = {
            "total_reviews": total,
            "accurate": accurate,
            "partial": partial,
            "wrong": wrong,
            "analyses_reviewed": analyses_reviewed,
            "training_pairs_with_corrections": partial + wrong,
        }
        return json.dumps(stats, indent=2)
    finally:
        Session.remove()


@mcp.tool()
def trigger_analysis(video_url: str, speed_tier: str = "standard", view_count: str = "", creator_note: str = "", goal: str = "") -> str:
    """Start a new video analysis. Returns the analysis ID which can be polled with check_analysis_status until status is 'completed'. Typical wait is 3-9 minutes depending on speed tier. Optional goal: viral_reach (default), sales_conversions, follower_growth, lead_generation, engagement."""
    import requests as req

    app_url = os.environ.get("APP_URL", "http://localhost:5000")
    auth_token = os.environ.get("MCP_AUTH_TOKEN", "")

    try:
        resp = req.post(
            f"{app_url}/api/analyze",
            json={
                "video_url": video_url,
                "speed_tier": speed_tier,
                "view_count": view_count,
                "creator_note": creator_note,
                "auth_token": auth_token,
                "goal": goal,
            },
            timeout=30,
        )
        if resp.ok:
            return json.dumps(resp.json(), indent=2)
        else:
            return f"Failed to trigger analysis: {resp.status_code} {resp.text}"
    except Exception as e:
        return f"Error triggering analysis: {str(e)}"


@mcp.tool()
def check_analysis_status(analysis_id: int) -> str:
    """Check the current status and progress of a running analysis. Returns stage, progress percentage, and any preview data available."""
    session = Session()
    try:
        analysis = session.query(Analysis).get(analysis_id)
        if not analysis:
            return "Analysis not found."

        result = {
            "id": analysis.id,
            "status": analysis.status,
            "video_url": analysis.video_url,
            "video_title": analysis.video_title,
            "created_at": analysis.created_at.isoformat() if analysis.created_at else None,
            "completed_at": analysis.completed_at.isoformat() if analysis.completed_at else None,
        }

        if analysis.status == "completed":
            data = analysis.analysis_data or {}
            result["scores"] = data.get("scores", {})
            result["hook"] = data.get("hook", "")

        return json.dumps(result, indent=2, default=str)
    finally:
        Session.remove()


@mcp.tool()
def get_analysis_frames(analysis_id: int, start: int = 0, count: int = 5) -> list:
    """Get frame images from an analysis. Returns up to `count` frames starting at index `start` as viewable images, plus a text summary. Frame images are extracted during video analysis and show key moments. Max 5 frames per call to keep response size manageable."""
    session = Session()
    try:
        analysis = session.query(Analysis).get(analysis_id)
        if not analysis:
            return ["Analysis not found."]
        if analysis.status != "completed":
            return [f"Analysis status is '{analysis.status}', frames only available for completed analyses."]

        data = analysis.analysis_data or {}
        frames = data.get("frame_gallery", [])
        if not frames:
            return ["No frames available for this analysis."]

        count = min(count, 5)
        start = max(0, min(start, len(frames) - 1))
        selected = frames[start:start + count]

        results = []
        results.append(f"Frames {start + 1}-{start + len(selected)} of {len(frames)} total for \"{analysis.video_title or 'Untitled'}\"")

        for i, frame in enumerate(selected):
            frame_data = frame if isinstance(frame, str) else frame.get("image", "")
            if not frame_data:
                results.append(f"Frame {start + i + 1}: (empty)")
                continue

            if frame_data.startswith("data:"):
                header, b64_str = frame_data.split(",", 1)
                mime = header.split(":")[1].split(";")[0] if ":" in header else "image/jpeg"
            else:
                b64_str = frame_data
                mime = "image/jpeg"

            fmt = mime.split("/")[1] if "/" in mime else "jpeg"
            if fmt == "jpg":
                fmt = "jpeg"

            try:
                img_bytes = base64.b64decode(b64_str)
                results.append(MCPImage(data=img_bytes, format=fmt))
            except Exception:
                results.append(f"Frame {start + i + 1}: (failed to decode)")

            if isinstance(frame, dict):
                desc = frame.get("description", "")
                ts = frame.get("timestamp", "")
                if desc or ts:
                    label = f"Frame {start + i + 1}"
                    if ts:
                        label += f" @ {ts}s"
                    if desc:
                        label += f": {desc}"
                    results.append(label)

        if start + count < len(frames):
            results.append(f"More frames available. Call get_analysis_frames(analysis_id={analysis_id}, start={start + count}) to see the next batch.")

        return results
    finally:
        Session.remove()


@mcp.tool()
def get_analysis_pdf(analysis_id: int) -> str:
    """Generate and retrieve the PDF report for a completed analysis. Returns the PDF as a downloadable link via the web app. Note: PDF generation requires the web app to be running."""
    import requests as req

    session = Session()
    try:
        analysis = session.query(Analysis).get(analysis_id)
        if not analysis:
            return "Analysis not found."
        if analysis.status != "completed":
            return f"Analysis status is '{analysis.status}', PDF only available for completed analyses."

        cache_key = analysis.pdf_cache_key
        if not cache_key:
            return "No PDF cache key found for this analysis. The PDF may not have been generated yet."

        app_url = os.environ.get("APP_URL", "http://localhost:5000")
        auth_token = os.environ.get("MCP_AUTH_TOKEN", "")

        try:
            resp = req.get(
                f"{app_url}/api/analysis/{analysis_id}/pdf",
                params={"auth_token": auth_token},
                timeout=60,
            )
            if resp.ok:
                result = resp.json()
                return json.dumps({
                    "status": "success",
                    "analysis_id": analysis_id,
                    "video_title": analysis.video_title,
                    "pdf_url": result.get("pdf_url", ""),
                    "message": "PDF is available at the URL above. Share this link to view/download the report.",
                }, indent=2)
            else:
                return f"Failed to get PDF: {resp.status_code} {resp.text}"
        except Exception as e:
            return f"Error getting PDF (is the web app running?): {str(e)}"
    finally:
        Session.remove()


# ---------------------------------------------------------------------------
# Health check route (so Render's health check doesn't 404)
# ---------------------------------------------------------------------------

@mcp.custom_route("/health", methods=["GET", "HEAD"])
async def health(request):
    from starlette.responses import JSONResponse
    return JSONResponse({"status": "ok", "server": "video-analysis-mcp"})


@mcp.custom_route("/", methods=["GET", "HEAD"])
async def root(request):
    from starlette.responses import JSONResponse
    return JSONResponse({"status": "ok", "server": "video-analysis-mcp", "transport": "/mcp"})


# ---------------------------------------------------------------------------
# Entry point — streamable HTTP for remote, stdio for local
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "--stdio" in sys.argv:
        logger.info("Starting MCP Server (stdio mode)")
        mcp.run(transport="stdio")
    else:
        port = int(os.environ.get("MCP_PORT", os.environ.get("PORT", "8001")))
        transport = os.environ.get("MCP_TRANSPORT", "streamable-http")
        logger.info(f"Starting MCP Server ({transport} mode) on port {port}")
        if transport == "sse":
            mcp.run(transport="sse", host="0.0.0.0", port=port)
        else:
            mcp.run(transport="streamable-http", host="0.0.0.0", port=port)
