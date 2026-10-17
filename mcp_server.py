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
import time
import asyncio
import logging
from datetime import datetime

from mcp.server import Server
from mcp.types import Tool, TextContent, ImageContent

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

# Import models after engine is created — they use Flask-SQLAlchemy's db,
# but we query with raw SQLAlchemy sessions here.
from models import Analysis, ReviewFeedback, KnowledgePattern

# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------
MCP_AUTH_TOKEN = os.environ.get("MCP_AUTH_TOKEN", "")

# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------
server = Server("video-analysis")


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

    # Core analysis sections
    for key in [
        "what_this_video_is", "why_it_performed", "viral_mechanics",
        "performance_prediction", "improvements", "transcript",
        "scores", "all_hooks_identified", "exact_hook_breakdown",
        "replication_formula", "timing_mastery", "format_fingerprint",
        "format_match", "comment_insights",
    ]:
        if data.get(key):
            result[key] = data[key]

    # Metrics
    for key in ["view_count", "like_count", "comment_count", "share_count"]:
        if data.get(key):
            result[key] = data[key]

    # Metadata
    for key in ["content_type", "niche", "platform", "audio_type",
                 "music_info", "video_description", "hashtags"]:
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
# Tool definitions
# ---------------------------------------------------------------------------

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="get_analysis",
            description="Get the full analysis for a specific video by analysis ID. Returns all sections: what it is, why it performed, hooks, timing, replication formula, transcript, scores, comments, and frame descriptions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "analysis_id": {"type": "integer", "description": "The analysis ID to retrieve"},
                    "include_frames": {"type": "boolean", "description": "Include base64 frame images (large). Default false.", "default": False},
                },
                "required": ["analysis_id"],
            },
        ),
        Tool(
            name="list_analyses",
            description="List recent video analyses. Returns summaries with IDs, titles, scores, and URLs. Use get_analysis for full details on a specific one.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max results (default 10)", "default": 10},
                    "status": {"type": "string", "enum": ["completed", "processing", "failed", "all"], "description": "Filter by status (default completed)", "default": "completed"},
                },
            },
        ),
        Tool(
            name="search_analyses",
            description="Search analyses by video title, URL, or content. Returns matching analysis summaries.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search term to match against video titles, URLs, and descriptions"},
                    "limit": {"type": "integer", "description": "Max results (default 10)", "default": 10},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="compare_analyses",
            description="Compare two video analyses side by side. Returns both analyses with key differences highlighted: hook structure, scores, timing, format fingerprint.",
            inputSchema={
                "type": "object",
                "properties": {
                    "analysis_id_1": {"type": "integer", "description": "First analysis ID"},
                    "analysis_id_2": {"type": "integer", "description": "Second analysis ID"},
                },
                "required": ["analysis_id_1", "analysis_id_2"],
            },
        ),
        Tool(
            name="submit_review",
            description="Submit a review/correction for a specific section of an analysis. Corrections build fine-tuning training data. Only rate sections you have a correction for — unrated sections are left as-is.",
            inputSchema={
                "type": "object",
                "properties": {
                    "analysis_id": {"type": "integer", "description": "The analysis to review"},
                    "section_name": {"type": "string", "description": "Section to review", "enum": [
                        "what_this_video_is", "why_it_performed", "viral_mechanics",
                        "hook_breakdown", "timing_mastery", "replication_formula",
                        "improvements", "comment_ai_insights",
                    ]},
                    "rating": {"type": "string", "enum": ["accurate", "partial", "wrong"], "description": "How correct the section is"},
                    "correction": {"type": "string", "description": "What the section should have said (required for partial/wrong)"},
                },
                "required": ["analysis_id", "section_name", "rating"],
            },
        ),
        Tool(
            name="get_review_stats",
            description="Get statistics on accumulated review/training data: total reviews, ratings breakdown, number of analyses reviewed.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="trigger_analysis",
            description="Start a new video analysis. Returns the analysis ID which can be polled with get_analysis until status is 'completed'. Typical wait is 3-9 minutes depending on speed tier.",
            inputSchema={
                "type": "object",
                "properties": {
                    "video_url": {"type": "string", "description": "TikTok, YouTube, or Instagram video URL"},
                    "speed_tier": {"type": "string", "enum": ["transcript_only", "ultra_fast", "fast", "standard", "thorough"], "description": "Analysis depth (default standard)", "default": "standard"},
                    "view_count": {"type": "string", "description": "Video view count if known (e.g. '1.5M', '500000')"},
                    "creator_note": {"type": "string", "description": "Optional context about the video from the creator"},
                },
                "required": ["video_url"],
            },
        ),
        Tool(
            name="check_analysis_status",
            description="Check the current status and progress of a running analysis. Returns stage, progress percentage, and any preview data available.",
            inputSchema={
                "type": "object",
                "properties": {
                    "analysis_id": {"type": "integer", "description": "The analysis ID to check"},
                },
                "required": ["analysis_id"],
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    session = Session()
    try:
        if name == "get_analysis":
            analysis = session.query(Analysis).get(arguments["analysis_id"])
            if not analysis:
                return [TextContent(type="text", text="Analysis not found.")]

            result = _analysis_to_dict(analysis)

            # Optionally include frame images
            if arguments.get("include_frames"):
                data = analysis.analysis_data or {}
                frames = data.get("frame_gallery", [])
                if frames:
                    result["frame_count"] = len(frames)
                    contents = [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
                    for i, frame_b64 in enumerate(frames[:10]):
                        if frame_b64.startswith("data:image"):
                            media_type = frame_b64.split(";")[0].split(":")[1]
                            b64_data = frame_b64.split(",", 1)[1]
                            contents.append(ImageContent(
                                type="image",
                                mimeType=media_type,
                                data=b64_data,
                            ))
                    return contents

            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        elif name == "list_analyses":
            limit = arguments.get("limit", 10)
            status = arguments.get("status", "completed")
            q = session.query(Analysis).order_by(Analysis.created_at.desc())
            if status != "all":
                q = q.filter(Analysis.status == status)
            rows = q.limit(limit).all()
            results = [_analysis_summary(r) for r in rows]
            return [TextContent(type="text", text=json.dumps(results, indent=2, default=str))]

        elif name == "search_analyses":
            query = arguments["query"]
            limit = arguments.get("limit", 10)
            pattern = f"%{query}%"
            rows = session.query(Analysis).filter(
                (Analysis.video_title.ilike(pattern)) |
                (Analysis.video_url.ilike(pattern))
            ).order_by(Analysis.created_at.desc()).limit(limit).all()
            results = [_analysis_summary(r) for r in rows]
            return [TextContent(type="text", text=json.dumps(results, indent=2, default=str))]

        elif name == "compare_analyses":
            a1 = session.query(Analysis).get(arguments["analysis_id_1"])
            a2 = session.query(Analysis).get(arguments["analysis_id_2"])
            if not a1 or not a2:
                return [TextContent(type="text", text="One or both analyses not found.")]

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
            return [TextContent(type="text", text=json.dumps(comparison, indent=2, default=str))]

        elif name == "submit_review":
            analysis = session.query(Analysis).get(arguments["analysis_id"])
            if not analysis:
                return [TextContent(type="text", text="Analysis not found.")]

            rating = arguments["rating"]
            correction = arguments.get("correction")
            if rating in ("partial", "wrong") and not correction:
                return [TextContent(type="text", text="Correction text is required for partial/wrong ratings.")]

            data = analysis.analysis_data or {}
            section = arguments["section_name"]

            # Get original text for the section
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
            original_text = section_map.get(section, "")

            video_context = {
                "video_url": analysis.video_url,
                "content_type": data.get("content_type", ""),
                "transcript_snippet": (data.get("transcript") or "")[:500],
                "view_count": data.get("view_count", ""),
                "niche": data.get("niche", ""),
            }

            fb = ReviewFeedback(
                analysis_id=arguments["analysis_id"],
                section_name=section,
                rating=rating,
                original_text=original_text,
                correction=correction if rating != "accurate" else None,
                video_context=video_context,
            )
            session.add(fb)
            session.commit()

            return [TextContent(type="text", text=f"Review saved: {section} rated as '{rating}'.")]

        elif name == "get_review_stats":
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
            return [TextContent(type="text", text=json.dumps(stats, indent=2))]

        elif name == "trigger_analysis":
            # This tool needs to call the Flask app's API to trigger analysis.
            # It creates a processing record and calls the background analysis.
            import requests as req

            app_url = os.environ.get("APP_URL", "http://localhost:5000")
            auth_token = os.environ.get("MCP_AUTH_TOKEN", "")

            resp = req.post(
                f"{app_url}/api/analyze",
                json={
                    "video_url": arguments["video_url"],
                    "speed_tier": arguments.get("speed_tier", "standard"),
                    "view_count": arguments.get("view_count", ""),
                    "creator_note": arguments.get("creator_note", ""),
                    "auth_token": auth_token,
                },
                timeout=30,
            )

            if resp.ok:
                result = resp.json()
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            else:
                return [TextContent(type="text", text=f"Failed to trigger analysis: {resp.status_code} {resp.text}")]

        elif name == "check_analysis_status":
            analysis = session.query(Analysis).get(arguments["analysis_id"])
            if not analysis:
                return [TextContent(type="text", text="Analysis not found.")]

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

            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        logger.error(f"Tool error ({name}): {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]
    finally:
        Session.remove()


# ---------------------------------------------------------------------------
# Entry point — SSE transport for remote deployment, stdio for local
# ---------------------------------------------------------------------------

def create_starlette_app():
    """Create a Starlette app that serves the MCP server over SSE."""
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.routing import Mount, Route
    from starlette.responses import JSONResponse

    sse = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await server.run(
                streams[0], streams[1], server.create_initialization_options()
            )

    async def health(request):
        return JSONResponse({"status": "ok", "server": "video-analysis-mcp"})

    return Starlette(
        debug=False,
        routes=[
            Route("/health", health),
            Mount("/sse", app=sse.handle_post_message, name="sse_post"),
            Route("/sse", endpoint=handle_sse),
        ],
    )


if __name__ == "__main__":
    import sys

    if "--stdio" in sys.argv:
        from mcp.server.stdio import run_server
        logger.info("Starting MCP Server (stdio mode)")
        asyncio.run(run_server(server))
    else:
        import uvicorn
        port = int(os.environ.get("MCP_PORT", os.environ.get("PORT", "8001")))
        logger.info(f"Starting MCP Server (SSE mode) on port {port}")
        app = create_starlette_app()
        uvicorn.run(app, host="0.0.0.0", port=port)
