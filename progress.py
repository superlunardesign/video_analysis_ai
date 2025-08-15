import os
import json
import redis

# Configure Redis URL (Render provides REDIS_URL in environment if you add Redis service)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
r = redis.Redis.from_url(REDIS_URL, decode_responses=True)

# TTL for job keys in seconds (1 day = 86400 seconds)
JOB_TTL = int(os.getenv("JOB_TTL", "86400"))


def _job_key(job_id):
    """Return the Redis key for a given job_id."""
    return f"job:{job_id}"


def start(job_id):
    """Initialize a new job in Redis."""
    data = {
        "stage": "Starting…",
        "percent": 0,
        "done": False,
        "error": None,
        "result": None
    }
    r.set(_job_key(job_id), json.dumps(data), ex=JOB_TTL)


def set_progress(job_id, stage, percent):
    """Update the job progress."""
    data = get(job_id) or {}
    data.update({
        "stage": stage,
        "percent": percent,
        "done": False
    })
    r.set(_job_key(job_id), json.dumps(data), ex=JOB_TTL)


def set_error(job_id, error_msg):
    """Mark the job as failed."""
    data = get(job_id) or {}
    data.update({
        "stage": "Error",
        "error": error_msg,
        "done": True
    })
    r.set(_job_key(job_id), json.dumps(data), ex=JOB_TTL)


def set_result(job_id, result_data):
    """Save the job result and mark as done."""
    data = get(job_id) or {}
    data.update({
        "stage": "Completed",
        "percent": 100,
        "done": True,
        "result": result_data
    })
    r.set(_job_key(job_id), json.dumps(data), ex=JOB_TTL)


def get(job_id):
    """Retrieve the job data from Redis."""
    raw = r.get(_job_key(job_id))
    if not raw:
        return None
    return json.loads(raw)


def pop_result(job_id):
    """Get and remove the job result from Redis."""
    data = get(job_id)
    if not data:
        return None
    result = data.get("result")
    # Remove the job key entirely
    r.delete(_job_key(job_id))
    return result
