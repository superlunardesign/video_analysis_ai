import sqlite3
import json
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "jobs.db")


def _init_db():
    """Initialize the jobs table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            stage TEXT,
            percent INTEGER,
            done INTEGER,
            error TEXT,
            result TEXT
        )
    """)
    conn.commit()
    conn.close()


def start(job_id):
    _init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT OR REPLACE INTO jobs (job_id, stage, percent, done, error, result)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (job_id, "Starting…", 0, 0, None, None))
    conn.commit()
    conn.close()


def set_progress(job_id, stage, percent):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        UPDATE jobs SET stage=?, percent=?, done=0 WHERE job_id=?
    """, (stage, percent, job_id))
    conn.commit()
    conn.close()


def set_error(job_id, error_msg):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        UPDATE jobs SET stage=?, error=?, done=1 WHERE job_id=?
    """, ("Error", error_msg, job_id))
    conn.commit()
    conn.close()


def set_result(job_id, result_data):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        UPDATE jobs SET stage=?, percent=?, done=1, result=? WHERE job_id=?
    """, ("Completed", 100, json.dumps(result_data), job_id))
    conn.commit()
    conn.close()


def get(job_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT stage, percent, done, error, result FROM jobs WHERE job_id=?", (job_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    stage, percent, done, error, result = row
    return {
        "stage": stage,
        "percent": percent,
        "done": bool(done),
        "error": error,
        "result": json.loads(result) if result else None
    }


def pop_result(job_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT result FROM jobs WHERE job_id=?", (job_id,))
    row = c.fetchone()
    c.execute("DELETE FROM jobs WHERE job_id=?", (job_id,))
    conn.commit()
    conn.close()
    if not row:
        return None
    result = row[0]
    return json.loads(result) if result else None