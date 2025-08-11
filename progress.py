# progress.py — tiny in-memory progress + result store (thread-safe)

from threading import Lock

_store = {}
_lock = Lock()

def start(job_id: str):
    with _lock:
        _store[job_id] = {
            "stage": "Queued",
            "percent": 0,
            "done": False,
            "error": None,
            "result": None,   # put final payload here
        }

def set_progress(job_id: str, stage: str, percent: int):
    with _lock:
        if job_id in _store:
            _store[job_id]["stage"] = stage
            _store[job_id]["percent"] = max(0, min(100, percent))

def set_error(job_id: str, msg: str):
    with _lock:
        if job_id in _store:
            _store[job_id]["error"] = msg
            _store[job_id]["done"] = True
            _store[job_id]["percent"] = 100

def set_result(job_id: str, result_dict: dict):
    with _lock:
        if job_id in _store:
            _store[job_id]["result"] = result_dict
            _store[job_id]["done"] = True
            _store[job_id]["percent"] = 100
            _store[job_id]["stage"] = "Complete"

def get(job_id: str):
    with _lock:
        return _store.get(job_id)

def pop_result(job_id: str):
    with _lock:
        data = _store.get(job_id)
        if not data:
            return None
        # keep minimal status but return result once
        return data.get("result")
