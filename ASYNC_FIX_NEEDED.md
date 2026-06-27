# Async Processing Fix Needed

## Current Problem
`/process` endpoint runs synchronously for 2-4 minutes → causes 502 timeout on Render

## Current Flow (BROKEN)
```
Form → /analyze_async → progress.html
progress.html → fetch(/process) → WAITS 2-4 min → timeout ❌
```

## What Needs to Change

### Option 1: Background Thread (Quick Fix)
```python
import threading

@app.route("/process", methods=["POST"])
def process():
    # Create processing record
    analysis_id = create_processing_analysis(user_id, video_url)
    
    # Run analysis in background thread
    thread = threading.Thread(
        target=run_analysis_background,
        args=(analysis_id, form_data)
    )
    thread.daemon = True
    thread.start()
    
    # Return immediately
    return jsonify({
        'analysis_id': analysis_id,
        'status': 'processing'
    })

def run_analysis_background(analysis_id, form_data):
    """Runs in background thread"""
    try:
        # ... do all the analysis ...
        complete_analysis(analysis_id, results)
    except Exception as e:
        fail_analysis(analysis_id, str(e))
```

Then update `progress.html` to poll:
```javascript
// Instead of waiting for /process
fetch("/process", {method: "POST", body: formData})
    .then(r => r.json())
    .then(data => {
        // Start polling
        const analysisId = data.analysis_id;
        pollForCompletion(analysisId);
    });

function pollForCompletion(id) {
    const interval = setInterval(() => {
        fetch(`/api/analysis_status/${id}`)
            .then(r => r.json())
            .then(data => {
                if (data.status === 'completed') {
                    clearInterval(interval);
                    window.location.href = `/analysis/${id}`;
                } else if (data.status === 'failed') {
                    clearInterval(interval);
                    showError(data.error);
                }
            });
    }, 2000); // Poll every 2 seconds
}
```

### Option 2: Use Celery/RQ (Better Long-term)
For production, use a proper task queue like Celery or RQ.

### Option 3: Increase Render Timeout (Temporary)
Only works on paid plans, max 300s. Analysis can exceed this.

## Recommended: Option 1
- Quick to implement
- Works with current infrastructure
- No new dependencies
- Solves timeout immediately
