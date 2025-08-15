import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from analyze_tiktok import analyze_tiktok_video

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # URL or file upload
        tiktok_url = request.form.get("tiktok_url", "").strip()
        uploaded_file = request.files.get("video_file")
        goal = request.form.get("goal", "General Analysis")

        if tiktok_url:
            # Analyze from URL
            analysis_results = analyze_tiktok_video(tiktok_url, goal=goal)
            return render_template("results.html", **analysis_results)

        elif uploaded_file and allowed_file(uploaded_file.filename):
            filename = secure_filename(uploaded_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(filepath)
            analysis_results = analyze_tiktok_video(filepath, local_file=True, goal=goal)
            return render_template("results.html", **analysis_results)

        else:
            return "Please provide a valid TikTok URL or upload a valid video file.", 400

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
