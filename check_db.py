"""
Temporary script to check what's in the database.
Run this to see if transcripts are being saved.
"""
import os
os.environ.setdefault('DATABASE_URL', 'your_database_url_here')

from models import db, Analysis, init_db
from flask import Flask

app = Flask(__name__)
init_db(app)

with app.app_context():
    # Get most recent analysis
    latest = Analysis.query.order_by(Analysis.id.desc()).first()

    if latest:
        print(f"\n=== Latest Analysis (ID: {latest.id}) ===")
        print(f"Video: {latest.video_title}")
        print(f"Status: {latest.status}")
        print(f"Created: {latest.created_at}")

        if latest.analysis_data:
            print(f"\n=== Saved Fields ===")
            print(f"Keys: {list(latest.analysis_data.keys())}")

            print(f"\n=== Transcript Check ===")
            if 'transcript' in latest.analysis_data:
                transcript = latest.analysis_data['transcript']
                if transcript:
                    print(f"✅ Transcript saved ({len(transcript)} chars)")
                    print(f"Preview: {transcript[:200]}...")
                else:
                    print("❌ Transcript field exists but is EMPTY")
            else:
                print("❌ Transcript field NOT in analysis_data")

            print(f"\n=== Video Description Check ===")
            if 'video_description' in latest.analysis_data:
                desc = latest.analysis_data['video_description']
                if desc:
                    print(f"✅ Description saved: {desc[:100]}...")
                else:
                    print("❌ Description field exists but is EMPTY")
            else:
                print("❌ Description field NOT in analysis_data")
        else:
            print("❌ No analysis_data saved at all")
    else:
        print("No analyses found in database")
