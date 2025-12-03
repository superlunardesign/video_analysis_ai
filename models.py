"""Database models for user accounts and analysis history."""
import os
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


class User(UserMixin, db.Model):
    """User account model."""
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationship to analyses
    analyses = db.relationship('Analysis', backref='user', lazy='dynamic', cascade='all, delete-orphan')

    def set_password(self, password):
        """Hash and store password."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """Verify password against hash."""
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.email}>'


class Analysis(db.Model):
    """Video analysis history model."""
    __tablename__ = 'analyses'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)

    # Video info
    video_url = db.Column(db.String(500), nullable=False)
    video_title = db.Column(db.String(500))
    thumbnail_url = db.Column(db.Text)  # Base64 or URL

    # Analysis status: 'processing', 'completed', 'failed'
    status = db.Column(db.String(20), default='processing', index=True)

    # Current processing stage for real-time progress
    current_stage = db.Column(db.String(50), default='queued')
    stage_progress = db.Column(db.Integer, default=0)  # 0-100 for current stage

    # Analysis data stored as JSON
    analysis_data = db.Column(db.JSON)

    # PDF cache key for downloading
    pdf_cache_key = db.Column(db.String(32))

    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    completed_at = db.Column(db.DateTime)

    def __repr__(self):
        return f'<Analysis {self.id}: {self.video_title[:30] if self.video_title else "Untitled"}>'


def init_db(app):
    """Initialize database with app context."""
    # Get database URL from environment or use default
    raw_url = os.environ.get('DATABASE_URL', '')
    database_url = raw_url.strip()

    # Debug: detailed logging
    print(f"[DB] Raw DATABASE_URL length: {len(raw_url)}")
    print(f"[DB] Stripped DATABASE_URL length: {len(database_url)}")

    if database_url:
        # Check for common issues
        if database_url.startswith('"') or database_url.startswith("'"):
            print("[DB WARNING] DATABASE_URL starts with quote - removing quotes")
            database_url = database_url.strip('"\'')

        # Mask password for logging but show structure
        if '://' in database_url:
            protocol = database_url.split('://')[0]
            print(f"[DB] Protocol detected: {protocol}")
        else:
            print(f"[DB ERROR] No '://' found in URL. First 50 chars: {database_url[:50]}")

        # Show masked version
        masked = database_url[:25] + '...' if len(database_url) > 25 else database_url
        print(f"[DB] DATABASE_URL starts with: {masked}")

    # Handle Heroku-style postgres:// URLs (need to be postgresql://)
    if database_url and database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)
        print("[DB] Converted postgres:// to postgresql://")

    # Validate URL format
    if database_url and not database_url.startswith(('postgresql://', 'sqlite://')):
        print(f"[DB ERROR] Invalid DATABASE_URL format. Must start with 'postgresql://' or 'sqlite://'")
        print(f"[DB ERROR] First 50 chars: {database_url[:50]}")
        print("[DB] Falling back to SQLite")
        database_url = ''

    # Default to SQLite for local development if no DATABASE_URL
    if not database_url:
        database_url = 'sqlite:///video_analysis.db'
        print("[DB] No valid DATABASE_URL found, using SQLite for local development")

    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
        'pool_pre_ping': True,  # Verify connections before use
        'pool_recycle': 300,    # Recycle connections every 5 minutes
    }

    db.init_app(app)

    with app.app_context():
        db.create_all()
        print("[DB] Database tables created/verified")

        # Run migrations for new columns
        _run_migrations(app, database_url)


def _run_migrations(app, database_url):
    """Run database migrations to add new columns to existing tables."""
    from sqlalchemy import text, inspect

    # Check if we're using PostgreSQL
    if not database_url.startswith('postgresql'):
        return

    try:
        inspector = inspect(db.engine)
        columns = [col['name'] for col in inspector.get_columns('analyses')]
    except Exception as e:
        print(f"[DB MIGRATION] Could not inspect table: {e}")
        return

    # Migration: Add 'status' column if it doesn't exist
    if 'status' not in columns:
        try:
            print("[DB MIGRATION] Adding 'status' column to analyses table...")
            db.session.execute(text(
                "ALTER TABLE analyses ADD COLUMN status VARCHAR(20) DEFAULT 'completed'"
            ))
            db.session.commit()
            print("[DB MIGRATION] Added 'status' column")
        except Exception as e:
            db.session.rollback()
            if 'already exists' in str(e).lower() or 'duplicate' in str(e).lower():
                print("[DB MIGRATION] 'status' column already exists (added by another worker)")
            else:
                print(f"[DB MIGRATION ERROR] Failed to add 'status': {e}")

    # Migration: Add 'completed_at' column if it doesn't exist
    if 'completed_at' not in columns:
        try:
            print("[DB MIGRATION] Adding 'completed_at' column to analyses table...")
            db.session.execute(text(
                "ALTER TABLE analyses ADD COLUMN completed_at TIMESTAMP"
            ))
            db.session.commit()
            print("[DB MIGRATION] Added 'completed_at' column")
        except Exception as e:
            db.session.rollback()
            if 'already exists' in str(e).lower() or 'duplicate' in str(e).lower():
                print("[DB MIGRATION] 'completed_at' column already exists (added by another worker)")
            else:
                print(f"[DB MIGRATION ERROR] Failed to add 'completed_at': {e}")

    # Migration: Add 'current_stage' column if it doesn't exist
    if 'current_stage' not in columns:
        try:
            print("[DB MIGRATION] Adding 'current_stage' column to analyses table...")
            db.session.execute(text(
                "ALTER TABLE analyses ADD COLUMN current_stage VARCHAR(50) DEFAULT 'queued'"
            ))
            db.session.commit()
            print("[DB MIGRATION] Added 'current_stage' column")
        except Exception as e:
            db.session.rollback()
            if 'already exists' in str(e).lower() or 'duplicate' in str(e).lower():
                print("[DB MIGRATION] 'current_stage' column already exists (added by another worker)")
            else:
                print(f"[DB MIGRATION ERROR] Failed to add 'current_stage': {e}")

    # Migration: Add 'stage_progress' column if it doesn't exist
    if 'stage_progress' not in columns:
        try:
            print("[DB MIGRATION] Adding 'stage_progress' column to analyses table...")
            db.session.execute(text(
                "ALTER TABLE analyses ADD COLUMN stage_progress INTEGER DEFAULT 0"
            ))
            db.session.commit()
            print("[DB MIGRATION] Added 'stage_progress' column")
        except Exception as e:
            db.session.rollback()
            if 'already exists' in str(e).lower() or 'duplicate' in str(e).lower():
                print("[DB MIGRATION] 'stage_progress' column already exists (added by another worker)")
            else:
                print(f"[DB MIGRATION ERROR] Failed to add 'stage_progress': {e}")
