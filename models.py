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

    # Analysis data stored as JSON
    analysis_data = db.Column(db.JSON)

    # PDF cache key for downloading
    pdf_cache_key = db.Column(db.String(32))

    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    def __repr__(self):
        return f'<Analysis {self.id}: {self.video_title[:30] if self.video_title else "Untitled"}>'


def init_db(app):
    """Initialize database with app context."""
    # Get database URL from environment or use default
    database_url = os.environ.get('DATABASE_URL')

    # Handle Heroku-style postgres:// URLs (need to be postgresql://)
    if database_url and database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)

    # Default to SQLite for local development if no DATABASE_URL
    if not database_url:
        database_url = 'sqlite:///video_analysis.db'
        print("[DB] No DATABASE_URL found, using SQLite for local development")

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
