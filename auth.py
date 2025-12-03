"""Authentication routes for user signup, login, logout, and account management."""
import re
from datetime import datetime, timedelta
from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from models import db, User, Analysis
from email_utils import generate_code, send_password_reset_email, send_email_change_confirmation

auth_bp = Blueprint('auth', __name__)
login_manager = LoginManager()


def init_auth(app):
    """Initialize authentication with the Flask app."""
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Please log in to access this page.'
    login_manager.login_message_category = 'info'

    # Set secret key for sessions
    import os
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', os.urandom(24).hex())


@login_manager.user_loader
def load_user(user_id):
    """Load user by ID for Flask-Login."""
    return User.query.get(int(user_id))


def is_valid_email(email):
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


@auth_bp.route('/signup', methods=['GET', 'POST'])
def signup():
    """Handle user registration."""
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        # Validation
        errors = []

        if not email:
            errors.append('Email is required.')
        elif not is_valid_email(email):
            errors.append('Please enter a valid email address.')

        if not password:
            errors.append('Password is required.')
        elif len(password) < 8:
            errors.append('Password must be at least 8 characters.')

        if password != confirm_password:
            errors.append('Passwords do not match.')

        # Check if user already exists
        if User.query.filter_by(email=email).first():
            errors.append('An account with this email already exists.')

        if errors:
            for error in errors:
                flash(error, 'error')
            return render_template('signup.html', email=email)

        # Create user
        user = User(email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        # Log in the new user
        login_user(user)
        flash('Account created successfully!', 'success')
        return redirect(url_for('index'))

    return render_template('signup.html')


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login."""
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        remember = request.form.get('remember', False)

        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            login_user(user, remember=remember)
            flash('Logged in successfully!', 'success')

            # Redirect to next page if specified
            next_page = request.args.get('next')
            if next_page:
                return redirect(next_page)
            return redirect(url_for('index'))
        else:
            flash('Invalid email or password.', 'error')
            return render_template('login.html', email=email)

    return render_template('login.html')


@auth_bp.route('/logout')
@login_required
def logout():
    """Handle user logout."""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))


# ==================== Account Management ====================

@auth_bp.route('/account')
@login_required
def account():
    """Display account settings page."""
    # Get user stats
    analysis_count = Analysis.query.filter_by(user_id=current_user.id, status='completed').count()
    pdf_count = Analysis.query.filter_by(user_id=current_user.id).filter(Analysis.pdf_cache_key.isnot(None)).count()

    return render_template('account.html',
                           analysis_count=analysis_count,
                           pdf_count=pdf_count,
                           pending_email=current_user.pending_email)


@auth_bp.route('/account/change-password', methods=['POST'])
@login_required
def change_password():
    """Handle password change for logged-in users."""
    current_password = request.form.get('current_password', '')
    new_password = request.form.get('new_password', '')
    confirm_password = request.form.get('confirm_password', '')

    # Validate current password
    if not current_user.check_password(current_password):
        flash('Current password is incorrect.', 'error')
        return redirect(url_for('auth.account'))

    # Validate new password
    if len(new_password) < 6:
        flash('New password must be at least 6 characters.', 'error')
        return redirect(url_for('auth.account'))

    if new_password != confirm_password:
        flash('New passwords do not match.', 'error')
        return redirect(url_for('auth.account'))

    # Update password
    current_user.set_password(new_password)
    db.session.commit()

    flash('Password updated successfully!', 'success')
    return redirect(url_for('auth.account'))


@auth_bp.route('/account/change-email', methods=['POST'])
@login_required
def change_email():
    """Initiate email change process - sends confirmation to new email."""
    new_email = request.form.get('new_email', '').strip().lower()
    password = request.form.get('password', '')

    # Validate password
    if not current_user.check_password(password):
        flash('Password is incorrect.', 'error')
        return redirect(url_for('auth.account'))

    # Validate new email
    if not is_valid_email(new_email):
        flash('Please enter a valid email address.', 'error')
        return redirect(url_for('auth.account'))

    if new_email == current_user.email:
        flash('New email is the same as your current email.', 'error')
        return redirect(url_for('auth.account'))

    # Check if email is already in use
    if User.query.filter_by(email=new_email).first():
        flash('This email is already in use by another account.', 'error')
        return redirect(url_for('auth.account'))

    # Generate confirmation code
    code = generate_code()
    current_user.pending_email = new_email
    current_user.email_confirm_token = code
    current_user.email_confirm_expires = datetime.utcnow() + timedelta(minutes=15)
    db.session.commit()

    # Send confirmation email to new address
    if send_email_change_confirmation(new_email, code, new_email):
        flash(f'Confirmation code sent to {new_email}. Check your inbox.', 'success')
    else:
        flash('Confirmation code generated. Check the server logs for the code (SMTP not configured).', 'info')

    return redirect(url_for('auth.account'))


@auth_bp.route('/account/confirm-email', methods=['POST'])
@login_required
def confirm_email():
    """Confirm email change with verification code."""
    code = request.form.get('code', '').strip()

    if not current_user.pending_email:
        flash('No pending email change.', 'error')
        return redirect(url_for('auth.account'))

    # Check if code expired
    if current_user.email_confirm_expires < datetime.utcnow():
        current_user.pending_email = None
        current_user.email_confirm_token = None
        current_user.email_confirm_expires = None
        db.session.commit()
        flash('Confirmation code has expired. Please try again.', 'error')
        return redirect(url_for('auth.account'))

    # Verify code
    if code != current_user.email_confirm_token:
        flash('Invalid confirmation code.', 'error')
        return redirect(url_for('auth.account'))

    # Update email
    old_email = current_user.email
    current_user.email = current_user.pending_email
    current_user.pending_email = None
    current_user.email_confirm_token = None
    current_user.email_confirm_expires = None
    db.session.commit()

    flash(f'Email changed from {old_email} to {current_user.email}!', 'success')
    return redirect(url_for('auth.account'))


@auth_bp.route('/account/cancel-email-change', methods=['POST'])
@login_required
def cancel_email_change():
    """Cancel pending email change."""
    current_user.pending_email = None
    current_user.email_confirm_token = None
    current_user.email_confirm_expires = None
    db.session.commit()

    flash('Email change cancelled.', 'info')
    return redirect(url_for('auth.account'))


# ==================== Password Reset ====================

@auth_bp.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    """Handle forgot password - send reset code to email."""
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()

        user = User.query.filter_by(email=email).first()

        if user:
            # Generate reset code
            code = generate_code()
            user.reset_token = code
            user.reset_token_expires = datetime.utcnow() + timedelta(minutes=15)
            db.session.commit()

            # Send reset email
            send_password_reset_email(email, code)

        # Always show the same message for security (don't reveal if email exists)
        flash('If an account exists with that email, a reset code has been sent.', 'info')
        return render_template('reset_password.html', email=email)

    return render_template('forgot_password.html')


@auth_bp.route('/reset-password', methods=['POST'])
def reset_password():
    """Handle password reset with verification code."""
    email = request.form.get('email', '').strip().lower()
    code = request.form.get('code', '').strip()
    password = request.form.get('password', '')
    confirm_password = request.form.get('confirm_password', '')

    user = User.query.filter_by(email=email).first()

    if not user:
        flash('Invalid reset request.', 'error')
        return redirect(url_for('auth.forgot_password'))

    # Check if code expired
    if not user.reset_token_expires or user.reset_token_expires < datetime.utcnow():
        user.reset_token = None
        user.reset_token_expires = None
        db.session.commit()
        flash('Reset code has expired. Please request a new one.', 'error')
        return redirect(url_for('auth.forgot_password'))

    # Verify code
    if code != user.reset_token:
        flash('Invalid reset code.', 'error')
        return render_template('reset_password.html', email=email)

    # Validate new password
    if len(password) < 6:
        flash('Password must be at least 6 characters.', 'error')
        return render_template('reset_password.html', email=email)

    if password != confirm_password:
        flash('Passwords do not match.', 'error')
        return render_template('reset_password.html', email=email)

    # Update password and clear reset token
    user.set_password(password)
    user.reset_token = None
    user.reset_token_expires = None
    db.session.commit()

    flash('Password reset successfully! You can now log in.', 'success')
    return redirect(url_for('auth.login'))
