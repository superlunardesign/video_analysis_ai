"""Email utility functions for sending verification codes and notifications."""
import os
import random
import string
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta


def generate_code(length=6):
    """Generate a random numeric code."""
    return ''.join(random.choices(string.digits, k=length))


def get_smtp_config():
    """Get SMTP configuration from environment variables."""
    return {
        'host': os.environ.get('SMTP_HOST', 'smtp.gmail.com'),
        'port': int(os.environ.get('SMTP_PORT', 587)),
        'username': os.environ.get('SMTP_USERNAME', ''),
        'password': os.environ.get('SMTP_PASSWORD', ''),
        'from_email': os.environ.get('SMTP_FROM_EMAIL', os.environ.get('SMTP_USERNAME', '')),
        'from_name': os.environ.get('SMTP_FROM_NAME', 'TikTok Analyzer'),
    }


def send_email(to_email, subject, html_body, text_body=None):
    """Send an email using SMTP.

    Returns True if successful, False otherwise.
    """
    config = get_smtp_config()

    if not config['username'] or not config['password']:
        print(f"[EMAIL] SMTP not configured - would send to {to_email}: {subject}")
        print(f"[EMAIL] Set SMTP_USERNAME and SMTP_PASSWORD environment variables")
        # Return True in dev mode so the flow continues
        return True

    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = f"{config['from_name']} <{config['from_email']}>"
        msg['To'] = to_email

        # Plain text version
        if text_body:
            msg.attach(MIMEText(text_body, 'plain'))

        # HTML version
        msg.attach(MIMEText(html_body, 'html'))

        # Connect and send
        with smtplib.SMTP(config['host'], config['port']) as server:
            server.starttls()
            server.login(config['username'], config['password'])
            server.send_message(msg)

        print(f"[EMAIL] Sent email to {to_email}: {subject}")
        return True

    except Exception as e:
        print(f"[EMAIL ERROR] Failed to send email to {to_email}: {e}")
        return False


def send_password_reset_email(to_email, code):
    """Send password reset code email."""
    subject = "Your Password Reset Code - TikTok Analyzer"

    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; margin: 0; padding: 20px; }}
            .container {{ max-width: 500px; margin: 0 auto; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; text-align: center; }}
            .header h1 {{ color: white; margin: 0; font-size: 24px; }}
            .content {{ padding: 30px; }}
            .code {{ background: #f0f0f0; font-size: 32px; font-weight: bold; letter-spacing: 8px; text-align: center; padding: 20px; border-radius: 8px; margin: 20px 0; color: #333; }}
            .footer {{ padding: 20px 30px; background: #f9f9f9; font-size: 12px; color: #666; text-align: center; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Password Reset</h1>
            </div>
            <div class="content">
                <p>Hi there,</p>
                <p>We received a request to reset your password. Enter this code to set a new password:</p>
                <div class="code">{code}</div>
                <p>This code expires in <strong>15 minutes</strong>.</p>
                <p>If you didn't request this, you can safely ignore this email.</p>
            </div>
            <div class="footer">
                <p>TikTok Analyzer by Superlunar Design Co.</p>
            </div>
        </div>
    </body>
    </html>
    """

    text_body = f"""
Password Reset - TikTok Analyzer

Your password reset code is: {code}

This code expires in 15 minutes.

If you didn't request this, you can safely ignore this email.

---
TikTok Analyzer by Superlunar Design Co.
    """

    return send_email(to_email, subject, html_body, text_body)


def send_email_change_confirmation(to_email, code, new_email):
    """Send email change confirmation code to the NEW email address."""
    subject = "Confirm Your New Email - TikTok Analyzer"

    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; margin: 0; padding: 20px; }}
            .container {{ max-width: 500px; margin: 0 auto; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; text-align: center; }}
            .header h1 {{ color: white; margin: 0; font-size: 24px; }}
            .content {{ padding: 30px; }}
            .code {{ background: #f0f0f0; font-size: 32px; font-weight: bold; letter-spacing: 8px; text-align: center; padding: 20px; border-radius: 8px; margin: 20px 0; color: #333; }}
            .footer {{ padding: 20px 30px; background: #f9f9f9; font-size: 12px; color: #666; text-align: center; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Confirm Your Email</h1>
            </div>
            <div class="content">
                <p>Hi there,</p>
                <p>Someone requested to change their TikTok Analyzer account email to <strong>{new_email}</strong>.</p>
                <p>Enter this code to confirm:</p>
                <div class="code">{code}</div>
                <p>This code expires in <strong>15 minutes</strong>.</p>
                <p>If you didn't request this, you can safely ignore this email.</p>
            </div>
            <div class="footer">
                <p>TikTok Analyzer by Superlunar Design Co.</p>
            </div>
        </div>
    </body>
    </html>
    """

    text_body = f"""
Confirm Your Email - TikTok Analyzer

Someone requested to change their account email to: {new_email}

Your confirmation code is: {code}

This code expires in 15 minutes.

If you didn't request this, you can safely ignore this email.

---
TikTok Analyzer by Superlunar Design Co.
    """

    return send_email(to_email, subject, html_body, text_body)
