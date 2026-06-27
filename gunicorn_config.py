# Gunicorn configuration file

# Increase timeout to 5 minutes (300 seconds)
timeout = 300

# Worker configuration
# Single worker so /process and /status polls share the same
# in-memory analysis_progress dict. Multiple threads let the
# status polls run while /process blocks on one thread.
workers = 1
threads = 8
worker_class = 'gthread'
worker_connections = 1000

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# Keep alive
keepalive = 5

# Graceful timeout
graceful_timeout = 30
