# Gunicorn configuration file

# Increase timeout to 5 minutes (300 seconds)
timeout = 300

# Worker configuration
workers = 2
worker_class = 'sync'
worker_connections = 1000

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# Keep alive
keepalive = 5

# Graceful timeout
graceful_timeout = 30
