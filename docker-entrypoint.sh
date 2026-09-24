#!/bin/bash
# Start Nginx in the background, then hand PID 1 over to the container command
# (python -m data_discovery_ai.server), so the Python process keeps receiving
# SIGTERM directly and shuts down gracefully, exactly as it did without Nginx.
#
# Nginx is only a thin front door on port 8000:
#   - it serves /api/v1/ml/health from the static file /tmp/status/health.json,
#     so the health check answers even while Python is busy loading models;
#   - it proxies everything else to Uvicorn on 127.0.0.1:$APP_PORT.
# Both run as appuser, so no supervisord/root process is needed. If Nginx dies
# the health check on port 8000 stops answering and the orchestrator replaces
# the container, which is the same signal a dead Python process gives.
set -euo pipefail

: "${APP_HOST:=127.0.0.1}"
: "${APP_PORT:=9000}"
export APP_HOST APP_PORT

nginx || {
    echo "Failed to start Nginx" >&2
    exit 1
}

exec "$@"
