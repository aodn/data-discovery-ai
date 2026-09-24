FROM python:3.11-slim

WORKDIR /app

# Accept the build argument from GitHub Actions
ARG APP_VERSION=unknown
ARG GIT_SHA=unknown

# Set it as an environment variable for Python to read
ENV APP_VERSION=$APP_VERSION
ENV GIT_SHA=$GIT_SHA

# Uvicorn listens on loopback only; Nginx is the public listener on 8000.
ENV APP_HOST=127.0.0.1
ENV APP_PORT=9000

RUN useradd -l -m -s /bin/bash appuser

COPY pyproject.toml poetry.lock ./

RUN apt update && \
    apt -y install --no-install-recommends curl build-essential nginx && \
    pip3 install --upgrade pip && \
    pip3 install poetry && \
    poetry config virtualenvs.create false && \
    poetry lock && \
    poetry install --no-root && \
    rm -rf /var/lib/apt/lists/*

COPY . /app

RUN chown -R appuser:appuser /app

COPY log_config.yaml /app/log_config.yaml
COPY ddai_site.conf /etc/nginx/sites-available/
# Nginx runs as appuser, so point the pid file at a writable path and drop the
# "user" directive (only the root master process could honour it).
RUN ln -s /etc/nginx/sites-available/ddai_site.conf /etc/nginx/sites-enabled/ && \
    rm -f /etc/nginx/sites-enabled/default && \
    sed -i 's#^pid .*#pid /tmp/nginx.pid;#; /^user /d' /etc/nginx/nginx.conf && \
    nginx -t && \
    rm -f /tmp/nginx.pid

# appuser also needs to write the Nginx logs and temp paths.
RUN chmod +x /app/docker-entrypoint.sh && \
    mkdir -p /tmp/status /var/lib/nginx /var/log/nginx && \
    chown -R appuser:appuser /tmp/status /var/lib/nginx /var/log/nginx

USER appuser

EXPOSE 8000

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["python", "-m", "data_discovery_ai.server"]
