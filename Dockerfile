FROM python:3.11-slim

WORKDIR /app

# Accept the build argument from GitHub Actions
ARG APP_VERSION=unknown
ARG GIT_SHA=unknown

# Set it as an environment variable for Python to read
ENV APP_VERSION=$APP_VERSION
ENV GIT_SHA=$GIT_SHA

COPY pyproject.toml poetry.lock ./

RUN apt update && \
    apt -y install --no-install-recommends curl build-essential nginx supervisor netcat-openbsd && \
    pip3 install --upgrade pip && \
    pip3 install poetry && \
    poetry config virtualenvs.create false && \
    poetry lock && \
    poetry install --no-root && \
    rm -rf /var/lib/apt/lists/*

COPY . /app
COPY ddai_site.conf /etc/nginx/sites-available/
RUN ln -s /etc/nginx/sites-available/ddai_site.conf /etc/nginx/sites-enabled/ && \
    rm -f /etc/nginx/sites-enabled/default && \
    nginx -t

EXPOSE 8000

CMD ["/usr/bin/supervisord", "-c", "/app/supervisord.conf"]
