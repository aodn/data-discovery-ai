FROM python:3.11-slim

WORKDIR /app

# Accept the build argument from GitHub Actions
ARG APP_VERSION=unknown
ARG GIT_SHA=unknown

# Set it as an environment variable for Python to read
ENV APP_VERSION=$APP_VERSION
ENV GIT_SHA=$GIT_SHA

RUN useradd -l -m -s /bin/bash appuser

COPY pyproject.toml poetry.lock ./

RUN apt update && \
    apt -y install --no-install-recommends curl build-essential && \
    pip3 install --upgrade pip && \
    pip3 install poetry && \
    poetry config virtualenvs.create false && \
    poetry lock && \
    poetry install --no-root

COPY . /app

RUN chown -R appuser:appuser /app
USER appuser

COPY log_config.yaml /app/log_config.yaml

EXPOSE 8000
