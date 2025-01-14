FROM python:3.10-slim

WORKDIR /app

RUN useradd -l -m -s /bin/bash appuser

COPY pyproject.toml poetry.lock ./

RUN apt update && \
    apt -y install --no-install-recommends curl build-essential && \
    pip3 install --upgrade pip && \
    pip3 install poetry && \
    poetry config virtualenvs.create false && \
    poetry lock && \
    poetry install

COPY . /app

RUN chown -R appuser:appuser /app
USER appuser

COPY log_config.yaml /app/log_config.yaml

EXPOSE 8000
