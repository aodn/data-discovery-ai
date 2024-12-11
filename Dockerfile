# Build stage
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy only necessary files for dependency installation
COPY pyproject.toml poetry.lock ./

# Install system dependencies and Poetry
RUN apt update && \
    apt -y install --no-install-recommends curl build-essential && \
    pip3 install --upgrade pip && \
    pip3 install poetry && \
    poetry config virtualenvs.create false && \
    poetry lock && \
    poetry install && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the rest of the application code
COPY . /app

# Make sure log config file (if provided) is available in the working directory
COPY log_config.yaml /app/log_config.yaml

# Expose the port the app runs on
EXPOSE 8000

# Set environment variables
ENV GUNICORN_TIMEOUT=3600
ENV GUNICORN_WORKERS_NUM=4

# Default command to run the application
CMD ["poetry", "run", "uvicorn", "--host", "0.0.0.0", "data_discovery_ai.server:app", "--log-config", "log_config.yaml"]
