# Use the official Python base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the pyproject.toml and poetry.lock files into the container
COPY pyproject.toml poetry.lock ./

# Install system dependencies and Python dependencies
RUN apt update && \
    apt -y upgrade && \
    pip3 install --upgrade pip && \
    pip3 install poetry && \
    poetry config virtualenvs.create false && \
    poetry lock && \
    poetry install

# Copy the rest of the application code into the container
COPY . /app

# Make sure log config file (if provided) is available in the working directory
COPY log_config.yaml /app/log_config.yaml

# Expose the port the app runs on
EXPOSE 8000

# Set environment variables from a .env file or directly
ENV GUNICORN_TIMEOUT=3600
ENV GUNICORN_WORKERS_NUM=4

# Default command to run the application
CMD ["poetry", "run", "uvicorn", "--host", "0.0.0.0", "data_discovery_ai.server:app", "--log-config", "log_config.yaml"]
