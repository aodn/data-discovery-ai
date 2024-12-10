# Use the official Ubuntu 22.04 base image
FROM ubuntu:22.04

# Set environment variables to prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory in the container
WORKDIR /app

# Install system dependencies, Python 3.10+, and Poetry
RUN apt update && apt -y upgrade && \
    apt install -y python3 python3-venv python3-distutils python3-pip curl && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy the pyproject.toml and poetry.lock files into the container
COPY pyproject.toml poetry.lock ./

# Install Python dependencies using Poetry
RUN poetry config virtualenvs.create false && \
    poetry lock && \
    poetry install

# Copy the rest of the application code into the container
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
