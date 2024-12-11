# Build stage
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Create a non-root user with login capabilities
RUN useradd -l -m -s /bin/bash appuser

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

# Ensure correct permissions for the non-root user
RUN chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# Expose the port the app runs on
EXPOSE 8000

# Default command to run the application
#CMD ["poetry", "run", "uvicorn", "--host", "0.0.0.0", "data_discovery_ai.server:app"]
