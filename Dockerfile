FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the pyproject.toml and poetry.lock files into the container
COPY pyproject.toml poetry.lock ./

# Install system dependencies
RUN apt update && \
    apt -y upgrade && \
    pip3 install --upgrade pip && \
    pip3 install poetry && \
    poetry config virtualenvs.create false && \
    poetry lock --no-update && \
    poetry install

# Expose the port the app runs on
EXPOSE 8000

# Copy the rest of the application code into the container
COPY . /app
