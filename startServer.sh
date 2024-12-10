#!/bin/bash

CONTAINER_NAME="data-discovery-ai"
IMAGE_NAME="data-discovery-ai"

# Check if a container with the same name already exists
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
  echo "Removing existing container with name: $CONTAINER_NAME"
  # Stop and remove the existing container
  docker stop $CONTAINER_NAME
  docker rm $CONTAINER_NAME
fi

# Build the Docker image without using the cache
echo "Building the Docker image: $IMAGE_NAME (no cache)"
docker build --no-cache -t $IMAGE_NAME .

# Run the container with the .env file
docker run -d --name $CONTAINER_NAME \
  --env-file .env \
  -p 8000:8000 \
  $IMAGE_NAME

echo "Container $CONTAINER_NAME is up and running."
