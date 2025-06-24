#!/bin/bash

# driver.sh - Script to build and run Docker image from Dockerfile
# Created: April 28, 2025

# Exit immediately if a command exits with a non-zero status
set -e

# Define variables
IMAGE_NAME="multiboost"
CONTAINER_NAME="multiboost-dev"
MEMORY_RESERVATION="12g"  # Reserve 12GB
CPU_LIMIT="4"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker and try again."
    exit 1
fi

# Build the Docker image from the existing Dockerfile
echo "Building Docker image $IMAGE_NAME from Dockerfile..."
docker build -t $IMAGE_NAME -f Dockerfile .

# Check if the build was successful
if [ $? -ne 0 ]; then
    echo "Docker build failed. Please check the Dockerfile and try again."
    exit 1
fi

echo "Docker image $IMAGE_NAME built successfully."

# Run a new container with memory reservation but no hard limit
echo "Running a new container with name $CONTAINER_NAME..."
echo "Memory reservation: $MEMORY_RESERVATION (no maximum limit)"
echo "CPUs: $CPU_LIMIT"

docker run -it \
    --name $CONTAINER_NAME \
    --memory-reservation=$MEMORY_RESERVATION \
    --cpus=$CPU_LIMIT \
    $IMAGE_NAME

echo "Container $CONTAINER_NAME stopped."
