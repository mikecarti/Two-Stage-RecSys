#!/bin/bash

# Stop and remove the existing recsys container if it exists
docker stop recsys
docker rm recsys

# Build the Docker image
docker build -t recsys .

# Run the Docker container with the specified mounts and ports
docker run -it \
    --name recsys \
    --mount type=bind,source="$(pwd)"/app/model_files,target=/app/model_files \
    --mount type=bind,source="$(pwd)"/app/logs,target=/app/logs \
    -p 80:80 \
    recsys:latest
