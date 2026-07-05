#!/usr/bin/env bash
# Start the MotionMillion container with GPU access and the repo mounted.
# The repo is mounted (not baked in) so code/checkpoints/datasets stay on the host.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_NAME="${IMAGE_NAME:-motionmillion:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-motionmillion}"

# Reuse an already-running container if present.
if [ -n "$(docker ps -q -f name="^${CONTAINER_NAME}$")" ]; then
    echo "Container '${CONTAINER_NAME}' is already running; attaching a shell."
    exec docker exec -it "${CONTAINER_NAME}" /bin/bash
fi

# Remove a stopped container with the same name, if any.
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

echo "Starting ${CONTAINER_NAME} from ${IMAGE_NAME}"
exec docker run -it \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    --ipc=host \
    --shm-size=16g \
    -v "${REPO_ROOT}:/workspace" \
    -w /workspace \
    "${IMAGE_NAME}" \
    /bin/bash
