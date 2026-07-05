#!/usr/bin/env bash
# Build the MotionMillion Docker image.
set -euo pipefail

# Repo root = parent of this docker-shell directory. Used as the build context so
# the Dockerfile can COPY requirements.txt.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_NAME="${IMAGE_NAME:-motionmillion:latest}"

echo "Building ${IMAGE_NAME} (context: ${REPO_ROOT})"
docker build \
    -f "${SCRIPT_DIR}/Dockerfile" \
    -t "${IMAGE_NAME}" \
    "${REPO_ROOT}"

echo "Done. Image: ${IMAGE_NAME}"
