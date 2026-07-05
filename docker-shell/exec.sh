#!/usr/bin/env bash
# Open an extra shell inside the already-running MotionMillion container.
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-motionmillion}"

if [ -z "$(docker ps -q -f name="^${CONTAINER_NAME}$")" ]; then
    echo "Container '${CONTAINER_NAME}' is not running. Start it first with:"
    echo "    bash docker-shell/run.sh"
    exit 1
fi

# Pass through a command if given, otherwise drop into an interactive bash.
if [ "$#" -gt 0 ]; then
    exec docker exec -it "${CONTAINER_NAME}" "$@"
else
    exec docker exec -it "${CONTAINER_NAME}" /bin/bash
fi
