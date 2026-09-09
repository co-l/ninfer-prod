#!/bin/bash
# ninfer reference server configuration for concurrent agent sessions
# with a RAM hot-KV tier (see README.md for rationale).
#
# Run on the GPU box (rootful podman + CDI). Model id is qwen3.8-27b on
# port 8000, NVFP4 weights, MTP4 spec-decode, vision.
set -euo pipefail

# Adjust: path to your ninfer source tree and model directory.
NINFER_DIR="${NINFER_DIR:-/home/your-user/ninfer}"
MODELS_DIR="${MODELS_DIR:-$NINFER_DIR/models}"
LOGS_DIR="${LOGS_DIR:-$NINFER_DIR/logs}"
IMAGE="${IMAGE:-localhost/ninfer:local}"

cd "$NINFER_DIR"
sudo podman run --name ninfer-serve --rm \
  --device nvidia.com/gpu=0 \
  --security-opt label=disable \
  -p 8000:8000 \
  -v "$MODELS_DIR:/models:ro" \
  -v "$LOGS_DIR:/logs" \
  "$IMAGE" \
  ninfer-serve /models/qwen3_8_27b_nvfp4.ninfer \
  --host 0.0.0.0 \
  --port 8000 \
  --model-id qwen3.8-27b \
  --max-context 252928 \
  --kv-capacity 480000 \
  --max-concurrency 2 \
  --max-pending-requests 16 \
  --pending-timeout-ms 600000 \
  --device-state-slots 4 \
  --host-state-slots 96 \
  --host-kv-mib 12288 \
  --max-private-continuations 128 \
  --max-shared-prefixes 64 \
  --request-log-jsonl /logs/requests.jsonl \
  --kv-dtype nvfp4 \
  --spec mtp \
  --draft-tokens 4 \
  --lm-head-draft \
  --vision
