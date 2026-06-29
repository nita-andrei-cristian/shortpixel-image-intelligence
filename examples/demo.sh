#!/usr/bin/env bash
# Send a product image + taxonomy to the running API.
set -e

IMAGE="${1:-examples/images/apple.jpg}"
URL="${2:-http://localhost:8000/analyze}"

curl -s -X POST "$URL" \
  -F "image=@${IMAGE}" \
  -F "payload=$(cat examples/request.json)" | python -m json.tool
