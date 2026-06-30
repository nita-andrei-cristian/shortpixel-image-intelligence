#!/usr/bin/env bash
# Create a local venv and install dependencies.
set -e
cd "$(dirname "$0")/.."

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo "Done. Start the API with scripts/run.sh"
