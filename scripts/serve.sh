#!/usr/bin/env bash
# Start llama-server in the foreground for one local model, with the pinned flags from models.yaml.
# Usage: scripts/serve.sh gemma4-e4b
# Prints the exact command before running it. Ctrl-C to stop.
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run python -m harness serve "$@"
