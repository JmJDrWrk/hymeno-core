#!/usr/bin/env bash
# Sets up the model server on Linux (including WSL): installs Ollama if it is
# missing and pulls the vision model.
#
#   scripts/setup-ollama.sh [model] [allowed-origins]
#
# allowed-origins is only needed when web pages (not this brain) call Ollama
# from the browser, e.g. "http://192.168.1.15,http://robot-head.local".
set -euo pipefail

MODEL=${1:-qwen2.5vl:7b}
ORIGINS=${2:-}

if ! command -v zstd >/dev/null; then
    echo "== Installing zstd (the Ollama installer needs it)"
    sudo apt-get update && sudo apt-get install -y zstd
fi

if ! command -v ollama >/dev/null; then
    echo "== Installing Ollama"
    curl -fsSL https://ollama.com/install.sh | sh
fi

if [ -n "$ORIGINS" ]; then
    echo "== Letting web pages from $ORIGINS call Ollama"
    sudo mkdir -p /etc/systemd/system/ollama.service.d
    printf '[Service]\nEnvironment="OLLAMA_ORIGINS=%s"\n' "$ORIGINS" |
        sudo tee /etc/systemd/system/ollama.service.d/override.conf >/dev/null
    sudo systemctl daemon-reload
    sudo systemctl restart ollama
fi

echo "== Pulling $MODEL"
ollama pull "$MODEL"
echo "== Ready: Ollama at http://localhost:11434 with $MODEL"
