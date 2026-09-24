#!/bin/bash
# Launches the two long-running processes needed for this demo (Ollama, then
# the webui) as background jobs with log files - the same manual sequence
# documented in SESSION_NOTES.md, just automated. Deliberately not
# systemd/tmux: nohup + a log file per process, so anyone can see exactly
# what's running (`ps aux`) and kill it by hand if this script isn't used.
#
# Safe to double-click again while the demo is already running - each step
# checks first and skips itself if there's nothing to do.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OLLAMA_DIR="$HOME/ollama"
MODEL="qwen2.5vl:3b"
API_BASE="http://localhost:11434/v1"
UI_PORT="8090"
UI_URL="https://localhost:${UI_PORT}"

echo "== VLM Demo: starting =="
echo

if curl -sf http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
    echo "Ollama already running - skipping clock reset and Ollama startup."
else
    # GPU clocks reset on every reboot - only worth doing (and worth the sudo
    # password prompt) on a genuinely fresh start.
    echo "Setting max power mode + locking clocks (may prompt for your sudo password)..."
    sudo nvpmodel -m 0
    sudo jetson_clocks
    echo

    echo "Starting Ollama..."
    export OLLAMA_MODELS="$OLLAMA_DIR/models"
    export OLLAMA_FLASH_ATTENTION=1
    export OLLAMA_KV_CACHE_TYPE=q8_0
    nohup "$OLLAMA_DIR/bin/ollama" serve > "$OLLAMA_DIR/serve.log" 2>&1 &
    disown

    echo -n "Waiting for Ollama to come up"
    until curl -sf http://127.0.0.1:11434/api/version >/dev/null 2>&1; do
        echo -n "."
        sleep 1
    done
    echo " ready."
fi
echo

if curl -sfk "$UI_URL" >/dev/null 2>&1; then
    echo "Webui already running."
else
    echo "Starting webui..."
    cd "$REPO_DIR"
    source .venv/bin/activate
    nohup live-vlm-webui --port "$UI_PORT" --model "$MODEL" --api-base "$API_BASE" \
        > "$REPO_DIR/run.log" 2>&1 &
    disown
fi

echo -n "Waiting for webui to come up"
until curl -sfk "$UI_URL" >/dev/null 2>&1; do
    echo -n "."
    sleep 1
done
echo " ready."
echo

echo "Demo is up: $UI_URL"
firefox --new-window "$UI_URL" >/dev/null 2>&1 &
disown

echo
echo "This window can be closed - both services keep running in the background."
echo "Logs: $OLLAMA_DIR/serve.log , $REPO_DIR/run.log"
echo "To stop: use the 'VLM Demo (Stop)' desktop icon, or scripts/stop_demo.sh"
