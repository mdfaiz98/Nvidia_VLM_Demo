#!/bin/bash
# Makes sure Ollama (the ollama.service systemd unit) is up, then launches the
# webui as a background job with a log file - the same manual sequence
# documented in SESSION_NOTES.md, just automated. The webui is deliberately
# not systemd/tmux: nohup + a log file, so anyone can see exactly what's
# running (`ps aux`) and kill it by hand if this script isn't used.
#
# Safe to double-click again while the demo is already running - each step
# checks first and skips itself if there's nothing to do.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="qwen2.5vl:3b"
API_BASE="http://localhost:11434/v1"
UI_PORT="8090"
UI_URL="https://localhost:${UI_PORT}"

echo "== VLM Demo: starting =="
echo

# GPU clocks reset on every reboot - only worth doing (and worth the sudo
# password prompt) once per boot. The marker is on tmpfs, so a reboot clears it.
CLOCKS_MARKER="/run/user/$(id -u)/vlm-demo-clocks-set"
if [ -e "$CLOCKS_MARKER" ]; then
    echo "Clocks already locked this boot - skipping."
else
    echo "Setting max power mode + locking clocks (may prompt for your sudo password)..."
    sudo nvpmodel -m 0 && sudo jetson_clocks && touch "$CLOCKS_MARKER"
fi
echo

# Ollama is the systemd service (models on /data/ollama/models, tuning in
# /etc/systemd/system/ollama.service.d/override.conf) and starts at boot -
# this only covers it having been stopped.
if curl -sf http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
    echo "Ollama already running."
else
    echo "Starting Ollama service..."
    sudo systemctl start ollama

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
echo "This window can be closed - the demo keeps running in the background."
echo "Logs: journalctl -u ollama , $REPO_DIR/run.log"
echo "To stop: use the 'VLM Demo (Stop)' desktop icon, or scripts/stop_demo.sh"
