#!/bin/bash
# Makes sure Ollama (the ollama.service systemd unit) is up, then launches the
# webui as a background job with a log file - the same manual sequence
# documented in SESSION_NOTES.md, just automated. The webui is deliberately
# not systemd/tmux: nohup + a log file, so anyone can see exactly what's
# running (`ps aux`) and kill it by hand if this script isn't used.
#
# Safe to double-click again while the demo is already running - each step
# checks first and skips itself if there's nothing to do.
#
# Several versions of the demo can live side by side as git worktrees (e.g.
# ~/live-vlm-webui on main, ~/live-vlm-webui-hermes on hermes-agent), each
# with its own desktop icon. They share port 8090 and the cameras, so only
# one runs at a time: this script starts the version in ITS OWN folder and
# stops any other version first. Worktrees reuse the main checkout's .venv
# (an editable install); PYTHONPATH makes it load this folder's code.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
BRANCH="$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
MAIN_DIR="$(git -C "$REPO_DIR" worktree list --porcelain 2>/dev/null | awk 'NR==1 {print $2}')"
VENV="$REPO_DIR/.venv"
[ -d "$VENV" ] || VENV="${MAIN_DIR:-$REPO_DIR}/.venv"
MODEL="qwen2.5vl:3b"
API_BASE="http://localhost:11434/v1"
UI_PORT="8090"
UI_URL="https://localhost:${UI_PORT}"
WEBUI_PATTERN="live-vlm-webui --port ${UI_PORT}"
source "$REPO_DIR/scripts/demo_services.sh"

echo "== VLM Demo: starting =="
echo "Version: $BRANCH ($REPO_DIR)"
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

# The webui is launched from its own folder, so its working directory tells
# which version is running.
WEBUI_PID="$(pgrep -f "$WEBUI_PATTERN" | head -1)"
if [ -n "$WEBUI_PID" ]; then
    RUNNING_DIR="$(readlink -f "/proc/$WEBUI_PID/cwd" 2>/dev/null)"
    if [ "$RUNNING_DIR" != "$REPO_DIR" ]; then
        echo "Another version is running ($RUNNING_DIR) - stopping it..."
        pkill -f "$WEBUI_PATTERN"
        while pgrep -f "$WEBUI_PATTERN" >/dev/null 2>&1; do sleep 0.5; done
        stop_demo_services "$RUNNING_DIR"
        WEBUI_PID=""
    fi
fi

# Extra services this version needs (scripts/demo_services, if any).
start_demo_services "$REPO_DIR"

if [ -n "$WEBUI_PID" ]; then
    echo "Webui already running."
else
    echo "Starting webui..."
    cd "$REPO_DIR"
    source "$VENV/bin/activate"
    PYTHONPATH="$REPO_DIR/src" nohup live-vlm-webui --port "$UI_PORT" --model "$MODEL" \
        --api-base "$API_BASE" > "$REPO_DIR/run.log" 2>&1 &
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
