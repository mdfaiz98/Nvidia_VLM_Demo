#!/bin/bash
# Stops the VLM demo started by run_demo.sh - whichever version (worktree) is
# running - by matching the webui's exact launch command line, not by port
# alone, so this can't accidentally kill an unrelated process that happens to
# be listening on the same port.
#
# Ollama is left running: it's the ollama.service systemd unit, shared with
# the Hermes agent (`sudo systemctl stop ollama` if you really want it off).

WEBUI_PATTERN="live-vlm-webui --port 8090"

if pgrep -f "$WEBUI_PATTERN" >/dev/null 2>&1; then
    RUNNING_DIR="$(readlink -f "/proc/$(pgrep -f "$WEBUI_PATTERN" | head -1)/cwd" 2>/dev/null)"
    pkill -f "$WEBUI_PATTERN"
    echo "Webui stopped (${RUNNING_DIR:-unknown folder})."
else
    echo "Demo wasn't running."
fi
