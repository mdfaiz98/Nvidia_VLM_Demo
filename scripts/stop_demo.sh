#!/bin/bash
# Stops the VLM demo started by run_demo.sh - whichever version (worktree) is
# running - by matching the webui's exact launch command line, not by port
# alone, so this can't accidentally kill an unrelated process that happens to
# be listening on the same port.
#
# Also stops that version's extra services (its scripts/demo_services), except
# ones marked 'keep'. Ollama is left running: it's the ollama.service systemd
# unit, shared with the Hermes agent (`sudo systemctl stop ollama` to stop it).

WEBUI_PATTERN="live-vlm-webui --port 8090"
source "$(dirname "${BASH_SOURCE[0]}")/demo_services.sh"

if pgrep -f "$WEBUI_PATTERN" >/dev/null 2>&1; then
    RUNNING_DIR="$(readlink -f "/proc/$(pgrep -f "$WEBUI_PATTERN" | head -1)/cwd" 2>/dev/null)"
    pkill -f "$WEBUI_PATTERN"
    echo "Webui stopped (${RUNNING_DIR:-unknown folder})."
    # ...and that version's extra services (its scripts/demo_services).
    [ -n "$RUNNING_DIR" ] && stop_demo_services "$RUNNING_DIR"
else
    echo "Demo wasn't running."
fi
