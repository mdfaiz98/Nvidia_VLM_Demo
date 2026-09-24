#!/bin/bash
# Stops the VLM demo started by run_demo.sh - kills Ollama and the webui by
# matching their exact launch command lines (the same processes run_demo.sh
# backgrounds), not by port alone, so this can't accidentally kill an
# unrelated process that happens to be listening on the same port.

OLLAMA_PATTERN="ollama/bin/ollama serve"
WEBUI_PATTERN="live-vlm-webui --port 8090"

stopped_any=false

if pgrep -f "$WEBUI_PATTERN" >/dev/null 2>&1; then
    pkill -f "$WEBUI_PATTERN"
    echo "Webui stopped."
    stopped_any=true
fi

if pgrep -f "$OLLAMA_PATTERN" >/dev/null 2>&1; then
    pkill -f "$OLLAMA_PATTERN"
    echo "Ollama stopped."
    stopped_any=true
fi

if [ "$stopped_any" = false ]; then
    echo "Demo wasn't running."
fi
