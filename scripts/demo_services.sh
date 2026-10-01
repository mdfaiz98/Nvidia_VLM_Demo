#!/bin/bash
# Helpers sourced by run_demo.sh / stop_demo.sh: per-version extra services.
#
# A demo version (worktree) may list systemd --user units it needs in
# <its folder>/scripts/demo_services, one per line:
#     hermes-panel          started with the demo, stopped with it
#     hermes-gateway keep   started if needed, but left running afterwards
# '#' starts a comment. Versions without the file need nothing extra.

_demo_services_list() {   # <repo_dir> -> "unit mode" lines
    local file="$1/scripts/demo_services"
    [ -f "$file" ] || return 0
    sed -e 's/#.*//' "$file" | awk 'NF {print $1, ($2 == "keep" ? "keep" : "stop")}'
}

start_demo_services() {   # <repo_dir>
    local unit mode
    while read -r unit mode; do
        if systemctl --user is-active --quiet "$unit"; then
            echo "$unit running."
        else
            echo "Starting $unit..."
            systemctl --user start "$unit" \
                || echo "!! $unit failed to start (journalctl --user -u $unit)"
        fi
    done < <(_demo_services_list "$1")
}

stop_demo_services() {    # <repo_dir> - stops the units not marked 'keep'
    local unit mode
    while read -r unit mode; do
        [ "$mode" = keep ] && continue
        if systemctl --user is-active --quiet "$unit"; then
            systemctl --user stop "$unit" && echo "$unit stopped."
        fi
    done < <(_demo_services_list "$1")
}
