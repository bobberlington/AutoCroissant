#!/usr/bin/env bash
# Deploy on the jetson: pull, psd-tools pin, incremental Cython build, restart, check the journal.
#   ~/Documents/AutoCroissant/scripts/deploy.sh [--bot autocroissant|ojwikibot]
#   ssh jetson Documents/AutoCroissant/scripts/deploy.sh
# Prints OK or FAILED (exit 0 / 1). Refuses 05:00-05:40 Pacific (the scheduled /update_stats).
set -euo pipefail

in_window() {
    local t
    t=$(TZ=America/Los_Angeles date +%H%M)
    [ "$((10#$t))" -ge 500 ] && [ "$((10#$t))" -lt 540 ]
}

fail() {
    echo "FAILED: $*"
    exit 1
}

pin_matches() {     # the installed psd-tools came from requirements-cards.txt's URL and tag
    "$PY" - "$1" <<'PY'
import json, sys
from importlib.metadata import distribution
url, _, rev = sys.argv[1].split("@ git+", 1)[1].rpartition("@")
try:
    d = json.loads(distribution("psd-tools").read_text("direct_url.json") or "{}")
except Exception:
    sys.exit(1)
ok = d.get("url", "").rstrip("/") == url.rstrip("/") and d.get("vcs_info", {}).get("requested_revision") == rev
sys.exit(0 if ok else 1)
PY
}

main() {
    local bot=autocroissant
    if [ "${1:-}" = "--bot" ]; then bot=${2:-}; fi
    case "$bot" in
        autocroissant) dir=~/Documents/AutoCroissant ;;
        ojwikibot) dir=~/Documents/OJWikibot ;;
        *) fail "unknown bot '$bot' (autocroissant or ojwikibot)" ;;
    esac
    PY=~/venvs/$bot/bin/python
    in_window && fail "05:00-05:40 Pacific: the scheduled /update_stats runs then"
    cd "$dir"
    export SSL_CERT_FILE=${SSL_CERT_FILE:-/etc/ssl/certs/ca-certificates.crt}
    if [ "$bot" = autocroissant ] && [ -z "${SSH_AUTH_SOCK:-}" ] && [ -S /run/autocroissant/ssh-agent.sock ]; then
        export SSH_AUTH_SOCK=/run/autocroissant/ssh-agent.sock
    fi

    echo "== $bot: pull"
    git pull --ff-only || fail "git pull --ff-only"
    git log --oneline -1

    if [ -f requirements-cards.txt ]; then
        local pin
        pin=$(grep '^psd-tools' requirements-cards.txt)
        if pin_matches "$pin"; then
            echo "== psd-tools: at the pin"
        else
            echo "== psd-tools: reinstalling $pin"
            "$PY" -m pip install --force-reinstall --no-deps "$pin" || fail "psd-tools install"
            "$PY" -m pip check || echo "(pip check reported the above)"
        fi
    fi

    echo "== build"
    "$PY" setup.py build_ext --inplace || fail "build (the service was not restarted)"

    in_window && fail "reached 05:00-05:40 Pacific during the build; restart after 05:40: sudo systemctl restart $bot"
    echo "== restart"
    local since pid log=""
    since=$(date '+%Y-%m-%d %H:%M:%S')
    sudo systemctl restart "$bot"
    pid=$(systemctl show -p MainPID --value "$bot")     # only the new process's lines: the old one's shutdown may log a Traceback
    [ "${pid:-0}" != 0 ] || fail "$bot did not start (journalctl -u $bot)"
    for _ in $(seq 1 45); do
        sleep 2
        log=$(sudo journalctl _PID="$pid" --since "$since" --no-pager -o cat 2>/dev/null || true)
        if grep -qE "Registered [0-9]+ slash commands" <<<"$log" && grep -q "Bot initialization complete" <<<"$log"; then
            grep -E "Registered [0-9]+ slash commands|Bot initialization complete" <<<"$log"
            echo "OK: $bot deployed at $(git rev-parse --short HEAD)"
            return 0
        fi
        if grep -q "Traceback" <<<"$log" || [ "$(systemctl show -p MainPID --value "$bot")" != "$pid" ]; then
            break
        fi
    done
    tail -n 30 <<<"$log"
    fail "$bot did not log its startup lines (journalctl -u $bot)"
}

main "$@"
exit $?
