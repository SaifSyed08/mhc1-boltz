#!/bin/bash
#----------------------------------------------------------------------------
# One TACC login per sitting instead of one per command.
#
# ControlMaster is the normal answer to "MFA on every ssh", but Git Bash's MSYS
# OpenSSH cannot multiplex (see the note in ~/.ssh/config). This does the same
# job by hand: a single ssh keeps one remote bash open, its stdin is fed from a
# local file that commands get appended to, and its output lands in another.
#
#   bash scripts/ls6_bridge.sh open             # in its own Git Bash window:
#                                               # log in once, leave it open
#   bash scripts/ls6_bridge.sh run 'squeue -u $USER'
#   bash scripts/ls6_bridge.sh status
#
# `run` executes in $WORK/mhc1-boltz inside a subshell, so a `cd` in one command
# does not leak into the next. It waits for the command to finish (LS6_TIMEOUT
# seconds, default 300) and exits with the remote exit code. Every command is
# echoed in the bridge window, so whoever owns the login sees what runs on it.
#
# Re-opening truncates the command file, so anything queued while the bridge
# was down is DROPPED, not replayed. That is deliberate: a stale sbatch firing
# on reconnect is worse than having to resend it.
#
# TACC drops a connection that has not finished password + token within ~3 min
# (seen as "Connection closed by 129.114.62.201 port 22" at 195 s). So
# `open` waits for Enter before dialling -- the clock starts when you are ready,
# not when the window happens to appear behind something else.
#----------------------------------------------------------------------------
set -u
B="${LS6_BRIDGE_DIR:-$HOME/.ls6-bridge}"
mkdir -p "$B"

case "${1:-}" in
open)
    # Two bridges reading one command file would run every command twice.
    if [ -f "$B/pid" ] && kill -0 "$(cat "$B/pid")" 2>/dev/null; then
        echo "A bridge is already open (pid $(cat "$B/pid")). Close that window first."
        read -rp "press Enter to close " _
        exit 1
    fi
    : > "$B/in"; : > "$B/out"; : > "$B/log"
    echo "LS6 bridge: you will log in ONCE, here, and then leave this window open."
    echo
    read -rp "Press Enter when you are ready to type your TACC password and token... " _
    echo

    printf 'echo __ls6_hello\n' >> "$B/in"
    ( until grep -q __ls6_hello "$B/out" 2>/dev/null; do sleep 1; done
      printf '\a\n== Connected. Minimize this window; closing it ends the bridge. ==\n'
      printf '== Commands run through it appear below. ==\n\n'
      exec tail -n +1 -f "$B/log" ) &
    watcher=$!

    exec 3< <(tail -n +1 -f "$B/in")
    feeder=$!
    # Password/token prompts go to /dev/tty, so they appear here even though
    # stdin is the command feed. ssh's own stderr (TACC's banner, errors) is
    # shown here AND kept in the out file.
    echo $$ > "$B/pid"
    ssh ls6 'exec bash -l' <&3 >> "$B/out" 2> >(tee -a "$B/out" >&2)
    rc=$?
    kill "$feeder" "$watcher" 2>/dev/null
    rm -f "$B/pid"
    echo
    echo "bridge closed ($(date), ssh exit $rc)."
    read -rp "press Enter to close this window " _
    ;;

run|status)
    if [ "$1" = status ]; then cmd='echo "bridge up: $(hostname) $(date)"'; timeout=20
    else shift; cmd="$*"; timeout="${LS6_TIMEOUT:-300}"; fi
    [ -n "$cmd" ] || { echo "usage: $0 run '<command>'" >&2; exit 64; }

    id="$(date +%s)_$RANDOM"
    start=$(wc -c < "$B/out")
    printf '[%s] %s\n' "$(date +%H:%M:%S)" "$cmd" >> "$B/log"
    # $? and $WORK are single-quoted on purpose: they expand on LS6, not here.
    printf '( cd "$WORK/mhc1-boltz" 2>/dev/null\n%s\n) 2>&1 < /dev/null; echo "__ls6_done_%s rc=$?"\n' \
        "$cmd" "$id" >> "$B/in"

    deadline=$(( $(date +%s) + timeout ))
    until tail -c +"$((start + 1))" "$B/out" | grep -q "__ls6_done_$id"; do
        if [ "$(date +%s)" -ge "$deadline" ]; then
            echo "!! no answer after ${timeout}s. Bridge down? (bash $0 open)" >&2
            exit 75
        fi
        sleep 1
    done
    out=$(tail -c +"$((start + 1))" "$B/out" | sed "/__ls6_done_$id/q")
    printf '%s\n' "$out" | grep -v -e "__ls6_done_$id" -e "^__ls6_hello$"
    rc=$(printf '%s\n' "$out" | sed -n "s/.*__ls6_done_$id rc=\([0-9]*\).*/\1/p")
    exit "${rc:-1}"
    ;;

*)
    sed -n '2,27p' "$0" | sed 's/^# \{0,1\}//'
    exit 64
    ;;
esac
