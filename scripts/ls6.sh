#!/bin/bash
#----------------------------------------------------------------------------
# Run a command on Lonestar6 with no prompt at all.
#
#   bash scripts/ls6.sh 'squeue -u $USER'
#   bash scripts/ls6.sh 'sacct -j 3615394 --format=State -X --noheader'
#
# Authentication is the key in ~/.ssh/id_ed25519_tacc for the first factor and
# a locally generated TOTP code for the second, fed to ssh through SSH_ASKPASS.
# Each call is its own connection, so there is no session to keep alive and
# nothing to re-open after a laptop sleeps. That is the point: it replaces
# scripts/ls6_bridge.sh, which existed only to amortise one MFA prompt.
#
# The bridge still works and is still the right thing when the login is being
# done by hand, or if the TOTP secret is ever removed from this machine.
#
# Commands run in $WORK/mhc1-boltz inside a subshell, same as the bridge, so a
# cd in one call cannot leak into the next.
#----------------------------------------------------------------------------
set -uo pipefail

cmd="${*:-}"
[ -n "$cmd" ] || { echo "usage: $0 '<command>'" >&2; exit 64; }

for f in "$HOME/.ssh/id_ed25519_tacc" "$HOME/.ssh/tacc_totp_secret" "$HOME/.ssh/tacc_askpass.sh"; do
    [ -r "$f" ] || { echo "!! missing $f; use scripts/ls6_bridge.sh instead" >&2; exit 70; }
done

# force makes OpenSSH use askpass even with a terminal attached and no DISPLAY.
# Needs OpenSSH 8.4 or newer; this laptop has 9.9.
SSH_ASKPASS="$HOME/.ssh/tacc_askpass.sh" \
SSH_ASKPASS_REQUIRE=force \
    ssh -o BatchMode=no -o NumberOfPasswordPrompts=1 ls6 \
        "cd \$WORK/mhc1-boltz 2>/dev/null; ( $cmd )"
