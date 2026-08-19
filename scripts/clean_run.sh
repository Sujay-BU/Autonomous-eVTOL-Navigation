#!/usr/bin/env bash
# Kill every stray evaluation / simulator, then run exactly one evaluation.
#
# Identification is done by reading /proc/<pid>/cmdline rather than with
# `pgrep -f`, because a shell invoked to manage these jobs carries the search
# pattern in its own command line and would otherwise match itself.
set -u
ROOT=/home/glaze/Desktop/github_projects/Phy_WAM
cd "$ROOT"
ME=$$

reap() {
  for p in $(ls /proc | grep -E '^[0-9]+$'); do
    [ "$p" = "$ME" ] && continue
    [ -r "/proc/$p/cmdline" ] || continue
    c=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null) || continue
    case "$c" in
      *"python scripts/evaluate.py"*|*"gz sim -s"*|*"python scripts/diagnose_cost.py"*)
        echo "  killing $p"; kill -9 "$p" 2>/dev/null ;;
    esac
  done
}
echo "[clean] reaping strays"
reap
sleep 3
rm -f /tmp/phywam_probe.pid

source scripts/env.sh
export PYTHONUNBUFFERED=1
echo "[clean] running: $*"
exec python scripts/evaluate.py "$@"
