#!/usr/bin/env bash
# Kill stale watcher shells, then run one scored MPPI evaluation with training
# suspended.
#
# Uses a PID FILE rather than pgrep to decide whether a probe is already
# running. pgrep -f matches any process whose command line contains the pattern,
# and a watcher shell built to wait for "probe_plan.sh" contains that string
# itself -- so the guard matches the watcher, never clears, and the thing it was
# guarding never launches. That failure is silent: the log file is simply never
# created, which reads identically to "still starting up".
set -u
ROOT=/home/glaze/Desktop/github_projects/Phy_WAM
cd "$ROOT"
PIDFILE=/tmp/phywam_probe.pid
LOG="${5:-$ROOT/logs/probe_run.log}"

# --- reap watcher shells left over from earlier polling -------------------
ME=$$
for p in $(ls /proc | grep -E '^[0-9]+$'); do
  [ "$p" = "$ME" ] && continue
  c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null) || continue
  case "$c" in
    *"until ! pgrep"*|*"until [ \"\$(grep -cE"*)
      kill -9 "$p" 2>/dev/null && echo "[probe] reaped watcher $p" ;;
  esac
done

if [ -f "$PIDFILE" ] && kill -0 "$(cat $PIDFILE)" 2>/dev/null; then
  echo "[probe] already running as $(cat $PIDFILE)"; exit 1
fi
echo $$ > "$PIDFILE"

source scripts/env.sh
export PYTHONUNBUFFERED=1

TR=$(pgrep -x python | while read p; do
       tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q 'train.py --name main' && echo $p; done | head -1)
GZ=""
[ -n "${TR:-}" ] && GZ=$(pgrep -P "$TR" | head -1)

resume() {
  [ -n "${TR:-}" ] && kill -CONT "$TR" 2>/dev/null
  [ -n "${GZ:-}" ] && kill -CONT "$GZ" 2>/dev/null
  rm -f "$PIDFILE"
  echo "[probe] trainer resumed, pidfile cleared"
}
trap resume EXIT INT TERM

if [ -n "${TR:-}" ]; then
  echo "[probe] suspending trainer $TR (gz $GZ)"
  kill -STOP "$TR" 2>/dev/null
  [ -n "${GZ:-}" ] && kill -STOP "$GZ" 2>/dev/null
fi
sleep 2

python scripts/evaluate.py --episodes "${1:-3}" --mode plan --rtf "${2:-0.5}" \
    --seed "${3:-777}" --max-time "${4:-150}" \
    --outdir /tmp/phywam_probe --tag probe 2>&1 | tee -a "$LOG"
echo "[probe] evaluation finished"
