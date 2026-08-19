#!/usr/bin/env bash
# Measure the MPPI controller with the current checkpoint, with training
# suspended so the timing is clean. The trap guarantees the trainer is resumed
# even if this script is killed -- leaving it SIGSTOPped is silent and costly.
set -u
cd /home/glaze/Desktop/github_projects/Phy_WAM
source scripts/env.sh
export PYTHONUNBUFFERED=1   # log lines appear as they happen, not at exit
TR=$(pgrep -f "[t]rain.py --name main" | head -1)
GZ=$(pgrep -P "${TR:-0}" | head -1)
resume() { [ -n "${TR:-}" ] && kill -CONT "$TR" 2>/dev/null
           [ -n "${GZ:-}" ] && kill -CONT "$GZ" 2>/dev/null; echo "[probe] trainer resumed"; }
trap resume EXIT INT TERM
if [ -n "${TR:-}" ]; then kill -STOP "$TR" 2>/dev/null; [ -n "${GZ:-}" ] && kill -STOP "$GZ" 2>/dev/null; fi
sleep 2
python scripts/evaluate.py --episodes "${1:-3}" --mode plan --rtf "${2:-0.5}" \
   --seed "${3:-777}" --max-time "${4:-150}" --outdir /tmp/phywam_probe --tag probe
