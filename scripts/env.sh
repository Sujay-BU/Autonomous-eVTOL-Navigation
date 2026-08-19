# source this to get a working Phy-WAM shell
export PHYWAM_ROOT=/home/glaze/Desktop/github_projects/Phy_WAM
export CONDA_PREFIX_PHYWAM=/home/glaze/miniconda3/envs/phywam
export PATH=$CONDA_PREFIX_PHYWAM/bin:${PATH:-}
export GZ_SIM_SYSTEM_PLUGIN_PATH=$PHYWAM_ROOT/sim/plugins/build
export GZ_SIM_RESOURCE_PATH=$PHYWAM_ROOT/sim
export LD_LIBRARY_PATH=$CONDA_PREFIX_PHYWAM/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$PHYWAM_ROOT:${PYTHONPATH:-}
export GZ_PARTITION=${GZ_PARTITION:-phywam}

# Kill any headless simulators left behind by an interrupted run.
phywam_reap() {
  for g in $(pgrep -f "[g]z sim -s"); do
    pp=$(ps -o ppid= -p "$g" | tr -d ' ')
    if ! ps -p "$pp" -o cmd= 2>/dev/null | grep -q "phywam\|train.py\|evaluate.py\|gui"; then
      echo "reaping orphan gz sim $g"; kill -9 "$g" 2>/dev/null
    fi
  done
}
export PYTHONUNBUFFERED=1
