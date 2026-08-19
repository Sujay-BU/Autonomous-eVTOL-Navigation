# Autonomous eVTOL Navigation & Collision Avoidance Using a Physics-informed World-Action Model

Autonomous vertiport-to-vertiport eVTOL flight from a camera, driven by a
physics-informed world-action model, planned with sampling-based MPC, and
shielded by a control-barrier filter.

Full design rationale, derivations and limitations: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

---

## What it does

An eVTOL lifts off a vertiport, transits a 3 km urban corridor at 120 m AGL,
and lands on a different vertiport. It avoids buildings, terrain, cooperative
eVTOL traffic and non-cooperative small UAS, including fourteen construction
cranes that exist in the world but are deliberately withheld from its obstacle
database, so the camera has to earn its place.

Every control decision is explained four ways: what the perception encoder
found dangerous, what the world model expects to see next, an exact itemisation
of why this action beat the alternatives, and what six other manoeuvres would
have cost.

---

## Result
[![Result Video](https://raw.githubusercontent.com/Sujay-BU/Autonomous-eVTOL-Navigation/blob/main/recordings/cover.jpg)](https://raw.githubusercontent.com/Sujay-BU/Autonomous-eVTOL-Navigation/blob/main/recordings/final_run2_VP3_to_VP2.mp4)

---

## Layout

```
phywam/
  config.py       single source of truth: airframe, sensors, world, learning
  physics.py      differentiable analytic 6-DOF eVTOL model (the "prior")
  worldmodel.py   PI-RSSM: categorical RSSM + physics-informed dynamics head
  agent.py        actor and critic, trained inside imagination
  planner.py      MPPI over the world model + high-order CBF safety filter
  route.py        A* global route over the known obstacle database
  geometry.py     analytic signed-distance fields, camera projection
  env.py          the flight task: reward, termination, safety bookkeeping
  bridge.py       Gazebo <-> Python transport
  runner.py       the control loop; also the classical baseline controller
  replay.py       memory-mapped sequence replay
  xai.py          Grad-CAM, imagination decode, cost attribution, counterfactuals
  dashboard.py    the 1600x900 operator display (GUI and video share this)
  instrument.py   wires the control loop to the XAI suite and the dashboard
  metrics.py      DAA-vocabulary safety metrics and reporting
  gui.py          live PySide6 operator console
sim/
  plugins/        C++ Gazebo systems: the plant, and the traffic
  worlds/         generated worlds (SDF + obstacle metadata)
  materials/      procedurally generated textures
scripts/
  feasibility.py  the arithmetic gate; run this first
  gen_world.py    procedural city, vertiports, traffic, unmapped cranes
  gen_textures.py procedural ground and facade textures
  train.py        training
  evaluate.py     scored evaluation, with optional video recording
```

---

## Setup

Everything installs into a conda environment without root.

```bash
conda create -y -n phywam -c conda-forge \
    python=3.11 gz-sim8 gz-tools2 gz-transport13 gz-transport13-python \
    gz-msgs10 gz-msgs10-python cmake ninja pkg-config gxx_linux-64 make \
    numpy scipy protobuf pyyaml tqdm rich ffmpeg

conda activate phywam
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install opencv-python-headless PySide6 pyqtgraph imageio imageio-ffmpeg \
            matplotlib tensorboard einops scikit-image psutil
```

Build the two Gazebo plugins:

```bash
source scripts/env.sh
cmake -S sim/plugins -B sim/plugins/build -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build sim/plugins/build -j8
```

`scripts/env.sh` sets `GZ_SIM_SYSTEM_PLUGIN_PATH`, `PYTHONPATH` and a
`GZ_PARTITION`; source it in any shell that talks to the simulator.

---

## Run it

**Check the arithmetic first.** It is a gate, not a formality.

```bash
python scripts/feasibility.py
```

**Generate a world** (seeded, so you can make as many as you like):

```bash
python scripts/gen_textures.py
python scripts/gen_world.py 1
```

**Train:**

```bash
python scripts/train.py --name main --hours 9 --resume
```

Seeds the replay with a scripted pilot, then alternates collection with
gradient steps, handing over from the scripted pilot to the learned actor and
walking a spawn-point curriculum back from short final to the departure pad.
Checkpoints every ten minutes to `runs/main/ckpt.pt`; `--resume` picks up where
it left off, and `SIGTERM` checkpoints cleanly.

**Evaluate and record:**

```bash
python scripts/evaluate.py --record --episodes 4 --mode plan \
       --ckpt runs/main/ckpt.pt --outdir recordings
```

Writes one MP4 per flight plus a metrics JSON and a formatted safety report.
`--mode baseline` runs the classical comparator through the identical code path.

**Live console:**

```bash
python -m phywam.gui --ckpt runs/main/ckpt.pt --mode plan
```

---

## A note on the simulator's clock

The simulator free-runs by default, which is what makes training cheap: the
control loop's ~7 ms of work fits inside the ~12.5 ms that one 50 ms control
period costs at RTF 4, so the loop always waits for the simulator.

That inverts as soon as the loop gets expensive. An MPPI solve plus a Grad-CAM
pass costs ~150 ms, during which a free-running simulator advances half a
second and the aircraft coasts on a stale command. `evaluate.py` therefore
throttles the simulator (`--rtf 0.25`) and **measures the realised control
period**, reporting it as `dt` and `overrun` on every run. If those numbers
drift, the flight is not being controlled at the rate the design assumes, and
the safety numbers should not be believed.
