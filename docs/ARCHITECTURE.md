# Architecture

This document explains what the system is, the arithmetic that had to work out
before any of it was written, why the pieces are arranged the way they are, and
where it falls short. It is written for someone who has to decide whether to
trust the thing, so the failures are in here alongside the results.

---

## 1. The problem, stated precisely

Launch a lift+cruise eVTOL from one vertiport, transit an urban corridor, and
land on a different vertiport. Do it from a camera and the sensors such an
aircraft actually carries. Avoid buildings, the ground, other eVTOLs, and small
uncrewed aircraft that nobody has told us about. Explain each decision well
enough that a person could audit it.

Three constraints shaped everything:

| Constraint | Consequence |
|---|---|
| 6 GB VRAM, shared between renderer and network | Unreal-based simulators are out, the renderer would take 4-5 GB and leave nothing to learn in |
| 15 GB RAM | The replay buffer is memory-mapped to disk, not held resident |
| No root access | Every dependency comes from conda-forge or pip, nothing is installed system-wide |

---

## 2. The arithmetic, done first

`scripts/feasibility.py` is executable and runs as a gate. Every number below
comes out of it.

### 2.1 Can the aircraft fly the mission?

The vehicle is a lift+cruise configuration: eight lift rotors, a fixed wing,
one pusher propeller. Lift+cruise rather than tiltrotor because the hover and
cruise effectors are physically separate, so the control-allocation matrix is
block-diagonal in the two regimes instead of being a function of tilt angle.
The transition becomes a blend of two well-conditioned problems rather than one
badly-conditioned one.

**Hover**, from actuator-disk momentum theory. Conservation of momentum through
the disk gives the induced velocity

$$v_i = \sqrt{\frac{T}{2\rho A}}, \qquad P_{\text{ideal}} = T v_i, \qquad P_{\text{shaft}} = \frac{P_{\text{ideal}}}{\mathrm{FM}}$$

With $m = 1000$ kg, eight rotors of radius 0.9 m ($A = 20.36\ \mathrm{m^2}$),
figure of merit 0.75 and 88 % electrical efficiency:

- disk loading 482 N/m² (48 kg/m²), mid-range for this class
- $v_i = 14.02$ m/s, ideal power 137.5 kW, shaft 183.3 kW, **electrical 208 kW**

**Cruise**, from a lifting-line drag polar $C_D = C_{D0} + C_L^2/(\pi e AR)$
with $C_{D0} = 0.035$ (lift rotors sit exposed in cruise, so parasite drag is
high), $e = 0.80$, $AR = 13.1$. At 42 m/s: $C_L = 0.83$, $L/D = 14.8$,
**38.5 kW electrical**.

**Mission**, 3 km at 120 m AGL (the FAA UAM corridor altitude):

| phase | t (s) | P (kW) | E (kWh) |
|---|---|---|---|
| vertical climb to 120 m | 24.0 | 239.6 | 1.597 |
| transition hover → wing | 15.0 | 150.0 | 0.625 |
| wingborne cruise | 56.4 | 38.5 | 0.604 |
| transition wing → hover | 15.0 | 150.0 | 0.625 |
| vertical descent and land | 40.0 | 187.5 | 2.083 |
| reserve (20 %) | | | 1.107 |
| **total** | **150.4** | | **6.642** |

6.64 kWh of a 60 kWh pack is **11.1 % SoC**, and still-air range at cruise is
188 km. Energy is not the binding constraint, it has a factor of nine in hand.


### 2.2 Can the sensors see a threat in time to avoid it?

A target of size $s$ at range $r$ subtends $s/r$ radians. A camera with
horizontal field of view $F$ across $W$ pixels resolves $F/W$ per pixel, so
requiring $N$ pixels for detection gives a detection range

$$r_{\text{det}} = \frac{s\,W}{N\,F}$$

Avoidance kinematics: at a 30° bank the available lateral acceleration is
$g\tan 30° = 5.66\ \mathrm{m/s^2}$, so displacing laterally by a required miss
distance $m$ takes $t = \sqrt{2m/a} + \tau$, with $\tau = 0.35$ s of
sense-decide-actuate latency.

| threat | sensor | detect at | time available | time needed | margin |
|---|---|---|---|---|---|
| crossing eVTOL (11 m, cooperative) | ADS-B + DAA cam | 5000 m | 59.5 s | 7.63 s | **7.80×** |
| small UAS (2 m, non-cooperative) | DAA camera | 244 m | 4.7 s | 2.65 s | **1.77×** |
| tower / crane (12 m, static) | DAA camera | 1467 m | 34.9 s | 3.01 s | **11.61×** |

The **binding case is the 2 m non-cooperative UAS at 1.77×**. That single
number set the DAA camera's 25° field of view, and it is why there are two
cameras rather than one: a 90° navigation camera cannot see a 2 m object beyond
68 m, which is 1.3 s at closing speed, less than the 2.65 s the manoeuvre
takes.

**Birds are explicitly out of scope.** A 0.5 m target spans three pixels only
inside 61 m, about 1.5 s before impact. No camera I can afford avoids that,
and real aircraft do not avoid birds either, they are certified to tolerate
the strike. Treating this as a structural case rather than a guidance case is
the honest answer, and pretending otherwise would have quietly inflated every
safety number.

### 2.3 Does the control loop close on the GPU?

Capturing the whole rollout as a CUDA graph replays it from a single launch and
brings it to **75 ms**, at which point it is genuinely compute-bound at 46 % of
the card's fp32 peak. (`torch.compile` fixes the same problem but has to trace
a 64-fold unrolled graph, which costs minutes of compile time for the same
result.)

Even at 75 ms the planner does not fit a 50 ms control period, so **MPPI
replans at 10 Hz behind a barrier filter that runs every control step at
20 Hz**, a slow deliberative planner behind a fast reactive shield, which is
the standard arrangement and leaves 25 % headroom.

| loop | rate |
|---|---|
| Gazebo physics, inner attitude loop | 200 Hz |
| camera render | 30 Hz |
| perception encode, barrier filter | 20 Hz |
| MPPI replan | 10 Hz |
| explanations and GUI | 4 Hz |

### 2.4 Is the lookahead long enough?

The gate compares the planner's horizon against the time the avoidance manoeuvre
actually takes. The tactical cases need 3.01 s. The horizon went to **64 steps = 3.2 s**.

The 150 m well-clear buffer for cooperative traffic needs 7.63 s. That is not a 
local-planner job, it
is resolved strategically off a 5 km ADS-B picture with tens of seconds of
warning. The gate separates tactical from strategic responsibility rather
than demanding one horizon cover both.

**Memory budget** measured: world-model training step 2094 MiB, planner
164 MiB, Gazebo ~700 MiB, Xorg ~300 MiB. Total 3258 MiB of 6141, leaving 47 %
headroom.

---

## 3. Simulator and the plant/model split

**Gazebo Harmonic** (`gz-sim8` from conda-forge), headless, driven over
`gz-transport`. It is the de-facto standard for aerial-vehicle SITL, it
installs without root, and it leaves the GPU free. The cost is photorealism,
which is why every large surface in the world carries a procedurally generated
texture, an untextured grey box against a flat blue sky gives an encoder
nothing to lock onto.

The important structural decision is what is *real* and what is *modelled*:

```
        PLANT  (C++ plugin, in-process, 200 Hz)     "the aircraft"
          rigid-body 6-DOF
        + momentum-theory rotors      + ground effect (Cheeseman-Bennett)
        + blade-element-ish wing      + rotor-wing interference
        + pusher with thrust lapse    + first-order actuator lag
        + post-stall blending         + Dryden-shaped turbulence
                        |
                        |  what the controller is allowed to know
                        v
        MODEL  (PyTorch, differentiable)            "what I believe"
          rigid-body 6-DOF
        + momentum-theory rotors      ( no ground effect )
        + linear-then-clipped wing    ( no interference  )
        + pusher                      ( no actuator lag  )
        + NN residual  <-- learns exactly the difference
```

The model is **deliberately incomplete**. Those four omissions are real,
structured, physically meaningful dynamics, so the residual network has
something to discover rather than noise to memorise. If the model matched the
plant exactly, the "physics-informed" claim would be decoration.

Running the 200 Hz inner loop inside the plugin, rather than in Python, is what
makes the simulator run **4.5x faster than real time** with three cameras
rendering: Python never sits in the inner loop, it only sets attitude and
thrust references at 20 Hz, exactly the split between a flight control
computer and a mission computer on a real aircraft.

---

## 4. The world-action model

### 4.1 Why it is split in two

Two predictors share one recurrent state. A categorical RSSM models the
**visual world**, buildings, ground, traffic, how the scene flows. A
physics-informed head models the **aircraft**.

The reasoning is simply that rigid-body flight dynamics are known in closed
form. Making a network rediscover gravity from pixels wastes both capacity and
samples. Conversely, no closed form predicts what a camera will see next. Each
half gets the representation it deserves.

### 4.2 The stochastic half

Following DreamerV3, because that recipe is what makes a single set of
hyper-parameters train stably across wildly different signal scales:

$$h_t = \mathrm{GRU}(h_{t-1}, [z_{t-1}, a_{t-1}])$$
$$\text{prior } p(z_t \mid h_t), \qquad \text{posterior } q(z_t \mid h_t, e_t)$$

with $z_t$ a set of 32 categoricals of 32 classes, sampled with
straight-through gradients and blended with 1 % uniform ("unimix") so no class
can collapse to exactly zero probability and kill its gradient.

The KL is **balanced**: the dynamics term trains the prior toward the
posterior, the representation term trains the posterior toward the prior, at
different rates, each clamped by a free-nats floor:

$$\mathcal{L}_{KL} = \beta_{\text{dyn}} \max(\mathrm{KL}(\mathrm{sg}[q] \Vert p), \lambda) + \beta_{\text{rep}} \max(\mathrm{KL}(q \Vert \mathrm{sg}[p]), \lambda)$$

Reward, clearance and value are predicted as **two-hot distributions over a
symlog-spaced support** rather than regressed. Returns here span from
single-digit shaping terms to a +60 goal bonus, an MSE head handles that badly,
cross-entropy over a fixed support is scale-free.

### 4.3 The physics-informed half

$$x_{t+1} = \underbrace{\mathrm{RK2}\big(f_{\text{phys}}\big)(x_t, a_t)}_{\text{analytic, exact, no parameters}} + \Delta t \cdot \underbrace{g_\theta(h_t, z_t, x_t, a_t)}_{\text{learned residual}}$$

where $x \in \mathbb{R}^{12}$ is position, Euler angles, body velocity and body
rates. $f_{\text{phys}}$ is the full analytic model, rotor thrust and torque,
wing lift and drag, the cascade attitude controller, and the rigid-body
equations

$$\dot v_b = \frac{F}{m} - \omega \times v_b + R^\top g, \qquad \dot\omega = I^{-1}\big(M - \omega \times I\omega\big)$$

integrated with midpoint RK2 rather than Euler, because the rotational modes
sit near 5-6 rad/s and Euler at $\Delta t = 50$ ms is only marginally stable
for them.

**The residual's output layer is initialised to exactly zero.** At step 0 of
training the world model already predicts flight correctly from first
principles, and learning only ever has to improve on that. There is no phase
where the agent must discover gravity. This is what buys the sample efficiency,
and it is directly visible in training: the planner flies competently before a
single gradient step.

Two losses, in the PINN tradition, a data term and a term that prefers the
physics to do the explaining:

$$\mathcal{L}_{\text{phys}} = \left\Vert \frac{\hat x_{t+1} - x_{t+1}}{\sigma} \right\Vert^2, \qquad \mathcal{L}_{\text{res}} = \Vert g_\theta \Vert^2$$

The normaliser $\sigma$ weights position, attitude, velocity and rate errors
onto comparable scales. Given two models that fit the data equally well, the
regulariser picks the one that leans on the analytic term, because that is the
part that extrapolates.

### 4.4 How well does it actually predict?

`scripts/validate_model.py` rolls the analytic model, the analytic-plus-residual
model, and a hold-state floor forward over recorded plant data and measures the
drift. It runs off the replay buffer, so it needs no simulator.

There is an important ceiling on what the residual can achieve here. The plant is driven by Dryden-shaped turbulence
with sigma = 4 m/s. Over a 2 s horizon an unpredictable 4 m/s lateral gust
displaces the aircraft by roughly

$$\tfrac12 \cdot \frac{\sigma}{\tau}\, t^2 \,\sim\, \text{several metres}$$

which is the same order as the total observed position error. That component is
**irreducible**: the residual is conditioned on the latent state and the action,
and no amount of training lets it predict white noise it has not yet been hit
by. The residual can only recover the *deterministic* modelling gap, ground
effect, rotor-wing interference, actuator lag, the post-stall blend.

This is why the architecture does not lean on open-loop prediction accuracy.
The planner replans at 10 Hz, so it never rides a single 3.2 s prediction to the
end, and the barrier filter operates on measured state rather than predicted
state. Long-horizon prediction is used to *rank* candidate manoeuvres, a job
that tolerates a common-mode error far better than trajectory tracking would.

---

## 5. Control

### 5.1 Three layers, by design

```
  A*  over the known obstacle database      static, global, seconds ahead
       |   subgoal ~160 m ahead
  MPPI over the learned model, 3.2 s        dynamic, sensed, anticipatory
       |   one action
  HOCBF barrier filter, every step          hard constraint, last resort
       |
  aircraft
```

A 3.2 s planner cannot route a 3 km flight, and a 3 km router cannot dodge a
crossing drone. Real eVTOLs fly published corridors computed against an
obstacle database and use onboard sensing for what the database cannot contain,
I mirror that exactly. Handing the learner a subgoal 160 m ahead instead of a
pad 3 km away is also what makes the reward informative, progress toward a
waypoint is dense, progress toward a distant vertiport is nearly flat across an
entire episode.

### 5.2 MPPI

**The sampling mean is the prior, and the prior decides everything.** This is
the single most important thing learned building this system, and it cost
seven evaluation rounds to find.

MPPI explores only the neighbourhood of whatever mean it is given. Warm-starting
purely from the previous solution is standard advice, and it is fine once the
plan is good -- but a plan that has drifted keeps being refined inside the same
drifted neighbourhood. Measured: **0 completed missions out of 18**, across six
configurations, while the geometric guidance law the planner was ignoring
completed **34 out of 34**.

Seeding the mean from that guidance law instead -- a 60/40 blend of the shifted
previous solution and the nominal action -- took the same checkpoint from **0 %
to 100 % mission success** with no change to the world model, the cost weights,
or the barrier filter. MPPI's job then becomes what sampling-based MPC is
actually good at: improving a competent nominal using a learned model of what
is about to happen, rather than rediscovering flight from scratch every
control step. This is the role TD-MPC2 gives its policy prior, here the prior is
a controller whose behaviour we can already vouch for, which also means the
system degrades gracefully if the learned components are poor.



512 action sequences are sampled around the previous solution, rolled 64 steps
through the physics head and the RSSM prior, and scored. The update uses
**elite selection**: the best 12 % are kept and softmax-averaged within that
set, with a temperature scaled by the elite spread rather than fixed.


The sampling prior sits at the **hover trim action**, not at zero. With $a = 0$
the schedule channel decodes to half wing-borne, which scales the collective by
0.54, below the 0.576 needed to hold weight. A zero-mean prior literally
describes an aircraft that cannot hover, and the planner dutifully flew it into
the ground.

Cost terms are named and separable, which is what makes the attribution
explanation exact rather than approximate: corridor tracking (horizontal and
vertical separately), velocity-direction alignment, speed-to-go, obstacle
clearance from the mapped SDF, ground floor, the *learned* hazard head,
predicted traffic separation, control effort, energy, a stall guard, and
envelope limits.

Two of those exist because of specific failures. **Vertical corridor tracking**
is separate from horizontal because a single 3-D distance term is nearly flat
in the vertical over 3.2 s, so the planner traded altitude for energy, every
replan, until it hit the ground. It was minimising the cost correctly, the cost
was wrong. **Velocity alignment** exists because distance alone does not say
which way to point: at 42 m/s the turn radius is 311 m, so a planner that only
shrinks range will happily orbit the target, and did, 3× the direct distance.

### 5.3 The barrier filter

With $b(p) = d(p) - d_{\text{safe}}$ and acceleration as the control, $b$ has
relative degree 2, so a first-order condition is not enforceable. The
high-order condition

$$\ddot b + \alpha_1 \dot b + \alpha_0 b \ge 0$$

expands, dropping the curvature term, which vanishes away from box corners,
into a **linear inequality in the commanded acceleration**:

$$\nabla d \cdot u \,\ge\, -\alpha_1 (\nabla d \cdot v) \,-\, \alpha_0 (d - d_{\text{safe}})$$

One such row per nearby hazard (buildings, ground, each tracked intruder, using
*relative* motion for the moving ones), and the desired acceleration is
projected onto the intersection:

$$u^\star = \arg\min_u \tfrac12\Vert u - u_{\text{des}}\Vert^2 \quad \text{s.t.}\quad Gu \ge h,\ \Vert u \Vert \le a_{\max}$$

With at most a handful of rows this is solved **exactly by enumerating active
sets** and checking KKT sign conditions, deterministic, and far cheaper than
calling a general QP solver in the control loop. Measured at 200 µs.

The correction is mapped back to attitude references through the same tilt
relations the guidance uses, $a_{\text{right}} = g\tan\phi$ and
$a_{\text{fwd}} = -g\tan\theta$.

**The ground floor is position-dependent**, and this matters more than it
sounds. A constant 20 m floor makes landing formally impossible: the barrier
will hold the aircraft above the pad indefinitely, and it did exactly that,
parked 0.2 m laterally from the pad at 19 m altitude, hovering, forever. Real
vertiports have a protected-surface carve-out over the FATO for precisely this
reason, so the floor relaxes to 1 m inside the pad and returns to 20 m
en route.

---

## 6. Perception and detect-and-avoid

The sensor suite mirrors what a certified eVTOL carries, and each element earns
its place from §2.2:

- **navigation camera**, 90° FOV, RGB + depth at 160×120, close-in geometry
- **DAA camera**, 25° FOV, RGB at 160×120, long-range traffic, this is the one
  that makes the binding threat pass
- **ADS-B / Remote-ID feed**, 5 km, 1 Hz, with broadcast position noise,
  cooperative traffic only
- IMU with per-axis noise, GNSS/INS, barometer

Detections are fused into eight ranked threat tracks, ordered by time-to-closest
approach, each carrying range, bearing, elevation, range-rate, cooperative flag
and inverse time-to-collision. The policy sees *tracks*, not raw pixels, for
traffic, which is how real DAA systems work and is far more sample-efficient
than asking a 64×64 encoder to notice a three-pixel aircraft.

**Unmapped obstacles.** Fourteen construction cranes exist in the world and in
collision truth but are deliberately withheld from the obstacle database that
the route planner and the barrier filter consume. The only way to avoid them is
to see them. Without this the camera would be decorative, every hazard would
already be in the map, and a purely geometric controller would score just as
well.

---

## 7. Explainability

Four views, chosen because between them they cover every stage of the decision.
The distinction between them is stated plainly in the interface, because an
explanation that looks authoritative but is merely plausible is worse than none.

| view | what it answers | faithful? |
|---|---|---|
| **Grad-CAM** on the encoder, w.r.t. the model's *own predicted clearance* | which pixels made it think this was dangerous | post-hoc, approximate |
| **Imagination decode**, the frames the world model expects over the next 3.2 s, beside what actually arrived | does the model understand the situation it is in | exact model output |
| **Cost attribution**, the chosen trajectory's cost itemised into named terms | why this action and not another | *is* the decision, itemised |
| **Counterfactuals**, the cost of six fixed alternative manoeuvres under the same model | what would banking left have cost | exact, same model |

Grad-CAM is taken with respect to predicted clearance rather than generic
saliency, so it answers "what looks dangerous" rather than the vaguer "what is
salient". Rising **prediction error** between the reconstruction and the actual
frame is a direct, honest signal that the controller is operating outside what
it has learned, arguably the single most useful number on the display.

---

## 8. Safety metrics

Reported in detect-and-avoid vocabulary rather than RL vocabulary, because
"mean episode return" tells an aviation reader nothing:

- **MAC / NMAC**, mid-air collision, and near mid-air within 30 m
- **LoWC**, loss of well clear: below 150 m horizontal *and* 30 m vertical
  against cooperative traffic
- **minimum obstacle clearance**, closest approach to any real obstacle,
  mapped or not, against a 20 m requirement
- **shield engagement rate**, the fraction of control steps on which the
  barrier filter had to repair the planner's action. This is a direct measure
  of how often the learned layer proposed something unsafe, and it is the
  number to watch: a controller that is only safe because the shield is
  constantly intervening has not learned to fly.

Obstacle clearance and ground clearance are tracked separately. An aircraft
sitting on a pad is 1.7 m from the ground by construction, and reporting that
as a flight's minimum clearance would make every run look equally dangerous.

Results are compared against a **classical baseline**, geometric guidance plus
an artificial potential field with reactive avoidance driven straight off the
depth image. It is built to be a fair comparator rather than a straw man: same
obstacle database, same tracked traffic, same barrier filter, and a genuine
camera-based reactive term. What it cannot do is anticipate. A potential field
reacts to the gradient it is standing in, so it cannot start a manoeuvre three
seconds before a conflict develops, and cannot reason about where an intruder
*will be* rather than where it is.

---

## 9. Limitations

- The world is a procedurally generated city with box buildings and OGRE2
  materials. Sim-to-real transfer is not claimed and has not been tested.
- Traffic intent is not modelled, intruders are extrapolated at constant
  velocity over the planning horizon.
- The barrier filter guarantees forward invariance only for the constraints it
  is given, under the acceleration model it assumes. It cannot protect against a
  hazard that was never detected, and it saturates: at 42 m/s closing on a wall
  45 m away, 6 m/s² is not enough and it correctly reports the state as one the
  planner should never have entered.
- Wind is a first-order Dryden-shaped approximation, not the full spectrum.
- Sub-metre objects are tolerated, not avoided (§2.2).
- **Known defect, not fixed:** the actor-critic's actor never became competent
  (§9.16). The deployed controller does not depend on it -- MPPI runs with
  actor seeding disabled -- but the imagination-trained policy in `agent.py`
  should be treated as non-functional until its gradient path is repaired and
  retrained.
