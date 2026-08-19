"""
Gazebo <-> Python bridge.

Owns the simulator process and the gz-transport plumbing. Everything above
this file sees numpy arrays and never touches protobuf.

Each bridge gets its own GZ_PARTITION so several environments can run side by
side without their topics colliding, which is how we get parallel data
collection out of one machine.
"""
import os, sys, time, signal, subprocess, threading
import numpy as np

from gz.transport13 import Node
from gz.msgs10.float_v_pb2 import Float_V
from gz.msgs10.image_pb2 import Image
from gz.msgs10.world_control_pb2 import WorldControl
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.physics_pb2 import Physics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# state vector layout -- must match sim/plugins/PhyWamPlant.cc
S_POS, S_QUAT, S_VEL_W, S_VEL_B = 0, 3, 7, 10
S_OMEGA, S_ACC_B = 13, 16
S_VAIR, S_ALPHA, S_BETA, S_AGL, S_SOC, S_POWER = 19, 20, 21, 22, 23, 24
S_ROTOR, S_PUSH, S_AIL, S_ELE, S_RUD = 25, 33, 34, 35, 36
S_WIND, S_TIME, S_LEN = 37, 40, 41


def _sim_env(partition):
    e = dict(os.environ)
    e["GZ_SIM_SYSTEM_PLUGIN_PATH"] = os.path.join(ROOT, "sim", "plugins", "build")
    e["GZ_SIM_RESOURCE_PATH"] = os.path.join(ROOT, "sim")
    e["GZ_PARTITION"] = partition
    e["PATH"] = os.path.join(sys.prefix, "bin") + ":" + e.get("PATH", "")
    return e


class GazeboBridge:
    def __init__(self, world_sdf, partition=None, ns="phywam",
                 headless=True, verbose=1, start=True, world_name="urban",
                 lockstep=False):
        self.world_sdf = world_sdf
        self.world_name = world_name
        self.lockstep = lockstep
        self.ns = ns
        self.partition = partition or f"phywam_{os.getpid()}_{id(self)&0xffff}"
        self.headless = headless
        self.verbose = verbose
        self.proc = None
        self._lock = threading.Lock()

        self.state = np.zeros(S_LEN, np.float64)
        self.rgb_nav = np.zeros((1, 1, 3), np.uint8)
        self.depth_nav = np.zeros((1, 1), np.float32)
        self.rgb_daa = np.zeros((1, 1, 3), np.uint8)
        self.traffic = np.zeros((0, 7), np.float64)
        self.n_state = self.n_rgb = self.n_depth = self.n_daa = 0

        if start:
            self.start()

    # ------------------------------------------------------------- process --
    def start(self):
        os.environ["GZ_PARTITION"] = self.partition   # node inherits this
        # Without lockstep the server free-runs while Python is busy. A single
        # 75 ms MPPI solve then lets ~4 control periods of simulated time slip
        # past unactuated, so the aircraft is really being flown at 4-5 Hz on
        # stale commands. Starting paused and stepping explicitly makes the
        # control period exactly 1/ctrl_hz of simulated time, always.
        cmd = ["gz", "sim", "-s", "-v", str(self.verbose)]
        if not self.lockstep:
            cmd.insert(3, "-r")
        cmd.append(self.world_sdf)
        if self.headless:
            cmd.insert(3, "--headless-rendering")
        self.proc = subprocess.Popen(
            cmd, env=_sim_env(self.partition),
            stdout=subprocess.DEVNULL if self.verbose < 2 else None,
            stderr=subprocess.STDOUT if self.verbose >= 2 else subprocess.DEVNULL,
            preexec_fn=os.setsid)
        self._connect()
        return self

    def _connect(self):
        self.node = Node()
        n, ns = self.node, self.ns
        n.subscribe(Float_V, f"{ns}/state",    self._on_state)
        n.subscribe(Float_V, f"{ns}/traffic",  self._on_traffic)
        n.subscribe(Image,   f"{ns}/cam_nav",  self._on_rgb)
        n.subscribe(Image,   f"{ns}/depth_nav", self._on_depth)
        n.subscribe(Image,   f"{ns}/cam_daa",  self._on_daa)
        self.pub_cmd = n.advertise(f"{ns}/cmd", Float_V)
        self.pub_reset = n.advertise(f"{ns}/reset", Float_V)
        self.ctrl_srv = f"/world/{self.world_name}/control"
        self.phys_srv = f"/world/{self.world_name}/set_physics"

    def set_rtf(self, rtf, dt=None):
        """Throttle the simulator to a fraction of real time.

        The server free-runs by default, which is what makes training cheap:
        the control loop's ~7 ms of work fits comfortably inside the ~12.5 ms
        of wall time that one 50 ms control period costs at RTF 4, so the loop
        always waits for the simulator rather than the other way round.

        That inverts the moment the loop gets expensive. An MPPI solve plus a
        Grad-CAM pass costs ~150 ms, during which a free-running simulator
        advances half a second and the aircraft coasts on a stale command.
        Slowing the simulator restores the invariant instead of papering over
        it, and costs nothing during evaluation, where wall-clock time is not
        the scarce resource.
        """
        from .config import CFG
        req = Physics()
        req.real_time_factor = float(rtf)
        req.max_step_size = float(dt or 1.0 / CFG.lrn.phys_hz)
        ok, _ = self.node.request(self.phys_srv, req, Physics, Boolean, 3000)
        return bool(ok)

    def step(self, n=1, timeout_ms=4000):
        """Advance exactly n physics iterations and block until they are done."""
        req = WorldControl()
        req.pause = True          # stay paused after the burst
        req.multi_step = int(n)
        ok, _ = self.node.request(self.ctrl_srv, req, WorldControl, Boolean,
                                  timeout_ms)
        return bool(ok)

    def step_and_sense(self, n, want_image=True, timeout=1.5):
        """Step, then wait for the resulting sensor messages to arrive.

        Stepping is synchronous but publication is not, so without this the
        controller would act on the previous frame while believing it was
        current."""
        s0, r0, d0, a0 = self.n_state, self.n_rgb, self.n_depth, self.n_daa
        self.step(n)
        t0 = time.time()
        while time.time() - t0 < timeout:
            fresh = self.n_state > s0
            if want_image:
                fresh = fresh and self.n_rgb > r0 and self.n_depth > d0 \
                        and self.n_daa > a0
            if fresh:
                return True
            time.sleep(0.0005)
        return False

    def wait_ready(self, timeout=60.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError("gz sim exited early")
            if self.n_state > 0 and self.n_rgb > 0 and self.n_depth > 0 \
                    and self.n_daa > 0:
                return True
            time.sleep(0.10)
        raise TimeoutError(
            f"sim not ready: state={self.n_state} rgb={self.n_rgb} "
            f"depth={self.n_depth} daa={self.n_daa}")

    def close(self):
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                self.proc.wait(timeout=6)
            except Exception:
                try: os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except Exception: pass
        self.proc = None

    # ----------------------------------------------------------- callbacks --
    def _on_state(self, m):
        if len(m.data) < S_LEN: return
        with self._lock:
            self.state = np.asarray(m.data, np.float64)
            self.n_state += 1

    def _on_traffic(self, m):
        d = np.asarray(m.data, np.float64)
        if d.size % 7: return
        with self._lock:
            self.traffic = d.reshape(-1, 7)

    @staticmethod
    def _rgb(m):
        a = np.frombuffer(m.data, np.uint8)
        if a.size < m.height * m.width * 3: return None
        return a[:m.height * m.width * 3].reshape(m.height, m.width, 3)

    def _on_rgb(self, m):
        a = self._rgb(m)
        if a is None: return
        with self._lock: self.rgb_nav = a.copy(); self.n_rgb += 1

    def _on_daa(self, m):
        a = self._rgb(m)
        if a is None: return
        with self._lock: self.rgb_daa = a.copy(); self.n_daa += 1

    def _on_depth(self, m):
        a = np.frombuffer(m.data, np.float32)
        if a.size < m.height * m.width: return
        d = a[:m.height * m.width].reshape(m.height, m.width).copy()
        d[~np.isfinite(d)] = np.inf
        with self._lock: self.depth_nav = d; self.n_depth += 1

    # -------------------------------------------------------------- actions --
    def send(self, thrust_col, roll_ref, pitch_ref, yawrate_ref, push, sched):
        m = Float_V()
        m.data.extend([float(thrust_col), float(roll_ref), float(pitch_ref),
                       float(yawrate_ref), float(push), float(sched)])
        self.pub_cmd.publish(m)

    def reset_to(self, x, y, z, yaw, soc=1.0):
        m = Float_V()
        m.data.extend([float(x), float(y), float(z), float(yaw), float(soc)])
        self.pub_reset.publish(m)

    def snapshot(self):
        with self._lock:
            return dict(state=self.state.copy(),
                        rgb=self.rgb_nav.copy(),
                        depth=self.depth_nav.copy(),
                        daa=self.rgb_daa.copy(),
                        traffic=self.traffic.copy())

    # ------------------------------------------------------------ accessors --
    @property
    def pos(self):   return self.state[S_POS:S_POS+3]
    @property
    def quat(self):  return self.state[S_QUAT:S_QUAT+4]      # w,x,y,z
    @property
    def vel_w(self): return self.state[S_VEL_W:S_VEL_W+3]
    @property
    def vel_b(self): return self.state[S_VEL_B:S_VEL_B+3]    # FRD
    @property
    def omega(self): return self.state[S_OMEGA:S_OMEGA+3]    # FRD
    @property
    def soc(self):   return self.state[S_SOC]
    @property
    def sim_time(self): return self.state[S_TIME]

    def euler(self):
        """roll, pitch, yaw in the FRD sense (pitch positive = nose up)."""
        w, x, y, z = self.quat
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        s = np.clip(2*(w*y - z*x), -1, 1)
        pitch_flu = np.arcsin(s)
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return roll, -pitch_flu, yaw
