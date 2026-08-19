"""
Vertiport-to-vertiport flight environment.

One episode = launch from a vertiport, transit the urban corridor, land on a
different vertiport, without hitting a building, the ground, another eVTOL or a
small UAS, and without running the battery below reserve.

The environment advances on SIM time, not wall time. With the plant running
in-process at 250 Hz the simulator is ~4x faster than real time, so pacing on
sim time keeps the control period exactly 50 ms regardless of machine load.
"""
import os, time
import numpy as np

from .bridge import GazeboBridge
from .config import CFG
from .geometry import WorldGeom, Camera, quat_to_R, FLU2FRD

A, S, L, SF, WD = CFG.air, CFG.sen, CFG.lrn, CFG.saf, CFG.wld

# episode outcomes
OK, CRASH_BLD, CRASH_GND, CRASH_AIR, BATTERY, UPSET, TIMEOUT, GOAL, OOB = range(9)
OUTCOME = ["running", "hit_building", "hit_ground", "midair_collision",
           "battery", "upset", "timeout", "goal", "out_of_bounds"]


class TrackFuser:
    """Turns raw intruder truth into the track tensor the policy actually sees.

    Cooperative eVTOLs arrive twice: over a simulated ADS-B/Remote-ID link
    (long range, 1 Hz, position noise) and, when close and in frame, from the
    camera detector. Non-cooperative sUAS arrive ONLY from the camera, which
    is exactly why the feasibility analysis made them the binding threat.
    """

    def __init__(self, cam_daa, cam_nav, rng, detector=None):
        self.daa, self.nav, self.rng = cam_daa, cam_nav, rng
        self.detector = detector          # optional learned CNN detector
        self.adsb = {}                    # id -> (pos, vel, t)
        self.last_world = []
        self.t_adsb = -1e9

    def reset(self):
        self.adsb.clear(); self.t_adsb = -1e9; self.last_world = []

    def __call__(self, t, pos, R_wb, vel_w, truth, det_tracks=None):
        """truth: (K,7) x,y,z,vx,vy,vz,kind  (kind 0 = cooperative eVTOL)."""
        out = np.zeros((S.max_tracks, 8), np.float32)
        if truth.size == 0:
            return out
        P, V, kind = truth[:, :3], truth[:, 3:6], truth[:, 6].astype(int)

        # --- cooperative surveillance: 1 Hz, ranged, noisy -------------------
        if t - self.t_adsb >= 1.0 / S.adsb_hz:
            self.t_adsb = t
            for i in np.where(kind == 0)[0]:
                if np.linalg.norm(P[i] - pos) < S.adsb_range:
                    self.adsb[i] = (P[i] + self.rng.normal(0, S.adsb_pos_sigma, 3),
                                    V[i], t)

        cands = []
        for i in range(len(P)):
            rel = P[i] - pos
            rng_ = float(np.linalg.norm(rel))
            if rng_ < 1e-3:
                continue
            size = 11.0 if kind[i] == 0 else 2.0
            seen = False
            est = P[i]
            # camera: in frame AND big enough in pixels to be detectable
            for cam in (self.daa, self.nav):
                uv, fx, vis = cam.project(P[i][None], pos, R_wb)
                if vis[0] and cam.angular_size(size, rng_) >= 3.0:
                    seen = True
                    est = P[i] + self.rng.normal(0, 0.012 * rng_, 3)
                    break
            if not seen and i in self.adsb:
                pa, va, ta = self.adsb[i]
                if t - ta < 3.0:
                    seen, est = True, pa + va * (t - ta)
            if not seen:
                continue
            rel = est - pos
            rng_ = float(np.linalg.norm(rel))
            rel_b = FLU2FRD @ (R_wb.T @ rel)          # body FRD
            vrel = V[i] - vel_w
            rdot = float(np.dot(vrel, rel) / max(rng_, 1e-3))
            ttc = rng_ / max(-rdot, 1e-3) if rdot < 0 else 1e6
            cands.append((ttc, rng_, rel_b, rdot, kind[i], est, V[i]))

        cands.sort(key=lambda c: c[0])                # most urgent first
        # world-frame estimates for the planner / barrier filter. These are the
        # ESTIMATES, never the truth: the controller must live with what it can
        # actually observe.
        self.last_world = [(c[5], c[6], c[4] == 0) for c in cands[:S.max_tracks]]
        for j, (ttc, rng_, rb, rdot, kd, _e, _v) in enumerate(cands[:S.max_tracks]):
            az = np.arctan2(-rb[1], rb[0])
            el = np.arctan2(-rb[2], np.hypot(rb[0], rb[1]))
            out[j] = [1.0, rng_ / 1000.0, np.sin(az), np.cos(az), el / 1.5,
                      rdot / 100.0, 1.0 if kd == 0 else 0.0,
                      float(np.clip(1.0 / max(ttc, 0.2), 0, 5)) / 5.0]
        return out


class VertiportEnv:
    def __init__(self, world_sdf, seed=0, headless=True, max_time=210.0,
                 verbose=0, detector=None):
        self.geom = WorldGeom(world_sdf.replace(".sdf", ".json"))
        self.rng = np.random.default_rng(seed)
        self.max_time = max_time
        self.dt = 1.0 / L.ctrl_hz
        self.br = GazeboBridge(world_sdf, headless=headless, verbose=verbose)
        self.cam_nav = Camera(S.nav_w, S.nav_h, S.nav_hfov_deg, [4.15, 0, 0.15], 0.22)
        self.cam_daa = Camera(S.daa_w, S.daa_h, S.daa_hfov_deg, [4.15, 0, 0.42], 0.0)
        self.fuser = TrackFuser(self.cam_daa, self.cam_nav, self.rng, detector)
        self.br.wait_ready(90)
        self.n_ep = 0

    # ------------------------------------------------------------------ obs --
    @staticmethod
    def _resize(img, r):
        import cv2
        return cv2.resize(img, (r, r), interpolation=cv2.INTER_AREA)

    def _observe(self):
        import cv2
        snap = self.br.snapshot()
        st = snap["state"]
        pos, q = st[0:3], st[3:7]
        R_wb = quat_to_R(q)
        vel_w, vel_b, omega = st[7:10], st[10:13], st[13:16]

        rgb = self._resize(snap["rgb"], L.img_res).astype(np.float32) / 255.0
        d = snap["depth"].copy()
        d[~np.isfinite(d)] = S.nav_far
        d = np.clip(d, 0, S.nav_far) / S.nav_far
        # log-compress: near obstacles are what matters, far ones are all "far"
        d = np.log1p(d * 20.0) / np.log(21.0)
        dep = self._resize(d.astype(np.float32), L.img_res)[..., None]
        img = np.concatenate([rgb, dep], -1).transpose(2, 0, 1)   # (4,R,R)

        # --- proprioception (22) ------------------------------------------
        roll, pitch, yaw = self.br.euler()
        gvec = self.subgoal - pos
        gh = np.linalg.norm(gvec[:2])
        gb = FLU2FRD @ (R_wb.T @ gvec)
        gdir = gb / max(np.linalg.norm(gb), 1e-6)
        pro = np.array([
            vel_b[0]/50, vel_b[1]/20, vel_b[2]/20,
            omega[0], omega[1], omega[2],
            roll, pitch,
            pos[2]/200.0,
            st[19]/50.0, st[20], st[21],
            gdir[0], gdir[1], gdir[2],
            np.log1p(gh)/8.0,
            st[23],
            *self.prev_a
        ], np.float32)

        trk = self.fuser(st[40], pos, R_wb, vel_w, snap["traffic"])
        self._snap, self._st, self._pos, self._R = snap, st, pos, R_wb
        self.phys_state = np.array([
            pos[0], pos[1], pos[2], roll, pitch, yaw,
            vel_b[0], vel_b[1], vel_b[2], omega[0], omega[1], omega[2]],
            np.float32)
        return dict(img=img, pro=pro, trk=trk,
                    rgb_full=snap["rgb"], depth_full=snap["depth"],
                    daa_full=snap["daa"], state=st, traffic=snap["traffic"])

    # ---------------------------------------------------------------- reset --
    def reset(self, start_vp=None, goal_vp=None, start_frac=None):
        """start_frac: if given, spawn AIRBORNE that fraction of the way along
        the direct line to the goal, at corridor altitude.

        Curriculum. A scripted pilot good enough to fly 1.5 km and land is
        nearly as hard to write as the controller we are trying to learn, so
        with a from-the-pad-only reset the agent may never once observe a
        successful arrival -- and a critic that has never seen the goal bonus
        has nothing to propagate backwards. Starting some episodes on short
        final makes success reachable immediately, and the spawn point is
        walked back toward the departure pad as competence grows."""
        vps = self.geom.vp
        n = len(vps)
        if start_vp is None:
            start_vp = int(self.rng.integers(n))
        if goal_vp is None:
            far = [j for j in range(n) if j != start_vp and
                   np.linalg.norm(vps[j][:2] - vps[start_vp][:2]) > 700.0]
            goal_vp = int(self.rng.choice(far if far else
                                          [j for j in range(n) if j != start_vp]))
        self.start_vp, self.goal_vp = start_vp, goal_vp
        self.goal = np.array([vps[goal_vp][0], vps[goal_vp][1],
                              vps[goal_vp][2] + 0.4])
        sp = vps[start_vp]
        yaw = float(np.arctan2(self.goal[1]-sp[1], self.goal[0]-sp[0]))
        yaw += float(self.rng.normal(0, 0.25))

        self.prev_a = np.zeros(L.act_dim, np.float32)
        self.subgoal = self.goal.copy()
        self._route_rem = None
        self.airborne_start = start_frac is not None
        if start_frac is None:
            rx, ry, rz = (sp[0] + self.rng.normal(0, 4),
                          sp[1] + self.rng.normal(0, 4), sp[2] + 0.9)
        else:
            f = float(np.clip(start_frac, 0.0, 0.98))
            rx = sp[0] + f * (self.goal[0] - sp[0]) + self.rng.normal(0, 25)
            ry = sp[1] + f * (self.goal[1] - sp[1]) + self.rng.normal(0, 25)
            # Spawn on the glide slope, not at cruise altitude. A curriculum
            # episode that begins 60 m from the pad but 120 m up still contains
            # the whole descent problem, so the agent would never actually
            # practise the easy end of the task the curriculum is meant to
            # expose it to.
            d_rem = float(np.linalg.norm(self.goal[:2] -
                                         np.array([sp[0] + f*(self.goal[0]-sp[0]),
                                                   sp[1] + f*(self.goal[1]-sp[1])])))
            rz = float(np.clip(self.goal[2] + 4.0 + 0.34 * (d_rem - 60.0),
                               self.goal[2] + 12.0, WD.corridor_alt))
            rz += self.rng.normal(0, 6)
            for _ in range(30):        # keep the spawn clear of a building
                if self.geom.true_sdf(np.array([rx, ry, rz]))[0] > 45.0:
                    break
                rx += self.rng.normal(0, 45); ry += self.rng.normal(0, 45)
            yaw = float(np.arctan2(self.goal[1]-ry, self.goal[0]-rx)) \
                + float(self.rng.normal(0, 0.2))
        self.br.reset_to(rx, ry, rz, yaw, 1.0)
        # let the teleport settle: hold trim for ~0.45 s of simulated time
        t0 = self.br.sim_time
        while self.br.sim_time - t0 < 0.45:
            self.br.send(A.W/(A.n_rotor*A.k_rotor*A.w_rotor_max**2), 0, 0, 0, 0, 0)
            time.sleep(0.003)

        self.t0 = self.br.sim_time
        self.n_ep += 1
        self.outcome = OK
        self.fuser.reset()
        self.dt_err = []
        self.last_dt = self.dt
        self.min_clr = 1e9
        self.min_obs = 1e9
        self.min_obs_mapped = 1e9
        self.min_obs_unmapped = 1e9
        self.nearest_unmapped = False
        self.hit_unmapped = False
        self.min_agl = 1e9
        self.enroute = False
        self.min_sep = 1e9
        self.nmac = 0
        self.lox = 0            # loss-of-well-clear events
        self.path_len = 0.0
        self.jerk_acc = 0.0
        self._prev_pos = None
        obs = self._observe()
        self.d_goal = float(np.linalg.norm(self.goal[:2] - self._pos[:2]))
        self.d0 = self.d_goal
        self.phys_state = np.zeros(12, np.float32)
        return obs

    # ----------------------------------------------------------------- step --
    # exactly this many physics iterations per control step
    @property
    def n_sub(self):
        return max(1, int(round(L.phys_hz / L.ctrl_hz)))

    def _wait(self):
        """Block until one control period of SIMULATED time has elapsed.

        Also records how much simulated time actually passed. If the loop is
        too slow for the simulator's real-time factor this overshoots, the
        control period silently stretches, and the aircraft flies on stale
        commands -- so it is measured rather than assumed.
        """
        t_start = self.br.sim_time
        target = t_start + self.dt
        t_wall = time.time()
        while self.br.sim_time < target:
            if time.time() - t_wall > 3.0:          # simulator stalled
                break
            time.sleep(0.0005)
        self.last_dt = self.br.sim_time - t_start
        self.dt_err.append(self.last_dt)

    def set_nav(self, subgoal, route_remaining=None):
        """Called by the runner before each step. Progress reward is measured
        along the planned route, not as straight-line distance to the pad: a
        route that must detour around a building block would otherwise be
        punished for the detour it was required to make."""
        self.subgoal = np.asarray(subgoal, np.float64)
        self._route_rem = route_remaining

    def step(self, a):
        a = np.clip(np.asarray(a, np.float32), -1, 1)
        # denormalise to physical references
        col   = float((a[0] + 1) * 0.5)
        roll  = float(a[1] * np.radians(SF.bank_max_deg))
        pitch = float(a[2] * np.radians(18.0))
        yawr  = float(a[3] * 0.45)
        push  = float((a[4] + 1) * 0.5)
        sched = float((a[5] + 1) * 0.5)
        self.br.send(col, roll, pitch, yawr, push, sched)
        self._wait()

        a_prev = self.prev_a.copy()
        obs = self._observe()
        st, pos = self._st, self._pos
        self.prev_a = a

        # ---- geometry / safety bookkeeping ---------------------------------
        clr_b, i_obs = self.geom.true_sdf(pos)   # truth: includes unmapped cranes
        # Which obstacle is nearest matters for the comparison: an obstacle in
        # the database can be avoided geometrically by anything, whereas an
        # unmapped crane can only be avoided by something that sees it.
        unmapped = i_obs >= len(self.geom.bc)
        if clr_b < self.min_obs:
            self.nearest_unmapped = bool(unmapped)
        if unmapped:
            self.min_obs_unmapped = min(self.min_obs_unmapped, clr_b)
        else:
            self.min_obs_mapped = min(self.min_obs_mapped, clr_b)
        clr = min(clr_b, float(pos[2]))
        # Obstacle clearance and ground clearance are different hazards and
        # deserve separate numbers. Sitting on a pad is 1.7 m from the ground
        # by construction; reporting that as the flight's minimum clearance
        # would make every run look equally dangerous.
        self.min_clr = min(self.min_clr, clr)
        self.min_obs = min(self.min_obs, clr_b)
        d_pad = float(np.min(np.linalg.norm(self.geom.vp[:, :2] - pos[:2], axis=1)))
        self.enroute = d_pad > 60.0
        if self.enroute:
            self.min_agl = min(self.min_agl, float(pos[2]))

        traffic = obs["traffic"]
        sep = 1e9; sep_h = 1e9; sep_v = 1e9; hit_air = False
        if traffic.size:
            rel = traffic[:, :3] - pos
            dist = np.linalg.norm(rel, axis=1)
            kind = traffic[:, 6].astype(int)
            j = int(np.argmin(dist)); sep = float(dist[j])
            self.min_sep = min(self.min_sep, sep)
            hit_r = np.where(kind == 0, 8.0, 3.0)
            hit_air = bool(np.any(dist < hit_r))
            coop = kind == 0
            if coop.any():
                h = np.linalg.norm(rel[coop][:, :2], axis=1)
                v = np.abs(rel[coop][:, 2])
                if np.any((h < SF.r_wellclear) & (v < SF.h_wellclear)):
                    self.lox += 1
                if np.any(np.linalg.norm(rel[coop], axis=1) < 30.0):
                    self.nmac += 1
            if np.any(dist[kind == 1] < SF.r_suas):
                self.lox += 1

        if self._prev_pos is not None:
            self.path_len += float(np.linalg.norm(pos - self._prev_pos))
        self._prev_pos = pos.copy()

        # ---- termination ----------------------------------------------------
        t = st[40] - self.t0
        roll_e, pitch_e, _ = self.br.euler()
        d_goal = float(np.linalg.norm(self.goal[:2] - pos[:2]))
        dz = float(pos[2] - self.goal[2])
        spd = float(np.linalg.norm(st[7:10]))

        done = False; term = OK
        if hit_air:                                   term, done = CRASH_AIR, True
        elif pos[2] <= 0.6 and d_goal > 30.0:         term, done = CRASH_GND, True
        elif clr_b <= 0.0:
            term, done = CRASH_BLD, True
            self.hit_unmapped = bool(unmapped)
        elif st[23] <= SF.soc_reserve:                term, done = BATTERY, True
        elif abs(roll_e) > 1.4 or abs(pitch_e) > 1.05 or pos[2] > 420:
                                                      term, done = UPSET, True
        elif (d_goal < 20.0 and abs(dz) < 2.0 and spd < 2.0):
                                                      term, done = GOAL, True
        elif t > self.max_time:                       term, done = TIMEOUT, True
        elif np.abs(pos[:2]).max() > self.geom.extent * 1.4:
                                                      term, done = OOB, True
        self.outcome = term

        # ---- reward ---------------------------------------------------------
        ref = self._route_rem if self._route_rem is not None else d_goal
        prog = self.d_goal - ref
        self.d_goal = ref
        r  = 0.020 * prog                                    # progress, m
        r -= 0.015                                           # time
        r -= 0.9 * max(0.0, SF.r_static - clr) / SF.r_static  # clearance shaping
        if sep < 1e8:
            r -= 0.7 * max(0.0, SF.r_suas * 2.5 - sep) / (SF.r_suas * 2.5)
        r -= 0.35 * float(np.mean(np.abs(a - a_prev)))  # smoothness
        r -= 0.10 * float(min(abs(roll_e) / 1.4, 1.0) ** 2)
        r -= 0.02 * st[24] / 2.0e5                            # energy
        if term == GOAL:                                      r += 60.0
        elif term in (CRASH_AIR, CRASH_BLD, CRASH_GND, UPSET): r -= 45.0
        elif term == BATTERY:                                  r -= 12.0

        info = dict(outcome=term, t=t, clr=clr, sep=sep, d_goal=d_goal,
                    soc=st[23], spd=spd, pos=pos.copy(),
                    min_clr=self.min_clr, min_obs=self.min_obs,
                    min_obs_mapped=self.min_obs_mapped,
                    min_obs_unmapped=self.min_obs_unmapped,
                    hit_unmapped=self.hit_unmapped,
                    min_agl=self.min_agl, min_sep=self.min_sep,
                    nmac=self.nmac, lox=self.lox, path_len=self.path_len)
        return obs, float(r), done, info

    def timing(self):
        """Control-period fidelity: how close the realised step was to 1/ctrl_hz."""
        if not self.dt_err:
            return dict(mean=self.dt, max=self.dt, over=0.0)
        a = np.asarray(self.dt_err)
        return dict(mean=float(a.mean()), max=float(a.max()),
                    over=float(np.mean(a > self.dt * 1.5)))

    def state_vel_w(self):
        return self._st[7:10].copy()

    def close(self):
        self.br.close()
