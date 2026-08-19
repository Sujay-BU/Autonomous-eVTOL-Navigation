"""
Runtime control loop.

One place where perception, the world model, the planner, the barrier filter
and the aircraft meet. The same object is used for data collection during
training and for scored evaluation runs, so what gets measured is exactly what
gets flown -- there is no separate "demo" path that could quietly differ.

Per control step (20 Hz):
    observe -> encode -> RSSM posterior update
             -> [10 Hz] MPPI replan over 3.2 s of imagined future
             -> barrier filter repairs the action
             -> actuate
"""
import time
import numpy as np
import torch

from .config import CFG
from .env import OUTCOME, GOAL
from .route import RoutePlanner, RouteTracker
from .planner import MPPIPlanner
from .replay import SequenceReplay

L, SF = CFG.lrn, CFG.saf


class FlightRunner:
    def __init__(self, env, wm, actor=None, critic=None, device="cuda",
                 route_planner=None, mode="plan", actor_seed_frac=0.25,
                 use_nominal=True):
        self.env, self.wm = env, wm
        self.actor, self.critic = actor, critic
        self.dev = device
        self.mode = mode                       # "plan" | "actor" | "scripted"
        self.rp = route_planner or RoutePlanner(env.geom)
        self.planner = MPPIPlanner(wm, env.geom, critic=critic, actor=actor,
                                   device=device,
                                   actor_seed_frac=actor_seed_frac)
        self.planner.H = L.horizon
        self.planner.N = L.n_samples
        self.plan_every = max(1, int(round(L.ctrl_hz / L.plan_hz)))
        self.use_nominal = bool(use_nominal)

    # ------------------------------------------------------------- helpers --
    @torch.no_grad()
    def _encode(self, obs, h, z, a_prev):
        t = lambda x: torch.as_tensor(np.asarray(x, np.float32),
                                      device=self.dev).unsqueeze(0)
        e = self.wm.enc(t(obs["img"]), t(obs["pro"]), t(obs["trk"]))
        h, z, _, _ = self.wm.rssm.obs_step(h, z, t(a_prev), e)
        return h, z

    # ------------------------------------------------------------- episode --
    def run(self, start_vp=None, goal_vp=None, replay=None, callback=None,
            explore=0.0, max_steps=None, start_frac=None):
        env = self.env
        obs = env.reset(start_vp, goal_vp, start_frac)
        wpts = self.rp.plan(env._pos, env.goal, CFG.wld.corridor_alt,
                            land_z=env.goal[2] + 2.5)
        tracker = RouteTracker(wpts)
        self.planner.reset()

        h, z = self.wm.rssm.initial(1, self.dev)
        a = np.zeros(L.act_dim, np.float32)
        prev_x = env.phys_state.copy()
        max_steps = max_steps or int(env.max_time * L.ctrl_hz)

        log = dict(pos=[], clr=[], sep=[], soc=[], spd=[], shield=[], cost=[],
                   act=[], t=[], rew=[])
        stats = dict(shield_engaged=0, steps=0, plan_ms=[], route=wpts)
        first = 1

        for k in range(max_steps):
            sg = tracker.update(env._pos)
            rem = tracker.remaining(env._pos) + np.linalg.norm(
                env.goal[:2] - tracker.w[-1][:2])
            env.set_nav(sg, rem)

            h, z = self._encode(obs, h, z, a)
            # disturbance observer: what did the last action actually do?
            if k > 0:
                self.planner.update_bias(prev_x, a, env.br.state[16:19],
                                         1.0 / L.ctrl_hz)
            prev_x = env.phys_state.copy()
            x0 = torch.as_tensor(env.phys_state, device=self.dev).unsqueeze(0)
            tracks = env.fuser.last_world

            # ---- decide ----------------------------------------------------
            t_plan = time.time()
            if self.mode == "plan":
                if k % self.plan_every == 0:
                    self.planner.set_obstacles(env._pos)
                    # nominal = the geometric guidance law, which flies the
                    # mission reliably; MPPI's job is to improve on it using
                    # the world model's view of what is coming
                    nom = self._scripted(env, sg) if self.use_nominal else None
                    self.planner.plan(h, z, x0, sg, tracks, env.goal,
                                      nominal=nom)
                sub = min(k % self.plan_every, self.planner.mean.shape[0] - 1)
                a_raw = self.planner.mean[sub].clamp(-1, 1).cpu().numpy()
            elif self.mode == "actor" and self.actor is not None:
                with torch.no_grad():
                    a_raw = self.actor(self.wm.feat(h, z),
                                       sample=explore > 0).cpu().numpy()[0]
            elif self.mode == "baseline":
                a_raw = self._baseline(env, sg, obs)
            else:
                a_raw = self._scripted(env, sg)
            stats["plan_ms"].append(1e3 * (time.time() - t_plan))

            if explore > 0:
                a_raw = np.clip(a_raw + np.random.randn(L.act_dim) * explore, -1, 1)

            # ---- shield ----------------------------------------------------
            a, engaged, u_safe = self.planner.shield(
                a_raw.astype(np.float32), env.phys_state, env._pos,
                env.state_vel_w(), tracks)
            stats["shield_engaged"] += engaged

            # ---- act -------------------------------------------------------
            nobs, r, done, info = env.step(a)

            if replay is not None:
                replay.add(SequenceReplay.encode_img(obs["img"]), obs["pro"],
                           obs["trk"], a, r, 0 if done else 1,
                           min(info["clr"], 200.0), env.phys_state, first)
            first = 0

            log["pos"].append(info["pos"].copy()); log["clr"].append(info["clr"])
            log["sep"].append(min(info["sep"], 9999)); log["soc"].append(info["soc"])
            log["spd"].append(info["spd"]); log["shield"].append(engaged)
            log["act"].append(a.copy()); log["t"].append(info["t"])
            log["rew"].append(r)
            stats["steps"] = k + 1

            if callback is not None:
                callback(dict(obs=obs, info=info, action=a, action_raw=a_raw,
                              h=h, z=z, engaged=engaged, subgoal=sg,
                              route=wpts, planner=self.planner, step=k,
                              tracks=tracks, u_safe=u_safe, reward=r))
            obs = nobs
            if done:
                break

        stats.update(outcome=OUTCOME[info["outcome"]],
                     success=int(info["outcome"] == GOAL),
                     min_clr=info["min_clr"], min_obs=info["min_obs"],
                     min_obs_mapped=info["min_obs_mapped"],
                     min_obs_unmapped=info["min_obs_unmapped"],
                     hit_unmapped=int(info["hit_unmapped"]),
                     min_agl=info["min_agl"], min_sep=info["min_sep"],
                     lox=info["lox"], nmac=info["nmac"],
                     path_len=info["path_len"], t=info["t"],
                     soc_used=1.0 - info["soc"],
                     ret=float(np.sum(log["rew"])),
                     shield_rate=stats["shield_engaged"] / max(stats["steps"], 1),
                     start_vp=env.start_vp, goal_vp=env.goal_vp,
                     direct=float(env.d0))
        return stats, log

    # --------------------------------------------------- scripted bootstrap --
    def _scripted(self, env, sg):
        """Phased guidance autopilot used to seed the replay buffer.

        Commands ACCELERATION, not heading. The obvious approach -- bank in
        proportion to heading error -- does not work on this airframe: the yaw
        loop asks for I_zz * 2.6 * r_err which reaches ~5400 Nm, while eight
        lift rotors at +/-30% differential produce about 547 Nm. The yaw
        channel saturates by an order of magnitude, so the nose simply does not
        follow, and any law that assumes it does flies the aircraft sideways
        into the countryside.

        A hover-borne eVTOL does not need to point where it is going. So we
        compute the acceleration we want in the world frame, resolve it into
        the body frame, and use the standard tilt relations

            a_right = g tan(phi)        a_forward = -g tan(theta)

        which is exactly the mapping the barrier filter uses when it repairs an
        action. Yaw is then commanded gently, purely to keep the camera looking
        where the aircraft is travelling.
        """
        A = CFG.air
        G = 9.80665
        col0 = 2*A.W/(A.n_rotor*A.k_rotor*A.w_rotor_max**2) - 1.0
        p = env._pos
        st = env.br.state
        roll, pitch, yaw = env.br.euler()
        vel = st[7:10]
        vz, V = st[9], st[19]
        d_goal = float(np.linalg.norm(env.goal[:2] - p[:2]))
        z_pad = env.goal[2]

        # ---- phase schedule ----------------------------------------------
        if d_goal > 340.0:
            z_ref, v_ref, aim = CFG.wld.corridor_alt, 34.0, np.asarray(sg)
        elif d_goal > 70.0:
            z_ref = max(z_pad + 4.0 + 0.34 * (d_goal - 60.0), z_pad + 10.0)
            v_ref, aim = float(np.clip(0.075 * d_goal, 6.0, 26.0)), env.goal
        elif d_goal > 16.0:
            z_ref, v_ref, aim = z_pad + 9.0, float(np.clip(0.28*d_goal, 2.0, 6.0)), env.goal
        else:
            z_ref, v_ref, aim = z_pad + 0.2, 0.6, env.goal

        # ---- horizontal guidance: velocity -> acceleration ----------------
        to = np.asarray(aim, np.float64)[:2] - p[:2]
        rng = max(float(np.linalg.norm(to)), 1e-3)
        v_des = to / rng * v_ref
        acc = 0.62 * (v_des - vel[:2])
        n = float(np.linalg.norm(acc))
        if n > 5.2:                       # stay inside the bank limit
            acc = acc / n * 5.2

        c, sn = np.cos(yaw), np.sin(yaw)
        a_fwd = float(acc[0]*c + acc[1]*sn)
        a_rgt = float(acc[0]*sn - acc[1]*c)

        a = np.zeros(L.act_dim, np.float32)
        a[1] = np.clip(np.arctan2(a_rgt, G) / np.radians(CFG.saf.bank_max_deg), -1, 1)
        a[2] = np.clip(np.arctan2(-a_fwd, G) / np.radians(18.0), -1, 1)

        # ---- vertical: altitude -> vertical speed -> collective -----------
        # Commanding thrust straight from an altitude error saturates the
        # collective to zero the instant a large descent is asked for, and the
        # aircraft falls. Rate-limiting through vz_ref prevents that.
        vz_ref = float(np.clip(0.32 * (z_ref - p[2]), -4.5, 6.5))
        a[0] = np.clip(col0 + 0.115 * (vz_ref - vz)
                       + 0.16 * (1.0 - np.cos(np.clip(roll, -1.0, 1.0))), -1, 1)

        # ---- pusher: forward thrust when aligned and wanting speed --------
        brg = np.arctan2(to[1], to[0])
        dyaw = (brg - yaw + np.pi) % (2*np.pi) - np.pi
        aligned = float(np.cos(np.clip(dyaw, -np.pi, np.pi)))
        thr = np.clip(0.045 * (v_ref - V) * max(aligned, 0.0), 0.0, 1.0)
        a[4] = np.clip(2.0*thr - 1.0, -1, 1)

        # ---- yaw: slow, cosmetic, keeps the camera on the flight path -----
        a[3] = np.clip(0.35 * dyaw, -0.6, 0.6)
        a[5] = -1.0                                  # hover-borne throughout
        a += np.random.randn(L.act_dim) * 0.035
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    # ------------------------------------------------- classical baseline --
    def _baseline(self, env, sg, obs):
        """Conventional autopilot: geometric guidance + artificial potential
        field, with reactive avoidance driven directly by the depth image.

        This is the comparator the learned stack has to beat, and it is built
        to be a fair one rather than a straw man. It gets the same obstacle
        database, the same tracked traffic, the same barrier filter and a
        genuine camera-based reactive term -- everything except a world model.
        What it cannot do is anticipate: a potential field reacts to the
        gradient it is standing in, so it has no way to start a manoeuvre three
        seconds before the conflict develops, and no way to reason about where
        an intruder will be rather than where it is.
        """
        G = 9.80665
        p = env._pos
        st = env.br.state
        roll, pitch, yaw = env.br.euler()
        vel = st[7:10]

        a = self._scripted(env, sg)          # nominal guidance and vertical loop

        # --- attractive term is already in `a`; build the repulsive one -----
        rep = np.zeros(2)

        # (i) mapped obstacles, inverse-square potential
        d_b, idx = env.geom.building_sdf(p)
        if d_b < 110.0:
            n = env.geom.building_grad(p)[:2]
            rep += n * 210.0 / max(d_b, 8.0) ** 1.35

        # (ii) reactive term straight off the depth image -- this is what lets
        #      the baseline see obstacles that are not in the database.
        #      Gated to en-route flight: the navigation camera is pitched 12.6
        #      deg down, so on approach it is looking mostly at the ground, and
        #      an ungated repulsion treats the landing site itself as the thing
        #      to escape from. The learned controller needs no such gate
        #      because its clearance head is trained on what actually hurt it.
        d_pad_own = float(np.min(np.linalg.norm(
            env.geom.vp[:, :2] - p[:2], axis=1)))
        dep = obs.get("depth_full")
        if dep is not None and dep.size > 4 and p[2] > 45.0 and d_pad_own > 90.0:
            d = dep.copy()
            d[~np.isfinite(d)] = CFG.sen.nav_far
            h, w = d.shape
            band = d[int(h*0.18):int(h*0.58)]     # above the ground return
            cols = band.min(axis=0)
            near = cols.min()
            if near < 95.0:
                u = float(np.argmin(cols))
                # bearing of the closest return, relative to boresight
                ang = (u - w/2.0) / (w/2.0) * np.radians(CFG.sen.nav_hfov_deg/2.0)
                brg = yaw - ang
                # push perpendicular, toward whichever side is more open
                left = float(cols[:w//2].min()); right = float(cols[w//2:].min())
                sgn = 1.0 if left > right else -1.0
                perp = np.array([-np.sin(brg), np.cos(brg)]) * sgn
                away = -np.array([np.cos(brg), np.sin(brg)])
                mag = 220.0 / max(near, 8.0) ** 1.2
                rep += (0.75 * perp + 0.45 * away) * mag

        # (iii) tracked traffic, present position only (no extrapolation)
        for (p_i, v_i, coop) in env.fuser.last_world:
            rel = p[:2] - np.asarray(p_i)[:2]
            dist = float(np.linalg.norm(rel))
            r_min = CFG.saf.r_wellclear if coop else CFG.saf.r_suas
            if dist < r_min * 2.2 and dist > 1e-3:
                rep += rel / dist * (150.0 * (r_min * 2.2 - dist) / (r_min * 2.2))

        n = float(np.linalg.norm(rep))
        if n < 1e-6:
            return a
        if n > 5.0:
            rep = rep / n * 5.0

        c, sn = np.cos(yaw), np.sin(yaw)
        a_fwd = float(rep[0]*c + rep[1]*sn)
        a_rgt = float(rep[0]*sn - rep[1]*c)
        a[1] = np.clip(a[1] + np.arctan2(a_rgt, G)
                       / np.radians(CFG.saf.bank_max_deg), -1, 1)
        a[2] = np.clip(a[2] + np.arctan2(-a_fwd, G) / np.radians(18.0), -1, 1)
        return a.astype(np.float32)
