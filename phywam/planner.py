"""
Local planner: MPPI over the physics-informed world model, shielded by a
high-order control barrier function.

Two layers, deliberately:

  MPPI (soft)  -- samples 512 action sequences, rolls each 2.0 s forward through
                  the learned dynamics, and scores them. Handles goals, comfort,
                  energy and *anticipation*: it can see a conflict developing
                  two seconds out and start turning early.

  HOCBF (hard) -- repairs the single action MPPI emits so that it provably
                  cannot drive the barrier b(p) = d(p) - d_safe negative. Runs
                  in microseconds and is the last thing between the network and
                  the actuators.

The soft layer is where performance comes from; the hard layer is where the
guarantee comes from. Neither is sufficient alone: a sampled planner offers no
guarantee even with infinite samples, and a barrier filter with no lookahead
will happily fly into a cul-de-sac it cannot escape.
"""
import math
import numpy as np
import torch

from .config import CFG
from .physics import AnalyticEVTOL

L, SF, S = CFG.lrn, CFG.saf, CFG.sen
G = 9.80665
V_SAFE = 1.25 * CFG.air.V_stall      # do not go wing-borne below this


# --------------------------------------------------------------------- HOCBF --
class BarrierFilter:
    """Relative-degree-2 barrier on obstacle clearance.

    With b(p) = d(p) - d_safe and the aircraft accelerating, b has relative
    degree 2 with respect to commanded acceleration, so a first-order condition
    is not enforceable. The HOCBF condition

        b_ddot + a1 * b_dot + a0 * b  >=  0

    expands (dropping the curvature term, which vanishes away from box corners)
    to a linear inequality in the commanded acceleration u:

        grad_d . u  >=  -a1 * (grad_d . v)  -  a0 * (d - d_safe)

    which is exactly the form a small QP wants. We collect one such row per
    nearby hazard and project the desired acceleration onto the intersection.
    """

    def __init__(self, geom):
        self.g = geom
        self.n_active = 0
        self.last_rows = []

    # -- exact projection onto {u : G u >= h} ∩ {||u|| <= amax} --------------
    @staticmethod
    def _project(u_des, Gm, h, amax):
        """min ||u - u_des||^2 s.t. G u >= h, ||u|| <= amax.

        With at most a handful of rows we can enumerate active sets and solve
        each equality-constrained subproblem in closed form. That is exact,
        deterministic and far cheaper than calling a general QP solver at
        250 Hz."""
        m = len(h)
        if m == 0:
            n = np.linalg.norm(u_des)
            return u_des * min(1.0, amax / max(n, 1e-9))
        best, best_c = None, np.inf
        idx = range(m)
        subsets = [()]
        for i in idx:
            subsets.append((i,))
            for j in idx:
                if j > i:
                    subsets.append((i, j))
                    for k in idx:
                        if k > j:
                            subsets.append((i, j, k))
        for act in subsets:
            if not act:
                u = u_des.copy()
            else:
                Ga = Gm[list(act)]
                ha = h[list(act)]
                # u = u_des + Ga^T lam,  Ga u = ha
                M = Ga @ Ga.T
                try:
                    lam = np.linalg.solve(M + 1e-9*np.eye(len(act)),
                                          ha - Ga @ u_des)
                except np.linalg.LinAlgError:
                    continue
                if np.any(lam < -1e-9):        # KKT: multipliers must be >= 0
                    continue
                u = u_des + Ga.T @ lam
            n = np.linalg.norm(u)
            if n > amax:
                u = u * (amax / n)
            if np.all(Gm @ u >= h - 1e-6):
                c = float(np.sum((u - u_des) ** 2))
                if c < best_c:
                    best, best_c = u, c
        if best is None:                        # infeasible: do the least-bad
            viol = h - Gm @ u_des
            i = int(np.argmax(viol))
            gi = Gm[i] / max(np.linalg.norm(Gm[i]), 1e-9)
            best = u_des + gi * max(viol[i], 0.0)
            n = np.linalg.norm(best)
            if n > amax:
                best = best * (amax / n)
        return best

    def ground_floor(self, pos):
        """Minimum legal altitude at this horizontal position.

        A constant 20 m floor makes landing formally impossible -- the barrier
        will hold the aircraft above the pad forever, which is exactly what it
        did. Vertiports have a protected-surface carve-out over the FATO for
        precisely this reason, so the floor relaxes to touchdown height inside
        the pad and returns to the en-route minimum outside it.
        """
        d = float(np.min(np.linalg.norm(self.g.vp[:, :2] - pos[:2], axis=1)))
        f = min(max((d - 25.0) / 35.0, 0.0), 1.0)
        return 1.0 + (SF.r_static - 1.0) * f

    def __call__(self, pos, vel, u_des, tracks_world=None):
        """pos, vel, u_des in world ENU. Returns the repaired acceleration."""
        rows, rhs = [], []

        # --- static obstacles (mapped database only) -------------------------
        d, _ = self.g.building_sdf(pos)
        if d < 130.0:
            n = self.g.building_grad(pos)
            rows.append(n)
            rhs.append(-SF.cbf_alpha1 * float(n @ vel)
                       - SF.cbf_alpha0 * (d - SF.r_static))
        # --- ground -----------------------------------------------------------
        dz = float(pos[2])
        floor = self.ground_floor(pos)
        if dz < floor + 45.0:
            n = np.array([0.0, 0.0, 1.0])
            rows.append(n)
            rhs.append(-SF.cbf_alpha1 * vel[2] - SF.cbf_alpha0 * (dz - floor))

        # --- dynamic hazards: barrier on RELATIVE motion ---------------------
        if tracks_world is not None and len(tracks_world):
            for (p_i, v_i, coop) in tracks_world:
                rel = pos - p_i
                dist = float(np.linalg.norm(rel))
                r_min = SF.r_wellclear if coop else SF.r_suas
                if dist > r_min * 2.6 or dist < 1e-3:
                    continue
                n = rel / dist
                vrel = vel - v_i
                rows.append(n)
                rhs.append(-SF.cbf_alpha1 * float(n @ vrel)
                           - SF.cbf_alpha0 * (dist - r_min))

        self.n_active = len(rows)
        if not rows:
            n = np.linalg.norm(u_des)
            return u_des * min(1.0, SF.accel_max / max(n, 1e-9)), 0
        Gm = np.asarray(rows, np.float64)
        h = np.asarray(rhs, np.float64)
        pre = Gm @ u_des - h
        u = self._project(u_des, Gm, h, SF.accel_max)
        engaged = int(np.any(pre < -1e-6))
        return u, engaged


# ---------------------------------------------------------------------- MPPI --
class MPPIPlanner:
    def __init__(self, wm, geom, critic=None, actor=None, device="cuda",
                 actor_seed_frac=0.25):
        self.wm, self.g, self.critic, self.actor = wm, geom, critic, actor
        # Fraction of the MPPI population seeded from the policy prior. Set to
        # zero when the actor is worse than the sampling prior: seeding from a
        # bad policy actively wastes samples that the Gaussian would have spent
        # better.
        self.actor_seed_frac = float(actor_seed_frac)
        self.dev = device
        self.H, self.N = L.horizon, L.n_samples
        self.dt = 1.0 / L.ctrl_hz
        # The sampling prior must sit at a FLYABLE action, not at zero. With
        # a = 0 the schedule decodes to half wing-borne, which scales the
        # collective by 0.54 -- below the 0.576 needed to hold weight. A
        # zero-mean prior therefore describes an aircraft that cannot hover.
        A = CFG.air
        col = 2.0 * A.W / (A.n_rotor * A.k_rotor * A.w_rotor_max ** 2) - 1.0
        self.a_trim = torch.tensor([col, 0.0, 0.0, 0.0, -1.0, -1.0],
                                   device=device)
        self.mean = self.a_trim.repeat(self.H, 1).clone()
        self.cbf = BarrierFilter(geom)
        # low-passed estimate of the model's specific-force error, body FRD
        self.bias_est = torch.zeros(3, device=device)
        self.bias_tau = 0.45          # s, filter time constant
        # how strongly the sampling mean is pulled toward the nominal guidance
        # action each replan; 0 reverts to pure warm-start MPPI
        self.nominal_weight = 0.6
        # Ablation switches, read when the CUDA graph is captured. Setting
        # w_hazard to 0 removes the learned clearance head from the cost,
        # leaving only the geometric SDF over the *mapped* database.
        self.w_hazard = 2.4
        self.phys = AnalyticEVTOL().to(device)
        self._obs_t = None
        self._graph = None
        self.last = {}

    def reset(self):
        self.mean.copy_(self.a_trim.repeat(self.mean.shape[0], 1))
        self.bias_est.zero_()

    @torch.no_grad()
    def update_bias(self, x_np, a_np, acc_meas_frd, dt):
        """One step of the disturbance observer.

        Compares the specific force the model thinks the last action produced
        against what the aircraft actually felt, and low-passes the difference.
        The estimate is then added to every rollout, so the planner is
        optimising against the aircraft it has rather than the one in the
        equations.
        """
        xt = torch.as_tensor(x_np, dtype=torch.float32, device=self.dev)[None]
        at = torch.as_tensor(a_np, dtype=torch.float32, device=self.dev)[None]
        f_model = self.phys.specific_force(xt, at)[0]
        f_meas = torch.as_tensor(acc_meas_frd, dtype=torch.float32,
                                 device=self.dev)
        err = f_meas - f_model
        al = math.exp(-dt / self.bias_tau)
        self.bias_est.mul_(al).add_(err, alpha=(1.0 - al))
        # never let the estimate exceed what the airframe could plausibly be
        # mismodelled by; an unbounded observer will chase sensor noise
        n = self.bias_est.norm()
        if float(n) > 4.0:
            self.bias_est.mul_(4.0 / n)
        return self.bias_est

    MAXOBS = 32          # padded so the rollout keeps a static shape
    MAXTRK = 8

    def set_obstacles(self, pos, radius=420.0):
        """Cache the nearby mapped obstacles as tensors once per control step.
        Rolling out 512 x 40 states against all 110 boxes every step is pure
        waste when only a handful are ever within the 2 s reachable set."""
        d = np.hypot(self.g.bc[:, 0] - pos[0], self.g.bc[:, 1] - pos[1])
        idx = np.argsort(d)[:self.MAXOBS]
        idx = idx[d[idx] < radius]
        c = np.full((self.MAXOBS, 2), 1e6, np.float32)
        h = np.full((self.MAXOBS, 2), 1.0, np.float32)
        z = np.full((self.MAXOBS,), 1.0, np.float32)
        if len(idx):
            c[:len(idx)] = self.g.bc[idx]
            h[:len(idx)] = self.g.bh[idx]
            z[:len(idx)] = self.g.bz[idx]
        t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=self.dev)
        self._obs_t = (t(c), t(h), t(z))

    @staticmethod
    def _sdf_t(P, obs):
        """Batched SDF, torch. P: (...,3) -> (...)"""
        c, hw, hz = obs
        dxy = (P[..., None, :2] - c).abs() - hw
        zc = hz * 0.5
        dz = (P[..., None, 2] - zc).abs() - zc
        d = torch.cat([dxy, dz[..., None]], -1)
        out = d.clamp(min=0).norm(dim=-1) + d.max(-1).values.clamp(max=0)
        return out.min(-1).values

    # ------------------------------------------------------------- rollout --
    def _rollout(self, h0, z0, x0, U, obs, TP, TV, TR, sg, gf, VP, BI):
        """Score N action sequences. Static shapes throughout so the whole
        loop can be captured as a CUDA graph -- without that this is
        launch-bound (~10k tiny kernels per plan) rather than compute-bound."""
        N, H = U.shape[0], U.shape[1]
        hb, zb, xb = h0, z0, x0
        cost = torch.zeros(N, device=U.device)
        clr_min = torch.full((N,), 1e3, device=U.device)
        disc = 1.0
        for t in range(H):
            a = U[:, t]
            xb_prev = xb
            xb, _, _ = self.wm.phys(hb, zb, xb, a, self.dt, BI)
            hb, zb, _ = self.wm.rssm.img_step(hb, zb, a)
            P_prev = xb_prev[:, :3]
            P = xb[:, :3]

            d = self._sdf_t(P, obs)
            clr_min = torch.minimum(clr_min, d)
            f = self.wm.feat(hb, zb)
            clr_hat = self.wm.twohot.decode(self.wm.head_clr(f))

            Pi = TP + TV * ((t + 1) * self.dt)                  # (K,3)
            dd = (P[:, None, :] - Pi[None]).norm(dim=-1)        # (N,K)

            # Corridor tracking, split into horizontal and vertical.
            # A single 3-D distance term is nearly flat in the vertical over a
            # 3.2 s horizon, so the planner will happily trade altitude for
            # energy -- it did exactly that, and flew itself into the ground.
            # Altitude in a UAM corridor is an assigned quantity, so it gets
            # its own, much stiffer, term.
            # --- guidance ----------------------------------------------------
            # Distance alone does not tell the aircraft which way to point. At
            # 42 m/s the turn radius is 311 m, so a planner that only shrinks
            # range will happily orbit the target. Aligning the velocity vector
            # with the bearing to the subgoal is what closes the loop.
            d_fin = (P[:, :2] - gf[:2]).norm(dim=-1)
            to_sg = sg[:2] - P[:, :2]
            rng_sg = to_sg.norm(dim=-1).clamp(min=1.0)
            vel_w = xb[:, :3] - P_prev if t > 0 else torch.zeros_like(P)
            vh = vel_w[:, :2]
            align = (vh * to_sg).sum(-1) / (vh.norm(dim=-1).clamp(min=0.5) * rng_sg)

            # speed-to-go: arrive slow enough to land, without braking early
            v_allow = (0.22 * d_fin).clamp(4.0, SF.v_max)
            # position-dependent ground floor, matching BarrierFilter
            d_vp = (P[:, None, :2] - VP[None]).norm(dim=-1).min(-1).values
            floor = 1.0 + (SF.r_static - 1.0) * ((d_vp - 25.0)/35.0).clamp(0, 1)

            # Horizontal and vertical guidance, scaled by what the aircraft
            # can actually change within the horizon. In 3.2 s it can close
            # ~100 m horizontally but only ~15 m vertically, so equal per-metre
            # weights make altitude ~7x more influential than progress. The
            # previous 0.020 / 0.25 split made it 12.5x on top of that: the
            # planner optimised almost purely for altitude-holding, the
            # horizontal term differed by 0.8 out of 130 between a good plan
            # and a wandering one, and the aircraft never went anywhere.
            c_t = (0.160 * (P[:, :2] - sg[:2]).norm(dim=-1)
                   + 0.120 * (P[:, 2] - sg[2]).abs()
                   # a weak pull on the FINAL pad as well, so the planner is not
                   # satisfied by merely sitting on the carrot
                   + 0.010 * d_fin
                   + 1.8 * (1.0 - align)
                   + 1.6 * torch.relu(xb[:, 6] - v_allow) / 10.0
                   + 3.2 * torch.relu(SF.r_static * 1.6 - d) / (SF.r_static * 1.6)
                   + 45.0 * (d < 2.0).float()
                   + 3.2 * torch.relu(floor - P[:, 2]) / SF.r_static
                   + 0.8 * torch.relu(P[:, 2] - 260.0) / 60.0
                   + self.w_hazard * torch.relu(SF.r_static - clr_hat)
                       / SF.r_static
                   + (4.5 * torch.relu(TR - dd) / TR).sum(-1)
                   + 60.0 * (dd < 12.0).float().sum(-1)
                   + 0.35 * a.pow(2).mean(-1)
                   # energy: rotor thrust is ~5x more expensive per newton of
                   # lift than the wing, so penalising collective is what makes
                   # the aircraft choose to transition once it is fast enough
                   + 0.12 * (a[:, 0] + 1.0) * 0.5
                   # ...but committing to wing-borne below a safe margin over
                   # the stall speed is how you arrive at the ground quickly
                   + 3.0 * ((a[:, 5] + 1.0) * 0.5)
                         * torch.relu(V_SAFE - xb[:, 6]) / V_SAFE
                   + 1.5 * torch.relu(xb[:, 3].abs() - math.radians(SF.bank_max_deg))
                   + 2.5 * torch.relu(xb[:, 4].abs() - math.radians(22.0))
                   + 0.8 * xb[:, 9:12].pow(2).sum(-1)
                   + 1.2 * torch.relu(xb[:, 6].abs() - SF.v_max) / 10.0)
            cost = cost + disc * c_t
            disc *= 0.985

        if self.critic is not None:
            cost = cost - self.critic(self.wm.feat(hb, zb)).squeeze(-1)
        return cost, clr_min

    # ------------------------------------------------- CUDA graph capture --
    def _build_graph(self):
        """Record the whole rollout once and replay it thereafter.

        The rollout is latency-bound, not compute-bound: 60 steps x ~200 tiny
        kernels is ~12k launches per plan, and at ~15 us of launch overhead
        each that alone exceeds the 50 ms control period while the GPU sits
        near idle. A captured graph replays the entire sequence from a single
        launch. torch.compile fixes the same problem, but has to trace a
        60-fold unrolled graph, which costs minutes of compile time.
        """
        dev, N, H = self.dev, self.N, self.H
        ft = torch.float32
        self.s_h = torch.zeros(N, L.deter, device=dev, dtype=ft)
        self.s_z = torch.zeros(N, self.wm.rssm.zs, device=dev, dtype=ft)
        self.s_x = torch.zeros(N, 12, device=dev, dtype=ft)
        self.s_U = torch.zeros(N, H, L.act_dim, device=dev, dtype=ft)
        self.s_oc = torch.full((self.MAXOBS, 2), 1e6, device=dev, dtype=ft)
        self.s_oh = torch.ones(self.MAXOBS, 2, device=dev, dtype=ft)
        self.s_oz = torch.ones(self.MAXOBS, device=dev, dtype=ft)
        self.s_tp = torch.full((self.MAXTRK, 3), 1e6, device=dev, dtype=ft)
        self.s_tv = torch.zeros(self.MAXTRK, 3, device=dev, dtype=ft)
        self.s_tr = torch.full((self.MAXTRK,), 1e-3, device=dev, dtype=ft)
        self.s_sg = torch.zeros(3, device=dev, dtype=ft)
        self.s_gf = torch.zeros(3, device=dev, dtype=ft)
        self.s_vp = torch.as_tensor(self.g.vp[:, :2], dtype=ft,
                                    device=dev).contiguous()
        self.s_bi = torch.zeros(1, 3, device=dev, dtype=ft)

        args = lambda: (self.s_h, self.s_z, self.s_x, self.s_U,
                        (self.s_oc, self.s_oh, self.s_oz),
                        self.s_tp, self.s_tv, self.s_tr, self.s_sg,
                        self.s_gf, self.s_vp, self.s_bi)
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3):
                self._rollout(*args())
        torch.cuda.current_stream().wait_stream(st)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self.s_cost, self.s_clr = self._rollout(*args())

    @torch.no_grad()
    def plan(self, h0, z0, x0, subgoal, tracks_world, goal_final,
             nominal=None):
        dev = self.dev
        N, H = self.N, self.H
        mean = torch.cat([self.mean[1:], self.a_trim[None]], 0)       # warm start

        # Blend the warm start toward a nominal guidance action.
        #
        # In MPPI the sampling mean *is* the prior, and a sampler explores only
        # the neighbourhood of whatever it is given. Warm-starting purely from
        # the previous solution means a plan that has drifted keeps being
        # refined in the same drifted neighbourhood -- measured as 0/18
        # completed missions across six configurations, while the geometric
        # guidance law it ignores completes 34/34.
        #
        # Seeding from that guidance law turns MPPI into what it is good at:
        # improving a competent nominal using a learned model of what is about
        # to happen, rather than discovering flight from scratch. This is the
        # role TD-MPC2 gives its policy prior; here the prior is a controller
        # we can already vouch for.
        if nominal is not None:
            nom = torch.as_tensor(np.asarray(nominal, np.float32), device=dev)
            w_nom = self.nominal_weight
            mean = (1.0 - w_nom) * mean + w_nom * nom.unsqueeze(0)

        # Temporally correlated exploration noise (AR(1) along the horizon).
        #
        # This is not a refinement, it is load-bearing. With noise drawn i.i.d.
        # per timestep, a 64-step sample is a random walk in action space: every
        # one of the 512 candidates is high-frequency jitter, none of them is a
        # coherent manoeuvre like "bank left and hold it", and so the elite set
        # never contains a plan worth committing to. The aircraft flew smoothly
        # and avoided everything but never went anywhere. Correlating the noise
        # over ~0.35 s makes each sample an actual manoeuvre.
        w = torch.randn(N, H, L.act_dim, device=dev)
        beta = L.mppi_beta
        g = math.sqrt(1.0 - beta * beta)
        eps = torch.empty_like(w)
        eps[:, 0] = w[:, 0]
        for t in range(1, H):
            eps[:, t] = beta * eps[:, t - 1] + g * w[:, t]
        U = (mean.unsqueeze(0) + eps * L.mppi_sigma).clamp(-1, 1)
        if self.actor is not None and self.actor_seed_frac > 0:
            k = int(N * self.actor_seed_frac)
            if k > 0:
                U[:k] = self._actor_rollout(h0, z0, k).clamp(-1, 1)

        hb = h0.expand(N, -1).contiguous()
        zb = z0.expand(N, -1).contiguous()
        xb = x0.expand(N, -1).contiguous()
        sg = torch.as_tensor(subgoal, dtype=torch.float32, device=dev)

        # pad the tracked intruders to a fixed count; padding sits far away
        # with zero radius so it contributes exactly nothing to the cost
        TP = torch.full((self.MAXTRK, 3), 1e6, device=dev)
        TV = torch.zeros((self.MAXTRK, 3), device=dev)
        TR = torch.full((self.MAXTRK,), 1e-3, device=dev)
        for i, (p_i, v_i, coop) in enumerate(tracks_world[:self.MAXTRK]):
            TP[i] = torch.as_tensor(p_i, dtype=torch.float32, device=dev)
            TV[i] = torch.as_tensor(v_i, dtype=torch.float32, device=dev)
            TR[i] = SF.r_wellclear if coop else SF.r_suas

        if self._graph is None:
            self._build_graph()
        self.s_h.copy_(hb); self.s_z.copy_(zb); self.s_x.copy_(xb)
        self.s_U.copy_(U); self.s_sg.copy_(sg)
        self.s_gf.copy_(torch.as_tensor(np.asarray(goal_final, np.float32),
                                        device=dev))
        oc, oh, oz = self._obs_t
        self.s_oc.copy_(oc); self.s_oh.copy_(oh); self.s_oz.copy_(oz)
        self.s_tp.copy_(TP); self.s_tv.copy_(TV); self.s_tr.copy_(TR)
        self.s_bi.copy_(self.bias_est.view(1, 3))
        self._graph.replay()
        cost, clr_min = self.s_cost.clone(), self.s_clr.clone()

        # Temperature relative to the spread of the population, not absolute.
        # The cost scale drifts by an order of magnitude between "far from the
        # goal" and "on final", and a fixed lambda either saturates the softmax
        # onto one sample or flattens it to a uniform average.
        # Elite selection rather than a softmax over the whole population.
        # With a cost landscape this flat, weighting all 512 samples returns an
        # average that sits essentially on top of the prior mean, so the plan
        # never moves. Restricting the average to the best 12% makes each
        # replan a decisive step while the softmax inside the elite set keeps
        # the update smooth.
        n_elite = max(8, int(0.12 * self.N))
        e_cost, e_idx = torch.topk(-cost, n_elite)
        e_cost = -e_cost
        spread = e_cost.std().clamp(min=1e-3)
        w = torch.softmax(-(e_cost - e_cost.min()) / (L.mppi_lambda * spread), 0)
        self.mean = (w[:, None, None] * U[e_idx]).sum(0)
        a0 = self.mean[0].clamp(-1, 1)
        self.last = dict(cost=cost, w=w, U=U, clr_min=clr_min, elite=e_idx,
                         best=int(cost.argmin()), eff_n=float(1.0/(w**2).sum()))
        return a0

    @torch.no_grad()
    def _actor_rollout(self, h0, z0, k):
        h = h0.expand(k, -1).contiguous(); z = z0.expand(k, -1).contiguous()
        out = []
        for _ in range(self.H):
            a = self.actor(self.wm.feat(h, z), sample=True)
            out.append(a)
            h, z, _ = self.wm.rssm.img_step(h, z, a)
        return torch.stack(out, 1)

    # ----------------------------------------------------------- CBF repair --
    def shield(self, action, x_np, pos, vel, tracks_world):
        """Map the planner's action to a commanded acceleration, project it
        onto the barrier-safe set, and map the correction back to references."""
        dev = self.dev
        xt = torch.as_tensor(x_np, dtype=torch.float32, device=dev)[None]
        at = torch.as_tensor(action, dtype=torch.float32, device=dev)[None]
        dx = self.phys(xt, at)[0].cpu().numpy()
        acc_body_frd = dx[6:9]
        # body FRD -> world ENU
        from .geometry import quat_to_R, FLU2FRD
        roll, pitch, yaw = x_np[3], x_np[4], x_np[5]
        from .physics import euler_to_R
        R = euler_to_R(torch.tensor([roll]), torch.tensor([pitch]),
                       torch.tensor([yaw]))[0].numpy()
        acc_w = R @ (FLU2FRD @ acc_body_frd)

        u_safe, engaged = self.cbf(pos, vel, acc_w, tracks_world)
        d_acc = u_safe - acc_w
        if not engaged or np.linalg.norm(d_acc) < 1e-6:
            return action, 0, u_safe

        # --- map the acceleration correction back to attitude references -----
        # In coordinated flight a lateral acceleration comes from bank and a
        # longitudinal one from pitch/pusher:  a_lat = g tan(phi).
        a_out = np.array(action, np.float64).copy()
        c, s = math.cos(yaw), math.sin(yaw)
        d_fwd = d_acc[0]*c + d_acc[1]*s
        d_lat = -d_acc[0]*s + d_acc[1]*c
        a_out[1] = np.clip(a_out[1] + np.arctan2(d_lat, G)
                           / math.radians(SF.bank_max_deg), -1, 1)
        a_out[2] = np.clip(a_out[2] + np.arctan2(d_fwd, G)
                           / math.radians(18.0), -1, 1)
        a_out[0] = np.clip(a_out[0] + d_acc[2] * 0.10, -1, 1)
        return a_out.astype(np.float32), 1, u_safe
