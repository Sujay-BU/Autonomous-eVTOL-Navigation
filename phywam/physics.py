"""
Differentiable analytic eVTOL model -- the "physics" half of the
physics-informed world model.

This is a batched PyTorch reimplementation of the same momentum-theory rotor
model, drag-polar wing and cascade attitude loop that the plant uses, and it is
DELIBERATELY INCOMPLETE. It omits ground effect, rotor-wing interference,
actuator lag, turbulence and the post-stall blend. Those omissions are the
whole point: they are the residual the network is asked to learn, so the
learned part is correcting real, structured, physically-meaningful error rather
than fitting noise.

State (B,12), all SI, world frame ENU, body frame FRD:
    0:3  position  (x_E, y_N, z_U)
    3:6  euler     (roll, pitch_frd, yaw_enu)     pitch_frd > 0 = nose up
    6:9  body velocity FRD (u, v, w)
    9:12 body rates FRD    (p, q, r)
"""
import math
import torch
import torch.nn as nn

from .config import CFG

G, RHO = 9.80665, 1.225


def euler_to_R(roll, pitch_frd, yaw):
    """ENU <- body FLU rotation matrix, built from FRD-sense Euler angles.

    Mirrors the plant exactly: the aircraft's FLU pitch is the negative of the
    aerospace FRD pitch, and yaw is measured in ENU (from East, CCW).
    """
    th = -pitch_frd                                    # FLU pitch
    cr, sr = torch.cos(roll), torch.sin(roll)
    ct, st = torch.cos(th), torch.sin(th)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    R = torch.stack([
        torch.stack([cy*ct, cy*st*sr - sy*cr, cy*st*cr + sy*sr], -1),
        torch.stack([sy*ct, sy*st*sr + cy*cr, sy*st*cr - cy*sr], -1),
        torch.stack([-st,   ct*sr,            ct*cr           ], -1)], -2)
    return R                                            # (B,3,3)


class AnalyticEVTOL(nn.Module):
    """Nominal dynamics f_phys(x, a) -> dx/dt."""

    def __init__(self):
        super().__init__()
        A = CFG.air
        self.A = A
        # register as buffers so .to(device) moves them with the module
        rp = A.rotor_positions()
        self.register_buffer("rx", torch.tensor([r[0] for r in rp]))
        self.register_buffer("ry", torch.tensor([r[1] for r in rp]))
        self.register_buffer("rs", torch.tensor([r[3] for r in rp]))
        self.register_buffer("I", torch.tensor([A.Ixx, A.Iyy, A.Izz]))
        # FLU <-> FRD sign flip and the gravity vector, as buffers rather than
        # literals: creating a tensor from a Python list inside forward() does
        # a host-to-device copy, which CUDA graph capture forbids.
        self.register_buffer("flip", torch.tensor([1.0, -1.0, -1.0]))
        self.register_buffer("g_up", torch.tensor([0.0, 0.0, -G]))
        self.Tmax_tot = A.n_rotor * A.k_rotor * A.w_rotor_max ** 2
        self.AR = A.AR

    # ------------------------------------------------------------ actuators --
    def _inner_loop(self, x, a):
        """Replicate the plant's cascade: attitude error -> rate -> torque."""
        A = self.A
        roll, pitch = x[:, 3], x[:, 4]
        p, q, r = x[:, 9], x[:, 10], x[:, 11]
        col   = (a[:, 0] + 1) * 0.5
        r_ref = a[:, 1] * math.radians(CFG.saf.bank_max_deg)
        p_ref = a[:, 2] * math.radians(18.0)
        y_ref = a[:, 3] * 0.45
        sched = ((a[:, 5] + 1) * 0.5).clamp(0, 1)

        p_des = (6.0 * (r_ref - roll)).clamp(-1.4, 1.4)
        q_des = (6.0 * (p_ref - pitch)).clamp(-1.2, 1.2)
        r_des = y_ref.clamp(-0.9, 0.9)
        Mx = A.Ixx * 5.5 * (p_des - p)
        My = A.Iyy * 5.0 * (q_des - q)
        Mz = A.Izz * 2.6 * (r_des - r)
        wr = 1.0 - 0.85 * sched
        # total upward rotor force (FRD z is down, so thrust is -z)
        Fz = -col.clamp(0, 1) * self.Tmax_tot * (1.0 - 0.92 * sched)
        return Fz, Mx * wr, My * wr, Mz * wr, sched

    # ---------------------------------------------------------------- forces --
    def specific_force(self, x, a):
        """Body-frame FRD force per unit mass produced by the model, excluding
        gravity and Coriolis. Directly comparable to what the plant publishes,
        which is what makes the online bias estimate possible."""
        d = self(x, a, bias=None, _force_only=True)
        return d

    def forward(self, x, a, bias=None, _force_only=False):
        A = self.A
        B = x.shape[0]
        roll, pitch, yaw = x[:, 3], x[:, 4], x[:, 5]
        u, v, w = x[:, 6], x[:, 7], x[:, 8]
        p, q, r = x[:, 9], x[:, 10], x[:, 11]

        Fz_rot, Mx, My, Mz, sched = self._inner_loop(x, a)
        push = ((a[:, 4] + 1) * 0.5).clamp(0, 1)

        # --- aerodynamics (no wind: the model does not know the gust field) --
        V = torch.sqrt(u*u + v*v + w*w + 1e-6)
        act = (V > 0.5).float()
        al = torch.atan2(w, u.clamp(min=1e-3))
        be = torch.asin((v / V).clamp(-0.999, 0.999))
        qbar = 0.5 * RHO * V * V

        # linear lift curve with a hard clip -- the plant uses a smooth
        # post-stall blend, so this is one of the residual's jobs
        CL = (A.CL0 + A.CL_alpha * al).clamp(-A.CL_max, A.CL_max)
        CD = A.CD0 + CL * CL / (math.pi * A.oswald * self.AR)
        L = qbar * A.S_wing * CL * act
        D = qbar * A.S_wing * CD * act
        Y = qbar * A.S_fin * (-0.62 * be) * act
        ca, sa = torch.cos(al), torch.sin(al)

        Fx = L * sa - D * ca + A.T_push_max * push
        Fy = Y
        Fz = -L * ca - D * sa + Fz_rot

        # aerodynamic moments (rate damping + static stability)
        bh = A.b_span / (2.0 * V.clamp(min=3.0))
        chh = A.chord / (2.0 * V.clamp(min=3.0))
        Cl = A.Cl_beta * be + (-0.48) * p * bh
        Cm = A.Cm0 + A.Cm_alpha * al + (-12.5) * q * chh
        Cn = A.Cn_beta * be + (-0.19) * r * bh
        Mx = Mx + qbar * A.S_wing * A.b_span * Cl * act
        My = My + qbar * A.S_wing * A.chord * Cm * act
        Mz = Mz + qbar * A.S_wing * A.b_span * Cn * act

        # --- rigid-body equations of motion ---------------------------------
        R = euler_to_R(roll, pitch, yaw)                       # ENU <- FLU
        # gravity in body FRD: rotate world -g*z_up into FLU then flip y,z
        g_w = self.g_up.expand(B, 3)
        g_flu = torch.einsum("bji,bj->bi", R, g_w)             # R^T g
        g_frd = g_flu * self.flip

        om = torch.stack([p, q, r], -1)
        vb = torch.stack([u, v, w], -1)
        F = torch.stack([Fx, Fy, Fz], -1)
        if _force_only:
            return F / A.mass
        # Online disturbance estimate, in body FRD force-per-mass. The model
        # deliberately omits rotor-wing interference (up to -16 % thrust),
        # actuator lag and ground effect, so it is systematically optimistic
        # about how much lift a given collective buys. Open-loop rollouts then
        # predict level flight while the aircraft sinks, and replanning does not
        # help because the error repeats every cycle. Feeding the measured
        # discrepancy back in is what a disturbance observer is for.
        vdot = F / A.mass - torch.cross(om, vb, dim=-1) + g_frd
        if bias is not None:
            vdot = vdot + bias

        M = torch.stack([Mx, My, Mz], -1)
        Iom = om * self.I
        omdot = (M - torch.cross(om, Iom, dim=-1)) / self.I

        # --- kinematics -------------------------------------------------------
        v_flu = vb * self.flip
        pdot = torch.einsum("bij,bj->bi", R, v_flu)            # ENU

        om_flu = om * self.flip
        pf, qf, rf = om_flu[:, 0], om_flu[:, 1], om_flu[:, 2]
        th = -pitch
        ct = torch.cos(th).clamp(min=0.15)
        tt = torch.tan(th.clamp(-1.4, 1.4))
        cr, sr = torch.cos(roll), torch.sin(roll)
        roll_d = pf + sr * tt * qf + cr * tt * rf
        th_d   = cr * qf - sr * rf
        yaw_d  = (sr * qf + cr * rf) / ct
        eul_d  = torch.stack([roll_d, -th_d, yaw_d], -1)       # back to FRD pitch

        return torch.cat([pdot, eul_d, vdot, omdot], -1)       # (B,12)

    def step(self, x, a, dt, bias=None):
        """One explicit RK2 (midpoint) step. RK2 rather than Euler because the
        rotational modes here sit near 5-6 rad/s and Euler at dt=50 ms is only
        marginally stable for them."""
        k1 = self(x, a, bias)
        k2 = self(x + 0.5 * dt * k1, a, bias)
        return x + dt * k2
