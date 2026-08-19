"""
Explainability for the Phy-WAM controller.

Four complementary views, chosen because between them they cover every stage of
the decision -- what was seen, what was predicted, why an action was preferred,
and what overrode it:

  1. Grad-CAM on the perception encoder, taken with respect to the model's own
     predicted clearance. Answers "which pixels made it think this was
     dangerous", not the vaguer "which pixels were salient".

  2. Imagination decode. The world model is generative, so we can render the
     frames it expects to see over the next 3.2 s and put them next to what
     actually arrives. Divergence between the two is a direct, honest signal
     that the model does not understand the situation it is in.

  3. Planner cost attribution. The MPPI cost is a sum of named terms, so the
     chosen trajectory's cost can be decomposed exactly into goal / obstacle /
     traffic / energy / envelope contributions. This is not an approximation of
     the decision -- it IS the decision, itemised.

  4. Barrier attribution. Which constraints were active, and how far the shield
     had to move the action to satisfy them.

(1) is post-hoc and approximate; (2)-(4) are exact readouts of the quantities
the controller actually used. That distinction is stated plainly in the GUI,
because an explanation that looks authoritative but is merely plausible is
worse than none.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F

from .config import CFG
from .planner import V_SAFE

L, SF = CFG.lrn, CFG.saf

COST_TERMS = ["subgoal_xy", "subgoal_z", "goal_pull", "align", "speed", "obstacle",
              "obstacle_hit", "ground", "ceiling", "hazard_pred", "traffic",
              "traffic_hit", "effort", "energy", "stall_guard", "bank",
              "pitch", "rates", "vmax"]


class GradCAM:
    """Grad-CAM over the encoder's last convolutional block."""

    def __init__(self, wm):
        self.wm = wm
        self.act = None
        self.grad = None
        target = wm.enc.conv[-3]                 # last Conv2d in the stack
        target.register_forward_hook(self._fwd)
        target.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o):
        self.act = o.detach()

    def _bwd(self, m, gi, go):
        self.grad = go[0].detach()

    def __call__(self, img, pro, trk, h, z, a, mode="hazard", smooth=3,
                 sigma=0.04):
        """Returns a (R,R) map in [0,1]. `mode` selects what is explained:
        'hazard' -> the predicted clearance head (low clearance = danger)
        'value'  -> the reward head."""
        wm = self.wm
        acc = None
        for s in range(smooth):
            x = img.clone()
            if s > 0:                            # SmoothGrad: average over
                x = x + torch.randn_like(x) * sigma   # noisy copies
            x.requires_grad_(True)
            with torch.enable_grad():
                e = wm.enc(x, pro, trk)
                h2, z2, _, _ = wm.rssm.obs_step(h, z, a, e)
                f = wm.feat(h2, z2)
                if mode == "hazard":
                    # large when the model expects LITTLE clearance
                    y = -wm.twohot.decode(wm.head_clr(f)).sum()
                else:
                    y = wm.twohot.decode(wm.head_rew(f)).sum()
                wm.zero_grad(set_to_none=True)
                y.backward()
            if self.act is None or self.grad is None:
                return np.zeros((L.img_res, L.img_res), np.float32)
            w = self.grad.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((w * self.act).sum(1, keepdim=True))
            cam = F.interpolate(cam, size=(L.img_res, L.img_res),
                                mode="bilinear", align_corners=False)
            acc = cam if acc is None else acc + cam
        cam = acc[0, 0].cpu().numpy()
        if cam.max() > 1e-8:
            cam = cam / cam.max()
        return cam.astype(np.float32)


class Imagination:
    """Decode what the world model expects the camera to show next."""

    def __init__(self, wm, device="cuda"):
        self.wm, self.dev = wm, device

    @torch.no_grad()
    def rollout(self, h, z, actions, every=8):
        """actions: (1,H,A). Returns decoded frames at `every` steps."""
        wm = self.wm
        hs, zs = h, z
        frames, steps = [], []
        for t in range(actions.shape[1]):
            hs, zs, _ = wm.rssm.img_step(hs, zs, actions[:, t])
            if (t + 1) % every == 0:
                f = wm.feat(hs, zs)
                img = wm.dec_img(f).clamp(0, 1)[0].cpu().numpy()
                frames.append(img)
                steps.append(t + 1)
        return frames, steps

    @torch.no_grad()
    def surprise(self, wm_recon, actual):
        """Per-pixel prediction error, i.e. where the world stopped matching
        the model. Rising surprise is the cue that the controller is operating
        outside what it has learned."""
        e = np.abs(wm_recon - actual).mean(0)
        return e / max(e.max(), 1e-6), float(e.mean())


class CostAttribution:
    """Itemise the MPPI cost of a single action sequence.

    Deliberately a separate, un-captured, single-sample rollout: it mirrors the
    terms in MPPIPlanner._rollout exactly but keeps them apart instead of
    summing. Run at GUI rate (4 Hz), so its inefficiency does not matter.
    """

    def __init__(self, planner):
        self.p = planner

    @torch.no_grad()
    def __call__(self, h, z, x, U, obs, TP, TV, TR, sg, gf, VP, BI=None):
        p = self.p
        wm = p.wm
        dt = p.dt
        terms = {k: 0.0 for k in COST_TERMS}
        hb, zb, xb = h.clone(), z.clone(), x.clone()
        disc = 1.0
        traj = []
        for t in range(U.shape[1]):
            a = U[:, t]
            xp = xb
            xb, _, _ = wm.phys(hb, zb, xb, a, dt, BI)
            hb, zb, _ = wm.rssm.img_step(hb, zb, a)
            P = xb[:, :3]
            traj.append(P[0].cpu().numpy().copy())
            d = p._sdf_t(P, obs)
            f = wm.feat(hb, zb)
            clr_hat = wm.twohot.decode(wm.head_clr(f))
            to_sg = sg[:2] - P[:, :2]
            rng_sg = to_sg.norm(dim=-1).clamp(min=1.0)
            vh = (xb[:, :3] - xp[:, :3])[:, :2]
            align = (vh * to_sg).sum(-1) / (vh.norm(dim=-1).clamp(min=0.5) * rng_sg)
            d_fin = (P[:, :2] - gf[:2]).norm(dim=-1)
            v_allow = (0.22 * d_fin).clamp(4.0, SF.v_max)
            d_vp = (P[:, None, :2] - VP[None]).norm(dim=-1).min(-1).values
            floor = 1.0 + (SF.r_static - 1.0) * ((d_vp - 25.0)/35.0).clamp(0, 1)
            Pi = TP + TV * ((t + 1) * dt)
            dd = (P[:, None, :] - Pi[None]).norm(dim=-1)
            g = lambda v: float(v.reshape(-1)[0]) * disc
            terms["subgoal_xy"]  += g(0.160 * (P[:, :2] - sg[:2]).norm(dim=-1))
            terms["subgoal_z"]   += g(0.120 * (P[:, 2] - sg[2]).abs())
            terms["goal_pull"]   += g(0.010 * d_fin)
            terms["align"]       += g(1.8 * (1.0 - align))
            terms["speed"]       += g(1.6 * torch.relu(xb[:, 6] - v_allow) / 10.0)
            terms["obstacle"]    += g(3.2 * torch.relu(SF.r_static*1.6 - d)
                                      / (SF.r_static*1.6))
            terms["obstacle_hit"]+= g(45.0 * (d < 2.0).float())
            terms["ground"]      += g(3.2 * torch.relu(floor - P[:, 2])
                                      / SF.r_static)
            terms["ceiling"]     += g(0.8 * torch.relu(P[:, 2] - 260.0) / 60.0)
            terms["hazard_pred"] += g(2.4 * torch.relu(SF.r_static - clr_hat)
                                      / SF.r_static)
            terms["traffic"]     += g((4.5 * torch.relu(TR - dd) / TR).sum(-1))
            terms["traffic_hit"] += g(60.0 * (dd < 12.0).float().sum(-1))
            terms["effort"]      += g(0.35 * a.pow(2).mean(-1))
            terms["energy"]      += g(0.12 * (a[:, 0] + 1.0) * 0.5)
            terms["stall_guard"] += g(3.0 * ((a[:, 5] + 1.0) * 0.5)
                                      * torch.relu(V_SAFE - xb[:, 6]) / V_SAFE)
            terms["bank"]        += g(1.5 * torch.relu(xb[:, 3].abs()
                                      - math.radians(SF.bank_max_deg)))
            terms["pitch"]       += g(2.5 * torch.relu(xb[:, 4].abs()
                                      - math.radians(22.0)))
            terms["rates"]       += g(0.8 * xb[:, 9:12].pow(2).sum(-1))
            terms["vmax"]        += g(1.2 * torch.relu(xb[:, 6].abs() - SF.v_max)
                                      / 10.0)
            disc *= 0.985
        return terms, np.asarray(traj)


class Counterfactual:
    """'What would have happened if...' -- evaluate a handful of fixed
    alternative manoeuvres against the chosen one under the same model.

    This is the explanation a pilot would actually ask for: not which pixels
    mattered, but what the alternatives would have cost.
    """

    OPTIONS = {
        "hold heading": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "bank left":    np.array([0.0, -0.8, 0.0, -0.5, 0.0, 0.0]),
        "bank right":   np.array([0.0, 0.8, 0.0, 0.5, 0.0, 0.0]),
        "climb":        np.array([0.6, 0.0, 0.3, 0.0, 0.0, 0.0]),
        "descend":      np.array([-0.6, 0.0, -0.3, 0.0, 0.0, 0.0]),
        "decelerate":   np.array([0.3, 0.0, 0.4, 0.0, -1.0, -1.0]),
    }

    def __init__(self, planner, attribution):
        self.p, self.attr = planner, attribution

    @torch.no_grad()
    def __call__(self, h, z, x, chosen_U, obs, TP, TV, TR, sg, gf, VP, BI=None):
        out = {}
        base, _ = self.attr(h, z, x, chosen_U, obs, TP, TV, TR, sg, gf, VP, BI)
        out["chosen"] = sum(base.values())
        for name, delta in self.OPTIONS.items():
            U = chosen_U.clone()
            d = torch.as_tensor(delta, dtype=U.dtype, device=U.device)
            # apply the manoeuvre for the first second, then revert to the plan
            U[:, :20] = (U[:, :20] + d).clamp(-1, 1)
            t, _ = self.attr(h, z, x, U, obs, TP, TV, TR, sg, gf, VP, BI)
            out[name] = sum(t.values())
        return out


class XAISuite:
    def __init__(self, wm, planner, device="cuda"):
        self.cam = GradCAM(wm)
        self.imag = Imagination(wm, device)
        self.attr = CostAttribution(planner)
        self.cf = Counterfactual(planner, self.attr)
        self.wm, self.planner, self.dev = wm, planner, device
