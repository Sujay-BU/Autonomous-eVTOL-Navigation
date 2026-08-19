"""
Instrumented flight: wires the control loop to the XAI suite and the dashboard.

The same object drives the live GUI and the video recorder, so what is recorded
is byte-identical to what an operator watching the screen would have seen.

Cost discipline: the dashboard frame is composed every control step (20 Hz) so
the video plays back in real time, but the expensive explanations -- Grad-CAM,
imagination decode, cost attribution, counterfactuals -- are recomputed at
4 Hz and held in between. None of them sit on the control path.
"""
import numpy as np
import torch
import cv2

from .config import CFG
from .dashboard import DashboardRenderer
from .xai import XAISuite
from .geometry import quat_to_R

L, SF, S = CFG.lrn, CFG.saf, CFG.sen


def depth_to_vis(d, far=None):
    far = far or S.nav_far
    m = np.isfinite(d) & (d > 0)
    v = np.zeros(d.shape, np.uint8)
    if m.any():
        v[m] = (255 * (1.0 - np.clip(d[m] / far, 0, 1)) ** 1.6).astype(np.uint8)
    return cv2.applyColorMap(v, cv2.COLORMAP_TURBO)


def cam_overlay(rgb, cam):
    """Grad-CAM heat map over the camera image."""
    base = cv2.cvtColor(cv2.resize(rgb, (cam.shape[1], cam.shape[0])),
                        cv2.COLOR_RGB2BGR)
    hm = cv2.applyColorMap((np.clip(cam, 0, 1) * 255).astype(np.uint8),
                           cv2.COLORMAP_INFERNO)
    return cv2.addWeighted(base, 0.45, hm, 0.55, 0)


def wm_img_to_bgr(x):
    """(4,R,R) float -> BGR preview of the RGB channels."""
    r = np.clip(x[:3].transpose(1, 2, 0), 0, 1)
    return cv2.cvtColor((r * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


class Instrumented:
    def __init__(self, runner, device="cuda", xai_every=5, title="Phy-WAM"):
        self.run = runner
        self.env = runner.env
        self.wm = runner.wm
        self.planner = runner.planner
        self.dev = device
        self.xai = XAISuite(self.wm, self.planner, device)
        self.dash = DashboardRenderer(self.env.geom, title)
        self.every = xai_every
        self.cache = {}
        self.trail = []
        self.n_steps = 0
        self.n_engaged = 0

    # ------------------------------------------------------------ per step --
    def _explain(self, d):
        """Recompute the expensive explanations. Called at ~4 Hz."""
        env, wm, pl = self.env, self.wm, self.planner
        obs, h, z = d["obs"], d["h"], d["z"]
        t = lambda x: torch.as_tensor(np.asarray(x, np.float32),
                                      device=self.dev).unsqueeze(0)
        img_t, pro_t, trk_t = t(obs["img"]), t(obs["pro"]), t(obs["trk"])
        a_t = t(d["action"])

        # --- Grad-CAM on the hazard head ---------------------------------
        cam = self.xai.cam(img_t, pro_t, trk_t, h, z, a_t, mode="hazard")
        self.cache["cam_vis"] = cv2.resize(
            cam_overlay(obs["rgb_full"], cam), (252, 190))

        # --- world-model reconstruction and imagination ------------------
        with torch.no_grad():
            e = wm.enc(img_t, pro_t, trk_t)
            h2, z2, _, _ = wm.rssm.obs_step(h, z, a_t, e)
            f = wm.feat(h2, z2)
            rec = wm.dec_img(f).clamp(0, 1)[0].cpu().numpy()
        U = pl.mean.unsqueeze(0)
        frames, _ = self.xai.imag.rollout(h2, z2, U, every=16)
        self.cache["wm_obs"] = wm_img_to_bgr(obs["img"])
        self.cache["wm_rec"] = wm_img_to_bgr(rec)
        self.cache["wm_imag"] = [wm_img_to_bgr(fr) for fr in frames]
        _, surp = self.xai.imag.surprise(rec[:3], obs["img"][:3])
        self.cache["surprise"] = surp

        # --- planner attribution + counterfactuals -----------------------
        if pl._obs_t is not None and pl._graph is not None:
            x0 = torch.as_tensor(env.phys_state, device=self.dev).unsqueeze(0)
            sg = torch.as_tensor(np.asarray(d["subgoal"], np.float32),
                                 device=self.dev)
            gf = torch.as_tensor(np.asarray(env.goal, np.float32),
                                 device=self.dev)
            args = (pl._obs_t, pl.s_tp, pl.s_tv, pl.s_tr, sg, gf,
                    pl.s_vp, pl.s_bi)
            terms, traj = self.xai.attr(h2, z2, x0, U, *args)
            self.cache["cost_terms"] = terms
            self.cache["plan_traj"] = traj
            self.cache["counterfactual"] = self.xai.cf(h2, z2, x0, U, *args)
            # a handful of elite samples, for the plan view
            el = pl.last.get("elite")
            if el is not None:
                k = min(10, el.numel())
                Ue = pl.last["U"][el[:k]]
                xb = x0.expand(k, -1).contiguous()
                hb = h2.expand(k, -1).contiguous()
                zb = z2.expand(k, -1).contiguous()
                trs = []
                with torch.no_grad():
                    for tt in range(Ue.shape[1]):
                        xb, _, _ = wm.phys(hb, zb, xb, Ue[:, tt], pl.dt)
                        hb, zb, _ = wm.rssm.img_step(hb, zb, Ue[:, tt])
                        trs.append(xb[:, :3].cpu().numpy())
                trs = np.stack(trs, 1)
                self.cache["samples"] = [(trs[i], i < 3) for i in range(k)]

    def callback(self, d):
        env = self.env
        self.n_steps += 1
        self.n_engaged += int(bool(d["engaged"]))
        obs, info = d["obs"], d["info"]
        pos = info["pos"]
        self.trail.append(pos.copy())
        if d["step"] % self.every == 0:
            try:
                self._explain(d)
            except Exception as exc:              # never let XAI stop a flight
                self.cache["xai_error"] = repr(exc)

        # project the tracked intruders into the nav image
        R_wb = quat_to_R(env.br.quat)
        proj = []
        for (p_i, v_i, coop) in d["tracks"]:
            uv, rng_, vis = env.cam_nav.project(np.asarray(p_i)[None], pos, R_wb)
            proj.append((uv[0], float(rng_[0]), coop, bool(vis[0])))
        sg_uv, _, sg_vis = env.cam_nav.project(
            np.asarray(d["subgoal"], np.float64)[None], pos, R_wb)
        roll, pitch, yaw = env.br.euler()

        frame_d = dict(
            rgb_full=obs["rgb_full"],
            depth_vis=depth_to_vis(obs["depth_full"]),
            proj=proj, sg_uv=(sg_uv[0][0], sg_uv[0][1], bool(sg_vis[0])),
            roll=roll, pitch=pitch, yaw=yaw,
            pos=pos, route=d["route"], tracks=d["tracks"],
            trail=self.trail, info=info,
            alt=pos[2], spd=info["spd"], clr=info["min_obs"] if False else info["clr"],
            sep=min(info["sep"], 9999), soc=info["soc"],
            engaged=d["engaged"], n_cbf=self.planner.cbf.n_active,
            u_safe=d["u_safe"], action=d["action"], action_raw=d["action_raw"],
            shield_frac=self.n_engaged / max(self.n_steps, 1),
            status="SHIELD ACTIVE" if d["engaged"] else "NOMINAL",
            banner=d.get("banner", ""),
        )
        frame_d.update(self.cache)
        self.dash.push(frame_d)
        return self.dash(frame_d)
