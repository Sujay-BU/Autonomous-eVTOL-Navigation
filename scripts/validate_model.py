"""
Validate the physics-informed dynamics head against recorded plant data.

Rolls three predictors forward over the same recorded action sequences and
measures how far each drifts from what the plant actually did:

  analytic      the closed-form model alone, no learning
  + residual    analytic plus the learned correction
  constant      hold the current state (a floor, to show the task is non-trivial)

Uses the replay buffer, so it needs no simulator and can run while training
continues. This is the number that says whether "physics-informed" is doing any
work, or is just a label.
"""
import os, sys, json, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
torch.set_float32_matmul_precision("high")

from phywam.config import CFG
from phywam.worldmodel import WorldModel
from phywam.replay import SequenceReplay

L = CFG.lrn
GROUPS = [("position", slice(0, 3), "m"),
          ("attitude", slice(3, 6), "rad"),
          ("body velocity", slice(6, 9), "m/s"),
          ("body rates", slice(9, 12), "rad/s")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="main")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rb = SequenceReplay(os.path.join(ROOT, "runs", a.name, "replay"), create=False)
    print(f"replay: {len(rb)} transitions")
    wm = WorldModel(dev).to(dev).eval()
    ck = a.ckpt or os.path.join(ROOT, "runs", a.name, "ckpt.pt")
    trained = os.path.exists(ck)
    if trained:
        c = torch.load(ck, map_location=dev, weights_only=False)
        wm.load_state_dict(c["wm"])
        print(f"checkpoint: step {c['step']}, ep {c['ep']}")
    else:
        print("no checkpoint — residual is at its zero initialisation")

    dt = 1.0 / L.ctrl_hz
    H = a.horizon
    err = {k: np.zeros((3, H)) for k in ("analytic", "residual", "constant")}
    cnt = 0
    rng = np.random.default_rng(0)

    for _ in range(a.reps):
        b = rb.sample(batch=a.batch, seq=H + 1, rng=rng)
        if b is None:
            print("not enough data yet"); return
        phys = torch.as_tensor(b["phys"], dtype=torch.float32, device=dev)
        act = torch.as_tensor(b["act"], dtype=torch.float32, device=dev)
        img = torch.as_tensor(b["img"], dtype=torch.float32, device=dev)
        pro = torch.as_tensor(b["pro"], dtype=torch.float32, device=dev)
        trk = torch.as_tensor(b["trk"], dtype=torch.float32, device=dev)
        B = phys.shape[0]

        with torch.no_grad():
            # latent trajectory, needed by the residual
            h, z, _, _ = wm.observe(img, pro, trk, act)
            x_an = phys[:, 0].clone()
            x_re = phys[:, 0].clone()
            x_c0 = phys[:, 0].clone()
            for t in range(H):
                a_t = act[:, t + 1]
                x_an = wm.phys.phys.step(x_an, a_t, dt)
                x_re, _, _ = wm.phys(h[:, t], z[:, t], x_re, a_t, dt)
                truth = phys[:, t + 1]
                for name, xp in (("analytic", x_an), ("residual", x_re),
                                 ("constant", x_c0)):
                    d = (xp - truth).abs().cpu().numpy()
                    for gi, (_, sl, _u) in enumerate(GROUPS[:3]):
                        err[name][gi, t] += d[:, sl].mean()
        cnt += 1

    for k in err:
        err[k] /= max(cnt, 1)

    print(f"\n  open-loop prediction error, averaged over {a.reps*a.batch} windows")
    print(f"  {'horizon':>9} {'analytic':>26} {'analytic + residual':>26}")
    print(f"  {'(s)':>9} {'pos(m)  att(rad)  vel(m/s)':>26} "
          f"{'pos(m)  att(rad)  vel(m/s)':>26}")
    for t in (4, 9, 19, min(39, H - 1)):
        if t >= H: continue
        A_ = err["analytic"][:, t]; R_ = err["residual"][:, t]
        print(f"  {(t+1)*dt:9.2f} {A_[0]:8.3f} {A_[1]:9.4f} {A_[2]:9.3f}   "
              f"{R_[0]:11.3f} {R_[1]:9.4f} {R_[2]:9.3f}")
    tl = H - 1
    imp = 100 * (1 - err["residual"][:, tl] / np.maximum(err["analytic"][:, tl], 1e-12))
    print(f"\n  at {H*dt:.1f} s the residual reduces error by: "
          f"position {imp[0]:+.1f} %, attitude {imp[1]:+.1f} %, "
          f"velocity {imp[2]:+.1f} %")

    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    ts = (np.arange(H) + 1) * dt
    for gi, ax in enumerate(axes):
        ax.plot(ts, err["constant"][gi], color="#666666", lw=1.1, ls=":",
                label="hold state")
        ax.plot(ts, err["analytic"][gi], color="#e0a24a", lw=1.5, label="analytic")
        ax.plot(ts, err["residual"][gi], color="#7ad07a", lw=1.7,
                label="analytic + residual")
        ax.set_title(f"{GROUPS[gi][0]} error ({GROUPS[gi][2]})", fontsize=10,
                     loc="left")
        ax.set_xlabel("prediction horizon (s)", fontsize=9)
        ax.grid(alpha=0.15); ax.legend(fontsize=8, frameon=False)
        ax.set_yscale("log")
    fig.suptitle("Open-loop prediction against recorded plant data", y=1.0)
    fig.tight_layout()
    out = a.out or os.path.join(ROOT, "docs", "model_validation.png")
    fig.savefig(out, dpi=110, facecolor="#151311")
    print("wrote", out)


if __name__ == "__main__":
    main()
