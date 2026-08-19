"""Training curves from runs/<name>/hist.json."""
import os, sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PANELS = [
    ("success",     None,        "mission success (rolling)",   "#4aa3e0"),
    ("ret",         "ret",       "episode return",              "#4aa3e0"),
    ("img",         "img",       "image reconstruction loss",   "#e0a24a"),
    ("kl",          "kl",        "KL (dynamics + representation)", "#e0a24a"),
    ("physpair",    None,        "1-step state error: analytic vs +residual", "#7ad07a"),
    ("val",         "val",       "critic value estimate",       "#c98ad0"),
    ("min_obs",     "min_obs",   "min obstacle clearance (m)",  "#7ad07a"),
    ("shield_rate", "shield_rate","barrier engagement (frac)",  "#e06a6a"),
]


def roll(a, k=25):
    a = np.asarray(a, float)
    if a.size < 2:
        return a
    k = max(1, min(k, a.size // 3 or 1))
    c = np.convolve(a, np.ones(k) / k, mode="valid")
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="main")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    hp = os.path.join(ROOT, "runs", a.name, "hist.json")
    h = json.load(open(hp))
    if not h:
        print("no history yet"); return
    print(f"{len(h)} logged episodes, {h[-1]['step']} gradient steps, "
          f"{h[-1]['t']/3600:.2f} h")

    plt.style.use("dark_background")
    fig, axes = plt.subplots(4, 2, figsize=(13, 11))
    fig.suptitle(f"Phy-WAM training — run '{a.name}'", fontsize=14, y=0.995)
    ep = [r["ep"] for r in h]
    for ax, (key, field, label, col) in zip(axes.ravel(), PANELS):
        if key == "success":
            y = roll([r["success"] for r in h])
            ax.plot(ep[len(ep)-len(y):], y, color=col, lw=1.4)
            ax.set_ylim(-0.05, 1.05)
        elif key == "physpair":
            n = roll([r["nom"] for r in h])
            p = roll([r["phys"] for r in h])
            x = ep[len(ep)-len(n):]
            ax.plot(x, n, color="#888888", lw=1.2, label="analytic only")
            ax.plot(x, p, color=col, lw=1.4, label="analytic + residual")
            ax.set_yscale("log"); ax.legend(fontsize=8, frameon=False)
        else:
            y = roll([r[field] for r in h])
            ax.plot(ep[len(ep)-len(y):], y, color=col, lw=1.3)
            if field in ("img", "kl"):
                ax.set_yscale("log")
        ax.set_title(label, fontsize=10, loc="left")
        ax.grid(alpha=0.15)
        ax.set_xlabel("episode", fontsize=8)
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    out = a.out or os.path.join(ROOT, "docs", f"training_{a.name}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=110, facecolor="#151311")
    print("wrote", out)


if __name__ == "__main__":
    main()
