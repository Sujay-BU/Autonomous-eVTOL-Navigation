"""
Procedural textures for the city.

Untextured primitives give a vision system almost nothing to lock onto: a flat
grey box against a flat blue sky has no gradients, no parallax cues and no
scale reference. These textures exist so the depth/flow structure the encoder
learns is grounded in something an actual camera would see.
"""
import os, sys
import numpy as np, cv2

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(root, "sim", "materials", "textures")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(7)


def grain(h, w, amp, scale):
    """Low-frequency mottling so flat surfaces still have gradient content."""
    s = max(2, int(scale))
    n = rng.normal(0, 1, (h // s + 2, w // s + 2)).astype(np.float32)
    n = cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
    return n * amp


# ------------------------------------------------------------------ ground --
# One tile = 400 m of city at 2048 px -> 0.195 m/px.
H = W = 2048
g = np.zeros((H, W, 3), np.float32)
g[:] = np.array([78, 96, 72], np.float32)                 # vegetation base
g += grain(H, W, 9.0, 26)[..., None]

# city blocks separated by a road grid
BLK = 256                                                  # 50 m blocks
for by in range(0, H, BLK):
    for bx in range(0, W, BLK):
        kind = rng.random()
        x0, y0 = bx + 22, by + 22
        x1, y1 = bx + BLK - 22, by + BLK - 22
        if kind < 0.60:                                    # built-up lot
            c = rng.uniform(88, 132)
            cv2.rectangle(g, (x0, y0), (x1, y1),
                          (float(c*0.96), float(c), float(c*1.04)), -1)
            for _ in range(rng.integers(3, 7)):            # rooftop plant
                px, py = rng.integers(x0, x1-20), rng.integers(y0, y1-20)
                s = int(rng.integers(10, 26))
                cv2.rectangle(g, (px, py), (px+s, py+s),
                              (float(c*0.7), float(c*0.72), float(c*0.75)), -1)
        elif kind < 0.78:                                  # park
            cv2.rectangle(g, (x0, y0), (x1, y1), (52, 104, 48), -1)
            for _ in range(rng.integers(8, 18)):
                px, py = rng.integers(x0, x1), rng.integers(y0, y1)
                cv2.circle(g, (px, py), int(rng.integers(6, 15)), (38, 78, 36), -1)
        else:                                              # parking / plaza
            cv2.rectangle(g, (x0, y0), (x1, y1), (96, 96, 100), -1)
            for yy in range(y0+8, y1-8, 18):
                cv2.line(g, (x0+6, yy), (x1-6, yy), (140, 140, 145), 1)

# roads over the block gaps
for k in range(0, H+1, BLK):
    cv2.rectangle(g, (0, k-20), (W, k+20), (58, 58, 62), -1)
    cv2.rectangle(g, (k-20, 0), (k+20, H), (58, 58, 62), -1)
for k in range(0, H+1, BLK):                               # lane markings
    for t in range(0, W, 60):
        cv2.line(g, (t, k), (t+34, k), (196, 190, 150), 2)
        cv2.line(g, (k, t), (k, t+34), (196, 190, 150), 2)

g += grain(H, W, 5.0, 7)[..., None]
cv2.imwrite(os.path.join(OUT, "ground.png"),
            np.clip(g, 0, 255).astype(np.uint8)[:, :, ::-1])

# ----------------------------------------------------------------- facades --
# One texture instance covers one building face, so the window pitch sets the
# apparent scale of the building in the image.
for idx, (base, win, cols, rows) in enumerate([
        ((122, 126, 134), (58, 74, 96), 14, 22),
        ((148, 140, 128), (70, 78, 88), 11, 18),
        ((96, 104, 116),  (176, 188, 200), 17, 26)]):
    h = w = 1024
    f = np.zeros((h, w, 3), np.float32)
    f[:] = np.array(base, np.float32)
    f += grain(h, w, 7.0, 24)[..., None]
    mw, mh = w // cols, h // rows
    for r in range(rows):
        for c in range(cols):
            x0, y0 = c*mw + int(mw*0.22), r*mh + int(mh*0.18)
            x1, y1 = c*mw + int(mw*0.82), r*mh + int(mh*0.74)
            lit = rng.random() < 0.16
            col = (np.array(win, np.float32) * (1.9 if lit else 1.0)
                   + rng.normal(0, 9, 3))
            cv2.rectangle(f, (x0, y0), (x1, y1),
                          tuple(float(v) for v in np.clip(col, 0, 255)), -1)
            cv2.rectangle(f, (x0, y0), (x1, y1),
                          tuple(float(v*0.72) for v in base), 1)
        cv2.line(f, (0, (r+1)*mh - 2), (w, (r+1)*mh - 2),
                 tuple(float(v*0.8) for v in base), 2)
    f += grain(h, w, 4.0, 6)[..., None]
    cv2.imwrite(os.path.join(OUT, f"facade{idx}.png"),
                np.clip(f, 0, 255).astype(np.uint8)[:, :, ::-1])

# ------------------------------------------------------------- vertiport pad --
h = w = 1024
p = np.full((h, w, 3), 46, np.float32)
p += grain(h, w, 6.0, 20)[..., None]
cv2.circle(p, (w//2, h//2), int(w*0.46), (58, 58, 62), -1)
cv2.circle(p, (w//2, h//2), int(w*0.40), (236, 236, 232), 14)
cv2.circle(p, (w//2, h//2), int(w*0.30), (236, 150, 40), 8)
cv2.rectangle(p, (int(w*.36), int(h*.30)), (int(w*.44), int(h*.70)), (240,240,238), -1)
cv2.rectangle(p, (int(w*.56), int(h*.30)), (int(w*.64), int(h*.70)), (240,240,238), -1)
cv2.rectangle(p, (int(w*.36), int(h*.46)), (int(w*.64), int(h*.54)), (240,240,238), -1)
cv2.imwrite(os.path.join(OUT, "pad.png"), np.clip(p,0,255).astype(np.uint8)[:,:,::-1])

for f in sorted(os.listdir(OUT)):
    print(f"  {f:14s} {os.path.getsize(os.path.join(OUT,f))/1024:7.0f} KB")
print(f"-> {OUT}")
