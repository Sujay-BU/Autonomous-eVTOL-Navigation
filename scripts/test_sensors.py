"""Hold a stable hover at corridor altitude and check the cameras actually
see the city: depth must return finite ranges, RGB must have structure."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phywam.bridge import GazeboBridge
from phywam.config import CFG
import cv2

A = CFG.air
COL0 = A.W / (A.n_rotor * A.k_rotor * A.w_rotor_max**2)
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
br = GazeboBridge(os.path.join(root, "sim", "worlds", "urban_1.sdf"), verbose=1)
try:
    br.wait_ready(60)
    dt, iz = 1/20, 0.0
    z_ref = 120.0
    yaw_ref = None
    for k in range(500):
        st = br.state; z, vz = st[2], st[9]
        ez = z_ref - z
        iz = float(np.clip(iz + ez*dt, -40, 40))
        col = float(np.clip(COL0 + 0.0022*ez + 0.0009*iz - 0.0055*vz, 0, 1))
        roll, pitch, yaw = br.euler()
        if yaw_ref is None: yaw_ref = yaw
        # gentle yaw sweep so the camera scans the skyline
        yr = 0.20 if k > 120 else 0.0
        br.send(col, -0.5*roll, -0.4*pitch, yr, 0.0, 0.0)
        if k % 100 == 0 and k > 0:
            d = br.depth_nav
            fin = np.isfinite(d) & (d > 0)
            rgb = br.rgb_nav
            print(f"k={k:4d} z={z:6.1f} yaw={np.degrees(yaw):7.1f} | "
                  f"depth finite {100*fin.mean():5.1f}%  "
                  f"min {d[fin].min() if fin.any() else -1:7.1f} "
                  f"med {np.median(d[fin]) if fin.any() else -1:7.1f} | "
                  f"rgb mean {rgb.mean():6.1f} std {rgb.std():6.1f}")
        time.sleep(dt)
    # save a frame set for inspection
    out = os.path.join(root, "logs")
    cv2.imwrite(f"{out}/probe_nav.png", cv2.cvtColor(br.rgb_nav, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{out}/probe_daa.png", cv2.cvtColor(br.rgb_daa, cv2.COLOR_RGB2BGR))
    d = br.depth_nav.copy(); m = np.isfinite(d) & (d > 0)
    vis = np.zeros(d.shape, np.uint8)
    if m.any():
        vis[m] = (255*(1 - np.clip(d[m]/CFG.sen.nav_far, 0, 1))).astype(np.uint8)
    cv2.imwrite(f"{out}/probe_depth.png", cv2.applyColorMap(vis, cv2.COLORMAP_TURBO))
    print("saved probe_nav.png / probe_daa.png / probe_depth.png")
finally:
    br.close()
