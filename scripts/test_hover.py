"""Validate the plant: hover hold, then a climb, then forward transition.
Also measures real-time factor, which sets the training data budget."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phywam.bridge import GazeboBridge
from phywam.config import CFG

A = CFG.air
W_over_Tmax = A.W / (A.n_rotor * A.k_rotor * A.w_rotor_max**2)
print(f"weight/Tmax = {W_over_Tmax:.4f}  (equilibrium collective)")

world = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "sim", "worlds", "urban_1.sdf")
br = GazeboBridge(world, verbose=1)
try:
    br.wait_ready(60)
    print(f"ready: state={br.n_state} rgb={br.n_rgb} depth={br.n_depth} daa={br.n_daa}")
    print(f"rgb {br.rgb_nav.shape} depth {br.depth_nav.shape} daa {br.rgb_daa.shape}")
    z0 = br.pos[2]
    print(f"start pos {br.pos.round(1)}  z0={z0:.2f}")

    dt = 1.0 / 20.0
    iz = 0.0
    t_wall0, t_sim0 = time.time(), br.sim_time
    log = []
    N = 700                                   # 35 s of sim time
    for k in range(N):
        t = k * dt
        # --- reference profile -------------------------------------------
        if   t < 4:   z_ref, push, sched = z0 + 2.0, 0.0, 0.0
        elif t < 16:  z_ref, push, sched = 120.0,    0.0, 0.0
        elif t < 24:  z_ref, push, sched = 120.0,    0.55, min((t-16)/8, 1.0)*0.85
        else:         z_ref, push, sched = 120.0,    0.75, 0.95

        st = br.state
        z, vz = st[2], st[9]
        ez = z_ref - z
        iz = np.clip(iz + ez * dt, -40, 40)
        # feed-forward the hover collective, PID the rest
        col = W_over_Tmax + 0.0022*ez + 0.0009*iz - 0.0055*vz
        col = float(np.clip(col, 0.0, 1.0))
        roll, pitch, yaw = br.euler()
        # hold wings level; small nose-down as speed builds
        br.send(col, -0.35*roll, -0.30*pitch, -0.25*yaw, push, sched)

        if k % 40 == 0:
            va, al = st[19], np.degrees(st[20])
            log.append((t, z, st[19], np.degrees(roll), np.degrees(pitch), col, st[23]))
            print(f"t={t:5.1f}  z={z:7.2f}  Vair={va:6.2f}  "
                  f"roll={np.degrees(roll):6.2f}  pitch={np.degrees(pitch):6.2f}  "
                  f"col={col:.3f}  soc={st[23]:.4f}")
        time.sleep(dt)

    t_wall = time.time() - t_wall0
    t_sim = br.sim_time - t_sim0
    print(f"\nRTF = {t_sim/t_wall:.2f}  (sim {t_sim:.1f}s in wall {t_wall:.1f}s)")
    print(f"final pos {br.pos.round(1)} vel_body {br.vel_b.round(2)}")
    d = br.depth_nav
    fin = d[np.isfinite(d)]
    print(f"depth: {d.shape} finite {fin.size}/{d.size} "
          f"min {fin.min() if fin.size else -1:.1f} max {fin.max() if fin.size else -1:.1f}")
    print(f"traffic tracks: {br.traffic.shape}")
finally:
    br.close()
