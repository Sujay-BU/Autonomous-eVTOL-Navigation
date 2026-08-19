"""Smoke-test the environment: reset, obs shapes, step rate, termination."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phywam.env import VertiportEnv, OUTCOME
from phywam.config import CFG

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env = VertiportEnv(os.path.join(root, "sim", "worlds", "urban_1.sdf"), seed=3)
try:
    for ep in range(2):
        o = env.reset()
        print(f"\n--- ep{ep}: VP{env.start_vp} -> VP{env.goal_vp}, "
              f"range {env.d0:.0f} m ---")
        print(f"  img {o['img'].shape} {o['img'].dtype} "
              f"[{o['img'].min():.2f},{o['img'].max():.2f}]  "
              f"pro {o['pro'].shape}  trk {o['trk'].shape}  "
              f"active tracks {int(o['trk'][:,0].sum())}")
        t0 = time.time(); tsim0 = env.br.sim_time
        # crude scripted climb-and-go so the episode does something sensible
        COL0 = 2*CFG.air.W/(CFG.air.n_rotor*CFG.air.k_rotor*CFG.air.w_rotor_max**2)-1
        R = 0
        for k in range(320):
            z = env._pos[2]
            a = np.zeros(6, np.float32)
            a[0] = np.clip(COL0 + 0.010*(120-z) - 0.020*env.br.state[9], -1, 1)
            a[5] = -1.0 if z < 90 else 0.5
            a[4] = -1.0 if z < 90 else 0.4
            a[2] = 0.15 if z > 90 else 0.0
            o, r, d, info = env.step(a)
            R += r
            if k % 80 == 0:
                print(f"  k={k:3d} t={info['t']:6.1f} z={info['pos'][2]:6.1f} "
                      f"d_goal={info['d_goal']:7.0f} clr={info['clr']:7.1f} "
                      f"sep={min(info['sep'],9999):6.0f} trk={int(o['trk'][:,0].sum())} "
                      f"R={R:7.1f}")
            if d:
                print(f"  DONE k={k} outcome={OUTCOME[info['outcome']]}")
                break
        tw, ts = time.time()-t0, env.br.sim_time-tsim0
        print(f"  steps/s wall = {(k+1)/tw:.1f}   RTF = {ts/tw:.2f}")
        print(f"  min_clr={info['min_clr']:.1f} min_sep={min(info['min_sep'],9999):.0f} "
              f"lox={info['lox']} nmac={info['nmac']}")
finally:
    env.close()
