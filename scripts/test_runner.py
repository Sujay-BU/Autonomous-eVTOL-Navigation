import os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phywam.env import VertiportEnv
from phywam.worldmodel import WorldModel
from phywam.runner import FlightRunner
from phywam.replay import SequenceReplay
torch.set_float32_matmul_precision('high')

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mode = sys.argv[1] if len(sys.argv) > 1 else "scripted"
env = VertiportEnv(os.path.join(root, "sim", "worlds", "urban_1.sdf"), seed=11)
wm = WorldModel("cuda").to("cuda").eval()
rb = SequenceReplay("/tmp/phywam_test_rb", capacity=20000)
run = FlightRunner(env, wm, device="cuda", mode=mode)
try:
    for ep in range(2):
        t0 = time.time()
        st, log = run.run(replay=rb, explore=0.0)
        wall = time.time() - t0
        print(f"ep{ep} {mode}: VP{st['start_vp']}->VP{st['goal_vp']} "
              f"direct {st['direct']:.0f} m")
        print(f"   outcome={st['outcome']:>16s} steps={st['steps']:4d} "
              f"t={st['t']:6.1f}s  return={st['ret']:8.1f}")
        print(f"   min_clr={st['min_clr']:7.1f} m  min_sep={min(st['min_sep'],9999):7.0f} m "
              f"LoWC={st['lox']:3d} NMAC={st['nmac']:2d}")
        print(f"   path={st['path_len']:7.0f} m  SoC used={100*st['soc_used']:5.1f}%  "
              f"shield engaged {100*st['shield_rate']:5.1f}% of steps")
        print(f"   plan {np.mean(st['plan_ms']):6.2f} ms/step  "
              f"wall {wall:5.1f}s  RTF {st['t']/wall:.2f}  route {len(st['route'])} wpts")
    rb.save_meta()
    print(f"replay: {len(rb)} transitions")
finally:
    env.close()
