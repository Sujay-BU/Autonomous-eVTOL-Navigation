"""
Is the MPPI cost wrong, or is the optimiser failing?

The scripted pilot completes the mission (34/34 seed episodes). MPPI, using a
world model whose one-step error is 61 % below the analytic prior, does not. So
one of two things is true:

  * the cost function does not actually prefer the behaviour that works -- in
    which case MPPI is optimising the wrong thing and finding it correctly;
  * the cost does prefer it, but the sampler never finds it -- in which case the
    search is at fault.

This scores three action sequences under the *same* cost at every control step
of a flight flown by the scripted pilot:

    scripted   the pilot's current action, held over the horizon
    mppi       whatever the planner would have chosen from this state
    trim       hover trim, held (a do-nothing reference)

If scripted beats mppi, the optimiser is broken. If mppi beats scripted while
the aircraft flies into the ground, the cost is broken. Either answer is
actionable; guessing is not.
"""
import os, sys, argparse
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
torch.set_float32_matmul_precision("high")

from phywam.config import CFG
from phywam.env import VertiportEnv
from phywam.worldmodel import WorldModel
from phywam.agent import Actor, Critic
from phywam.runner import FlightRunner
from phywam.route import RoutePlanner, RouteTracker
from phywam.xai import CostAttribution, COST_TERMS

L = CFG.lrn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "runs/main/ckpt.pt"))
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--start", type=int, default=5)
    ap.add_argument("--goal", type=int, default=3)
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--every", type=int, default=60)
    ap.add_argument("--rtf", type=float, default=1.0)
    a = ap.parse_args()

    dev = "cuda"
    world = os.path.join(ROOT, "sim", "worlds", f"urban_{a.world}.sdf")
    env = VertiportEnv(world, seed=5, max_time=160.0)
    env.br.set_rtf(a.rtf)
    wm = WorldModel(dev).to(dev).eval()
    actor = Actor(wm.feat_dim).to(dev).eval()
    critic = Critic(wm.feat_dim, dev).to(dev).eval()
    c = torch.load(a.ckpt, map_location=dev, weights_only=False)
    wm.load_state_dict(c["wm"]); actor.load_state_dict(c["actor"])
    critic.load_state_dict(c["critic"])
    print(f"ckpt step {c['step']} ep {c['ep']}")

    rp = RoutePlanner(env.geom)
    run = FlightRunner(env, wm, actor, critic, dev, route_planner=rp,
                       mode="scripted", actor_seed_frac=0.0)
    pl = run.planner
    attr = CostAttribution(pl)

    obs = env.reset(a.start, a.goal)
    wpts = rp.plan(env._pos, env.goal, CFG.wld.corridor_alt,
                   land_z=env.goal[2] + 2.5)
    trk = RouteTracker(wpts)
    pl.reset()
    h, z = wm.rssm.initial(1, dev)
    act = np.zeros(L.act_dim, np.float32)
    prev_x = env.phys_state.copy()

    print(f"\n route: {len(wpts)} waypoints, direct "
          f"{np.linalg.norm(env.goal[:2]-env._pos[:2]):.0f} m")
    print(f"\n{'step':>5} {'d_goal':>7} {'z':>6} {'V':>6} "
          f"{'C_script':>9} {'C_mppi':>9} {'C_trim':>9}  verdict")

    rows = []
    for k in range(a.steps):
        sg = trk.update(env._pos)
        env.set_nav(sg, trk.remaining(env._pos))
        h, z = run._encode(obs, h, z, act)
        if k > 0:
            pl.update_bias(prev_x, act, env.br.state[16:19], 1.0/L.ctrl_hz)
        prev_x = env.phys_state.copy()
        x0 = torch.as_tensor(env.phys_state, device=dev).unsqueeze(0)
        tracks = env.fuser.last_world

        a_scr = run._scripted(env, sg)

        if k % a.every == 0:
            pl.set_obstacles(env._pos)
            pl.plan(h, z, x0, sg, tracks, env.goal)          # fills statics
            sgt = torch.as_tensor(np.asarray(sg, np.float32), device=dev)
            gft = torch.as_tensor(np.asarray(env.goal, np.float32), device=dev)
            args = (pl._obs_t, pl.s_tp, pl.s_tv, pl.s_tr, sgt, gft, pl.s_vp,
                    pl.s_bi)

            def cost_of(seq):
                U = seq.unsqueeze(0)
                t, _ = attr(h, z, x0, U, *args)
                return sum(t.values()), t

            U_scr = torch.as_tensor(a_scr, device=dev).repeat(pl.H, 1)
            U_trim = pl.a_trim.repeat(pl.H, 1)
            c_scr, t_scr = cost_of(U_scr)
            c_mppi, t_mppi = cost_of(pl.mean)
            c_trim, _ = cost_of(U_trim)
            d = float(np.linalg.norm(env.goal[:2] - env._pos[:2]))
            verdict = "COST WRONG (mppi<script)" if c_mppi < c_scr \
                      else "SEARCH WEAK (script<mppi)"
            print(f"{k:5d} {d:7.0f} {env._pos[2]:6.1f} {env.br.state[19]:6.1f} "
                  f"{c_scr:9.1f} {c_mppi:9.1f} {c_trim:9.1f}  {verdict}")
            rows.append((k, c_scr, c_mppi, t_scr, t_mppi))

        obs, r, done, info = env.step(a_scr)
        act = a_scr
        if done:
            print(f"\n scripted pilot outcome: {info['outcome']} at step {k}")
            break

    if rows:
        n_wrong = sum(1 for _, cs, cm, _, _ in rows if cm < cs)
        print(f"\n mppi plan scored better than the working pilot in "
              f"{n_wrong}/{len(rows)} samples")
        print("\n where the two differ most (mean over samples):")
        acc = {t: [0.0, 0.0] for t in COST_TERMS}
        for _, _, _, ts, tm in rows:
            for t in COST_TERMS:
                acc[t][0] += ts.get(t, 0.0) / len(rows)
                acc[t][1] += tm.get(t, 0.0) / len(rows)
        diffs = sorted(acc.items(), key=lambda kv: -abs(kv[1][0] - kv[1][1]))
        print(f"   {'term':>14} {'scripted':>10} {'mppi':>10} {'delta':>10}")
        for t, (vs, vm) in diffs[:9]:
            print(f"   {t:>14} {vs:10.2f} {vm:10.2f} {vm-vs:+10.2f}")
    env.close()


if __name__ == "__main__":
    main()
