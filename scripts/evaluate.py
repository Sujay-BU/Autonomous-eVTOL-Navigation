"""
Scored evaluation, with optional video recording.

Runs the full stack -- perception, PI-RSSM, MPPI, barrier filter -- over
randomly drawn vertiport pairs and reports the safety metrics. The recorded
runs use exactly the same code path as the unrecorded ones; recording only
attaches a renderer to the callback.
"""
import os, sys, json, time, argparse, subprocess, signal, atexit
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_float32_matmul_precision("high")

from phywam.config import CFG
from phywam.env import VertiportEnv, OUTCOME
from phywam.worldmodel import WorldModel
from phywam.agent import Actor, Critic
from phywam.runner import FlightRunner
from phywam.route import RoutePlanner
from phywam.instrument import Instrumented
from phywam.metrics import summarise, format_report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class VideoWriter:
    """Raw frames -> ffmpeg -> H.264.

    Throughput matters more than it looks. A 1600x900 bgr24 frame is 4.3 MB, so
    20 fps is 86 MB/s into the pipe. With x264 on a 'medium' preset the encoder
    falls behind, the 64 KB pipe buffer fills, writes block, and close() ends up
    killing ffmpeg with most of the backlog unencoded -- which silently
    truncates the recording to a couple of frames. Hence the fast preset and
    the generous drain timeout.
    """

    def __init__(self, path, w, h, fps):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.p = subprocess.Popen(
            ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
             "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", str(fps), "-i", "-",
             "-an", "-vcodec", "libx264", "-preset", "veryfast", "-crf", "22",
             "-pix_fmt", "yuv420p", path],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=open(path + ".ffmpeg.log", "wb"))
        self.path = path
        self.n = 0
        self.err = None

    def write(self, frame):
        try:
            self.p.stdin.write(frame.tobytes())
            self.n += 1
        except (BrokenPipeError, ValueError) as e:
            if self.err is None:
                self.err = repr(e)

    def close(self):
        try:
            self.p.stdin.close(); self.p.wait(timeout=300)
        except Exception:
            self.p.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "runs/main/ckpt.pt"))
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--mode", default="plan",
                    choices=["plan", "actor", "baseline", "scripted"])
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--outdir", default=os.path.join(ROOT, "recordings"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-time", type=float, default=200.0)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--rtf", type=float, default=0.25)
    ap.add_argument("--actor-seed", type=float, default=0.25,
                    help="fraction of MPPI samples seeded from the policy; "
                         "0 disables the actor entirely")
    ap.add_argument("--no-nominal", action="store_true",
                    help="do not seed the MPPI mean from the guidance law")
    ap.add_argument("--ablate-learned", action="store_true",
                    help="disable the physics residual and the learned "
                         "clearance head, leaving analytic dynamics + the "
                         "geometric map")
    ap.add_argument("--no-critic", action="store_true",
                    help="drop the learned terminal value from the MPPI cost")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    world = os.path.join(ROOT, "sim", "worlds", f"urban_{args.world}.sdf")
    env = VertiportEnv(world, seed=args.seed, max_time=args.max_time)
    # Slow the simulator so the full stack (MPPI + barrier + XAI + dashboard)
    # comfortably fits inside every 50 ms control period.
    env.br.set_rtf(args.rtf)
    print(f"simulator throttled to RTF {args.rtf}")
    dev = "cuda"
    wm = WorldModel(dev).to(dev).eval()
    actor = Actor(wm.feat_dim).to(dev).eval()
    critic = Critic(wm.feat_dim, dev).to(dev).eval()
    if os.path.exists(args.ckpt):
        c = torch.load(args.ckpt, map_location=dev, weights_only=False)
        wm.load_state_dict(c["wm"]); actor.load_state_dict(c["actor"])
        critic.load_state_dict(c["critic"])
        print(f"loaded {args.ckpt}  (step {c['step']}, ep {c['ep']})")
    else:
        print(f"WARNING: no checkpoint at {args.ckpt}; running untrained")

    # Make sure the simulator dies with us. Without this, a SIGTERM kills
    # Python before the finally-block runs and leaves a headless gz sim behind
    # burning ~250% CPU and 1.2 GB -- and after a few interrupted runs the
    # machine is quietly saturated by ghosts.
    def _shutdown(*_):
        try:
            env.close()
        finally:
            os._exit(1)
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    atexit.register(lambda: env.close())

    if args.ablate_learned:
        # Silence the two learned terms that feed the planner: the physics
        # residual and the clearance head. What remains is the analytic model
        # plus the geometric SDF over the mapped obstacle database -- i.e. the
        # same planner with nothing learned inside it. This isolates what the
        # learning actually bought.
        wm.phys.log_scale.data.fill_(-20.0)
        print("ABLATION: physics residual and learned hazard head disabled")

    rp = RoutePlanner(env.geom)
    run = FlightRunner(env, wm, actor, critic if not args.no_critic else None,
                       dev, route_planner=rp, mode=args.mode,
                       actor_seed_frac=args.actor_seed,
                       use_nominal=not args.no_nominal)
    if args.ablate_learned:
        run.planner.w_hazard = 0.0
    inst = Instrumented(run, dev) if args.record else None

    n_vp = len(env.geom.vp)
    results, t_all = [], time.time()
    try:
        for ep in range(args.episodes):
            s = int(rng.integers(n_vp))
            g = int(rng.choice([j for j in range(n_vp) if j != s]))
            vw = None
            cb = None
            if inst:
                inst.trail = []; inst.cache = {}
                inst.n_steps = inst.n_engaged = 0
                inst.dash.hist = {k: [] for k in inst.dash.hist}
                tag = f"{args.tag}_" if args.tag else ""
                path = os.path.join(args.outdir,
                                    f"{tag}run{ep+1}_VP{s}_to_VP{g}.mp4")
                vw = VideoWriter(path, 1600, 900, args.fps)

                stride = max(1, int(round(CFG.lrn.ctrl_hz / args.fps)))

                def cb(d, _vw=vw, _s=s, _g=g, _ep=ep, _st=stride):
                    if d["step"] % _st:
                        return
                    d["banner"] = (f"run {_ep+1}/{args.episodes}   "
                                   f"vertiport {_s} -> vertiport {_g}   "
                                   f"mode {args.mode}")
                    _vw.write(inst.callback(d))

            t0 = time.time()
            st, log = run.run(start_vp=s, goal_vp=g, callback=cb)
            st["wall"] = time.time() - t0
            st["ep"] = ep + 1
            tm = env.timing()
            st["ctrl_dt_mean"] = tm["mean"]
            st["ctrl_dt_max"] = tm["max"]
            st["ctrl_overrun_frac"] = tm["over"]
            results.append(st)
            if vw:
                vw.close()
                st["video"] = os.path.basename(vw.path)
                st["frames_written"] = vw.n
                print(f"    video: {vw.n} frames -> {os.path.basename(vw.path)}"
                      + (f"   WRITE ERROR {vw.err}" if vw.err else ""), flush=True)
            print(f"run {ep+1}: VP{s}->VP{g} direct {st['direct']:6.0f} m | "
                  f"{st['outcome']:>16s} | t={st['t']:6.1f}s "
                  f"path={st['path_len']:6.0f}m "
                  f"minObs={st['min_obs']:6.1f}m "
                  f"minSep={min(st['min_sep'],9999):6.0f}m "
                  f"LoWC={st['lox']:3d} NMAC={st['nmac']:2d} "
                  f"shield={100*st['shield_rate']:4.1f}% "
                  f"| dt {1e3*st['ctrl_dt_mean']:.1f}ms "
                  f"(overrun {100*st['ctrl_overrun_frac']:.1f}%)", flush=True)
    finally:
        env.close()

    # strip the non-serialisable diagnostics before writing JSON
    for r in results:
        r.pop("route", None)
        pm = r.pop("plan_ms", None)
        if pm:
            r["plan_ms_mean"] = float(np.mean(pm))
            r["plan_ms_p95"] = float(np.percentile(pm, 95))
    summ = summarise(results)
    os.makedirs(args.outdir, exist_ok=True)
    tag = f"{args.tag}_" if args.tag else ""
    jp = os.path.join(args.outdir, f"{tag}{args.mode}_metrics.json")
    json.dump(dict(mode=args.mode, ckpt=args.ckpt, world=args.world,
                   seed=args.seed, runs=results, summary=summ),
              open(jp, "w"), indent=2, default=float)
    rep = format_report(summ, results, args.mode)
    open(os.path.join(args.outdir, f"{tag}{args.mode}_report.txt"), "w").write(rep)
    print("\n" + rep)
    print(f"\nwrote {jp}   ({time.time()-t_all:.0f}s total)")


if __name__ == "__main__":
    main()
