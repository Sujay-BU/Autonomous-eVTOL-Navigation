"""
Phy-WAM training.

  phase 0   seed the replay with a scripted pilot, so the world model sees
            flight, terrain and a few crashes before anything is learned
  phase 1   loop: collect one episode with the actor (fast, ~2 ms/step),
                  then run gradient steps proportional to what was collected
            periodically evaluate with the full MPPI + barrier stack

Collection uses the actor rather than MPPI because MPPI costs 45 ms/step
against the actor's 2 ms; spending the wall clock on gradient steps instead of
planning is worth far more at this stage. MPPI is what flies at evaluation.
"""
import os, sys, time, json, argparse, signal
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_float32_matmul_precision("high")

from phywam.config import CFG
from phywam.env import VertiportEnv
from phywam.worldmodel import WorldModel, symlog
from phywam.agent import Actor, Critic, ImagTrainer
from phywam.replay import SequenceReplay
from phywam.runner import FlightRunner
from phywam.route import RoutePlanner

L, SF = CFG.lrn, CFG.saf
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# normalisation for the 12-dim physics state, so the physics loss weights
# position, attitude, velocity and rate errors on comparable scales
PHYS_SCALE = torch.tensor([50., 50., 30., 0.5, 0.5, 3.14,
                           15., 6., 6., 0.6, 0.6, 0.6])


class Trainer:
    def __init__(self, args):
        self.a = args
        self.dev = "cuda"
        self.run_dir = os.path.join(ROOT, "runs", args.name)
        os.makedirs(self.run_dir, exist_ok=True)

        world = os.path.join(ROOT, "sim", "worlds", f"urban_{args.world}.sdf")
        self.env = VertiportEnv(world, seed=args.seed, max_time=args.max_time)
        self.wm = WorldModel(self.dev).to(self.dev)
        self.actor = Actor(self.wm.feat_dim).to(self.dev)
        self.critic = Critic(self.wm.feat_dim, self.dev).to(self.dev)
        self.imag = ImagTrainer(self.wm, self.actor, self.critic, self.dev,
                                horizon=args.imag_horizon)
        self.opt = torch.optim.Adam(self.wm.parameters(), L.lr_world, eps=1e-6)
        self.scaler = torch.amp.GradScaler("cuda")
        self.rb = SequenceReplay(os.path.join(self.run_dir, "replay"))
        self.rp = RoutePlanner(self.env.geom)
        self.runner = FlightRunner(self.env, self.wm, self.actor, self.critic,
                                   self.dev, route_planner=self.rp, mode="actor")
        self.ps = PHYS_SCALE.to(self.dev)
        self.step = 0
        self.ep = 0
        self.hist = []
        self.t0 = time.time()
        self.stop = False
        signal.signal(signal.SIGTERM, self._sig)
        signal.signal(signal.SIGINT, self._sig)
        if args.resume and os.path.exists(self.ckpt_path):
            self.load()

    def _sig(self, *_):
        print("\n[train] stop requested, checkpointing...", flush=True)
        self.stop = True

    @property
    def ckpt_path(self):
        return os.path.join(self.run_dir, "ckpt.pt")

    # -------------------------------------------------------- world model --
    def wm_loss(self, b):
        t = lambda k, d=torch.float32: torch.as_tensor(b[k], dtype=d,
                                                       device=self.dev)
        img, pro, trk = t("img"), t("pro"), t("trk")
        act, rew, cont = t("act"), t("rew"), t("cont")
        clr, phys = t("clr"), t("phys")
        B, T = img.shape[:2]

        h, z, pri, pos = self.wm.observe(img, pro, trk, act)
        f = self.wm.feat(h, z)
        ff = f.reshape(B*T, -1)

        rec = self.wm.dec_img(ff).reshape(B, T, 4, L.img_res, L.img_res)
        l_img = F.mse_loss(rec, img)
        l_pro = F.mse_loss(self.wm.dec_pro(f), symlog(pro))
        l_trk = F.mse_loss(self.wm.dec_trk(f).reshape(B, T, -1),
                           trk.reshape(B, T, -1))

        lr_ = self.wm.head_rew(f)
        l_rew = -(self.wm.twohot.encode(rew) * torch.log_softmax(lr_, -1)).sum(-1).mean()
        l_con = F.binary_cross_entropy_with_logits(
            self.wm.head_con(f).squeeze(-1), cont)
        lc_ = self.wm.head_clr(f)
        l_clr = -(self.wm.twohot.encode(clr) * torch.log_softmax(lc_, -1)).sum(-1).mean()

        kd, kr = self.wm.rssm.kl(pos, pri)
        l_kl = (L.kl_scale * kd + L.rep_scale * kr).mean()

        # --- physics-informed dynamics ------------------------------------
        # Predict x_{t+1} from x_t and compare against what the plant did. The
        # residual is penalised separately: given two models that fit the data
        # equally well we want the one that leans on the analytic term, because
        # that is the part that extrapolates.
        x_t = phys[:, :-1].reshape(-1, 12)
        a_t = act[:, 1:].reshape(-1, L.act_dim)
        h_t = h[:, :-1].reshape(-1, L.deter).detach()
        z_t = z[:, :-1].reshape(-1, self.wm.rssm.zs).detach()
        x_n, nom, g = self.wm.phys(h_t, z_t, x_t, a_t, 1.0 / L.ctrl_hz)
        x_true = phys[:, 1:].reshape(-1, 12)
        err = (x_n - x_true) / self.ps
        l_phys = err.pow(2).mean()
        l_nom = ((nom - x_true) / self.ps).pow(2).mean().detach()
        l_res = g.pow(2).mean()

        loss = (self.a.w_img * l_img + l_pro + l_trk + l_rew + l_con
                + self.a.w_clr * l_clr + l_kl
                + self.a.w_phys * l_phys + self.a.w_res * l_res)
        m = dict(img=float(l_img), pro=float(l_pro), trk=float(l_trk),
                 rew=float(l_rew), con=float(l_con), clr=float(l_clr),
                 kl=float(l_kl), phys=float(l_phys), nom=float(l_nom),
                 res=float(l_res), loss=float(loss))
        return loss, m, h.detach(), z.detach()

    def train_step(self):
        b = self.rb.sample()
        if b is None:
            return None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, m, h, z = self.wm_loss(b)
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(self.wm.parameters(), L.grad_clip)
        self.scaler.step(self.opt)
        self.scaler.update()
        m.update(self.imag.update(h, z))
        self.step += 1
        return m

    # -------------------------------------------------------------- loops --
    def collect(self, mode, explore, start_frac=None):
        self.runner.mode = mode
        st, _ = self.runner.run(replay=self.rb, explore=explore,
                                start_frac=start_frac)
        self.ep += 1
        return st

    def curriculum(self):
        """Walk the spawn point back from short final to the departure pad.
        Returns None once the agent should be flying the whole mission."""
        a = self.a
        prog = max(0.0, (self.ep - a.seed_eps) / a.curriculum_eps)
        if prog >= 1.0:
            return None
        if np.random.rand() < 0.35 * (1.0 - prog):
            return None                    # keep some full missions throughout
        hi = 0.94 - 0.94 * prog
        return float(np.random.uniform(max(hi - 0.28, 0.0), hi))

    def save(self):
        torch.save(dict(wm=self.wm.state_dict(), actor=self.actor.state_dict(),
                        critic=self.critic.state_dict(),
                        opt=self.opt.state_dict(), step=self.step, ep=self.ep,
                        hist=self.hist), self.ckpt_path)
        self.rb.save_meta()
        json.dump(self.hist, open(os.path.join(self.run_dir, "hist.json"), "w"))

    def load(self):
        c = torch.load(self.ckpt_path, map_location=self.dev, weights_only=False)
        self.wm.load_state_dict(c["wm"]); self.actor.load_state_dict(c["actor"])
        self.critic.load_state_dict(c["critic"]); self.opt.load_state_dict(c["opt"])
        self.step, self.ep, self.hist = c["step"], c["ep"], c.get("hist", [])
        print(f"[train] resumed at step {self.step}, ep {self.ep}", flush=True)

    def run(self):
        a = self.a
        print(f"[train] seeding with {a.seed_eps} scripted episodes", flush=True)
        while self.ep < a.seed_eps and not self.stop:
            sf = None if self.ep % 3 == 0 else float(np.random.uniform(0.55, 0.94))
            st = self.collect("scripted", 0.08, sf)
            print(f"  seed ep{self.ep:3d} {st['outcome']:>14s} "
                  f"steps={st['steps']:4d} frac={-1 if sf is None else sf:.2f} "
                  f"buf={len(self.rb)}", flush=True)

        print(f"[train] main loop, budget {a.hours:.1f} h", flush=True)
        last_save = time.time()
        while not self.stop and (time.time() - self.t0) < a.hours * 3600:
            # Hand over from the scripted pilot to the actor gradually. A
            # freshly initialised actor crashes within seconds, and a replay
            # full of 3-second episodes teaches the world model nothing about
            # cruise, transition or approach.
            frac = min(1.0, max(0.0, (self.ep - a.seed_eps) / a.handover))
            use_actor = np.random.rand() < frac
            mode = "actor" if use_actor else "scripted"
            st = self.collect(mode,
                              a.explore if use_actor else a.explore_scripted,
                              self.curriculum())
            n_grad = max(20, int(st["steps"] / a.replay_ratio))
            ms = []
            for _ in range(n_grad):
                m = self.train_step()
                if m: ms.append(m)
                if self.stop: break
            if ms:
                avg = {k: float(np.mean([x[k] for x in ms])) for k in ms[0]}
                rec = dict(ep=self.ep, step=self.step, t=time.time()-self.t0,
                           **{k: st[k] for k in ("outcome", "success", "steps",
                                                 "ret", "min_obs", "min_sep",
                                                 "lox", "nmac", "path_len",
                                                 "shield_rate", "soc_used")},
                           **avg)
                self.hist.append(rec)
                print(f"ep{self.ep:4d} step{self.step:6d} "
                      f"{st['outcome']:>14s} R={st['ret']:8.1f} "
                      f"obs={st['min_obs']:6.1f} sep={min(st['min_sep'],9999):6.0f} "
                      f"| img={avg['img']:.4f} kl={avg['kl']:.2f} "
                      f"phys={avg['phys']:.4f}(nom {avg['nom']:.4f}) "
                      f"V={avg['val']:7.2f} {mode[:3]} | "
                      f"{(time.time()-self.t0)/60:.0f}m",
                      flush=True)
            if time.time() - last_save > 600:
                self.save(); last_save = time.time()
        self.save()
        print(f"[train] done: {self.ep} episodes, {self.step} grad steps, "
              f"{(time.time()-self.t0)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="main")
    p.add_argument("--world", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hours", type=float, default=10.0)
    p.add_argument("--seed-eps", type=int, default=24)
    p.add_argument("--replay-ratio", type=float, default=3.0)
    p.add_argument("--explore", type=float, default=0.28)
    # The scripted pilot is deterministic; without injected noise the
    # world model only ever sees one narrow slice of state space, and
    # then meets something quite different when MPPI takes over.
    p.add_argument("--explore-scripted", type=float, default=0.18)
    p.add_argument("--imag-horizon", type=int, default=15)
    p.add_argument("--handover", type=float, default=140.0)
    p.add_argument("--curriculum-eps", type=float, default=420.0)
    p.add_argument("--max-time", type=float, default=170.0)
    p.add_argument("--w-img", type=float, default=12.0)
    p.add_argument("--w-clr", type=float, default=1.5)
    p.add_argument("--w-phys", type=float, default=6.0)
    p.add_argument("--w-res", type=float, default=0.02)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    t = Trainer(args)
    try:
        t.run()
    finally:
        t.env.close()
