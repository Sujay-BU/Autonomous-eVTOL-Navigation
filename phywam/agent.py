"""
Actor and critic, trained entirely inside the world model's imagination.

The planner does the real decision-making at runtime, so why train a policy at
all? Two reasons, both about the planner's blind spots:

  * MPPI can only see 3 seconds ahead. The critic supplies the value of the
    state it lands in, which is how a 3 s planner ends up behaving sensibly
    over a 3 km flight.
  * Sampling 512 sequences from a zero-mean Gaussian wastes most of them on
    obviously bad behaviour. Seeding a quarter of the population from the
    policy concentrates samples where the good trajectories actually are.

Training happens on imagined rollouts rather than replayed ones, so it costs no
simulator time -- the expensive resource here.
"""
import numpy as np
import torch
import torch.nn as nn

from .config import CFG
from .worldmodel import mlp, symlog, symexp, TwoHot

L = CFG.lrn


class Actor(nn.Module):
    def __init__(self, feat):
        super().__init__()
        self.net = mlp(feat, 512, 2 * L.act_dim, 3)
        self.min_std, self.max_std = 0.08, 0.9

    def dist(self, f):
        o = self.net(f)
        mu, ls = o.chunk(2, -1)
        mu = torch.tanh(mu)
        std = self.min_std + (self.max_std - self.min_std) * torch.sigmoid(ls)
        return torch.distributions.Normal(mu, std)

    def forward(self, f, sample=True):
        d = self.dist(f)
        a = d.rsample() if sample else d.mean
        return a.clamp(-1, 1)


class Critic(nn.Module):
    """Distributional value head over a symlog two-hot support -- the same
    trick as the reward head, for the same reason: returns here span from
    single-digit shaping terms to a +60 goal bonus."""

    def __init__(self, feat, device="cuda"):
        super().__init__()
        self.net = mlp(feat, 512, 41, 3)
        self.th = TwoHot(device=device)

    def forward(self, f):
        return self.th.decode(self.net(f))

    def logits(self, f):
        return self.net(f)


def lambda_return(rew, val, cont, lam=0.95):
    """Discounted lambda-return computed backwards through the imagined roll.

    cont is the model's own predicted continuation probability, so an imagined
    trajectory that the model believes ends in a crash stops accumulating value
    at that point without any hand-written episode logic.
    """
    T = rew.shape[1]
    out = [None] * T
    nxt = val[:, -1]
    for t in reversed(range(T)):
        nxt = rew[:, t] + cont[:, t] * ((1 - lam) * val[:, t] + lam * nxt)
        out[t] = nxt
    return torch.stack(out, 1)


class ImagTrainer:
    """One imagination update: roll the actor forward in latent space, score
    with the critic, then push both toward the lambda-return."""

    def __init__(self, wm, actor, critic, device="cuda", horizon=15):
        self.wm, self.actor, self.critic = wm, actor, critic
        self.horizon = horizon
        self.dev = device
        self.tgt = Critic(wm.feat_dim, device).to(device)
        self.tgt.load_state_dict(critic.state_dict())
        for p in self.tgt.parameters():
            p.requires_grad_(False)
        self.opt_a = torch.optim.Adam(actor.parameters(), L.lr_actor, eps=1e-6)
        self.opt_c = torch.optim.Adam(critic.parameters(), 3e-4, eps=1e-6)
        self.ret_lo = self.ret_hi = None

    def _norm(self, ret):
        """Scale advantages by the 5-95 percentile spread of returns, tracked
        with an EMA. Fixes the actor's effective learning rate as the reward
        scale drifts over training."""
        lo = torch.quantile(ret.detach(), 0.05)
        hi = torch.quantile(ret.detach(), 0.95)
        if self.ret_lo is None:
            self.ret_lo, self.ret_hi = lo, hi
        else:
            self.ret_lo = 0.99 * self.ret_lo + 0.01 * lo
            self.ret_hi = 0.99 * self.ret_hi + 0.01 * hi
        return torch.clamp(self.ret_hi - self.ret_lo, min=1.0)

    def update(self, h, z):
        """h,z: posterior states (B,T,*) from the world-model update."""
        wm = self.wm
        B, T = h.shape[:2]
        hs = h.reshape(B*T, -1).detach()
        zs = z.reshape(B*T, -1).detach()

        feats, acts, logps, ents = [], [], [], []
        for _ in range(self.horizon):
            f = wm.feat(hs, zs)
            d = self.actor.dist(f)
            a = d.rsample().clamp(-1, 1)
            feats.append(f); acts.append(a)
            logps.append(d.log_prob(a).sum(-1))
            ents.append(d.entropy().sum(-1))
            hs, zs, _ = wm.rssm.img_step(hs, zs, a)
        feats.append(wm.feat(hs, zs))
        F = torch.stack(feats, 1)                       # (BT, H+1, feat)

        with torch.no_grad():
            heads = wm.decode_heads(F)
            rew = heads["reward"][:, :-1]
            cont = heads["cont"][:, :-1].clamp(0.0, 1.0) * 0.997
        val_t = self.tgt(F)
        ret = lambda_return(rew, val_t, cont)

        # --- critic ---------------------------------------------------------
        vl = self.critic.logits(F[:, :-1].detach())
        tgt = self.critic.th.encode(ret.detach())
        loss_c = -(tgt * torch.log_softmax(vl, -1)).sum(-1).mean()
        self.opt_c.zero_grad(set_to_none=True)
        loss_c.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 100.0)
        self.opt_c.step()

        # --- actor ----------------------------------------------------------
        scale = self._norm(ret)
        adv = (ret - val_t[:, :-1]).detach() / scale
        logp = torch.stack(logps, 1)
        ent = torch.stack(ents, 1)
        # REINFORCE on the advantage plus the straight-through pathwise term:
        # the categorical latents pass gradients, so both routes are available
        loss_a = -(logp * adv).mean() - 3e-4 * ent.mean() - 0.1 * (ret / scale).mean()
        self.opt_a.zero_grad(set_to_none=True)
        loss_a.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0)
        self.opt_a.step()

        with torch.no_grad():                            # EMA target critic
            for p, q in zip(self.tgt.parameters(), self.critic.parameters()):
                p.mul_(0.98).add_(q.detach(), alpha=0.02)

        return dict(loss_actor=float(loss_a), loss_critic=float(loss_c),
                    ret_mean=float(ret.mean()), ent=float(ent.mean()),
                    val=float(val_t.mean()))
