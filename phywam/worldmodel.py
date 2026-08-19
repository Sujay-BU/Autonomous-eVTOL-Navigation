"""
Physics-Informed Recurrent State-Space Model (PI-RSSM).

Two coupled predictors sharing one recurrent state:

  * a categorical RSSM that models the VISUAL world -- buildings, ground,
    traffic, how the scene flows as the aircraft moves;
  * a physics-informed head that models the AIRCRAFT -- the analytic
    momentum-theory/drag-polar model plus a learned residual.

Splitting them this way is the central design choice. Rigid-body flight
dynamics are known in closed form, so making a network rediscover them from
pixels wastes both capacity and samples. Conversely no closed form predicts
what a camera will see next. Each half gets the representation it deserves,
and the residual network is left with exactly the physics the analytic model
omits: ground effect, rotor-wing interference, actuator lag and turbulence.

Follows the DreamerV3 recipe for the stochastic part (categorical latents with
straight-through gradients, symlog targets, two-hot reward, KL balancing with
free nats) because that combination is what makes a single hyper-parameter set
train stably across very different signal scales.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CFG
from .physics import AnalyticEVTOL

L = CFG.lrn

# Distribution argument validation calls .all() and branches on the result,
# which forces a device->host sync. That is merely slow in training and is
# outright illegal inside a CUDA graph capture, which the planner relies on.
torch.distributions.Distribution.set_default_validate_args(False)


# ------------------------------------------------------------------ helpers --
def symlog(x):  return torch.sign(x) * torch.log1p(torch.abs(x))
def symexp(x):  return torch.sign(x) * torch.expm1(torch.abs(x))


class TwoHot:
    """Two-hot encoding over a symlog-spaced support.

    Regressing reward with an MSE head is badly behaved when returns span
    orders of magnitude. Predicting a distribution over a fixed support and
    training with cross-entropy makes the loss scale-free.
    """
    def __init__(self, lo=-12.0, hi=12.0, n=41, device="cpu"):
        self.bins = torch.linspace(lo, hi, n, device=device)
        self.n = n

    def encode(self, x):
        x = symlog(x).clamp(self.bins[0], self.bins[-1]).unsqueeze(-1)
        d = (x - self.bins).abs()
        i = d.argmin(-1)
        below = (self.bins[i] <= x.squeeze(-1))
        lo = torch.where(below, i, (i - 1).clamp(min=0))
        hi = (lo + 1).clamp(max=self.n - 1)
        wl = (self.bins[hi] - x.squeeze(-1)) / (self.bins[hi] - self.bins[lo] + 1e-8)
        wl = wl.clamp(0, 1)
        out = torch.zeros(*x.shape[:-1], self.n, device=x.device)
        out.scatter_(-1, lo.unsqueeze(-1), wl.unsqueeze(-1))
        out.scatter_add_(-1, hi.unsqueeze(-1), (1 - wl).unsqueeze(-1))
        return out

    def decode(self, logits):
        return symexp((logits.softmax(-1) * self.bins).sum(-1))


def mlp(i, h, o, layers=2, out_zero=False):
    m, d = [], i
    for _ in range(layers):
        m += [nn.Linear(d, h), nn.LayerNorm(h), nn.SiLU()]
        d = h
    last = nn.Linear(d, o)
    if out_zero:
        nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
    m.append(last)
    return nn.Sequential(*m)


# ------------------------------------------------------------------ encoder --
class Encoder(nn.Module):
    """RGB+depth CNN, proprioception MLP and threat-track MLP -> one embedding."""

    def __init__(self):
        super().__init__()
        d = L.cnn_depth
        self.conv = nn.Sequential(
            nn.Conv2d(4, d, 4, 2, 1),      nn.GroupNorm(4, d),      nn.SiLU(),
            nn.Conv2d(d, 2*d, 4, 2, 1),    nn.GroupNorm(4, 2*d),    nn.SiLU(),
            nn.Conv2d(2*d, 4*d, 4, 2, 1),  nn.GroupNorm(4, 4*d),    nn.SiLU(),
            nn.Conv2d(4*d, 8*d, 4, 2, 1),  nn.GroupNorm(4, 8*d),    nn.SiLU())
        self.img_out = nn.Linear(8*d * 4 * 4, 512)
        self.pro = mlp(L.proprio_dim, 128, 128, 1)
        self.trk = mlp(CFG.sen.max_tracks * 8, 128, 128, 1)
        self.dim = 512 + 128 + 128

    def forward(self, img, pro, trk):
        B = img.shape[0]
        f = self.conv(img).reshape(B, -1)
        return torch.cat([self.img_out(f),
                          self.pro(symlog(pro)),
                          self.trk(trk.reshape(B, -1))], -1)


class ImageDecoder(nn.Module):
    """Reconstructs RGB+depth. Its output is also what the XAI panel shows as
    'what the model believes the world looks like'."""

    def __init__(self, feat):
        super().__init__()
        d = L.cnn_depth
        self.fc = nn.Linear(feat, 8*d * 4 * 4)
        self.d = d
        self.net = nn.Sequential(
            nn.ConvTranspose2d(8*d, 4*d, 4, 2, 1), nn.GroupNorm(4, 4*d), nn.SiLU(),
            nn.ConvTranspose2d(4*d, 2*d, 4, 2, 1), nn.GroupNorm(4, 2*d), nn.SiLU(),
            nn.ConvTranspose2d(2*d, d, 4, 2, 1),   nn.GroupNorm(4, d),   nn.SiLU(),
            nn.ConvTranspose2d(d, 4, 4, 2, 1))

    def forward(self, f):
        x = self.fc(f).reshape(-1, 8*self.d, 4, 4)
        return self.net(x)


# --------------------------------------------------------------------- RSSM --
class RSSM(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.dt = L.deter
        self.nc, self.ncl = L.stoch_cat, L.stoch_cls
        self.zs = self.nc * self.ncl
        self.in_proj = nn.Sequential(
            nn.Linear(self.zs + L.act_dim, L.hidden), nn.LayerNorm(L.hidden), nn.SiLU())
        self.gru = nn.GRUCell(L.hidden, self.dt)
        self.prior = mlp(self.dt, L.hidden, self.zs, 1)
        self.post = mlp(self.dt + embed_dim, L.hidden, self.zs, 1)

    def initial(self, B, device):
        return (torch.zeros(B, self.dt, device=device),
                torch.zeros(B, self.zs, device=device))

    def _probs(self, logits):
        """Unimix: blend 1% uniform into every categorical so no class can
        collapse to exactly zero probability and kill its gradient."""
        lg = logits.reshape(*logits.shape[:-1], self.nc, self.ncl)
        p = lg.softmax(-1)
        return 0.99 * p + 0.01 / self.ncl

    def _dist(self, logits):
        return torch.distributions.OneHotCategoricalStraightThrough(
            probs=self._probs(logits))

    def img_step(self, h, z, a):
        """Prior transition: where the model thinks it will end up, unaided."""
        # GRUCell is not autocast-eligible, so under bf16 the input arrives
        # half-precision while the hidden state is still fp32. The cell is a
        # tiny fraction of the rollout cost, so pin it to fp32 rather than
        # scatter casts through the caller.
        inp = self.in_proj(torch.cat([z, a.to(z.dtype)], -1))
        with torch.autocast("cuda", enabled=False):
            h = self.gru(inp.float(), h.float())
        lg = self.prior(h)
        z = self._dist(lg).rsample().reshape(h.shape[0], -1)
        return h, z, lg

    def obs_step(self, h, z, a, e):
        """Posterior: correct the prior with the observation actually seen."""
        h, _, prior_lg = self.img_step(h, z, a)
        post_lg = self.post(torch.cat([h, e.to(h.dtype)], -1))
        z = self._dist(post_lg).rsample().reshape(h.shape[0], -1)
        return h, z, prior_lg, post_lg

    def kl(self, post_lg, prior_lg):
        """KL balancing: the representation and the dynamics are pulled toward
        each other at different rates so neither collapses onto the other."""
        q = self._probs(post_lg)          # posterior  (.., nc, ncl)
        p = self._probs(prior_lg)         # prior
        lq, lp = q.log(), p.log()
        # KL(q || p) summed over classes, then over the nc independent groups.
        # "dyn" stops gradient on q (train the prior toward the posterior);
        # "rep" stops gradient on p (train the posterior toward the prior).
        kl_dyn = (q.detach() * (lq.detach() - lp)).sum(-1).sum(-1)
        kl_rep = (q * (lq - lp.detach())).sum(-1).sum(-1)
        free = L.kl_free
        return kl_dyn.clamp(min=free), kl_rep.clamp(min=free)


# --------------------------------------------- physics-informed dynamics head --
class PhysicsHead(nn.Module):
    """x_{t+1} = RK2(f_phys)(x_t, a_t) + dt * g_theta(h, z, x, a)

    The residual is initialised to exactly zero, so at step 0 of training the
    world model already predicts flight correctly from first principles and
    learning only ever has to improve on that. This is what buys the sample
    efficiency: there is no phase where the agent has to discover gravity.
    """

    def __init__(self, feat):
        super().__init__()
        self.phys = AnalyticEVTOL()
        self.res = mlp(feat + 12 + L.act_dim, 256, 12, 2, out_zero=True)
        self.log_scale = nn.Parameter(torch.full((12,), -2.0))

    def forward(self, h, z, x, a, dt, bias=None):
        nom = self.phys.step(x, a, dt, bias)
        g = self.res(torch.cat([h, z, x, a], -1)) * self.log_scale.exp()
        return nom + dt * g, nom, g


# ---------------------------------------------------------------- full model --
class WorldModel(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.enc = Encoder()
        self.rssm = RSSM(self.enc.dim)
        feat = L.deter + self.rssm.zs
        self.dec_img = ImageDecoder(feat)
        self.dec_pro = mlp(feat, 384, L.proprio_dim)
        self.dec_trk = mlp(feat, 384, CFG.sen.max_tracks * 8)
        self.head_rew = mlp(feat, 384, 41)
        self.head_con = mlp(feat, 384, 1)
        self.head_clr = mlp(feat, 384, 41)       # predicted min clearance
        self.phys = PhysicsHead(feat)
        self.twohot = TwoHot(device=device)
        self.feat_dim = feat
        self.device = device

    @staticmethod
    def feat(h, z):
        return torch.cat([h, z], -1)

    # ------------------------------------------------------------ observe ----
    def observe(self, img, pro, trk, act):
        """Roll the posterior over a batch of sequences. Shapes (B,T,...)."""
        B, T = img.shape[:2]
        e = self.enc(img.reshape(B*T, *img.shape[2:]),
                     pro.reshape(B*T, -1), trk.reshape(B*T, *trk.shape[2:]))
        e = e.reshape(B, T, -1)
        h, z = self.rssm.initial(B, img.device)
        hs, zs, pri, pos = [], [], [], []
        for t in range(T):
            h, z, plg, qlg = self.rssm.obs_step(h, z, act[:, t], e[:, t])
            hs.append(h); zs.append(z); pri.append(plg); pos.append(qlg)
        st = lambda l: torch.stack(l, 1)
        return st(hs), st(zs), st(pri), st(pos)

    # ------------------------------------------------------------ imagine ----
    @torch.no_grad()
    def imagine(self, h, z, actions, x=None, dt=None):
        """Prior-only rollout used by the planner. actions: (B,H,A)."""
        dt = dt or 1.0 / L.ctrl_hz
        H = actions.shape[1]
        hs, zs, xs = [], [], []
        for t in range(H):
            a = actions[:, t]
            if x is not None:
                x, _, _ = self.phys(h, z, x, a, dt)
                xs.append(x)
            h, z, _ = self.rssm.img_step(h, z, a)
            hs.append(h); zs.append(z)
        out = (torch.stack(hs, 1), torch.stack(zs, 1))
        return out + ((torch.stack(xs, 1),) if x is not None else ())

    def decode_heads(self, f):
        return dict(reward=self.twohot.decode(self.head_rew(f)),
                    cont=torch.sigmoid(self.head_con(f)).squeeze(-1),
                    clr=self.twohot.decode(self.head_clr(f)))
