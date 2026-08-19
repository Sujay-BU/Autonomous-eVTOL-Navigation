"""
Memory-mapped sequence replay.

400k transitions of 64x64 RGB+depth is ~6.5 GB, which does not fit in the 15 GB
this machine has once Gazebo, the trainer and the GUI have taken their share.
Backing the buffer with numpy memmaps pushes it onto disk (of which there is
705 GB) and lets the page cache decide what stays resident. Sampling is random
across episodes, so the access pattern is scattered either way -- there is no
locality to lose.
"""
import os
import numpy as np

from .config import CFG

L = CFG.lrn


class SequenceReplay:
    def __init__(self, path, capacity=None, res=None, create=True):
        self.path = path
        self.cap = int(capacity or L.replay_cap)
        self.res = int(res or L.img_res)
        os.makedirs(path, exist_ok=True)
        spec = dict(
            img=((self.cap, 4, self.res, self.res), np.uint8),
            pro=((self.cap, L.proprio_dim), np.float32),
            trk=((self.cap, CFG.sen.max_tracks, 8), np.float32),
            act=((self.cap, L.act_dim), np.float32),
            rew=((self.cap,), np.float32),
            cont=((self.cap,), np.uint8),
            clr=((self.cap,), np.float32),
            phys=((self.cap, 12), np.float32),
            first=((self.cap,), np.uint8),
        )
        self.f = {}
        for k, (shape, dt) in spec.items():
            fn = os.path.join(path, f"{k}.npy")
            mode = "r+" if os.path.exists(fn) else "w+"
            if mode == "w+" and not create:
                raise FileNotFoundError(fn)
            self.f[k] = np.lib.format.open_memmap(fn, mode=mode,
                                                  dtype=dt, shape=shape)
        self.meta_fn = os.path.join(path, "meta.npz")
        if os.path.exists(self.meta_fn):
            m = np.load(self.meta_fn)
            self.ptr, self.size = int(m["ptr"]), int(m["size"])
        else:
            self.ptr = self.size = 0

    def save_meta(self):
        np.savez(self.meta_fn, ptr=self.ptr, size=self.size)

    # ------------------------------------------------------------------ add --
    def add(self, img, pro, trk, act, rew, cont, clr, phys, first):
        i = self.ptr
        self.f["img"][i] = img
        self.f["pro"][i] = pro
        self.f["trk"][i] = trk
        self.f["act"][i] = act
        self.f["rew"][i] = rew
        self.f["cont"][i] = cont
        self.f["clr"][i] = clr
        self.f["phys"][i] = phys
        self.f["first"][i] = first
        self.ptr = (i + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    # --------------------------------------------------------------- sample --
    def sample(self, batch=None, seq=None, rng=None):
        """Contiguous windows that never straddle the write head or an episode
        boundary (a window starting mid-episode is fine; one that crosses INTO
        a new episode is not, because the recurrent state would be nonsense)."""
        batch = batch or L.batch
        seq = seq or L.seq_len
        rng = rng or np.random
        if self.size < seq + 2:
            return None
        idx = np.empty((batch, seq), np.int64)
        n = 0
        guard = 0
        while n < batch and guard < batch * 40:
            guard += 1
            s = int(rng.integers(0, self.size - seq)) if hasattr(rng, "integers") \
                else int(rng.randint(0, self.size - seq))
            w = np.arange(s, s + seq)
            if self.size == self.cap:
                # avoid windows that span the circular write head
                rel = (w - self.ptr) % self.cap
                if rel.max() - rel.min() != seq - 1:
                    continue
            if self.f["first"][w[1:]].any():         # boundary inside window
                continue
            idx[n] = w
            n += 1
        if n < batch:
            idx[n:] = idx[:1]
        out = {}
        for k in ("pro", "trk", "act", "rew", "cont", "clr", "phys"):
            out[k] = np.asarray(self.f[k][idx])
        out["img"] = np.asarray(self.f["img"][idx], np.float32) / 255.0
        out["cont"] = out["cont"].astype(np.float32)
        return out

    @staticmethod
    def encode_img(img_f32):
        """(4,R,R) float in [0,1] -> uint8. Depth is already log-compressed by
        the environment, so a linear 8-bit quantisation costs ~0.4% of range."""
        return np.clip(img_f32 * 255.0, 0, 255).astype(np.uint8)

    def __len__(self):
        return self.size
