"""
Dashboard renderer.

Composes one 1600x900 BGR frame from a control step. The live GUI displays
these frames and the recorder writes the same frames to disk, so the video is
not a reconstruction of what the operator saw -- it is exactly what they saw.
"""
import math
import numpy as np
import cv2

from .config import CFG
from .xai import COST_TERMS

SF, S = CFG.saf, CFG.sen
W, H = 1600, 900

BG      = (20, 19, 17)
PANEL   = (34, 32, 29)
EDGE    = (58, 55, 50)
TXT     = (232, 228, 222)
DIM     = (150, 146, 140)
ACCENT  = (66, 168, 238)      # BGR amber-blue
GOOD    = (120, 205, 130)
WARN    = (70, 190, 240)
BAD     = (72, 76, 240)
VIOLET  = (220, 130, 170)

F = cv2.FONT_HERSHEY_SIMPLEX
FD = cv2.FONT_HERSHEY_DUPLEX


def _t(im, s, x, y, sc=0.42, c=TXT, th=1, f=F):
    cv2.putText(im, s, (int(x), int(y)), f, sc, c, th, cv2.LINE_AA)


def _panel(im, x, y, w, h, title=None):
    cv2.rectangle(im, (x, y), (x+w, y+h), PANEL, -1)
    cv2.rectangle(im, (x, y), (x+w, y+h), EDGE, 1)
    if title:
        cv2.rectangle(im, (x, y), (x+w, y+19), (44, 42, 38), -1)
        _t(im, title, x+7, y+14, 0.40, ACCENT, 1)
    return x, y + (21 if title else 2)


def _blit(im, img, x, y, w, h):
    if img is None or img.size == 0:
        return
    r = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST
                   if img.shape[0] < h//2 else cv2.INTER_AREA)
    if r.ndim == 2:
        r = cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)
    im[y:y+h, x:x+w] = r


def _bar(im, x, y, w, h, frac, col, bg=(52, 50, 46)):
    cv2.rectangle(im, (x, y), (x+w, y+h), bg, -1)
    f = float(np.clip(frac, 0, 1))
    if f > 0:
        cv2.rectangle(im, (x, y), (x+int(w*f), y+h), col, -1)


def _spark(im, x, y, w, h, series, lo=None, hi=None, col=ACCENT, ref=None):
    cv2.rectangle(im, (x, y), (x+w, y+h), (26, 25, 23), -1)
    if not series:
        return
    v = np.asarray(series[-400:], np.float32)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return
    lo = np.min(v) if lo is None else lo
    hi = np.max(v) if hi is None else hi
    if hi - lo < 1e-6:
        hi = lo + 1.0
    xs = np.linspace(x, x+w, v.size).astype(np.int32)
    ys = (y + h - (v - lo) / (hi - lo) * h).clip(y, y+h).astype(np.int32)
    if ref is not None and lo <= ref <= hi:
        yr = int(y + h - (ref - lo)/(hi - lo)*h)
        cv2.line(im, (x, yr), (x+w, yr), (60, 90, 120), 1)
    cv2.polylines(im, [np.stack([xs, ys], 1)], False, col, 1, cv2.LINE_AA)


class DashboardRenderer:
    def __init__(self, geom, title="Phy-WAM"):
        self.g = geom
        self.title = title
        self.hist = dict(alt=[], spd=[], clr=[], sep=[], soc=[], surp=[])

    # ------------------------------------------------------------- overlays --
    def _nav_panel(self, im, d, x, y, w, h):
        rgb = d.get("rgb_full")
        if rgb is None:
            return
        v = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (w, h),
                       interpolation=cv2.INTER_LINEAR)
        sx, sy = w / S.nav_w, h / S.nav_h
        # projected threat tracks
        for (uv, rng_, coop, vis) in d.get("proj", []):
            if not vis:
                continue
            px, py = int(uv[0]*sx), int(uv[1]*sy)
            col = WARN if coop else BAD
            r = max(6, int(260/max(rng_, 12)))
            cv2.circle(v, (px, py), r, col, 2, cv2.LINE_AA)
            cv2.line(v, (px-r-5, py), (px-r-1, py), col, 1)
            cv2.line(v, (px+r+1, py), (px+r+5, py), col, 1)
            _t(v, f"{rng_:.0f}m {'ADS-B' if coop else 'sUAS'}",
               px-r, py-r-5, 0.34, col, 1)
        # subgoal reticle
        sg = d.get("sg_uv")
        if sg is not None and sg[2]:
            px, py = int(sg[0]*sx), int(sg[1]*sy)
            cv2.drawMarker(v, (px, py), GOOD, cv2.MARKER_CROSS, 22, 1, cv2.LINE_AA)
            _t(v, "SUBGOAL", px+13, py-6, 0.34, GOOD)
        # artificial horizon
        roll, pitch = d.get("roll", 0.0), d.get("pitch", 0.0)
        cx, cy = w//2, h//2 + int(pitch*h*0.9)
        dx, dy = int(math.cos(roll)*w*0.32), int(math.sin(roll)*w*0.32)
        cv2.line(v, (cx-dx, cy+dy), (cx+dx, cy-dy), (90, 200, 230), 1, cv2.LINE_AA)
        cv2.line(v, (w//2-28, h//2), (w//2-8, h//2), (90, 200, 230), 1)
        cv2.line(v, (w//2+8, h//2), (w//2+28, h//2), (90, 200, 230), 1)
        im[y:y+h, x:x+w] = v

    def _map_panel(self, im, d, x, y, w, h):
        # Rendered into a private canvas rather than straight onto the frame:
        # world coordinates routinely fall outside the panel, and cv2 happily
        # draws them across the rest of the dashboard. Drawing into a canvas of
        # exactly the panel size makes cv2 clip for us.
        canvas = np.full((h, w, 3), (24, 23, 21), np.uint8)
        im_full, im = im, canvas
        pos = d["pos"]
        span = 620.0
        to = lambda p: (int(w/2 + (p[0]-pos[0])/span*w),
                        int(h/2 - (p[1]-pos[1])/span*h))
        # mapped buildings (grey) and unmapped cranes (violet)
        for C, Hf, Z, col in ((self.g.bc, self.g.bh, self.g.bz, (62, 60, 56)),
                              (self.g.uc, self.g.uh, self.g.uz, (95, 60, 80))):
            for i in range(len(C)):
                if abs(C[i][0]-pos[0]) > span or abs(C[i][1]-pos[1]) > span:
                    continue
                a = to((C[i][0]-Hf[i][0], C[i][1]+Hf[i][1]))
                b = to((C[i][0]+Hf[i][0], C[i][1]-Hf[i][1]))
                shade = col if Z[i] < pos[2] else tuple(int(c*1.7) for c in col)
                cv2.rectangle(im, a, b, shade, -1)
        # vertiports
        for i, v in enumerate(self.g.vp):
            if abs(v[0]-pos[0]) > span or abs(v[1]-pos[1]) > span:
                continue
            cv2.circle(im, to(v), 8, (120, 150, 90), 1, cv2.LINE_AA)
        # route
        rt = d.get("route")
        if rt is not None and len(rt) > 1:
            pts = np.array([to(p) for p in rt], np.int32)
            cv2.polylines(im, [pts], False, (90, 110, 70), 1, cv2.LINE_AA)
            for p in pts:
                cv2.drawMarker(im, tuple(p), (110, 140, 90), cv2.MARKER_DIAMOND, 6, 1)
        # MPPI sampled trajectories, coloured by rank
        for tr, good in d.get("samples", []):
            pts = np.array([to(p) for p in tr[::3]], np.int32)
            if len(pts) > 1:
                cv2.polylines(im, [pts], False,
                              (70, 130, 90) if good else (52, 50, 60), 1)
        # chosen plan
        pl = d.get("plan_traj")
        if pl is not None and len(pl) > 1:
            pts = np.array([to(p) for p in pl[::2]], np.int32)
            cv2.polylines(im, [pts], False, ACCENT, 2, cv2.LINE_AA)
        # flown path
        tr = d.get("trail")
        if tr is not None and len(tr) > 1:
            pts = np.array([to(p) for p in tr[::2]], np.int32)
            cv2.polylines(im, [pts], False, (200, 200, 200), 1, cv2.LINE_AA)
        # traffic
        for (p_i, v_i, coop) in d.get("tracks", []):
            c = to(p_i)
            col = WARN if coop else BAD
            cv2.circle(im, c, 5, col, -1)
            e = to(p_i + v_i*4.0)
            cv2.line(im, c, e, col, 1, cv2.LINE_AA)
            rr = int((SF.r_wellclear if coop else SF.r_suas)/span*w)
            cv2.circle(im, c, max(rr, 3), col, 1)
        # ownship
        c = to(pos)
        yaw = d.get("yaw", 0.0)
        tip = (int(c[0]+math.cos(yaw)*13), int(c[1]-math.sin(yaw)*13))
        l = (int(c[0]+math.cos(yaw+2.4)*9), int(c[1]-math.sin(yaw+2.4)*9))
        r = (int(c[0]+math.cos(yaw-2.4)*9), int(c[1]-math.sin(yaw-2.4)*9))
        cv2.fillPoly(im, [np.array([tip, l, r], np.int32)], (245, 245, 245))
        _t(im, f"{int(span*2)} m across", 7, h-8, 0.34, DIM)
        im = im_full
        im[y:y+h, x:x+w] = canvas
        cv2.rectangle(im, (x, y), (x+w, y+h), EDGE, 1)

    # --------------------------------------------------------------- render --
    def __call__(self, d):
        im = np.full((H, W, 3), BG, np.uint8)
        info = d.get("info", {})

        # ---------- header ----------
        cv2.rectangle(im, (0, 0), (W, 46), (30, 28, 26), -1)
        _t(im, "Phy-WAM", 16, 31, 0.78, ACCENT, 1, FD)
        _t(im, "physics-informed world-action model  |  vertiport-to-vertiport autonomy",
           148, 30, 0.44, DIM)
        _t(im, d.get("banner", ""), 700, 30, 0.46, TXT)
        st = d.get("status", "NOMINAL")
        col = {"NOMINAL": GOOD, "SHIELD ACTIVE": WARN, "TERMINATED": BAD}.get(st, TXT)
        cv2.rectangle(im, (W-206, 11), (W-14, 35), (44, 42, 38), -1)
        cv2.rectangle(im, (W-206, 11), (W-14, 35), col, 1)
        _t(im, st, W-198, 28, 0.46, col)

        # ---------- left: cameras ----------
        px, py = _panel(im, 12, 56, 528, 330, "NAV CAMERA  90 deg FOV  RGB + overlays")
        self._nav_panel(im, d, px+4, py+2, 520, 302)

        px, py = _panel(im, 12, 392, 260, 218, "DEPTH  (log-compressed)")
        dep = d.get("depth_vis")
        _blit(im, dep, px+4, py+2, 252, 190)

        px, py = _panel(im, 280, 392, 260, 218, "GRAD-CAM  hazard attribution")
        _blit(im, d.get("cam_vis"), px+4, py+2, 252, 190)

        px, py = _panel(im, 12, 616, 528, 272, "COST ATTRIBUTION  why this action")
        terms = d.get("cost_terms") or {}
        if terms:
            items = sorted(terms.items(), key=lambda kv: -abs(kv[1]))[:11]
            mx = max(abs(v) for _, v in items) or 1.0
            for i, (k, v) in enumerate(items):
                yy = py + 8 + i*22
                _t(im, k[:15], px+8, yy+11, 0.38, TXT)
                _bar(im, px+128, yy+2, 300, 12, abs(v)/mx,
                     BAD if k in ("obstacle", "obstacle_hit", "traffic",
                                  "traffic_hit", "ground", "hazard_pred")
                     else ACCENT)
                _t(im, f"{v:8.1f}", px+436, yy+11, 0.38, DIM)
        else:
            _t(im, "planner idle", px+10, py+22, 0.40, DIM)

        # ---------- middle: model ----------
        px, py = _panel(im, 548, 56, 512, 330, "PLAN VIEW  route / samples / traffic")
        self._map_panel(im, d, px+4, py+2, 504, 302)

        px, py = _panel(im, 548, 392, 512, 218,
                        "WORLD MODEL  observed | reconstructed | imagined t+0.4 .. t+3.2 s")
        obs_s = d.get("wm_obs"); rec_s = d.get("wm_rec")
        _blit(im, obs_s, px+6, py+4, 108, 108)
        _t(im, "seen", px+6, py+126, 0.36, DIM)
        _blit(im, rec_s, px+122, py+4, 108, 108)
        _t(im, "recon", px+122, py+126, 0.36, DIM)
        for i, fr in enumerate((d.get("wm_imag") or [])[:4]):
            _blit(im, fr, px+248 + i*66, py+4, 62, 62)
            _t(im, f"+{(i+1)*0.8:.1f}s", px+248 + i*66, py+76, 0.32, DIM)
        sp = d.get("surprise")
        if sp is not None:
            _t(im, f"prediction error {sp:.4f}", px+248, py+126, 0.36,
               BAD if sp > 0.06 else DIM)
            _spark(im, px+248, py+132, 250, 46, self.hist["surp"], col=VIOLET)

        px, py = _panel(im, 548, 616, 512, 272, "COUNTERFACTUALS  cost of alternatives")
        cf = d.get("counterfactual") or {}
        if cf:
            base = cf.get("chosen", 0.0)
            items = [(k, v) for k, v in cf.items() if k != "chosen"]
            _t(im, f"chosen plan cost {base:9.1f}", px+8, py+18, 0.42, ACCENT)
            for i, (k, v) in enumerate(items):
                yy = py + 34 + i*24
                dlt = v - base
                _t(im, k, px+8, yy+12, 0.40, TXT)
                _t(im, f"{v:9.1f}", px+178, yy+12, 0.40, DIM)
                c = GOOD if dlt < 0 else BAD
                _t(im, f"{dlt:+9.1f}", px+282, yy+12, 0.40, c)
                _bar(im, px+380, yy+3, 118, 11,
                     min(abs(dlt)/max(abs(base)*0.35, 1e-6), 1.0), c)
            _t(im, "positive = worse than the chosen plan", px+8, py+254, 0.34, DIM)
        else:
            _t(im, "planner idle", px+10, py+22, 0.40, DIM)

        # ---------- right: telemetry + safety ----------
        px, py = _panel(im, 1068, 56, 520, 330, "TELEMETRY")
        rows = [("altitude AGL", "alt", "m", d.get("alt", 0), None, ACCENT,
                 CFG.wld.corridor_alt),
                ("airspeed", "spd", "m/s", d.get("spd", 0), None, ACCENT,
                 CFG.air.V_stall),
                ("obstacle clearance", "clr", "m", d.get("clr", 0), None, GOOD,
                 SF.r_static),
                ("traffic separation", "sep", "m", d.get("sep", 0), None, WARN,
                 SF.r_suas)]
        for i, (lab, key, unit, val, _, col, ref) in enumerate(rows):
            yy = py + 6 + i*76
            _t(im, lab, px+10, yy+13, 0.40, TXT)
            _t(im, f"{val:8.1f} {unit}", px+390, yy+13, 0.44,
               BAD if (key in ("clr", "sep") and val < ref) else TXT)
            _spark(im, px+10, yy+18, 494, 48, self.hist[key], col=col, ref=ref)

        px, py = _panel(im, 1068, 392, 520, 218, "SAFETY BARRIER  (HOCBF)")
        eng = d.get("engaged", 0)
        cv2.circle(im, (px+20, py+22), 9, BAD if eng else GOOD, -1)
        _t(im, "ENGAGED - action repaired" if eng else "inactive - planner in command",
           px+38, py+27, 0.44, BAD if eng else GOOD)
        _t(im, f"active constraints  {d.get('n_cbf', 0)}", px+10, py+56, 0.40, TXT)
        _t(im, f"engaged this flight {100*d.get('shield_frac', 0):5.1f} %",
           px+10, py+78, 0.40, TXT)
        u = d.get("u_safe")
        if u is not None:
            _t(im, f"commanded accel  [{u[0]:6.2f} {u[1]:6.2f} {u[2]:6.2f}] m/s2",
               px+10, py+100, 0.40, DIM)
        a_r, a_f = d.get("action_raw"), d.get("action")
        if a_r is not None and a_f is not None:
            _t(im, "action  planner -> shielded", px+10, py+126, 0.38, DIM)
            names = ["col", "roll", "ptch", "yaw", "push", "sched"]
            for i in range(6):
                xx = px + 12 + i*84
                _t(im, names[i], xx, py+144, 0.34, DIM)
                _bar(im, xx, py+150, 74, 8, (a_r[i]+1)/2, (90, 90, 96))
                _bar(im, xx, py+161, 74, 8, (a_f[i]+1)/2,
                     BAD if abs(a_f[i]-a_r[i]) > 0.02 else ACCENT)

        px, py = _panel(im, 1068, 616, 520, 272, "MISSION & SAFETY METRICS")
        m = [("outcome", d.get("outcome", "in flight")),
             ("elapsed", f"{info.get('t', 0):.1f} s"),
             ("range to goal", f"{info.get('d_goal', 0):.0f} m"),
             ("route flown", f"{info.get('path_len', 0):.0f} m"),
             ("battery used", f"{100*(1-info.get('soc', 1)):.1f} %"),
             ("min obstacle clearance", f"{info.get('min_obs', 0):.1f} m"),
             ("min traffic separation", f"{min(info.get('min_sep', 9999), 9999):.0f} m"),
             ("loss of well-clear", f"{info.get('lox', 0)}"),
             ("near mid-air (<30 m)", f"{info.get('nmac', 0)}")]
        for i, (k, v) in enumerate(m):
            yy = py + 12 + i*27
            _t(im, k, px+12, yy+12, 0.42, DIM)
            bad = (k.startswith("loss") and v != "0") or \
                  (k.startswith("near") and v != "0") or \
                  (k.startswith("min obstacle") and float(v.split()[0]) < SF.r_static)
            _t(im, v, px+330, yy+12, 0.46, BAD if bad else TXT)
        return im

    def push(self, d):
        self.hist["alt"].append(d.get("alt", 0))
        self.hist["spd"].append(d.get("spd", 0))
        self.hist["clr"].append(min(d.get("clr", 0), 300))
        self.hist["sep"].append(min(d.get("sep", 0), 1500))
        self.hist["soc"].append(d.get("soc", 1))
        if d.get("surprise") is not None:
            self.hist["surp"].append(d["surprise"])
