"""
Global route planner over the static obstacle database.

Real eVTOLs do not improvise a 3 km path from pixels; they fly a published
corridor computed against a known terrain/obstacle database, and use onboard
sensing for what the database cannot contain -- other traffic, cranes, weather.
We mirror that split exactly:

    A* here      -> static, known, global, seconds of lookahead
    MPPI + CBF   -> dynamic, sensed, local, 2 seconds of lookahead

Giving the learner a subgoal ~200 m ahead instead of a goal 3 km ahead is also
what makes the reward informative: progress toward a waypoint is dense, whereas
progress toward a distant vertiport is nearly flat across an entire episode.
"""
import heapq
import numpy as np

from .config import CFG

SF, WD = CFG.saf, CFG.wld


class RoutePlanner:
    def __init__(self, geom, cell=30.0, z_lo=45.0, z_hi=235.0, z_cell=25.0):
        self.g = geom
        self.cell, self.z_cell = cell, z_cell
        self.z_lo, self.z_hi = z_lo, z_hi
        ext = geom.extent
        self.x0, self.y0 = -ext, -ext
        self.nx = int(2 * ext / cell) + 1
        self.ny = self.nx
        self.nz = int((z_hi - z_lo) / z_cell) + 1
        self._build()

    def _build(self):
        """Occupancy from the building boxes, inflated by the required static
        clearance plus half a cell so a path through free cells is genuinely
        clear rather than merely collision-free at the sample points."""
        infl = SF.r_static + 0.5 * self.cell
        xs = self.x0 + self.cell * np.arange(self.nx)
        ys = self.y0 + self.cell * np.arange(self.ny)
        zs = self.z_lo + self.z_cell * np.arange(self.nz)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        occ = np.zeros((self.nx, self.ny, self.nz), bool)
        for (cx, cy), (hw, hd), h in zip(self.g.bc, self.g.bh, self.g.bz):
            m = (np.abs(X - cx) <= hw + infl) & (np.abs(Y - cy) <= hd + infl)
            if not m.any():
                continue
            kz = zs <= h + infl
            occ[m[..., None] & kz[None, None, :]] = True
        self.occ = occ
        self.free_frac = 1.0 - occ.mean()

    # ------------------------------------------------------------ indexing --
    def to_idx(self, p):
        return (int(round((p[0] - self.x0) / self.cell)),
                int(round((p[1] - self.y0) / self.cell)),
                int(round((np.clip(p[2], self.z_lo, self.z_hi) - self.z_lo) / self.z_cell)))

    def to_pos(self, i):
        return np.array([self.x0 + i[0]*self.cell,
                         self.y0 + i[1]*self.cell,
                         self.z_lo + i[2]*self.z_cell])

    def _ok(self, i):
        return (0 <= i[0] < self.nx and 0 <= i[1] < self.ny and
                0 <= i[2] < self.nz and not self.occ[i])

    # ------------------------------------------------------------------ A* --
    def plan(self, start, goal, cruise_z=None, land_z=None):
        cruise_z = cruise_z or WD.corridor_alt
        land_z = land_z if land_z is not None else (goal[2] + 3.0)
        s = self.to_idx([start[0], start[1], cruise_z])
        g = self.to_idx([goal[0], goal[1], cruise_z])
        if not self._ok(s):
            s = self._nearest_free(s)
        if not self._ok(g):
            g = self._nearest_free(g)
        if s is None or g is None:
            return self._terminals(self._straight(start, goal, cruise_z),
                                   start, goal, land_z)

        nbrs = [(dx, dy, dz)
                for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
                if (dx or dy or dz)]
        # climbing costs more than turning, so level routes are preferred
        wt = {n: np.sqrt(n[0]**2 + n[1]**2 + (2.5*n[2])**2) for n in nbrs}
        hcost = lambda a: np.sqrt((a[0]-g[0])**2 + (a[1]-g[1])**2 + (2.5*(a[2]-g[2]))**2)

        openq = [(hcost(s), 0.0, s)]
        came, gsc = {}, {s: 0.0}
        seen = set()
        while openq:
            _, gc, cur = heapq.heappop(openq)
            if cur in seen:
                continue
            seen.add(cur)
            if cur == g:
                return self._terminals(
                    self._smooth(self._trace(came, cur), start, goal),
                    start, goal, land_z)
            for n in nbrs:
                nx = (cur[0]+n[0], cur[1]+n[1], cur[2]+n[2])
                if not self._ok(nx) or nx in seen:
                    continue
                ng = gc + wt[n]
                if ng < gsc.get(nx, 1e18):
                    gsc[nx] = ng
                    came[nx] = cur
                    heapq.heappush(openq, (ng + hcost(nx), ng, nx))
        return self._terminals(self._straight(start, goal, cruise_z),
                               start, goal, land_z)

    def _nearest_free(self, i):
        for r in range(1, 14):
            for dx in range(-r, r+1):
                for dy in range(-r, r+1):
                    for dz in range(-2, 3):
                        c = (i[0]+dx, i[1]+dy, i[2]+dz)
                        if self._ok(c):
                            return c
        return None

    def _trace(self, came, cur):
        out = [cur]
        while cur in came:
            cur = came[cur]
            out.append(cur)
        return [self.to_pos(i) for i in reversed(out)]

    def _straight(self, start, goal, z):
        return [np.array([start[0], start[1], z]), np.array([goal[0], goal[1], z])]

    @staticmethod
    def _terminals(pts, start, goal, z_lo):
        """Bracket the cruise route with a vertical departure and arrival.

        Without these the route sits at corridor altitude directly over the
        landing pad and the aircraft is being told to hold 120 m while also
        being required to touch down -- two instructions it cannot satisfy at
        once. Real vertiport procedures have exactly this shape: climb in the
        departure volume, transit the corridor, descend in the arrival volume.
        """
        out = [np.array([start[0], start[1], max(start[2], 2.0)]),
               np.array([start[0], start[1], pts[0][2]])] + list(pts[1:-1]) + \
              [np.array([goal[0], goal[1], pts[-1][2]]),
               np.array([goal[0], goal[1], z_lo])]
        # drop duplicates that the brackets may have introduced
        ded = [out[0]]
        for p in out[1:]:
            if np.linalg.norm(p - ded[-1]) > 4.0:
                ded.append(p)
        return ded

    def _smooth(self, pts, start, goal):
        """String-pulling: drop any waypoint whose removal still leaves a
        collision-free straight segment. Turns the 26-connected staircase into
        a handful of long legs, which is both flyable and cheap to follow."""
        if len(pts) < 3:
            return pts
        out = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self._clear(pts[i], pts[j]):
                j -= 1
            out.append(pts[j])
            i = j
        out[0] = np.array([start[0], start[1], out[0][2]])
        out[-1] = np.array([goal[0], goal[1], out[-1][2]])
        return out

    def _clear(self, a, b, step=12.0):
        n = max(2, int(np.linalg.norm(b - a) / step))
        P = a[None] + (b - a)[None] * np.linspace(0, 1, n)[:, None]
        return bool(self.g.building_sdf_batch(P).min() > SF.r_static)


class RouteTracker:
    """Walks the aircraft along the route and exposes the active subgoal."""

    def __init__(self, wpts, capture=95.0, lead=170.0, final_capture=22.0):
        self.w = [np.asarray(p, np.float64) for p in wpts]
        self.i = 1 if len(self.w) > 1 else 0
        self.capture, self.lead = capture, lead
        self.final_capture = final_capture

    def _passed(self, pos, corridor=170.0):
        """True once the aircraft is beyond the current waypoint along its leg.

        Capture-radius-only sequencing does not work on this airframe: the turn
        radius at cruise is V^2/(g tan 30 deg) = 311 m, so a 95 m capture circle
        is routinely missed, the subgoal ends up BEHIND the aircraft, and it
        orbits. Measured: 3.6x the direct distance, then a timeout.

        The cross-track guard matters as much as the along-track test. Without
        it, "past the waypoint" is true for any aircraft far off to one side,
        including one that has wandered kilometres away, so a single excursion
        satisfies the test for every remaining leg at once.
        """
        a = self.w[max(self.i - 1, 0)]
        b = self.w[self.i]
        ab = b[:2] - a[:2]
        n2 = float(np.dot(ab, ab))
        if n2 < 1.0:
            return False
        rel = pos[:2] - a[:2]
        t = float(np.dot(rel, ab) / n2)
        cross = float(np.linalg.norm(rel - t * ab))
        return t > 1.02 and cross < corridor

    def update(self, pos):
        """Advance AT MOST ONE waypoint per call.

        The previous version looped. Combined with along-track sequencing that
        let one overshoot cascade through the entire route in a single step,
        landing the aircraft on the final descent leg -- whose subgoal is the
        pad at ~4 m altitude -- while still a kilometre out. It then flew that
        commanded descent straight into the ground.
        """
        if self.i < len(self.w) - 1:
            d_h = float(np.linalg.norm(self.w[self.i][:2] - pos[:2]))
            d_v = abs(self.w[self.i][2] - pos[2])
            leg = self.w[self.i] - self.w[max(self.i - 1, 0)]
            vertical = abs(leg[2]) > np.linalg.norm(leg[:2])
            # The arrival leg is committed to on proximity only: descending
            # early is unrecoverable, so it must never be entered on a guess.
            entering_final = (self.i + 1) >= len(self.w) - 1
            if vertical:
                hit = d_v < 12.0 and d_h < self.final_capture
            elif entering_final:
                hit = d_h < self.final_capture * 2.2
            else:
                hit = (d_h < self.capture) or self._passed(pos)
            if hit:
                self.i += 1
        return self.subgoal(pos)

    def subgoal(self, pos):
        """A carrot on the current leg, `lead` metres ahead of the projection
        of the aircraft onto that leg. Following a moving carrot rather than a
        fixed corner keeps the commanded heading continuous through turns."""
        a = self.w[max(self.i - 1, 0)]
        b = self.w[self.i]
        ab = b - a
        n = np.linalg.norm(ab)
        if n < 1e-6:
            return b.copy()
        t = float(np.clip(np.dot(pos - a, ab) / (n * n), 0.0, 1.0))
        proj = a + t * ab
        return proj + ab / n * min(self.lead, n * (1.0 - t) + 1e-6)

    @property
    def done(self):
        return self.i >= len(self.w) - 1

    def remaining(self, pos):
        d = np.linalg.norm(self.w[self.i][:2] - pos[:2])
        for k in range(self.i, len(self.w) - 1):
            d += np.linalg.norm(self.w[k+1][:2] - self.w[k][:2])
        return float(d)
