"""
World geometry: analytic clearance, signed distance fields and camera
projection.

Collision and clearance are computed here in closed form rather than read out
of the physics engine. Two reasons: it is exact and differentiable, which the
control-barrier filter needs, and it is ~100x cheaper than a contact query,
which matters when the planner evaluates hundreds of rollouts per control step.
"""
import json
import numpy as np


def quat_to_R(q):
    """w,x,y,z -> 3x3 rotation matrix, body(FLU) -> world(ENU)."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)]], np.float64)


FLU2FRD = np.diag([1.0, -1.0, -1.0])


class WorldGeom:
    """Static scenery + helpers. Buildings are axis-aligned boxes standing on
    the ground plane, which makes the signed distance a closed form."""

    def __init__(self, meta_path):
        m = json.load(open(meta_path))
        b = np.asarray(m["buildings"], np.float64)          # (N,5) x,y,w,d,h
        self.bc = b[:, :2].copy()                            # centres
        self.bh = b[:, 2:4] * 0.5                            # half extents xy
        self.bz = b[:, 4].copy()                             # heights
        self.vp = np.asarray(m["vertiports"], np.float64)    # (M,3)
        # Unmapped obstacles: excluded from the database the route planner and
        # the barrier filter see, but included in collision truth. `full_*`
        # arrays are the ground truth; `bc/bh/bz` stay database-only.
        u = np.asarray(m.get("unmapped", []), np.float64).reshape(-1, 6)
        self.uc = u[:, :2].copy() if len(u) else np.zeros((0, 2))
        self.uh = u[:, 2:4] * 0.5 if len(u) else np.zeros((0, 2))
        self.uz = u[:, 4].copy() if len(u) else np.zeros((0,))
        self.full_c = np.concatenate([self.bc, self.uc], 0)
        self.full_h = np.concatenate([self.bh, self.uh], 0)
        self.full_z = np.concatenate([self.bz, self.uz], 0)
        self.seed = m.get("seed", 0)
        self.n_traffic = m.get("n_traffic", 0)
        self.n_suas = m.get("n_suas", 0)
        self.extent = float(np.abs(self.bc).max() + 400.0)

    # ------------------------------------------------------------ clearance --
    def _sdf(self, p, C, H, Z):
        d_xy = np.abs(p[:2][None, :] - C) - H
        zc = Z * 0.5
        d_z = np.abs(p[2] - zc) - zc
        d = np.concatenate([d_xy, d_z[:, None]], 1)
        dist = np.linalg.norm(np.maximum(d, 0.0), axis=1) + np.minimum(d.max(1), 0.0)
        i = int(np.argmin(dist))
        return float(dist[i]), i

    def true_sdf(self, p):
        """Distance to the nearest REAL obstacle, mapped or not. Used for
        scoring and for termination -- never fed to the planner."""
        return self._sdf(p, self.full_c, self.full_h, self.full_z)

    def building_sdf(self, p):
        """Signed distance from p=(3,) to the nearest building.

        For an AABB the exterior distance is ||max(|p-c| - h, 0)|| and the
        interior distance is max(|p-c| - h) which is negative inside. We keep
        both branches so the barrier stays well-defined after a violation.
        """
        d_xy = np.abs(p[:2][None, :] - self.bc) - self.bh    # (N,2)
        # z: box spans [0, h]
        zc = self.bz * 0.5
        d_z = np.abs(p[2] - zc) - zc                          # (N,)
        d = np.concatenate([d_xy, d_z[:, None]], 1)           # (N,3)
        outside = np.maximum(d, 0.0)
        d_out = np.linalg.norm(outside, axis=1)
        d_in = np.minimum(d.max(axis=1), 0.0)
        dist = d_out + d_in
        i = int(np.argmin(dist))
        return float(dist[i]), i

    def building_sdf_batch(self, P):
        """Vectorised over P=(K,3). Returns (K,) nearest-building distance."""
        d_xy = np.abs(P[:, None, :2] - self.bc[None]) - self.bh[None]   # (K,N,2)
        zc = self.bz * 0.5
        d_z = np.abs(P[:, None, 2] - zc[None]) - zc[None]               # (K,N)
        d = np.concatenate([d_xy, d_z[..., None]], -1)                  # (K,N,3)
        d_out = np.linalg.norm(np.maximum(d, 0.0), axis=-1)
        d_in = np.minimum(d.max(-1), 0.0)
        return (d_out + d_in).min(-1)

    def building_grad(self, p, eps=0.35):
        """Gradient of the building SDF. Central differences: the SDF is
        piecewise-smooth and the kinks sit on measure-zero sets, so this is
        stable in practice and far simpler than case-splitting the box."""
        g = np.zeros(3)
        for k in range(3):
            e = np.zeros(3); e[k] = eps
            g[k] = (self.building_sdf(p + e)[0] - self.building_sdf(p - e)[0]) / (2*eps)
        n = np.linalg.norm(g)
        return g / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])

    def clearance(self, p):
        """Min of building distance and height above ground."""
        db, _ = self.building_sdf(p)
        return min(db, float(p[2]))

    def nearest_vertiport(self, p, exclude=None):
        d = np.linalg.norm(self.vp[:, :2] - p[:2], axis=1)
        if exclude is not None:
            d[exclude] = 1e12
        return int(np.argmin(d))


class Camera:
    """Pinhole camera rigidly mounted in the body frame."""

    def __init__(self, w, h, hfov_deg, pose_xyz, pitch_down_rad):
        self.w, self.h = int(w), int(h)
        self.f = (self.w * 0.5) / np.tan(np.radians(hfov_deg) * 0.5)
        self.cx, self.cy = self.w * 0.5, self.h * 0.5
        self.t_b = np.asarray(pose_xyz, np.float64)          # in body FLU
        cp, sp = np.cos(pitch_down_rad), np.sin(pitch_down_rad)
        # rotation from camera frame to body FLU: camera looks along +x
        self.R_bc = np.array([[cp, 0.0, sp],
                              [0.0, 1.0, 0.0],
                              [-sp, 0.0, cp]], np.float64)

    def project(self, P_world, pos, R_wb):
        """World points (K,3) -> pixel (K,2), depth (K,), visible mask (K,).

        Optical convention: camera looks down its +x, image u grows to the
        right (which is body -y) and v grows downward (body -z)."""
        d_w = np.atleast_2d(P_world) - pos[None, :]
        d_b = d_w @ R_wb                                    # world -> body FLU
        d_c = d_b @ self.R_bc                               # body -> camera
        fx = d_c[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.cx - self.f * d_c[:, 1] / fx
            v = self.cy - self.f * d_c[:, 2] / fx
        vis = (fx > 0.5) & (u >= 0) & (u < self.w) & (v >= 0) & (v < self.h)
        return np.stack([u, v], 1), fx, vis

    def angular_size(self, phys_size, rng):
        """Pixels spanned by an object of physical size at range."""
        return self.f * phys_size / np.maximum(rng, 1e-3)
