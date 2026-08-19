// ---------------------------------------------------------------------------
//  Phy-WAM  |  plant physics for a lift+cruise eVTOL
//
//  Pure math, no Gazebo API. All aerodynamics here use the STANDARD AEROSPACE
//  body frame FRD (x forward, y right, z down) so that every formula matches
//  the textbook sign conventions. The plugin converts to Gazebo's FLU/ENU at
//  the boundary by negating the y and z components.
//
//  This is the PLANT ("the real aircraft"). It deliberately contains effects
//  that the learned planner-side model does NOT get analytically -- ground
//  effect, rotor-wing interference, actuator lag, Dryden turbulence -- so the
//  physics-informed residual network has genuine unmodelled dynamics to learn
//  rather than fitting noise.
// ---------------------------------------------------------------------------
#ifndef PHYWAM_AERO_HH_
#define PHYWAM_AERO_HH_

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace phywam {

constexpr double kG   = 9.80665;
constexpr double kRho = 1.225;
constexpr int    kNRotor = 8;

struct V3 {
  double x{0}, y{0}, z{0};
  V3() = default;
  V3(double a, double b, double c) : x(a), y(b), z(c) {}
  V3 operator+(const V3 &o) const { return {x + o.x, y + o.y, z + o.z}; }
  V3 operator-(const V3 &o) const { return {x - o.x, y - o.y, z - o.z}; }
  V3 operator*(double s)    const { return {x * s, y * s, z * s}; }
  V3 cross(const V3 &o) const {
    return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
  }
  double norm() const { return std::sqrt(x * x + y * y + z * z); }
};

// --------------------------------------------------------------- parameters
struct Params {
  double mass = 1000.0;
  double Ixx = 2200.0, Iyy = 3000.0, Izz = 4600.0;

  // lift rotors
  double R_rotor = 0.90;
  double k_rotor = 3.030e-2;      // N/(rad/s)^2
  double b_rotor = 2.8164e-3;     // Nm/(rad/s)^2
  double w_max   = 265.0;         // rad/s
  double tau_rotor = 0.055;       // s
  double rx[kNRotor] = { 3.6,  3.6,  1.2,  1.2, -1.2, -1.2, -3.6, -3.6};
  double ry[kNRotor] = { 3.0, -3.0,  3.0, -3.0,  3.0, -3.0,  3.0, -3.0};
  double rz[kNRotor] = {-0.35,-0.35,-0.35,-0.35,-0.35,-0.35,-0.35,-0.35}; // FRD: up = -z
  double rs[kNRotor] = { 1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0};  // spin sign

  // wing / tail
  double S_wing = 11.0, b_span = 12.0, chord = 0.9167;
  double CL0 = 0.12, CL_alpha = 5.10, CL_max = 1.50;
  double CD0 = 0.035, oswald = 0.80;
  double Cm0 = 0.020, Cm_alpha = -0.85;
  double Cn_beta = 0.110, Cl_beta = -0.085;
  double S_fin = 1.40, l_tail = 4.30;
  // rate damping derivatives (per rad/s, non-dimensionalised inside)
  double Clp = -0.48, Cmq = -12.5, Cnr = -0.19;

  // pusher
  double T_push_max = 3600.0;
  double push_x = -4.60;

  // surfaces
  double Cl_dail = 0.150, Cm_dele = -1.100, Cn_drud = -0.075;
  double d_max = 0.4363;          // 25 deg
  double tau_surf = 0.040;
};

// ------------------------------------------------------------- actuator set
struct Actuators {
  double w[kNRotor] = {0};        // rad/s, first-order lagged
  double push = 0.0;              // 0..1
  double ail = 0.0, ele = 0.0, rud = 0.0;   // rad
};

// high-level command from the Python planner (20 Hz, zero-order held)
struct Command {
  double thrust_col = 0.0;   // 0..1 collective lift-rotor demand
  double roll_ref   = 0.0;   // rad
  double pitch_ref  = 0.0;   // rad
  double yawrate_ref= 0.0;   // rad/s
  double push_thr   = 0.0;   // 0..1 pusher throttle
  double sched      = 0.0;   // 0 = hover-borne, 1 = fully wing-borne
};

struct Wrench { V3 F, M; };

inline double clampd(double v, double lo, double hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// ---------------------------------------------------------------------------
//  Lift rotors: momentum theory + ground effect + rotor-wing interference.
//
//  Baseline thrust of rotor i:      T_i = k * w_i^2
//  Ground effect (Cheeseman-Bennett):
//      T_IGE / T_OGE = 1 / (1 - (R/4z)^2)          z = height above ground
//  Wing blanking: in forward flight the wing sits in the rotor wake and the
//  rotors lose effectiveness roughly with dynamic-pressure ratio.
// ---------------------------------------------------------------------------
inline Wrench rotorWrench(const Params &p, const Actuators &a,
                          double height_agl, double u_air, double *T_out) {
  Wrench wr;
  const double zc = std::max(height_agl, 0.30);
  double ige = 1.0;
  if (zc < 4.0 * p.R_rotor) {
    const double r4z = p.R_rotor / (4.0 * zc);
    ige = 1.0 / std::max(1.0 - r4z * r4z, 0.35);   // capped: no infinite thrust
  }
  // rotor-wing interference: wake impinges on the wing, net vertical force
  // recovered falls off with advance ratio mu = u / (w R)
  const double w_ref = std::max(1.0, p.w_max * 0.5);
  const double mu    = clampd(std::fabs(u_air) / (w_ref * p.R_rotor), 0.0, 1.0);
  const double intf  = 1.0 - 0.16 * mu;

  for (int i = 0; i < kNRotor; ++i) {
    const double T = p.k_rotor * a.w[i] * a.w[i] * ige * intf;
    const double Q = p.b_rotor * a.w[i] * a.w[i];
    if (T_out) T_out[i] = T;
    // thrust acts along body -z (upward) in FRD
    wr.F.z -= T;
    const V3 r(p.rx[i], p.ry[i], p.rz[i]);
    const V3 F(0.0, 0.0, -T);
    const V3 m = r.cross(F);
    wr.M.x += m.x; wr.M.y += m.y; wr.M.z += m.z;
    // reaction torque of the spinning rotor about body z
    wr.M.z += -p.rs[i] * Q;
  }
  return wr;
}

// ---------------------------------------------------------------------------
//  Fixed-wing aerodynamics with a smooth post-stall blend.
//
//  alpha = atan2(w, u),  beta = asin(v / V)
//  Attached:  CL = CL0 + CL_alpha * alpha
//  Post-stall: flat-plate CL = 2 sin a cos a, CD = 2 sin^2 a
//  Blended with a logistic in alpha so the transition is C1-continuous, which
//  matters because the planner differentiates through a learned copy of this.
// ---------------------------------------------------------------------------
inline Wrench aeroWrench(const Params &p, const Actuators &a,
                         const V3 &v_air, const V3 &omega) {
  Wrench wr;
  const double V = v_air.norm();
  if (V < 0.5) return wr;

  const double q  = 0.5 * kRho * V * V;
  const double al = std::atan2(v_air.z, v_air.x);
  const double be = std::asin(clampd(v_air.y / V, -1.0, 1.0));
  const double AR = p.b_span * p.b_span / p.S_wing;

  // --- stall blending -------------------------------------------------------
  const double a_stall = (p.CL_max - p.CL0) / p.CL_alpha;   // ~0.27 rad
  const double sig = 1.0 / (1.0 + std::exp(-(std::fabs(al) - a_stall) / 0.045));
  const double CL_att = p.CL0 + p.CL_alpha * al;
  const double CL_sep = 2.0 * std::sin(al) * std::cos(al);
  const double CL = (1.0 - sig) * CL_att + sig * CL_sep;

  const double CDi = CL * CL / (M_PI * p.oswald * AR);
  const double CD_att = p.CD0 + CDi;
  const double CD_sep = p.CD0 + 2.0 * std::sin(al) * std::sin(al);
  const double CD = (1.0 - sig) * CD_att + sig * CD_sep;

  const double CY = -0.62 * be;                 // fin + fuselage side force

  // --- forces: lift perpendicular to wind, drag along it (FRD, x-z plane) ---
  const double L = q * p.S_wing * CL;
  const double D = q * p.S_wing * CD;
  const double Y = q * p.S_fin  * CY;
  const double ca = std::cos(al), sa = std::sin(al);
  wr.F.x += L * sa - D * ca;
  wr.F.z += -L * ca - D * sa;
  wr.F.y += Y;

  // --- moments --------------------------------------------------------------
  // non-dimensional rates
  const double p_h = omega.x * p.b_span / (2.0 * V);
  const double q_h = omega.y * p.chord  / (2.0 * V);
  const double r_h = omega.z * p.b_span / (2.0 * V);

  const double Cl = p.Cl_beta * be + p.Clp * p_h + p.Cl_dail * a.ail;
  const double Cm = p.Cm0 + p.Cm_alpha * al * (1.0 - 0.7 * sig)
                  + p.Cmq * q_h + p.Cm_dele * a.ele;
  const double Cn = p.Cn_beta * be + p.Cnr * r_h + p.Cn_drud * a.rud;

  wr.M.x += q * p.S_wing * p.b_span * Cl;
  wr.M.y += q * p.S_wing * p.chord  * Cm;
  wr.M.z += q * p.S_wing * p.b_span * Cn;
  return wr;
}

// ------------------------------------------------------------------- pusher
inline Wrench pusherWrench(const Params &p, const Actuators &a, double u_air) {
  Wrench wr;
  // thrust lapses with forward speed (fixed-pitch prop unloading)
  const double lapse = clampd(1.0 - 0.55 * std::fabs(u_air) / 60.0, 0.25, 1.0);
  const double T = p.T_push_max * clampd(a.push, 0.0, 1.0) * lapse;
  wr.F.x += T;
  return wr;   // thrust line through CG in x, no moment by construction
}

// ---------------------------------------------------------------------------
//  Dryden-like turbulence. A first-order shaping filter driven by white noise
//  reproduces the low-frequency content of the Dryden spectrum, which is what
//  actually disturbs a vehicle of this size and time constant.
// ---------------------------------------------------------------------------
struct Gust {
  double s[3] = {0, 0, 0};
  uint64_t rng = 0x9E3779B97F4A7C15ull;
  double u01() {                                   // xorshift64*
    rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27;
    return double((rng * 2685821657736338717ull) >> 11) / 9007199254740992.0;
  }
  double gauss() {
    double a = std::max(u01(), 1e-12), b = u01();
    return std::sqrt(-2.0 * std::log(a)) * std::cos(2.0 * M_PI * b);
  }
  V3 step(double dt, double sigma, double L, double V) {
    const double tau = std::max(L / std::max(V, 5.0), 0.15);
    const double al  = std::exp(-dt / tau);
    const double gn  = sigma * std::sqrt(1.0 - al * al);
    for (int i = 0; i < 3; ++i) s[i] = al * s[i] + gn * gauss();
    return {s[0], s[1], s[2]};
  }
};

// ---------------------------------------------------------------------------
//  Control allocation.
//
//  Map the 8 lift-rotor thrusts to [Fz, Mx, My, Mz]:
//     Fz = -sum(T_i)
//     Mx = -sum(T_i * y_i)          (thrust up at y>0 rolls left)
//     My = +sum(T_i * x_i)
//     Mz =  sum(-s_i * (b/k) * T_i)
//  and invert with the damped pseudo-inverse  A^T (A A^T + eps I)^-1.
//  A A^T is only 4x4, so we solve it with Gauss-Jordan at configure time.
// ---------------------------------------------------------------------------
class Allocator {
 public:
  void build(const Params &p) {
    const double bk = p.b_rotor / p.k_rotor;
    for (int i = 0; i < kNRotor; ++i) {
      A_[0][i] = -1.0;
      A_[1][i] = -p.ry[i];
      A_[2][i] =  p.rx[i];
      A_[3][i] = -p.rs[i] * bk;
    }
    double M[4][8];   // [A A^T | I]
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) {
        double s = 0;
        for (int i = 0; i < kNRotor; ++i) s += A_[r][i] * A_[c][i];
        M[r][c] = s + (r == c ? 1e-6 : 0.0);
      }
      for (int c = 0; c < 4; ++c) M[r][4 + c] = (r == c) ? 1.0 : 0.0;
    }
    for (int c = 0; c < 4; ++c) {                    // Gauss-Jordan
      int piv = c;
      for (int r = c + 1; r < 4; ++r)
        if (std::fabs(M[r][c]) > std::fabs(M[piv][c])) piv = r;
      for (int k = 0; k < 8; ++k) std::swap(M[c][k], M[piv][k]);
      const double d = M[c][c];
      for (int k = 0; k < 8; ++k) M[c][k] /= d;
      for (int r = 0; r < 4; ++r) {
        if (r == c) continue;
        const double f = M[r][c];
        for (int k = 0; k < 8; ++k) M[r][k] -= f * M[c][k];
      }
    }
    for (int r = 0; r < 4; ++r)
      for (int c = 0; c < 4; ++c) Inv_[r][c] = M[r][4 + c];
    Tmax_ = p.k_rotor * p.w_max * p.w_max;
  }

  // desired [Fz, Mx, My, Mz] -> per-rotor thrust, saturated
  void solve(const double d[4], double T[kNRotor]) const {
    double y[4] = {0, 0, 0, 0};
    for (int r = 0; r < 4; ++r)
      for (int c = 0; c < 4; ++c) y[r] += Inv_[r][c] * d[c];
    for (int i = 0; i < kNRotor; ++i) {
      double t = 0;
      for (int r = 0; r < 4; ++r) t += A_[r][i] * y[r];
      T[i] = clampd(t, 0.0, Tmax_);
    }
  }
  double Tmax() const { return Tmax_; }

 private:
  double A_[4][kNRotor]{};
  double Inv_[4][4]{};
  double Tmax_{1.0};
};

}  // namespace phywam
#endif
