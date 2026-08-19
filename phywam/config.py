"""
Single source of truth for the Phy-WAM aircraft, sensors, world and learning
hyper-parameters.

Everything downstream reads from here: the feasibility analysis, the generated
SDF geometry, the C++ plant plugin (via a dumped JSON), the learned dynamics
prior, the planner cost and the GUI. Changing a number here changes it
everywhere, which is the only way a vehicle model and its controller stay
consistent.
"""
from dataclasses import dataclass, field, asdict
import math, json

G   = 9.80665
RHO = 1.225


# ---------------------------------------------------------------- airframe --
@dataclass
class Airframe:
    """Lift+cruise eVTOL: 8 lift rotors, fixed wing, 1 pusher propeller.

    Lift+cruise rather than tiltrotor because hover and cruise effectors are
    physically separate. The control-allocation matrix is then block-diagonal
    in the two regimes instead of being a function of tilt angle, which makes
    the transition a smooth blend of two well-conditioned problems rather than
    one badly-conditioned one.
    """
    mass:      float = 1000.0        # kg   MTOM
    Ixx:       float = 2200.0        # kg m^2  roll  (wide span, booms outboard)
    Iyy:       float = 3000.0        # kg m^2  pitch
    Izz:       float = 4600.0        # kg m^2  yaw
    Ixz:       float = 120.0         # kg m^2  product of inertia

    # --- lift rotor array ---
    n_rotor:   int   = 8
    R_rotor:   float = 0.90          # m
    rotor_x:   tuple = (3.6, 1.2, -1.2, -3.6)   # longitudinal stations
    rotor_y:   float = 3.00          # m  lateral offset (+/-)
    rotor_z:   float = 0.35          # m  above CG
    FM:        float = 0.75          # figure of merit
    k_rotor:   float = 3.030e-2      # N/(rad/s)^2 = C_T rho A R^2, C_T=0.012
    b_rotor:   float = 2.8164e-3     # Nm/(rad/s)^2 from P=T.v_i/FM at hover
    w_rotor_max: float = 265.0       # rad/s -> T/W 1.74, tip M0.70
    tau_rotor: float = 0.055         # s  first-order motor+ESC lag

    # --- wing / tail ---
    S_wing:    float = 11.00         # m^2
    b_span:    float = 12.00         # m
    CL0:       float = 0.12
    CL_alpha:  float = 5.10          # 1/rad  (finite-AR lifting line)
    CL_max:    float = 1.50
    CD0:       float = 0.035
    oswald:    float = 0.80
    S_tail:    float = 2.20          # m^2  horizontal tail
    l_tail:    float = 4.30          # m  tail arm
    S_fin:     float = 1.40          # m^2  vertical fin
    Cm0:       float = 0.020
    Cm_alpha:  float = -0.85         # 1/rad  static pitch stability (<0)
    Cn_beta:   float = 0.110         # 1/rad  weathercock stability (>0)
    Cl_beta:   float = -0.085        # 1/rad  dihedral effect (<0)

    # --- pusher propeller ---
    T_push_max: float = 3600.0       # N -> 0->V_cruise in 14.3 s
    eta_push:   float = 0.82
    push_x:     float = -4.60        # m

    # --- control surfaces ---
    d_ail_max: float = math.radians(25.0)
    d_ele_max: float = math.radians(25.0)
    d_rud_max: float = math.radians(25.0)
    Cl_dail:   float = 0.150         # 1/rad
    Cm_dele:   float = -1.100        # 1/rad
    Cn_drud:   float = -0.075        # 1/rad
    tau_surf:  float = 0.040         # s  servo lag

    # --- energy ---
    E_batt_kwh: float = 60.0
    eta_motor:  float = 0.88

    # --- geometry for rendering / collision ---
    fus_len:   float = 9.00
    fus_w:     float = 1.35
    fus_h:     float = 1.45

    @property
    def AR(self):     return self.b_span ** 2 / self.S_wing
    @property
    def chord(self):  return self.S_wing / self.b_span
    @property
    def A_disk(self): return self.n_rotor * math.pi * self.R_rotor ** 2
    @property
    def W(self):      return self.mass * G
    @property
    def V_stall(self):
        return math.sqrt(2 * self.W / (RHO * self.S_wing * self.CL_max))

    def rotor_positions(self):
        """(x, y, z, spin) for each lift rotor. Spin alternates for torque balance."""
        out, k = [], 0
        for x in self.rotor_x:
            for y in (self.rotor_y, -self.rotor_y):
                out.append((x, y, self.rotor_z, 1.0 if (k % 2 == 0) else -1.0))
                k += 1
        return out


# ----------------------------------------------------------------- sensors --
@dataclass
class Sensors:
    """Sensor suite mirrors what a certified eVTOL actually carries.

    Resolutions and FOVs are NOT arbitrary: they are the ones that made the
    detection-range gate in scripts/feasibility.py pass with >=1.5x time
    margin against a 2 m non-cooperative UAS, which is the binding threat.
    """
    nav_w: int = 160; nav_h: int = 120       # wide navigation camera
    nav_hfov_deg: float = 90.0
    nav_near: float = 0.35; nav_far: float = 300.0

    daa_w: int = 160; daa_h: int = 120       # narrow detect-and-avoid camera
    daa_hfov_deg: float = 25.0
    daa_near: float = 1.0;  daa_far: float = 3000.0

    cam_hz: float = 30.0
    imu_hz: float = 250.0

    gps_sigma_xy: float = 0.65               # m  1-sigma (RTK-less GNSS/INS)
    gps_sigma_z:  float = 1.10
    baro_sigma:   float = 0.40               # m
    imu_acc_sigma: float = 0.035             # m/s^2 /sqrt(Hz)
    imu_gyr_sigma: float = 0.0021            # rad/s /sqrt(Hz)

    adsb_range: float = 5000.0               # m  cooperative surveillance
    adsb_hz:    float = 1.0
    adsb_pos_sigma: float = 8.0              # m  broadcast position error

    max_tracks: int = 8                      # threat tracks fed to world model


# ------------------------------------------------------------------- world --
@dataclass
class World:
    extent:        float = 1600.0    # m  half-size of the city block region
    n_buildings:   int   = 110
    bld_h_range:   tuple = (30.0, 200.0)
    bld_w_range:   tuple = (22.0, 60.0)
    n_vertiports:  int   = 6
    n_unmapped:    int   = 14        # cranes/masts absent from the obstacle
                                     # database: only the camera can catch them
    corridor_alt:  float = 120.0     # m AGL  (FAA UAM corridor)
    n_traffic:     int   = 5         # cooperative eVTOL intruders
    n_suas:        int   = 6         # non-cooperative small UAS
    wind_mean:     float = 6.0       # m/s
    wind_gust:     float = 4.0       # m/s  Dryden sigma


# ---------------------------------------------------------------- learning --
@dataclass
class Learn:
    # RSSM (sizes fixed by the VRAM budget in scripts/feasibility.py)
    deter:      int = 512            # h_t
    stoch_cat:  int = 32             # z_t : 32 categoricals ...
    stoch_cls:  int = 32             # ... of 32 classes each
    hidden:     int = 512
    img_res:    int = 64             # world-model image resolution
    cnn_depth:  int = 48

    act_dim:    int = 6              # [T_col, phi_ref, th_ref, r_ref, push, sched]
    proprio_dim: int = 23

    horizon:    int = 64             # MPPI rollout steps = 3.2 s @ 20 Hz;
                                     # set by the 1.88 s avoidance manoeuvre
                                     # time in scripts/feasibility.py
    n_samples:  int = 512            # MPPI trajectories
    mppi_lambda: float = 0.65        # temperature
    mppi_sigma: float = 0.30         # exploration std in normalised action units
    mppi_beta:  float = 0.85         # AR(1) correlation of the sampled noise
                                     # along the horizon; ~0.33 s correlation
                                     # time, so each sample is a manoeuvre
                                     # rather than per-step jitter

    batch:      int = 20
    seq_len:    int = 40
    lr_world:   float = 3e-4
    lr_actor:   float = 8e-5
    kl_free:    float = 1.0          # free-nats floor
    kl_scale:   float = 1.0
    rep_scale:  float = 0.10         # DreamerV3 representation/dynamics split
    grad_clip:  float = 100.0

    ctrl_hz:    float = 20.0
    plan_hz:    float = 10.0         # MPPI replans at 10 Hz, the CBF shield
                                     # runs every control step at 20 Hz
    phys_hz:    float = 200.0   # 200/20 = exactly 10 physics substeps
                                 # per control step, for lockstep stepping

    replay_cap: int = 400_000
    vram_budget_mib: int = 4369      # 85% of measured free VRAM


# ----------------------------------------------------------------- safety ---
@dataclass
class Safety:
    """Separation standards. Values follow the UAM corridor / well-clear
    literature, scaled to the size of this simulated city."""
    r_static:    float = 20.0        # m  min clearance to buildings/terrain
    r_suas:      float = 15.0        # m  min clearance to non-cooperative UAS
    r_wellclear: float = 150.0       # m  horizontal well-clear vs other eVTOL
    h_wellclear: float = 30.0        # m  vertical well-clear vs other eVTOL
    # HOCBF gains. Roots of s^2 + a1 s + a0 set how early the barrier
    # starts pushing back. Aggressive gains make the filter a co-pilot
    # that fights the planner; these leave it as a last resort.
    cbf_alpha0:  float = 1.2
    cbf_alpha1:  float = 2.6
    bank_max_deg: float = 30.0
    accel_max:   float = 6.0         # m/s^2  commanded accel magnitude limit
    v_max:       float = 46.0        # m/s
    soc_reserve: float = 0.20        # abort mission below this


@dataclass
class Config:
    air:   Airframe = field(default_factory=Airframe)
    sen:   Sensors  = field(default_factory=Sensors)
    wld:   World    = field(default_factory=World)
    lrn:   Learn    = field(default_factory=Learn)
    saf:   Safety   = field(default_factory=Safety)

    def dump(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        return path


CFG = Config()

if __name__ == "__main__":
    a = CFG.air
    print(f"AR          = {a.AR:.2f}")
    print(f"chord       = {a.chord:.3f} m")
    print(f"disk area   = {a.A_disk:.2f} m^2")
    print(f"V_stall     = {a.V_stall:.1f} m/s")
    print(f"disk load   = {a.W/a.A_disk:.1f} N/m^2")
    # hover check: does the rotor coefficient actually lift the aircraft?
    w_hov = math.sqrt(a.W / (a.n_rotor * a.k_rotor))
    print(f"hover omega = {w_hov:.1f} rad/s  "
          f"({100*w_hov/a.w_rotor_max:.0f}% of max -> "
          f"{'OK' if w_hov < 0.8*a.w_rotor_max else 'RERATE ROTORS'})")
    print(f"thrust margin at max rpm = "
          f"{a.n_rotor*a.k_rotor*a.w_rotor_max**2/a.W:.2f}x weight")
    print("rotors:", a.rotor_positions())
