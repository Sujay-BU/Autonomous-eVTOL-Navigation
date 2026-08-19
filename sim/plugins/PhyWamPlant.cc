// ---------------------------------------------------------------------------
//  Phy-WAM plant plugin.
//
//  Runs INSIDE Gazebo at the physics rate (250 Hz) and plays the role of the
//  flight control computer plus the airframe itself:
//
//     Python (20 Hz)  --/phywam/cmd-->  [ attitude loop -> allocation ->
//                                         actuator lag -> aero+rotor+pusher
//                                         wrench -> link ]  --/phywam/state-->
//
//  Keeping the 250 Hz loop in-process is what lets the whole thing run many
//  times faster than real time during training: Python never sits in the
//  inner loop, it only sets references.
// ---------------------------------------------------------------------------
#include "PhyWamAero.hh"

#include <gz/plugin/Register.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/Util.hh>
#include <gz/transport/Node.hh>
#include <gz/msgs/float_v.pb.h>
#include <gz/math/Pose3.hh>
#include <gz/math/Vector3.hh>
#include <gz/sim/components/LinearVelocityCmd.hh>
#include <gz/sim/components/AngularVelocityCmd.hh>

#include <mutex>
#include <string>

namespace phywam {

using namespace gz;

// state vector layout published on /phywam/state  (see docs/ARCHITECTURE.md)
enum : int {
  S_POS = 0,        // 0..2   position, world ENU
  S_QUAT = 3,       // 3..6   orientation quaternion w,x,y,z (world->body FLU)
  S_VEL_W = 7,      // 7..9   linear velocity, world ENU
  S_VEL_B = 10,     // 10..12 body velocity FRD (u,v,w)
  S_OMEGA = 13,     // 13..15 body rates FRD (p,q,r)
  S_ACC_B = 16,     // 16..18 specific force FRD (what the IMU measures)
  S_VAIR = 19, S_ALPHA = 20, S_BETA = 21, S_AGL = 22,
  S_SOC = 23, S_POWER = 24,
  S_ROTOR = 25,     // 25..32 rotor speeds
  S_PUSH = 33, S_AIL = 34, S_ELE = 35, S_RUD = 36,
  S_WIND = 37,      // 37..39 wind, world ENU
  S_TIME = 40,
  S_LEN = 41
};

class PhyWamPlant : public sim::System,
                    public sim::ISystemConfigure,
                    public sim::ISystemPreUpdate,
                    public sim::ISystemPostUpdate {
 public:
  void Configure(const sim::Entity &entity,
                 const std::shared_ptr<const sdf::Element> &sdf,
                 sim::EntityComponentManager &ecm,
                 sim::EventManager &) override {
    model_ = sim::Model(entity);
    const std::string linkName =
        sdf->Get<std::string>("link_name", std::string("base_link")).first;
    link_ = sim::Link(model_.LinkByName(ecm, linkName));
    if (!link_.Valid(ecm)) {
      gzerr << "[phywam] link '" << linkName << "' not found\n";
      return;
    }
    link_.EnableVelocityChecks(ecm, true);
    link_.EnableAccelerationChecks(ecm, true);

    ns_ = sdf->Get<std::string>("namespace", std::string("phywam")).first;
    alloc_.build(par_);
    soc_ = 1.0;
    E_batt_J_ = sdf->Get<double>("battery_kwh", 60.0).first * 3.6e6;
    wind_mean_ = sdf->Get<double>("wind_mean", 6.0).first;
    wind_dir_  = sdf->Get<double>("wind_dir", 0.6).first;
    gust_sig_  = sdf->Get<double>("wind_gust", 4.0).first;
    gust_.rng ^= (uint64_t)sdf->Get<int>("seed", 1).first * 0x2545F4914F6CDD1Dull;

    node_.Subscribe(ns_ + "/cmd", &PhyWamPlant::OnCmd, this);
    node_.Subscribe(ns_ + "/reset", &PhyWamPlant::OnReset, this);
    linkE_ = model_.LinkByName(ecm, linkName);
    pubState_ = node_.Advertise<msgs::Float_V>(ns_ + "/state");
    gzmsg << "[phywam] plant ready on '" << ns_ << "'\n";
  }

  // -------------------------------------------------------------- PreUpdate
  void PreUpdate(const sim::UpdateInfo &info,
                 sim::EntityComponentManager &ecm) override {
    if (info.paused || !link_.Valid(ecm)) return;
    const double dt = std::chrono::duration<double>(info.dt).count();
    if (dt <= 0.0) return;

    // ---- episode reset ----------------------------------------------------
    // Teleport, then hold zero velocity for a few steps so the physics engine
    // actually clears the momentum instead of carrying it through the jump.
    {
      std::lock_guard<std::mutex> lk(mtx_);
      if (rstPending_) {
        model_.SetWorldPoseCmd(ecm, math::Pose3d(rx_, ry_, rz_, 0, 0, ryaw_));
        act_ = Actuators();
        cmd_ = Command();
        soc_ = rsoc_;
        rstHold_ = 6;
        rstPending_ = false;
      }
    }
    if (rstHold_ > 0) {
      --rstHold_;
      link_.SetLinearVelocity(ecm, math::Vector3d::Zero);
      link_.SetAngularVelocity(ecm, math::Vector3d::Zero);
      return;                       // no aero this step: the body is being placed
    }
    if (rstHold_ == 0 && linkE_ != sim::kNullEntity && clearCmd_) {
      ecm.RemoveComponent<sim::components::LinearVelocityCmd>(linkE_);
      ecm.RemoveComponent<sim::components::AngularVelocityCmd>(linkE_);
      clearCmd_ = false;
    }

    const auto poseO = link_.WorldPose(ecm);
    const auto velO  = link_.WorldLinearVelocity(ecm);
    const auto omgO  = link_.WorldAngularVelocity(ecm);
    if (!poseO || !velO || !omgO) return;
    const math::Pose3d pose = *poseO;
    const math::Quaterniond R = pose.Rot();

    // ---- world -> body. Gazebo body is FLU; our physics is FRD ------------
    const math::Vector3d vB_flu = R.RotateVectorReverse(*velO);
    const math::Vector3d wB_flu = R.RotateVectorReverse(*omgO);
    const V3 v_body(vB_flu.X(), -vB_flu.Y(), -vB_flu.Z());
    const V3 omega (wB_flu.X(), -wB_flu.Y(), -wB_flu.Z());

    // ---- atmosphere -------------------------------------------------------
    const double agl = std::max(pose.Pos().Z(), 0.0);
    const V3 gz_ = gust_.step(dt, gust_sig_, 200.0, std::max(v_body.x, 5.0));
    const math::Vector3d windW(wind_mean_ * std::cos(wind_dir_) + gz_.x,
                               wind_mean_ * std::sin(wind_dir_) + gz_.y,
                               0.35 * gz_.z);
    const math::Vector3d wB = R.RotateVectorReverse(windW);
    const V3 wind_body(wB.X(), -wB.Y(), -wB.Z());
    const V3 v_air = v_body - wind_body;

    // ---- Euler angles (FRD convention) ------------------------------------
    const math::Vector3d rpy = R.Euler();
    const double roll = rpy.X(), pitch = -rpy.Y();
    const double Vt = v_air.norm();

    Command c;
    { std::lock_guard<std::mutex> lk(mtx_); c = cmd_; }
    const double sch = clampd(c.sched, 0.0, 1.0);

    // ---- attitude -> rate -> torque (cascade P/PD) -------------------------
    const double p_des = clampd(6.0 * (c.roll_ref  - roll ), -1.4, 1.4);
    const double q_des = clampd(6.0 * (c.pitch_ref - pitch), -1.2, 1.2);
    const double r_des = clampd(c.yawrate_ref, -0.9, 0.9);

    const double Mx = par_.Ixx * (5.5 * (p_des - omega.x));
    const double My = par_.Iyy * (5.0 * (q_des - omega.y));
    const double Mz = par_.Izz * (2.6 * (r_des - omega.z));

    // ---- collective: hover-borne share of weight ---------------------------
    const double Tmax_tot = kNRotor * alloc_.Tmax();
    const double Fz_des = -clampd(c.thrust_col, 0.0, 1.0) * Tmax_tot;

    // rotors carry attitude authority in hover, surfaces take over in cruise
    const double wr = 1.0 - 0.85 * sch;
    const double d[4] = {Fz_des * (1.0 - 0.92 * sch),
                         Mx * wr, My * wr, Mz * wr};
    double Tcmd[kNRotor];
    alloc_.solve(d, Tcmd);

    // ---- actuator dynamics: first-order lag on rotors and servos ----------
    const double ar = std::exp(-dt / par_.tau_rotor);
    for (int i = 0; i < kNRotor; ++i) {
      const double w_t = std::sqrt(std::max(Tcmd[i], 0.0) / par_.k_rotor);
      act_.w[i] = ar * act_.w[i] + (1.0 - ar) * clampd(w_t, 0.0, par_.w_max);
    }
    const double as = std::exp(-dt / par_.tau_surf);
    const double qbar = 0.5 * kRho * Vt * Vt * par_.S_wing;
    const double ail_t = (qbar > 1.0)
        ? clampd(Mx * sch / (qbar * par_.b_span * par_.Cl_dail), -par_.d_max, par_.d_max) : 0.0;
    const double ele_t = (qbar > 1.0)
        ? clampd(My * sch / (qbar * par_.chord  * par_.Cm_dele), -par_.d_max, par_.d_max) : 0.0;
    const double rud_t = (qbar > 1.0)
        ? clampd(Mz * sch / (qbar * par_.b_span * par_.Cn_drud), -par_.d_max, par_.d_max) : 0.0;
    act_.ail = as * act_.ail + (1.0 - as) * ail_t;
    act_.ele = as * act_.ele + (1.0 - as) * ele_t;
    act_.rud = as * act_.rud + (1.0 - as) * rud_t;
    act_.push = clampd(c.push_thr, 0.0, 1.0);

    // ---- total wrench in FRD ----------------------------------------------
    double Tact[kNRotor];
    Wrench wR = rotorWrench(par_, act_, agl, v_air.x, Tact);
    Wrench wA = aeroWrench (par_, act_, v_air, omega);
    Wrench wP = pusherWrench(par_, act_, v_air.x);
    const V3 F = wR.F + wA.F + wP.F;
    const V3 M = wR.M + wA.M + wP.M;

    // ---- FRD -> FLU -> world, then apply -----------------------------------
    const math::Vector3d F_flu(F.x, -F.y, -F.z);
    const math::Vector3d M_flu(M.x, -M.y, -M.z);
    link_.AddWorldWrench(ecm, R.RotateVector(F_flu), R.RotateVector(M_flu));

    // ---- energy ------------------------------------------------------------
    double P = 0.0;
    for (int i = 0; i < kNRotor; ++i)
      P += par_.b_rotor * act_.w[i] * act_.w[i] * act_.w[i];
    P /= 0.88;
    P += par_.T_push_max * act_.push * std::max(Vt, 3.0) / (0.82 * 0.88);
    P += 2500.0;                                   // avionics + thermal
    soc_ = clampd(soc_ - P * dt / E_batt_J_, 0.0, 1.0);

    // cache for PostUpdate
    lastV_ = v_body; lastW_ = omega; lastAir_ = v_air; lastP_ = P;
    lastAgl_ = agl; lastWindW_ = windW;
    lastAcc_ = V3(F.x / par_.mass, F.y / par_.mass, F.z / par_.mass);
  }

  // ------------------------------------------------------------- PostUpdate
  void PostUpdate(const sim::UpdateInfo &info,
                  const sim::EntityComponentManager &ecm) override {
    if (info.paused || !link_.Valid(ecm)) return;
    const double t = std::chrono::duration<double>(info.simTime).count();
    if (t - tPub_ < 0.01) return;                 // 100 Hz
    tPub_ = t;
    const auto poseO = link_.WorldPose(ecm);
    const auto velO  = link_.WorldLinearVelocity(ecm);
    if (!poseO || !velO) return;

    msgs::Float_V m;
    m.mutable_data()->Resize(S_LEN, 0.0f);
    auto *d = m.mutable_data()->mutable_data();
    const auto &p = poseO->Pos(); const auto &q = poseO->Rot();
    d[S_POS + 0] = p.X(); d[S_POS + 1] = p.Y(); d[S_POS + 2] = p.Z();
    d[S_QUAT + 0] = q.W(); d[S_QUAT + 1] = q.X();
    d[S_QUAT + 2] = q.Y(); d[S_QUAT + 3] = q.Z();
    d[S_VEL_W + 0] = velO->X(); d[S_VEL_W + 1] = velO->Y(); d[S_VEL_W + 2] = velO->Z();
    d[S_VEL_B + 0] = lastV_.x; d[S_VEL_B + 1] = lastV_.y; d[S_VEL_B + 2] = lastV_.z;
    d[S_OMEGA + 0] = lastW_.x; d[S_OMEGA + 1] = lastW_.y; d[S_OMEGA + 2] = lastW_.z;
    d[S_ACC_B + 0] = lastAcc_.x; d[S_ACC_B + 1] = lastAcc_.y; d[S_ACC_B + 2] = lastAcc_.z;
    const double Vt = lastAir_.norm();
    d[S_VAIR] = Vt;
    d[S_ALPHA] = (Vt > 0.5) ? std::atan2(lastAir_.z, lastAir_.x) : 0.0;
    d[S_BETA]  = (Vt > 0.5) ? std::asin(clampd(lastAir_.y / Vt, -1.0, 1.0)) : 0.0;
    d[S_AGL] = lastAgl_; d[S_SOC] = soc_; d[S_POWER] = lastP_;
    for (int i = 0; i < kNRotor; ++i) d[S_ROTOR + i] = act_.w[i];
    d[S_PUSH] = act_.push; d[S_AIL] = act_.ail;
    d[S_ELE] = act_.ele;  d[S_RUD] = act_.rud;
    d[S_WIND + 0] = lastWindW_.X(); d[S_WIND + 1] = lastWindW_.Y();
    d[S_WIND + 2] = lastWindW_.Z();
    d[S_TIME] = t;
    pubState_.Publish(m);
  }

 private:
  void OnReset(const msgs::Float_V &m) {
    if (m.data_size() < 4) return;
    std::lock_guard<std::mutex> lk(mtx_);
    rx_ = m.data(0); ry_ = m.data(1); rz_ = m.data(2); ryaw_ = m.data(3);
    rsoc_ = (m.data_size() > 4) ? m.data(4) : 1.0;
    rstPending_ = true;
    clearCmd_ = true;
  }

  void OnCmd(const msgs::Float_V &m) {
    if (m.data_size() < 6) return;
    std::lock_guard<std::mutex> lk(mtx_);
    cmd_.thrust_col  = m.data(0);
    cmd_.roll_ref    = m.data(1);
    cmd_.pitch_ref   = m.data(2);
    cmd_.yawrate_ref = m.data(3);
    cmd_.push_thr    = m.data(4);
    cmd_.sched       = m.data(5);
  }

  sim::Model model_{sim::kNullEntity};
  sim::Link  link_{sim::kNullEntity};
  Params     par_;
  Actuators  act_;
  Allocator  alloc_;
  Command    cmd_;
  Gust       gust_;
  std::mutex mtx_;
  transport::Node node_;
  transport::Node::Publisher pubState_;
  std::string ns_{"phywam"};
  double soc_{1.0}, E_batt_J_{2.16e8}, tPub_{-1.0};
  double wind_mean_{6.0}, wind_dir_{0.6}, gust_sig_{4.0};
  double lastP_{0.0}, lastAgl_{0.0};
  sim::Entity linkE_{sim::kNullEntity};
  bool rstPending_{false}, clearCmd_{false};
  int rstHold_{0};
  double rx_{0}, ry_{0}, rz_{0}, ryaw_{0}, rsoc_{1.0};
  V3 lastV_, lastW_, lastAir_, lastAcc_;
  math::Vector3d lastWindW_;
};

}  // namespace phywam

GZ_ADD_PLUGIN(phywam::PhyWamPlant, gz::sim::System,
              phywam::PhyWamPlant::ISystemConfigure,
              phywam::PhyWamPlant::ISystemPreUpdate,
              phywam::PhyWamPlant::ISystemPostUpdate)
GZ_ADD_PLUGIN_ALIAS(phywam::PhyWamPlant, "phywam::PhyWamPlant")
