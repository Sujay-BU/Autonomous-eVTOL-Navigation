// ---------------------------------------------------------------------------
//  Phy-WAM traffic plugin.
//
//  Drives the intruders: cooperative eVTOLs (aircraft-sized, corridor
//  altitude, ADS-B equipped) and non-cooperative small UAS (small, erratic,
//  no broadcast -- these are the binding threat identified by the perception
//  analysis in scripts/feasibility.py).
//
//  A configurable fraction of legs are deliberately aimed to conflict with
//  the ownship, otherwise random traffic in a 3 km city almost never produces
//  an encounter and the avoidance behaviour would never be exercised.
//
//  Publishes ground truth on <ns>/traffic; the Python side degrades it into a
//  realistic ADS-B feed (cooperative only, 1 Hz, position noise) and uses the
//  untouched truth solely for scoring safety metrics.
// ---------------------------------------------------------------------------
#include "PhyWamAero.hh"

#include <gz/plugin/Register.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Pose.hh>
#include <gz/transport/Node.hh>
#include <gz/msgs/float_v.pb.h>
#include <gz/math/Pose3.hh>

#include <sstream>
#include <string>
#include <vector>

namespace phywam {
using namespace gz;

struct Bld { double x, y, hw, hd, h; };

class PhyWamTraffic : public sim::System,
                      public sim::ISystemConfigure,
                      public sim::ISystemPreUpdate {
 public:
  void Configure(const sim::Entity &, const std::shared_ptr<const sdf::Element> &sdf,
                 sim::EntityComponentManager &ecm, sim::EventManager &) override {
    nT_    = sdf->Get<int>("n_traffic", 5).first;
    nS_    = sdf->Get<int>("n_suas", 6).first;
    ext_   = sdf->Get<double>("extent", 1600.0).first;
    alt_   = sdf->Get<double>("corridor_alt", 120.0).first;
    pConf_ = sdf->Get<double>("p_conflict", 0.55).first;
    ns_    = sdf->Get<std::string>("namespace", std::string("phywam")).first;
    rng_.rng ^= (uint64_t)sdf->Get<int>("seed", 1).first * 0xD1B54A32D192ED03ull;

    // buildings, as "x,y,halfw,halfd,height x,y,..." so intruders can climb
    // over them instead of flying through
    std::string bs = sdf->Get<std::string>("buildings", std::string("")).first;
    std::stringstream ss(bs); std::string tok;
    while (ss >> tok) {
      Bld b; char c;
      std::stringstream t(tok);
      if (t >> b.x >> c >> b.y >> c >> b.hw >> c >> b.hd >> c >> b.h)
        blds_.push_back(b);
    }

    pub_ = node_.Advertise<msgs::Float_V>(ns_ + "/traffic");
    gzmsg << "[phywam] traffic configured: " << blds_.size()
          << " buildings, binding intruders on first update\n";
  }

  void PreUpdate(const sim::UpdateInfo &info,
                 sim::EntityComponentManager &ecm) override {
    if (info.paused) return;
    if (!bound_) {                       // model entities exist only now
      for (int i = 0; i < nT_; ++i) Add(ecm, "traffic_" + std::to_string(i), 0);
      for (int i = 0; i < nS_; ++i) Add(ecm, "suas_"    + std::to_string(i), 1);
      own_ = ByName(ecm, "ownship");
      if ((int)ag_.size() == nT_ + nS_ && own_ != sim::kNullEntity) {
        bound_ = true;
        gzmsg << "[phywam] traffic bound: " << ag_.size() << " intruders\n";
      } else {
        ag_.clear();
        return;
      }
    }
    if (ag_.empty()) return;
    const double dt = std::chrono::duration<double>(info.dt).count();
    const double t  = std::chrono::duration<double>(info.simTime).count();
    if (dt <= 0.0) return;

    // ownship position, used to seed conflicting legs
    V3 op(0, 0, alt_);
    if (own_ != sim::kNullEntity) {
      if (auto *pc = ecm.Component<sim::components::Pose>(own_))
        op = V3(pc->Data().Pos().X(), pc->Data().Pos().Y(), pc->Data().Pos().Z());
    }

    for (auto &a : ag_) {
      if (t >= a.t_next) NewLeg(a, op, t);
      a.p = a.p + a.v * dt;

      // climb over any building we are about to enter
      for (const auto &b : blds_) {
        if (std::fabs(a.p.x - b.x) < b.hw + 12.0 &&
            std::fabs(a.p.y - b.y) < b.hd + 12.0 && a.p.z < b.h + 22.0) {
          a.p.z += std::min(28.0 * dt, b.h + 22.0 - a.p.z);
        }
      }
      if (std::fabs(a.p.x) > ext_ * 1.25 || std::fabs(a.p.y) > ext_ * 1.25)
        a.t_next = t;                                   // force a new leg

      math::Pose3d np(a.p.x, a.p.y, a.p.z, 0, 0,
                      std::atan2(a.v.y, a.v.x));
      ecm.SetComponentData<sim::components::Pose>(a.e, np);
      ecm.SetChanged(a.e, sim::components::Pose::typeId,
                     sim::ComponentState::OneTimeChange);
    }

    if (t - tPub_ >= 0.05) {                            // 20 Hz truth feed
      tPub_ = t;
      msgs::Float_V m;
      for (const auto &a : ag_) {
        m.add_data(a.p.x); m.add_data(a.p.y); m.add_data(a.p.z);
        m.add_data(a.v.x); m.add_data(a.v.y); m.add_data(a.v.z);
        m.add_data(double(a.kind));
      }
      pub_.Publish(m);
    }
  }

 private:
  struct Agent { sim::Entity e; V3 p, v; double t_next{0}; int kind{0}; };

  sim::Entity ByName(sim::EntityComponentManager &ecm, const std::string &n) {
    return ecm.EntityByComponents(sim::components::Name(n),
                                  sim::components::Model());
  }
  void Add(sim::EntityComponentManager &ecm, const std::string &n, int kind) {
    auto e = ByName(ecm, n);
    if (e == sim::kNullEntity) return;
    Agent a; a.e = e; a.kind = kind; a.t_next = 0.0;
    a.p = V3(0, 0, kind ? 90.0 : alt_);
    ag_.push_back(a);
  }

  // Pick a fresh straight leg. With probability p_conflict, aim the leg so it
  // passes close to where the ownship will be, which is what actually
  // generates encounters worth avoiding.
  void NewLeg(Agent &a, const V3 &own, double t) {
    const bool suas = (a.kind == 1);
    const double sp = suas ? (6.0 + 8.0 * rng_.u01())
                           : (32.0 + 14.0 * rng_.u01());
    const double z  = suas ? (55.0 + 75.0 * rng_.u01())
                           : (alt_ + 20.0 + 70.0 * rng_.u01());
    if (rng_.u01() < pConf_) {
      // start far away on a random bearing, fly at the ownship with a miss
      // offset so encounters vary from near-miss to comfortable pass
      const double th = rng_.u01() * 2.0 * M_PI;
      const double R  = suas ? (260.0 + 200.0 * rng_.u01())
                             : (900.0 + 700.0 * rng_.u01());
      a.p = V3(own.x + R * std::cos(th), own.y + R * std::sin(th), z);
      const double off = (rng_.u01() - 0.5) * (suas ? 60.0 : 320.0);
      V3 aim(own.x - off * std::sin(th), own.y + off * std::cos(th), own.z);
      V3 d = aim - a.p;
      const double n = std::max(d.norm(), 1e-3);
      a.v = d * (sp / n);
    } else {
      const double th = rng_.u01() * 2.0 * M_PI;
      a.p = V3(ext_ * 1.15 * std::cos(th), ext_ * 1.15 * std::sin(th), z);
      const double ph = th + M_PI + (rng_.u01() - 0.5) * 1.1;
      a.v = V3(sp * std::cos(ph), sp * std::sin(ph), 0.0);
    }
    a.t_next = t + (2.0 * ext_) / sp;
  }

  std::vector<Agent> ag_;
  std::vector<Bld>   blds_;
  sim::Entity own_{sim::kNullEntity};
  Gust rng_;
  transport::Node node_;
  transport::Node::Publisher pub_;
  std::string ns_{"phywam"};
  int nT_{5}, nS_{6};
  double ext_{1600.0}, alt_{120.0}, pConf_{0.55}, tPub_{-1.0};
  bool bound_{false};
};

}  // namespace phywam

GZ_ADD_PLUGIN(phywam::PhyWamTraffic, gz::sim::System,
              phywam::PhyWamTraffic::ISystemConfigure,
              phywam::PhyWamTraffic::ISystemPreUpdate)
GZ_ADD_PLUGIN_ALIAS(phywam::PhyWamTraffic, "phywam::PhyWamTraffic")
