"""
Safety and performance metrics.

The vocabulary follows UAS detect-and-avoid practice rather than generic RL
reporting, because "mean episode return" tells an aviation reader nothing:

  MAC / NMAC   mid-air collision, and near mid-air (<30 m) -- the standard
               severity pair used in DAA evaluation
  LoWC         loss of well clear: horizontal separation below 150 m while
               vertical is below 30 m, against cooperative traffic
  min clearance  closest approach to any real obstacle, mapped or not
  shield rate  fraction of control steps on which the barrier filter had to
               repair the planner's action -- a direct measure of how often
               the learned layer proposed something unsafe
"""
import numpy as np

from .config import CFG

SF = CFG.saf


def summarise(runs):
    if not runs:
        return {}
    g = lambda k: np.array([r[k] for r in runs], float)
    out = {}
    out["n"] = len(runs)
    out["success_rate"] = float(np.mean([r["success"] for r in runs]))
    outc = {}
    for r in runs:
        outc[r["outcome"]] = outc.get(r["outcome"], 0) + 1
    out["outcomes"] = outc

    coll = sum(1 for r in runs if r["outcome"] in
               ("hit_building", "hit_ground", "midair_collision"))
    out["collision_rate"] = coll / len(runs)
    out["mac"] = sum(1 for r in runs if r["outcome"] == "midair_collision")
    out["nmac_total"] = int(g("nmac").sum())
    out["nmac_runs"] = int(np.sum(g("nmac") > 0))
    out["lowc_total"] = int(g("lox").sum())
    out["lowc_runs"] = int(np.sum(g("lox") > 0))

    out["hit_unmapped"] = int(sum(r.get("hit_unmapped", 0) for r in runs))
    out["hit_mapped"] = coll - out["hit_unmapped"] - out["mac"] - \
        sum(1 for r in runs if r["outcome"] == "hit_ground")
    mou = np.minimum(g("min_obs_unmapped"), 9999)
    out["min_unmapped_clearance_m"] = float(mou.min())
    out["unmapped_violations"] = int(np.sum(mou < SF.r_static))
    mo = g("min_obs"); ms = np.minimum(g("min_sep"), 9999)
    out["min_obs_clearance_m"] = float(mo.min())
    out["mean_min_obs_clearance_m"] = float(mo.mean())
    out["obs_violations"] = int(np.sum(mo < SF.r_static))
    out["min_separation_m"] = float(ms.min())
    out["mean_min_separation_m"] = float(ms.mean())

    out["shield_rate_mean"] = float(g("shield_rate").mean())
    out["shield_rate_max"] = float(g("shield_rate").max())

    ok = [r for r in runs if r["success"]]
    if ok:
        eff = np.array([r["path_len"] / max(r["direct"], 1.0) for r in ok])
        out["path_efficiency"] = float(eff.mean())
        out["mean_flight_time_s"] = float(np.mean([r["t"] for r in ok]))
        out["mean_energy_pct"] = float(100*np.mean([r["soc_used"] for r in ok]))
    out["mean_return"] = float(g("ret").mean())
    return out


def format_report(s, runs, mode):
    if not s:
        return "no runs"
    L = []
    A = L.append
    A("=" * 74)
    A(f"  PHY-WAM SAFETY REPORT   controller = {mode}   n = {s['n']} flights")
    A("=" * 74)
    A("")
    A("  MISSION")
    A(f"    success rate                {100*s['success_rate']:6.1f} %")
    A(f"    outcome breakdown           " +
      ", ".join(f"{k}={v}" for k, v in sorted(s["outcomes"].items())))
    if "path_efficiency" in s:
        A(f"    path efficiency             {s['path_efficiency']:6.2f} x direct")
        A(f"    mean flight time            {s['mean_flight_time_s']:6.1f} s")
        A(f"    mean energy used            {s['mean_energy_pct']:6.1f} % SoC")
    A("")
    A("  SAFETY  (aviation DAA vocabulary)")
    A(f"    collisions (any)            {100*s['collision_rate']:6.1f} %  "
      f"({int(s['collision_rate']*s['n'])}/{s['n']})")
    A(f"    mid-air collisions (MAC)    {s['mac']:6d}")
    A(f"      of which struck a MAPPED obstacle   {max(s['hit_mapped'],0):4d}")
    A(f"      of which struck an UNMAPPED crane   {s['hit_unmapped']:4d}"
      f"   <- only a camera can prevent these")
    A(f"    near mid-air (<30 m, NMAC)  {s['nmac_total']:6d} events in "
      f"{s['nmac_runs']} flights")
    A(f"    loss of well clear (LoWC)   {s['lowc_total']:6d} events in "
      f"{s['lowc_runs']} flights")
    A(f"      (well clear = {SF.r_wellclear:.0f} m horizontal "
      f"AND {SF.h_wellclear:.0f} m vertical vs cooperative traffic)")
    A("")
    A("  SEPARATION")
    A(f"    min obstacle clearance      {s['min_obs_clearance_m']:6.1f} m   "
      f"(requirement {SF.r_static:.0f} m)")
    A(f"    mean of per-flight minima   {s['mean_min_obs_clearance_m']:6.1f} m")
    A(f"    flights below requirement   {s['obs_violations']:6d} / {s['n']}")
    A(f"    min clearance to UNMAPPED   "
      f"{min(s['min_unmapped_clearance_m'], 9999):6.1f} m   "
      f"({s['unmapped_violations']}/{s['n']} flights below requirement)")
    A(f"    min traffic separation      {s['min_separation_m']:6.0f} m")
    A(f"    mean of per-flight minima   {s['mean_min_separation_m']:6.0f} m")
    A("")
    A("  BARRIER FILTER")
    A(f"    steps repaired (mean)       {100*s['shield_rate_mean']:6.1f} %")
    A(f"    steps repaired (worst run)  {100*s['shield_rate_max']:6.1f} %")
    A("")
    A("  PER-FLIGHT")
    A(f"    {'#':>2} {'route':>11} {'direct':>7} {'outcome':>17} {'t(s)':>6} "
      f"{'path(m)':>8} {'minObs':>7} {'minSep':>7} {'LoWC':>5} {'NMAC':>5}")
    for r in runs:
        A(f"    {r.get('ep','-'):>2} "
          f"{'VP%d->VP%d' % (r['start_vp'], r['goal_vp']):>11} "
          f"{r['direct']:7.0f} {r['outcome']:>17} {r['t']:6.1f} "
          f"{r['path_len']:8.0f} {r['min_obs']:7.1f} "
          f"{min(r['min_sep'],9999):7.0f} {r['lox']:5d} {r['nmac']:5d}")
    A("=" * 74)
    return "\n".join(L)
