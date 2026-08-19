"""
Side-by-side comparison of two evaluation runs.

Reads the JSON written by evaluate.py and prints a controller-vs-controller
table. Both sides must have been run with the same --seed so the drawn
vertiport pairs, and therefore the routes and traffic encounters, are identical.
"""
import os, sys, json, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phywam.config import CFG

SF = CFG.saf

ROWS = [
    ("success rate",              "success_rate",            "%",  1, "hi"),
    ("collisions",                "collision_rate",          "%",  1, "lo"),
    ("mid-air collisions",        "mac",                     "",   0, "lo"),
    ("NMAC events (<30 m)",       "nmac_total",              "",   0, "lo"),
    ("LoWC events",               "lowc_total",              "",   0, "lo"),
    ("worst obstacle clearance",  "min_obs_clearance_m",     "m",  1, "hi"),
    ("mean obstacle clearance",   "mean_min_obs_clearance_m","m",  1, "hi"),
    ("flights below 20 m clr",    "obs_violations",          "",   0, "lo"),
    ("struck unmapped crane",     "hit_unmapped",            "",   0, "lo"),
    ("worst UNMAPPED clearance",  "min_unmapped_clearance_m","m",  1, "hi"),
    ("worst traffic separation",  "min_separation_m",        "m",  0, "hi"),
    ("mean traffic separation",   "mean_min_separation_m",   "m",  0, "hi"),
    ("barrier engagement",        "shield_rate_mean",        "%",  1, "lo"),
    ("path efficiency",           "path_efficiency",         "x",  2, "lo"),
    ("mean flight time",          "mean_flight_time_s",      "s",  1, "lo"),
    ("mean energy used",          "mean_energy_pct",         "%",  1, "lo"),
]


def load(p):
    d = json.load(open(p))
    return d["mode"], d["summary"]


def fmt(v, unit, dp):
    if v is None:
        return "     -"
    if unit == "%":
        v = 100 * v if v <= 1.0001 else v
    return f"{v:.{dp}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ma, sa = load(args.a)
    mb, sb = load(args.b)

    L = []
    A = L.append
    A("=" * 78)
    A(f"  CONTROLLER COMPARISON      {ma}  vs  {mb}")
    A(f"  n = {sa.get('n','?')} and {sb.get('n','?')} flights, identical routes")
    A("=" * 78)
    A(f"  {'metric':<28}{ma:>14}{mb:>14}{'better':>12}")
    A("  " + "-" * 68)
    for label, key, unit, dp, want in ROWS:
        va, vb = sa.get(key), sb.get(key)
        if va is None and vb is None:
            continue
        if va is None or vb is None:
            win = ""
        elif abs(va - vb) < 1e-9:
            win = "tie"
        else:
            better_a = (va > vb) if want == "hi" else (va < vb)
            win = ma if better_a else mb
        u = f" {unit}" if unit and unit != "%" else ("%" if unit == "%" else "")
        A(f"  {label:<28}{fmt(va,unit,dp)+u:>14}{fmt(vb,unit,dp)+u:>14}{win:>12}")
    A("  " + "-" * 68)
    A(f"  well clear = {SF.r_wellclear:.0f} m horizontal AND "
      f"{SF.h_wellclear:.0f} m vertical vs cooperative traffic")
    A(f"  obstacle clearance requirement = {SF.r_static:.0f} m")
    A("=" * 78)
    txt = "\n".join(L)
    print(txt)
    if args.out:
        open(args.out, "w").write(txt)


if __name__ == "__main__":
    main()
