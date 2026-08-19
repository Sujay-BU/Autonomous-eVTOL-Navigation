"""
Procedurally generate the Gazebo world: urban block, vertiports, the eVTOL,
cooperative traffic and non-cooperative small UAS.

Seeded, so every evaluation run can get a different city while the vehicle and
sensor definitions stay byte-identical to what was trained on.
"""
import math, random, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phywam.config import CFG

A, S, W = CFG.air, CFG.sen, CFG.wld


def _mat(r, g, b, a=1.0):
    return (f"<material><ambient>{r*.4:.3f} {g*.4:.3f} {b*.4:.3f} {a}</ambient>"
            f"<diffuse>{r:.3f} {g:.3f} {b:.3f} {a}</diffuse>"
            f"<specular>0.12 0.12 0.12 1</specular></material>")


TEXDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "sim", "materials", "textures")


def _tex(png, r=0.85, m=0.0, tint=(1.0, 1.0, 1.0)):
    tint = tuple(min(1.0, max(0.0, float(t))) for t in tint)   # SDF caps at 1.0
    """PBR material with an albedo map. Untextured primitives give the encoder
    nothing to lock onto, so every large surface in this world carries one."""
    return (f"<material><ambient>0.5 0.5 0.5 1</ambient>"
            f"<diffuse>{tint[0]:.3f} {tint[1]:.3f} {tint[2]:.3f} 1</diffuse>"
            f"<specular>0.06 0.06 0.06 1</specular>"
            f"<pbr><metal><albedo_map>{os.path.join(TEXDIR, png)}</albedo_map>"
            f"<metalness>{m}</metalness><roughness>{r}</roughness>"
            f"</metal></pbr></material>")


def _box(name, sx, sy, sz, pose, mat, collide=True):
    col = (f"<collision name='{name}_c'><geometry><box><size>{sx} {sy} {sz}"
           f"</size></box></geometry></collision>") if collide else ""
    return (f"<visual name='{name}_v'><pose>{pose}</pose><geometry><box><size>"
            f"{sx} {sy} {sz}</size></box></geometry>{mat}</visual>"
            + (col.replace("<collision", f"<collision") if collide else ""))


def _cyl(name, r, l, pose, mat):
    return (f"<visual name='{name}_v'><pose>{pose}</pose><geometry><cylinder>"
            f"<radius>{r}</radius><length>{l}</length></cylinder></geometry>"
            f"{mat}</visual>")


# --------------------------------------------------------------- the eVTOL --
def evtol_model(name, x, y, z, yaw, ns, seed, sensors=True, plugin=True):
    """Lift+cruise eVTOL. Geometry is derived from phywam.config so the picture
    and the physics always describe the same aircraft."""
    hull  = _mat(0.90, 0.92, 0.95)
    wing  = _mat(0.82, 0.84, 0.88)
    dark  = _mat(0.15, 0.16, 0.19)
    accent= _mat(0.05, 0.45, 0.85)

    v = []
    # fuselage: nose cone + body + tail cone
    v.append(_cyl("fus", A.fus_w/2, A.fus_len*0.62, f"0.3 0 0 0 1.5708 0", hull))
    v.append(f"<visual name='nose_v'><pose>3.9 0 0 0 1.5708 0</pose><geometry>"
             f"<sphere><radius>{A.fus_w/2:.3f}</radius></sphere></geometry>{hull}</visual>")
    v.append(f"<visual name='canopy_v'><pose>2.2 0 0.55 0 0 0</pose><geometry>"
             f"<box><size>2.6 1.1 0.5</size></box></geometry>{accent}</visual>")
    # wing, horizontal tail, fin
    v.append(_box("wing", A.chord, A.b_span, 0.16, "0 0 0.15 0 0 0", wing, False))
    v.append(_box("htail", 0.62, 3.9, 0.12, f"{-A.l_tail} 0 0.55 0 0 0", wing, False))
    v.append(_box("fin", 0.9, 0.12, 1.5, f"{-A.l_tail+0.2} 0 1.15 0 0 0", wing, False))
    # booms carrying the lift rotors
    for sgn in (1, -1):
        v.append(_box(f"boom{sgn}", 8.2, 0.28, 0.24,
                      f"0 {sgn*A.rotor_y} 0.18 0 0 0", dark, False))
    # lift rotors (visual discs) + pusher
    for i, (rx, ry, rz, sp) in enumerate(A.rotor_positions()):
        v.append(_cyl(f"rot{i}", A.R_rotor, 0.05, f"{rx} {ry} {rz+0.12} 0 0 0",
                      _mat(0.10, 0.10, 0.12, 0.55)))
        v.append(_cyl(f"hub{i}", 0.13, 0.22, f"{rx} {ry} {rz} 0 0 0", dark))
    v.append(_cyl("push", 0.85, 0.05, f"{A.push_x-0.3} 0 0.15 0 1.5708 0",
                  _mat(0.10, 0.10, 0.12, 0.55)))

    # collision: coarse hull + wing box (cheap, convex)
    col = (f"<collision name='hull'><pose>0 0 0.3 0 1.5708 0</pose><geometry>"
           f"<cylinder><radius>{A.fus_w/2:.3f}</radius><length>{A.fus_len*0.8:.2f}"
           f"</length></cylinder></geometry></collision>"
           f"<collision name='span'><pose>0 0 0.2 0 0 0</pose><geometry><box>"
           f"<size>{A.chord+0.4:.2f} {A.b_span:.2f} 0.4</size></box></geometry></collision>")

    sen = ""
    if sensors:
        hf_n = math.radians(S.nav_hfov_deg)
        hf_d = math.radians(S.daa_hfov_deg)
        sen = f"""
      <sensor name='cam_nav' type='camera'>
        <pose>4.15 0 0.15 0 0.2200 0</pose>
        <topic>{ns}/cam_nav</topic><update_rate>{S.cam_hz}</update_rate>
        <camera><horizontal_fov>{hf_n:.5f}</horizontal_fov>
          <image><width>{S.nav_w}</width><height>{S.nav_h}</height>
          <format>R8G8B8</format></image>
          <clip><near>{S.nav_near}</near><far>{S.nav_far}</far></clip>
        </camera>
      </sensor>
      <sensor name='depth_nav' type='depth_camera'>
        <pose>4.15 0 0.15 0 0.2200 0</pose>
        <topic>{ns}/depth_nav</topic><update_rate>{S.cam_hz}</update_rate>
        <camera><horizontal_fov>{hf_n:.5f}</horizontal_fov>
          <image><width>{S.nav_w}</width><height>{S.nav_h}</height>
          <format>R_FLOAT32</format></image>
          <clip><near>{S.nav_near}</near><far>{S.nav_far}</far></clip>
        </camera>
      </sensor>
      <sensor name='cam_daa' type='camera'>
        <pose>4.15 0 0.42 0 0 0</pose>
        <topic>{ns}/cam_daa</topic><update_rate>{S.cam_hz}</update_rate>
        <camera><horizontal_fov>{hf_d:.5f}</horizontal_fov>
          <image><width>{S.daa_w}</width><height>{S.daa_h}</height>
          <format>R8G8B8</format></image>
          <clip><near>{S.daa_near}</near><far>{S.daa_far}</far></clip>
        </camera>
      </sensor>
      <sensor name='imu' type='imu'>
        <topic>{ns}/imu</topic><update_rate>{S.imu_hz}</update_rate>
        <imu>
          <angular_velocity>
            <x><noise type='gaussian'><stddev>{S.imu_gyr_sigma}</stddev></noise></x>
            <y><noise type='gaussian'><stddev>{S.imu_gyr_sigma}</stddev></noise></y>
            <z><noise type='gaussian'><stddev>{S.imu_gyr_sigma}</stddev></noise></z>
          </angular_velocity>
          <linear_acceleration>
            <x><noise type='gaussian'><stddev>{S.imu_acc_sigma}</stddev></noise></x>
            <y><noise type='gaussian'><stddev>{S.imu_acc_sigma}</stddev></noise></y>
            <z><noise type='gaussian'><stddev>{S.imu_acc_sigma}</stddev></noise></z>
          </linear_acceleration>
        </imu>
      </sensor>"""

    plg = ""
    if plugin:
        plg = f"""
    <plugin filename='PhyWamPlant' name='phywam::PhyWamPlant'>
      <link_name>base_link</link_name>
      <namespace>{ns}</namespace>
      <battery_kwh>{A.E_batt_kwh}</battery_kwh>
      <wind_mean>{W.wind_mean}</wind_mean>
      <wind_gust>{W.wind_gust}</wind_gust>
      <wind_dir>{(seed*0.7)%6.283:.3f}</wind_dir>
      <seed>{seed}</seed>
    </plugin>"""

    return f"""
  <model name='{name}'>
    <pose>{x} {y} {z} 0 0 {yaw}</pose>
    <self_collide>false</self_collide>
    <link name='base_link'>
      <inertial><mass>{A.mass}</mass>
        <inertia><ixx>{A.Ixx}</ixx><iyy>{A.Iyy}</iyy><izz>{A.Izz}</izz>
        <ixy>0</ixy><ixz>{A.Ixz}</ixz><iyz>0</iyz></inertia>
      </inertial>
      {''.join(v)}
      {col}{sen}
    </link>{plg}
  </model>"""


# ------------------------------------------------------------- other traffic --
def intruder_model(name, x, y, z, kind, idx):
    """kind: 'evtol' (cooperative, aircraft-sized) or 'suas' (non-cooperative)."""
    if kind == "evtol":
        m = _mat(0.95, 0.75, 0.10)
        body = (_cyl("f", 0.62, 5.6, "0 0 0 0 1.5708 0", m) +
                _box("w", 0.85, 11.0, 0.14, "0 0 0.1 0 0 0", m, False) +
                _box("t", 0.5, 3.2, 0.1, "-3.8 0 0.4 0 0 0", m, False))
        r = 5.5
    else:
        m = _mat(0.95, 0.15, 0.15)
        body = (_box("b", 0.9, 0.9, 0.28, "0 0 0 0 0 0", m, False) +
                _cyl("p1", 0.55, 0.03, " 0.7  0.7 0.16 0 0 0", _mat(.1,.1,.1,.5)) +
                _cyl("p2", 0.55, 0.03, " 0.7 -0.7 0.16 0 0 0", _mat(.1,.1,.1,.5)) +
                _cyl("p3", 0.55, 0.03, "-0.7  0.7 0.16 0 0 0", _mat(.1,.1,.1,.5)) +
                _cyl("p4", 0.55, 0.03, "-0.7 -0.7 0.16 0 0 0", _mat(.1,.1,.1,.5)))
        r = 1.1
    return f"""
  <model name='{name}'>
    <pose>{x} {y} {z} 0 0 0</pose>
    <static>true</static>
    <link name='body'>
      <inertial><mass>1</mass><inertia><ixx>1</ixx><iyy>1</iyy><izz>1</izz>
      <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
      {body}
      <collision name='c'><geometry><sphere><radius>{r}</radius></sphere>
      </geometry></collision>
    </link>
  </model>"""


# ----------------------------------------------------------------- scenery --
def make_world(seed=1, out=None):
    rng = random.Random(seed)
    parts, meta = [], {"buildings": [], "vertiports": [], "seed": seed}

    # --- buildings on a jittered grid, leaving the corridor spine clearer ---
    placed = []
    tries = 0
    while len(placed) < W.n_buildings and tries < 4000:
        tries += 1
        bx = rng.uniform(-W.extent, W.extent)
        by = rng.uniform(-W.extent, W.extent)
        bw = rng.uniform(*W.bld_w_range)
        bd = rng.uniform(*W.bld_w_range)
        bh = rng.uniform(*W.bld_h_range)
        if any(abs(bx-px) < (bw+pw)*0.62 and abs(by-py) < (bd+pd)*0.62
               for px, py, pw, pd, _ in placed):
            continue
        placed.append((bx, by, bw, bd, bh))
    for i, (bx, by, bw, bd, bh) in enumerate(placed):
        fac = f"facade{rng.randrange(3)}.png"
        g = rng.uniform(0.72, 1.0)
        tint = rng.choice([(g, g, g*1.04), (g*1.04, g, g*0.94), (g*0.94, g*0.99, g*1.06)])
        parts.append(f"""
  <model name='bld_{i}'><static>true</static>
    <pose>{bx:.2f} {by:.2f} {bh/2:.2f} 0 0 0</pose>
    <link name='l'>
      <collision name='c'><geometry><box><size>{bw:.2f} {bd:.2f} {bh:.2f}</size>
      </box></geometry></collision>
      <visual name='v'><geometry><box><size>{bw:.2f} {bd:.2f} {bh:.2f}</size>
      </box></geometry>{_tex(fac, tint=tint)}</visual>
      <visual name='roof'><pose>0 0 {bh/2+0.3:.2f} 0 0 0</pose><geometry>
      <box><size>{bw*0.97:.2f} {bd*0.97:.2f} 0.6</size></box></geometry>
      {_mat(0.22,0.24,0.26)}</visual>
    </link></model>""")
        meta["buildings"].append([bx, by, bw, bd, bh])

    # --- vertiports: well separated, clear of buildings ---------------------
    vps, guard = [], 0
    while len(vps) < W.n_vertiports and guard < 6000:
        guard += 1
        ang = rng.uniform(0, 2*math.pi)
        rad = rng.uniform(0.45*W.extent, 0.95*W.extent)
        vx, vy = rad*math.cos(ang), rad*math.sin(ang)
        if any(math.hypot(vx-ox, vy-oy) < 0.55*W.extent for ox, oy, _ in vps):
            continue
        if any(abs(vx-bx) < bw/2+45 and abs(vy-by) < bd/2+45
               for bx, by, bw, bd, _ in placed):
            continue
        vps.append((vx, vy, 0.0))
    for i, (vx, vy, vz) in enumerate(vps):
        hue = _mat(0.10, 0.55, 0.85) if i % 2 == 0 else _mat(0.85, 0.45, 0.10)
        parts.append(f"""
  <model name='vertiport_{i}'><static>true</static>
    <pose>{vx:.2f} {vy:.2f} 0 0 0 0</pose>
    <link name='l'>
      <collision name='c'><pose>0 0 0.6 0 0 0</pose><geometry><cylinder>
      <radius>26</radius><length>1.2</length></cylinder></geometry></collision>
      <visual name='pad'><pose>0 0 0.6 0 0 0</pose><geometry><box>
      <size>52 52 1.2</size></box></geometry>{_tex("pad.png")}</visual>
      <visual name='ring'><pose>0 0 1.25 0 0 0</pose><geometry><cylinder>
      <radius>21</radius><length>0.12</length></cylinder></geometry>{hue}</visual>
      <visual name='h1'><pose>-5 0 1.32 0 0 0</pose><geometry><box>
      <size>2.4 14 0.1</size></box></geometry>{_mat(0.95,0.95,0.95)}</visual>
      <visual name='h2'><pose> 5 0 1.32 0 0 0</pose><geometry><box>
      <size>2.4 14 0.1</size></box></geometry>{_mat(0.95,0.95,0.95)}</visual>
      <visual name='h3'><pose> 0 0 1.32 0 0 0</pose><geometry><box>
      <size>7.4 2.4 0.1</size></box></geometry>{_mat(0.95,0.95,0.95)}</visual>
    </link></model>""")
        meta["vertiports"].append([vx, vy, 1.2])

    # --- unmapped obstacles ------------------------------------------------
    # Construction cranes and masts. These are emitted into the world and
    # recorded under "unmapped" so the scorer counts collisions with them, but
    # they are withheld from the obstacle database the route planner and the
    # control-barrier filter consume. The only way to avoid them is to see them.
    meta["unmapped"] = []
    for i in range(W.n_unmapped):
        for _ in range(200):
            ux = rng.uniform(-W.extent, W.extent)
            uy = rng.uniform(-W.extent, W.extent)
            if all(math.hypot(ux-vx, uy-vy) > 120 for vx, vy, _ in vps):
                break
        uh = rng.uniform(120.0, 235.0)
        jib = rng.uniform(24.0, 42.0)
        ja  = rng.uniform(0, 2*math.pi)
        mast = _mat(0.92, 0.62, 0.08)
        parts.append(f"""
  <model name='crane_{i}'><static>true</static>
    <pose>{ux:.2f} {uy:.2f} 0 0 0 {ja:.3f}</pose>
    <link name='l'>
      <collision name='m'><pose>0 0 {uh/2:.2f} 0 0 0</pose><geometry><box>
      <size>3.0 3.0 {uh:.2f}</size></box></geometry></collision>
      <visual name='mv'><pose>0 0 {uh/2:.2f} 0 0 0</pose><geometry><box>
      <size>3.0 3.0 {uh:.2f}</size></box></geometry>{mast}</visual>
      <collision name='j'><pose>{jib/2-6:.2f} 0 {uh:.2f} 0 0 0</pose><geometry>
      <box><size>{jib:.2f} 2.2 2.2</size></box></geometry></collision>
      <visual name='jv'><pose>{jib/2-6:.2f} 0 {uh:.2f} 0 0 0</pose><geometry>
      <box><size>{jib:.2f} 2.2 2.2</size></box></geometry>{mast}</visual>
      <visual name='cv'><pose>0 0 {uh+3.2:.2f} 0 0 0</pose><geometry>
      <box><size>2.6 2.6 2.6</size></box></geometry>{_mat(0.85,0.1,0.1)}</visual>
    </link></model>""")
        # stored as an upright box enclosing mast+jib, for exact collision truth
        meta["unmapped"].append([ux, uy, max(jib, 6.0), 6.0, uh + 4.5, ja])

    # --- aircraft: ownship starts on vertiport 0 ---------------------------
    ox, oy, _ = vps[0]
    parts.append(evtol_model("ownship", ox, oy, 1.9, rng.uniform(0, 6.28),
                             "phywam", seed))
    for i in range(W.n_traffic):
        parts.append(intruder_model(f"traffic_{i}", 0, 0, -500 - 10*i, "evtol", i))
    for i in range(W.n_suas):
        parts.append(intruder_model(f"suas_{i}", 0, 0, -600 - 10*i, "suas", i))
    meta["n_traffic"], meta["n_suas"] = W.n_traffic, W.n_suas
    meta["start"] = [ox, oy]

    bld_str = ' '.join(f"{b[0]:.1f},{b[1]:.1f},{b[2]/2:.1f},{b[3]/2:.1f},{b[4]:.1f}"
                       for b in meta['buildings'])
    # 400 m ground tiles: one texture instance per tile keeps the city grid at
    # a believable 50 m block pitch instead of stretching one image over 4 km.
    TILE, NT = 400.0, 11
    gt = []
    for iy in range(NT):
        for ix in range(NT):
            gx = (ix - NT // 2) * TILE
            gy = (iy - NT // 2) * TILE
            gt.append(f"<visual name='g_{ix}_{iy}'><pose>{gx:.1f} {gy:.1f} "
                      f"-0.2 0 0 0</pose><geometry><box><size>{TILE} {TILE} 0.4"
                      f"</size></box></geometry>{_tex('ground.png', r=0.95)}</visual>")
    ground_tiles = "".join(gt)
    sky = ("<sky></sky>" )
    world = f"""<?xml version='1.0'?>
<sdf version='1.9'>
<world name='urban'>
  <physics name='fast' type='ignored'>
    <max_step_size>{1.0/CFG.lrn.phys_hz:.6f}</max_step_size>
    <real_time_factor>0</real_time_factor>
  </physics>
  <plugin filename='gz-sim-physics-system' name='gz::sim::systems::Physics'/>
  <plugin filename='gz-sim-user-commands-system'
          name='gz::sim::systems::UserCommands'/>
  <plugin filename='gz-sim-scene-broadcaster-system'
          name='gz::sim::systems::SceneBroadcaster'/>
  <plugin filename='gz-sim-sensors-system' name='gz::sim::systems::Sensors'>
    <render_engine>ogre2</render_engine>
  </plugin>
  <plugin filename='gz-sim-imu-system' name='gz::sim::systems::Imu'/>
  <plugin filename='PhyWamTraffic' name='phywam::PhyWamTraffic'>
    <seed>{seed}</seed>
    <n_traffic>{W.n_traffic}</n_traffic>
    <n_suas>{W.n_suas}</n_suas>
    <extent>{W.extent}</extent>
    <corridor_alt>{W.corridor_alt}</corridor_alt>
    <namespace>phywam</namespace>
    <p_conflict>0.55</p_conflict>
    <buildings>{bld_str}</buildings>
  </plugin>

  <gravity>0 0 -9.80665</gravity>
  <scene>
    <ambient>0.38 0.40 0.44 1</ambient>
    <background>0.46 0.60 0.78 1</background>
    <grid>false</grid>{sky}
  </scene>
  <light type='directional' name='sun'>
    <cast_shadows>true</cast_shadows>
    <pose>0 0 400 0 0 0</pose>
    <diffuse>0.88 0.86 0.80 1</diffuse><specular>0.18 0.18 0.18 1</specular>
    <direction>-0.42 0.38 -0.82</direction>
  </light>
  <model name='ground'><static>true</static>
    <link name='l'>
      <collision name='c'><geometry><plane><normal>0 0 1</normal>
      <size>9000 9000</size></plane></geometry></collision>
{ground_tiles}
    </link></model>
{''.join(parts)}
</world>
</sdf>"""
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        open(out, "w").write(world)
        open(out.replace(".sdf", ".json"), "w").write(json.dumps(meta))
    return world, meta


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = os.path.join(root, "sim", "worlds", f"urban_{seed}.sdf")
    _, meta = make_world(seed, p)
    print(f"wrote {p}")
    print(f"  buildings  : {len(meta['buildings'])}")
    print(f"  vertiports : {len(meta['vertiports'])}")
    for i, v in enumerate(meta["vertiports"]):
        print(f"     VP{i}: ({v[0]:8.1f}, {v[1]:8.1f})")
    d = [math.hypot(a[0]-b[0], a[1]-b[1])
         for i, a in enumerate(meta["vertiports"])
         for b in meta["vertiports"][i+1:]]
    print(f"  pair separation: min {min(d):.0f} m, max {max(d):.0f} m")
