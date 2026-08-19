"""
Live operator console.

Runs a flight in a worker thread and displays the dashboard frames as they are
produced. The rendering is done by DashboardRenderer, the same code the video
recorder uses, so the console and the recordings cannot drift apart.

    python -m phywam.gui --ckpt runs/main/ckpt.pt --mode plan
"""
import os, sys, time, argparse
import numpy as np
import torch

from PySide6 import QtCore, QtGui, QtWidgets

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from phywam.config import CFG
from phywam.env import VertiportEnv
from phywam.worldmodel import WorldModel
from phywam.agent import Actor, Critic
from phywam.runner import FlightRunner
from phywam.route import RoutePlanner
from phywam.instrument import Instrumented


class FlightWorker(QtCore.QThread):
    frame = QtCore.Signal(object)
    done = QtCore.Signal(object)
    status = QtCore.Signal(str)

    def __init__(self, args):
        super().__init__()
        self.args = args
        self._abort = False
        self._pending = None

    def abort(self):
        self._abort = True

    def request(self, mode, s, g):
        self._pending = (mode, s, g)

    def run(self):
        a = self.args
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_float32_matmul_precision("high")
        self.status.emit("starting simulator ...")
        world = os.path.join(ROOT, "sim", "worlds", f"urban_{a.world}.sdf")
        env = VertiportEnv(world, seed=a.seed, max_time=a.max_time)
        wm = WorldModel(dev).to(dev).eval()
        actor = Actor(wm.feat_dim).to(dev).eval()
        critic = Critic(wm.feat_dim, dev).to(dev).eval()
        if os.path.exists(a.ckpt):
            c = torch.load(a.ckpt, map_location=dev, weights_only=False)
            wm.load_state_dict(c["wm"]); actor.load_state_dict(c["actor"])
            critic.load_state_dict(c["critic"])
            self.status.emit(f"loaded checkpoint  step {c['step']}  ep {c['ep']}")
        else:
            self.status.emit("no checkpoint - running untrained")
        rp = RoutePlanner(env.geom)
        run = FlightRunner(env, wm, actor, critic, dev, route_planner=rp,
                           mode=a.mode)
        inst = Instrumented(run, dev, xai_every=a.xai_every)
        n_vp = len(env.geom.vp)
        rng = np.random.default_rng(a.seed)
        try:
            while not self._abort:
                if self._pending:
                    mode, s, g = self._pending
                    self._pending = None
                else:
                    mode = a.mode
                    s = int(rng.integers(n_vp))
                    g = int(rng.choice([j for j in range(n_vp) if j != s]))
                run.mode = mode
                inst.trail = []; inst.cache = {}
                inst.n_steps = inst.n_engaged = 0
                inst.dash.hist = {k: [] for k in inst.dash.hist}
                self.status.emit(f"flying  VP{s} -> VP{g}   mode={mode}")

                def cb(d, _s=s, _g=g, _m=mode):
                    if self._abort:
                        raise KeyboardInterrupt
                    d["banner"] = f"vertiport {_s} -> vertiport {_g}   mode {_m}"
                    self.frame.emit(inst.callback(d))
                try:
                    st, _ = run.run(start_vp=s, goal_vp=g, callback=cb)
                except KeyboardInterrupt:
                    break
                self.done.emit(st)
                self.status.emit(
                    f"VP{s}->VP{g}: {st['outcome']}  minObs={st['min_obs']:.1f} m  "
                    f"minSep={min(st['min_sep'],9999):.0f} m  "
                    f"shield={100*st['shield_rate']:.1f}%")
                time.sleep(1.0)
        finally:
            env.close()


class Console(QtWidgets.QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.setWindowTitle("Phy-WAM  |  operator console")
        self.setStyleSheet("QMainWindow{background:#151311;}"
                           "QLabel{color:#e8e4de;}"
                           "QPushButton{background:#2d2b28;color:#e8e4de;"
                           "border:1px solid #4a4740;padding:5px 12px;}"
                           "QPushButton:hover{background:#3c3934;}"
                           "QComboBox{background:#2d2b28;color:#e8e4de;"
                           "border:1px solid #4a4740;padding:3px 8px;}")
        cw = QtWidgets.QWidget(); self.setCentralWidget(cw)
        v = QtWidgets.QVBoxLayout(cw); v.setContentsMargins(8, 8, 8, 8)

        self.view = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.view.setMinimumSize(1200, 675)
        self.view.setText("starting ...")
        v.addWidget(self.view, 1)

        bar = QtWidgets.QHBoxLayout()
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["plan", "actor", "baseline", "scripted"])
        self.mode.setCurrentText(args.mode)
        self.vs = QtWidgets.QComboBox(); self.vg = QtWidgets.QComboBox()
        for i in range(CFG.wld.n_vertiports):
            self.vs.addItem(f"VP{i}"); self.vg.addItem(f"VP{i}")
        self.vg.setCurrentIndex(1)
        go = QtWidgets.QPushButton("fly this route")
        go.clicked.connect(self._go)
        stop = QtWidgets.QPushButton("stop")
        stop.clicked.connect(self.close)
        self.stat = QtWidgets.QLabel("...")
        for w, lab in ((self.mode, "controller"), (self.vs, "from"), (self.vg, "to")):
            bar.addWidget(QtWidgets.QLabel(lab)); bar.addWidget(w)
        bar.addWidget(go); bar.addWidget(stop)
        bar.addStretch(1); bar.addWidget(self.stat)
        v.addLayout(bar)

        self._shot = args.screenshot
        self._shot_after = args.screenshot_after
        self._frames = 0
        self.worker = FlightWorker(args)
        self.worker.frame.connect(self.on_frame)
        self.worker.status.connect(self.stat.setText)
        self.worker.start()
        self._last = 0.0
        self._t_start = time.time()

    def _go(self):
        self.worker.request(self.mode.currentText(),
                            self.vs.currentIndex(), self.vg.currentIndex())
        self.stat.setText("queued - will start after the current flight")

    def on_frame(self, bgr):
        now = time.time()
        if now - self._last < 0.045:            # cap the repaint rate
            return
        self._last = now
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        h, w, _ = rgb.shape
        qi = QtGui.QImage(rgb.data, w, h, 3*w, QtGui.QImage.Format_RGB888)
        pm = QtGui.QPixmap.fromImage(qi).scaled(
            self.view.size(), QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation)
        self.view.setPixmap(pm)
        self._frames += 1
        if self._shot and self._frames >= self._shot_after:
            # self-test: prove the console really renders on this display
            QtCore.QTimer.singleShot(400, self._grab)
            self._shot = None

    def _grab(self):
        self.grab().save(self._shot_path)
        print(f"screenshot -> {self._shot_path}", flush=True)
        QtWidgets.QApplication.quit()

    def closeEvent(self, e):
        self.worker.abort(); self.worker.wait(15000); e.accept()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "runs/main/ckpt.pt"))
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--mode", default="plan")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-time", type=float, default=200.0)
    ap.add_argument("--xai-every", type=int, default=5)
    ap.add_argument("--screenshot", default=None,
                    help="self-test: save the window after N frames and exit")
    ap.add_argument("--screenshot-after", type=int, default=25)
    args = ap.parse_args()
    app = QtWidgets.QApplication(sys.argv)
    c = Console(args); c._shot_path = args.screenshot
    c.resize(1620, 980); c.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
