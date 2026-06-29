#!/usr/bin/env python
"""
FET curve viewer — pure Qt5 (hardware-accelerated).

Usage:
    python viewer_qt.py <output_dir>

Controls:
    Mouse wheel / Up/Down  — scroll slices
    Click on image         — show TBR(t) curve
    R                      — reset crosshair
    Q / Escape             — quit
    Opacity slider (right) — adjust cluster overlay transparency
"""

import argparse
import json
import os
import sys

import numpy as np
import nibabel as nib

from PyQt5 import QtCore, QtGui, QtWidgets

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ── Cluster colours ──────────────────────────────────────────────────────
_CLUSTER_RGB = {   # alpha is controlled by slider
    1: (255, 51,  51),    # rising  — red
    2: (51,  102, 255),   # falling — blue
    3: (51,  204, 51),    # plateau — green
}

# Target display size for anti-aliased upscaling
_DISPLAY_TARGET = 600  # shorter dimension will be scaled to this


# ── Data loading ─────────────────────────────────────────────────────────

def load_vol(path: str) -> np.ndarray:
    img = nib.load(path)
    canonical = nib.as_closest_canonical(img)
    return canonical.get_fdata().astype(np.float64)


def find_files(output_dir: str, names: list[str]) -> dict[str, str]:
    found = {}
    for name in names:
        path = os.path.join(output_dir, name)
        if os.path.exists(path):
            found[name] = path
        else:
            alt = path.replace(".nii.gz", ".nii")
            if os.path.exists(alt):
                found[name] = alt
    return found


# ── Image widget ─────────────────────────────────────────────────────────

class SliceView(QtWidgets.QGraphicsView):
    """Scrollable axial slice display with click-to-query."""

    slice_changed = QtCore.pyqtSignal(int)
    voxel_clicked = QtCore.pyqtSignal(int, int, int)
    voxel_hovered = QtCore.pyqtSignal(int, int, int)

    def __init__(self):
        super().__init__()
        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)

        # Separated underlay + overlay for independent opacity control
        self._underlay_item = QtWidgets.QGraphicsPixmapItem()
        self._overlay_item = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._underlay_item)
        self._scene.addItem(self._overlay_item)

        # Crosshair
        self._cross_v = QtWidgets.QGraphicsLineItem()
        self._cross_h = QtWidgets.QGraphicsLineItem()
        pen = QtGui.QPen(QtGui.QColor(255, 255, 0, 150), 1)
        self._cross_v.setPen(pen)
        self._cross_h.setPen(pen)
        self._cross_v.hide()
        self._cross_h.hide()
        self._scene.addItem(self._cross_v)
        self._scene.addItem(self._cross_h)

        self.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
        self.setRenderHint(QtGui.QPainter.Antialiasing, True)
        self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setMouseTracking(True)

        self._underlay_pixmaps = []
        self._overlay_pixmaps = []
        self._nz = 0
        self._cur_z = 0
        self._nx = 0   # canvas size (after scaling)
        self._ny = 0
        self._vx = 0   # original voxel dimensions
        self._vy = 0
        self._scale = 1.0

    def set_overlay_opacity(self, opacity: float):
        self._overlay_item.setOpacity(opacity)

    def set_data(self, underlay_pixmaps: list, overlay_pixmaps: list,
                 nx: int, ny: int, nz: int, target_size: int = _DISPLAY_TARGET):
        # Scale to target size with smooth transformation
        self._vx, self._vy = nx, ny
        scale = target_size / min(nx, ny)
        self._scale = scale
        self._nx = int(nx * scale)
        self._ny = int(ny * scale)

        self._underlay_pixmaps = [
            p.scaled(self._nx, self._ny,
                     QtCore.Qt.IgnoreAspectRatio,
                     QtCore.Qt.SmoothTransformation)
            for p in underlay_pixmaps
        ]
        self._overlay_pixmaps = [
            p.scaled(self._nx, self._ny,
                     QtCore.Qt.IgnoreAspectRatio,
                     QtCore.Qt.SmoothTransformation)
            for p in overlay_pixmaps
        ]
        self._nz = nz
        self._cur_z = nz // 2
        self._show_slice()

    def _show_slice(self):
        if not self._underlay_pixmaps:
            return
        self._underlay_item.setPixmap(self._underlay_pixmaps[self._cur_z])
        self._overlay_item.setPixmap(self._overlay_pixmaps[self._cur_z])
        self._scene.setSceneRect(self._underlay_item.boundingRect())
        self.fitInView(self._underlay_item, QtCore.Qt.KeepAspectRatio)
        self.slice_changed.emit(self._cur_z)

    # ── Navigation ──

    def scroll_slice(self, dz: int):
        self._cur_z = max(0, min(self._nz - 1, self._cur_z + dz))
        self._show_slice()

    # ── Coordinate mapping ──

    def _display_to_voxel(self, dx: int, dy: int) -> tuple:
        """Convert display coords (scaled pixmap pixel) to canonical voxel coords.
        disp = flipud(fliplr(data.T)) means:
          disp[row, col] = data[nx-1-col, ny-1-row, z]
        => vx = nx-1 - (dx/scale),  vy = ny-1 - (dy/scale)
        """
        vx = self._vx - 1 - int(dx / self._scale)
        vy = self._vy - 1 - int(dy / self._scale)
        return vx, vy, self._cur_z

    # ── Crosshair ──

    def show_crosshair(self, dx: int, dy: int):
        if dx < 0 or dy < 0:
            self._cross_v.hide()
            self._cross_h.hide()
        else:
            self._cross_v.setLine(dx, 0, dx, self._ny)
            self._cross_h.setLine(0, dy, self._nx, dy)
            self._cross_v.show()
            self._cross_h.show()

    # ── Events ──

    def wheelEvent(self, event: QtGui.QWheelEvent):
        dz = -1 if event.angleDelta().y() > 0 else 1
        self.scroll_slice(dz)

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if event.key() == QtCore.Qt.Key_Up:
            self.scroll_slice(1)
        elif event.key() == QtCore.Qt.Key_Down:
            self.scroll_slice(-1)
        elif event.key() == QtCore.Qt.Key_R:
            self.show_crosshair(-1, -1)
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.LeftButton:
            pt = self.mapToScene(event.pos())
            dx, dy = int(pt.x()), int(pt.y())
            if 0 <= dx < self._nx and 0 <= dy < self._ny:
                vx, vy, vz = self._display_to_voxel(dx, dy)
                if 0 <= vx < self._vx and 0 <= vy < self._vy:
                    self.show_crosshair(dx, dy)
                    self.voxel_clicked.emit(vx, vy, vz)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        pt = self.mapToScene(event.pos())
        dx, dy = int(pt.x()), int(pt.y())
        if 0 <= dx < self._nx and 0 <= dy < self._ny:
            vx, vy, vz = self._display_to_voxel(dx, dy)
            if 0 <= vx < self._vx and 0 <= vy < self._vy:
                self.voxel_hovered.emit(vx, vy, vz)
        super().mouseMoveEvent(event)


# ── Main viewer ──────────────────────────────────────────────────────────

class FETQtViewer(QtWidgets.QMainWindow):
    def __init__(self, output_dir: str):
        super().__init__()
        self.output_dir = output_dir

        # ── Load report ──
        report_path = os.path.join(output_dir, "report.json")
        if os.path.exists(report_path):
            with open(report_path) as f:
                self.report = json.load(f)
        else:
            self.report = {}

        params = self.report.get("parameters", {})
        self.mode = params.get("mode", self.report.get("mode", "static"))

        # ── Time points ──
        if self.mode == "dynamic":
            tp = self.report.get("time_points_min",
                                 self.report.get("parameters", {}).get("time_points_min", []))
        else:
            tp = params.get("time_points_min", [20.0, 40.0, 60.0])
        tp = list(np.asarray(tp, dtype=float).ravel())
        if len(tp) < 3:
            tp = [20.0, 40.0, 60.0]
        self.time_points = tp

        # ── Files ──
        if self.mode == "dynamic":
            self.has_curves = False
            required = ["map_slope.nii.gz", "map_tbrmax.nii.gz", "mask_clusters.nii.gz"]
        else:
            self.has_curves = True
            required = ["map_tbr_t20.nii.gz", "map_tbr_t40.nii.gz", "map_tbr_t60.nii.gz",
                        "mask_clusters.nii.gz"]
        optional = ["map_t1.nii.gz", "map_sulmax.nii.gz", "mask_brain.nii.gz"]

        self.files = find_files(output_dir, required + optional)
        missing = [f for f in required if f not in self.files]
        if missing:
            print(f"ERROR: missing: {missing}")
            sys.exit(1)

        # ── Load volumes ──
        print("Loading data...")
        if self.mode == "static":
            tbrs = [load_vol(self.files[k]) for k in
                    ["map_tbr_t20.nii.gz", "map_tbr_t40.nii.gz", "map_tbr_t60.nii.gz"]]
            self.tbr_volumes = np.stack(tbrs, axis=-1)

        self.has_t1 = "map_t1.nii.gz" in self.files
        if self.has_t1:
            self.underlay = load_vol(self.files["map_t1.nii.gz"])
            self.underlay_label = "T1"
        elif self.mode == "dynamic":
            self.underlay = load_vol(self.files["map_tbrmax.nii.gz"])
            self.underlay_label = "TBRmax"
        else:
            self.underlay = np.mean(self.tbr_volumes, axis=-1)
            self.underlay_label = "TBR mean"

        self.clusters = load_vol(self.files["mask_clusters.nii.gz"]).astype(np.int8)
        self.nx, self.ny, self.nz = self.shape = self.underlay.shape
        print(f"  Volume: {self.shape}  ({self.underlay_label})")

        # ── Precompute underlay + overlay pixmaps (separated) ──
        print("  Precomputing slice pixmaps...")
        underlay_pixmaps = []
        overlay_pixmaps = []

        for z in range(self.nz):
            sl = self.underlay[:, :, z]
            disp = np.flipud(np.fliplr(sl.T))  # (ny, nx) — radiological + anterior up

            # Percentile-based windowing
            pos = sl[sl > 0]
            if pos.size > 0:
                vmin = np.percentile(pos, 5)
                vmax = np.percentile(pos, 95)
            else:
                vmin, vmax = sl.min(), sl.max()
            if vmax <= vmin:
                vmin, vmax = sl.min(), sl.max()

            # ── Underlay: gray only ──
            norm = np.clip((disp - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
            under_rgba = np.zeros((self.ny, self.nx, 4), dtype=np.uint8)
            under_rgba[:, :, 0] = norm
            under_rgba[:, :, 1] = norm
            under_rgba[:, :, 2] = norm
            under_rgba[:, :, 3] = 255
            qimg = QtGui.QImage(under_rgba.tobytes(), self.nx, self.ny, self.nx * 4,
                                QtGui.QImage.Format_RGBA8888)
            underlay_pixmaps.append(QtGui.QPixmap.fromImage(qimg))

            # ── Overlay: transparent bg + coloured clusters ──
            csl = np.flipud(np.fliplr(self.clusters[:, :, z].T))
            over_rgba = np.zeros((self.ny, self.nx, 4), dtype=np.uint8)
            for label, (r, g, b) in _CLUSTER_RGB.items():
                mask = csl == label
                over_rgba[mask, 0] = r
                over_rgba[mask, 1] = g
                over_rgba[mask, 2] = b
                over_rgba[mask, 3] = 200  # full opacity, controlled by slider
            qimg2 = QtGui.QImage(over_rgba.tobytes(), self.nx, self.ny, self.nx * 4,
                                 QtGui.QImage.Format_RGBA8888)
            overlay_pixmaps.append(QtGui.QPixmap.fromImage(qimg2))

        self._cur_z = self.nz // 2

        # ── Build UI ──
        self.setWindowTitle(f"FET Viewer — {os.path.basename(output_dir)}")
        self.setMinimumSize(1200, 700)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)

        # Left: image
        self.view = SliceView()
        self.view.set_data(underlay_pixmaps, overlay_pixmaps,
                           self.nx, self.ny, self.nz)
        layout.addWidget(self.view, 2)

        # Right panel
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Slice info
        self.info_label = QtWidgets.QLabel()
        self.info_label.setAlignment(QtCore.Qt.AlignCenter)
        self.info_label.setStyleSheet("font-size: 13px; padding: 6px;")
        right_layout.addWidget(self.info_label)

        # TBR curve
        if self.has_curves:
            self.fig = Figure(figsize=(5, 4), dpi=100)
            self.canvas = FigureCanvas(self.fig)
            self.ax = self.fig.add_subplot(111)
            self.ax.set_title("TBR(t) at clicked voxel")
            self.ax.set_xlabel("Time (min)")
            self.ax.set_ylabel("TBR")
            self.ax.grid(True, alpha=0.3)
            self.curve_line, = self.ax.plot([], [], "o-", color="#e74c3c", lw=2.5, ms=14,
                                             markeredgecolor="#c0392b", markeredgewidth=1.5)
            self.ax.set_xlim(min(self.time_points) - 5, max(self.time_points) + 5)
            self.ax.set_ylim(0, 5)
            self.fig.tight_layout()
            right_layout.addWidget(self.canvas, 1)
        else:
            no_curve = QtWidgets.QLabel(
                "Dynamic mode:\nmulti-frame TBR not saved.\n\n"
                "Use static mode for interactive TBR plots.")
            no_curve.setAlignment(QtCore.Qt.AlignCenter)
            no_curve.setStyleSheet("color: #888; font-size: 13px;")
            right_layout.addWidget(no_curve, 1)

        # ── Opacity slider ──
        opacity_w = QtWidgets.QWidget()
        opacity_l = QtWidgets.QHBoxLayout(opacity_w)
        opacity_l.setContentsMargins(0, 4, 0, 4)
        opacity_l.addWidget(QtWidgets.QLabel("Overlay:"))
        self.opacity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.opacity_slider.setTickInterval(10)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        opacity_l.addWidget(self.opacity_slider)
        self.opacity_label = QtWidgets.QLabel("50%")
        self.opacity_label.setFixedWidth(35)
        opacity_l.addWidget(self.opacity_label)
        right_layout.addWidget(opacity_w)

        # Legend
        legend_w = QtWidgets.QWidget()
        legend_l = QtWidgets.QHBoxLayout(legend_w)
        legend_l.setContentsMargins(0, 0, 0, 0)
        for label, name, color in [(1, "Rising", "#e33"), (2, "Falling", "#33e"),
                                     (3, "Plateau", "#3c3")]:
            swatch = QtWidgets.QLabel()
            swatch.setFixedSize(14, 14)
            swatch.setStyleSheet(f"background: {color}; border: 1px solid #666;")
            legend_l.addWidget(swatch)
            legend_l.addWidget(QtWidgets.QLabel(name))
            legend_l.addSpacing(12)
        legend_l.addStretch()
        right_layout.addWidget(legend_w)

        layout.addWidget(right, 1)

        # Status bar
        self.statusBar().showMessage(f"z={self._cur_z}/{self.nz - 1}  |  {self.underlay_label}")

        # ── Connect signals ──
        self.view.slice_changed.connect(self._on_slice_changed)
        self.view.voxel_clicked.connect(self._on_voxel_clicked)
        self.view.voxel_hovered.connect(self._on_voxel_hovered)

        self._update_info()

    # ── Slots ──

    def _on_opacity_changed(self, value: int):
        self.view.set_overlay_opacity(value / 100.0)
        self.opacity_label.setText(f"{value}%")

    def _on_slice_changed(self, z: int):
        self._cur_z = z
        self._update_info()

    def _on_voxel_clicked(self, vx: int, vy: int, vz: int):
        if not self.has_curves:
            return
        tbr_vals = self.tbr_volumes[vx, vy, vz, :]
        cluster = int(self.clusters[vx, vy, vz])
        label_map = {1: "Rising", 2: "Falling", 3: "Plateau", 0: "Background"}

        for txt in list(self.ax.texts):
            txt.remove()

        self.curve_line.set_data(self.time_points, list(tbr_vals))
        ymax = max(float(tbr_vals.max()) * 1.3, 2.0)
        self.ax.set_ylim(0, ymax)

        for t, v in zip(self.time_points, tbr_vals):
            self.ax.annotate(f"{v:.2f}", (t, v),
                             textcoords="offset points", xytext=(0, 12),
                             ha="center", fontsize=10, fontweight="bold",
                             color="#c0392b")

        self.ax.set_title(
            f"Voxel ({vx}, {vy}, {vz})  Cluster: {label_map.get(cluster, '?')}\n"
            f"TBR: {tbr_vals[0]:.3f} → {tbr_vals[1]:.3f} → {tbr_vals[2]:.3f}")
        self.fig.tight_layout()
        self.canvas.draw_idle()

        print(f"  Click: voxel=({vx},{vy},{vz}) cluster={cluster} "
              f"TBR=({tbr_vals[0]:.3f}, {tbr_vals[1]:.3f}, {tbr_vals[2]:.3f})")

    def _on_voxel_hovered(self, vx: int, vy: int, vz: int):
        val = self.underlay[vx, vy, vz]
        cls = int(self.clusters[vx, vy, vz])
        label = {1: "R", 2: "F", 3: "P", 0: "-"}.get(cls, "?")
        self.statusBar().showMessage(
            f"({vx}, {vy}, {vz})  val={val:.3f}  cluster={label}  "
            f"z={vz}/{self.nz - 1}  |  {self.underlay_label}")

    def _update_info(self):
        z = self._cur_z
        self.info_label.setText(
            f"<b>Axial slice</b><br>"
            f"z = {z} / {self.nz - 1}<br>"
            f"<span style='color:#888;'>{self.underlay_label}</span><br>"
            f"<span style='color:#888;font-size:11px;'>click for TBR curve</span>")

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if event.key() == QtCore.Qt.Key_Q or event.key() == QtCore.Qt.Key_Escape:
            self.close()
        elif event.key() == QtCore.Qt.Key_Up:
            self.view.scroll_slice(1)
        elif event.key() == QtCore.Qt.Key_Down:
            self.view.scroll_slice(-1)
        elif event.key() == QtCore.Qt.Key_R:
            self.view.show_crosshair(-1, -1)
        else:
            super().keyPressEvent(event)


# ── CLI ──

def main():
    print("FET Viewer — Qt5 hardware-accelerated")
    parser = argparse.ArgumentParser(description="Interactive FET curve viewer (Qt)")
    parser.add_argument("output_dir", help="Pipeline output directory with NIfTI maps")
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        print(f"ERROR: not a directory: {args.output_dir}")
        sys.exit(1)

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    viewer = FETQtViewer(args.output_dir)
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
