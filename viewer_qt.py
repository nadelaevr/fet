#!/usr/bin/env python
"""
FET curve viewer — pure Qt5 (hardware-accelerated).

Usage:
    python viewer_qt.py <output_dir>

Controls:
    Mouse wheel / Up/Down  — scroll slices
    Click on image         — show TBR(t) curve + voxel info
    R                      — reset crosshair
    Q / Escape             — quit
    Tab 1                  — T1 + Clusters overlay
    Tab 2                  — T1 + TBRmax overlay
    Opacity slider         — adjust overlay transparency
    Window sliders         — adjust T1 brightness/contrast
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

from scipy.ndimage import zoom

# ── Cluster colours ──────────────────────────────────────────────────────
_CLUSTER_RGB = {
    1: (255, 51,  51),    # rising  — red
    2: (51,  102, 255),   # falling — blue
    3: (51,  204, 51),    # plateau — green
}

# ── TBRmax colormap (jet-like, vectorized in _build_pixmaps) ────────────


# ── Data loading ─────────────────────────────────────────────────────────

def load_vol(path: str) -> np.ndarray:
    img = nib.load(path)
    canonical = nib.as_closest_canonical(img)
    return canonical.get_fdata().astype(np.float64)


def load_vol_with_affine(path: str) -> tuple:
    img = nib.load(path)
    canonical = nib.as_closest_canonical(img)
    return canonical.get_fdata().astype(np.float64), canonical.affine


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
    slice_changed = QtCore.pyqtSignal(int)
    voxel_clicked = QtCore.pyqtSignal(int, int, int, int, int)
    voxel_hovered = QtCore.pyqtSignal(int, int, int, int, int)

    def __init__(self):
        super().__init__()
        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        self._underlay_item = QtWidgets.QGraphicsPixmapItem()
        self._overlay_item = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._underlay_item)
        self._scene.addItem(self._overlay_item)
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
        self._nx = 0
        self._ny = 0
        self._vx = 0
        self._vy = 0

    def set_overlay_opacity(self, opacity: float):
        self._overlay_item.setOpacity(opacity)

    def set_data(self, underlay_pixmaps: list, overlay_pixmaps: list,
                 nx: int, ny: int, nz: int):
        self._vx, self._vy = nx, ny
        self._nx = underlay_pixmaps[0].width() if underlay_pixmaps else 0
        self._ny = underlay_pixmaps[0].height() if underlay_pixmaps else 0
        self._underlay_pixmaps = underlay_pixmaps
        self._overlay_pixmaps = overlay_pixmaps
        self._nz = nz
        self._cur_z = min(self._cur_z, nz - 1) if nz > 0 else 0
        self._show_slice()

    def swap_overlay(self, overlay_pixmaps: list):
        """Swap overlay pixmaps without changing slice."""
        self._overlay_pixmaps = overlay_pixmaps
        self._show_slice()

    def _show_slice(self):
        if not self._underlay_pixmaps:
            return
        z = min(self._cur_z, len(self._underlay_pixmaps) - 1)
        self._underlay_item.setPixmap(self._underlay_pixmaps[z])
        if z < len(self._overlay_pixmaps):
            self._overlay_item.setPixmap(self._overlay_pixmaps[z])
        self._scene.setSceneRect(self._underlay_item.boundingRect())
        self.fitInView(self._underlay_item, QtCore.Qt.KeepAspectRatio)
        self.slice_changed.emit(self._cur_z)

    def scroll_slice(self, dz: int):
        self._cur_z = max(0, min(self._nz - 1, self._cur_z + dz))
        self._show_slice()

    def _display_to_voxel(self, dx: int, dy: int) -> tuple:
        scale_x = self._nx / self._vx if self._vx else 1
        scale_y = self._ny / self._vy if self._vy else 1
        vx = self._vx - 1 - int(dx / scale_x)
        vy = self._vy - 1 - int(dy / scale_y)
        return vx, vy, self._cur_z

    def show_crosshair(self, dx: int, dy: int):
        if dx < 0 or dy < 0:
            self._cross_v.hide()
            self._cross_h.hide()
        else:
            self._cross_v.setLine(dx, 0, dx, self._ny)
            self._cross_h.setLine(0, dy, self._nx, dy)
            self._cross_v.show()
            self._cross_h.show()

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
                    self.voxel_clicked.emit(vx, vy, vz, dx, dy)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        pt = self.mapToScene(event.pos())
        dx, dy = int(pt.x()), int(pt.y())
        if 0 <= dx < self._nx and 0 <= dy < self._ny:
            vx, vy, vz = self._display_to_voxel(dx, dy)
            if 0 <= vx < self._vx and 0 <= vy < self._vy:
                self.voxel_hovered.emit(vx, vy, vz, dx, dy)
        super().mouseMoveEvent(event)


# ── Main viewer ──────────────────────────────────────────────────────────

class FETQtViewer(QtWidgets.QMainWindow):
    def __init__(self, output_dir: str, upsample: int = 2):
        super().__init__()
        self.output_dir = output_dir
        self._upsample = upsample

        report_path = os.path.join(output_dir, "report.json")
        if os.path.exists(report_path):
            with open(report_path) as f:
                self.report = json.load(f)
        else:
            self.report = {}

        params = self.report.get("parameters", {})
        self.mode = params.get("mode", self.report.get("mode", "static"))

        if self.mode == "dynamic":
            tp = self.report.get("time_points_min",
                                 self.report.get("parameters", {}).get("time_points_min", []))
        else:
            tp = params.get("time_points_min", [20.0, 40.0, 60.0])
        tp = list(np.asarray(tp, dtype=float).ravel())
        if len(tp) < 3:
            tp = [20.0, 40.0, 60.0]
        self.time_points = tp

        # For dynamic with --sig-time-from: only show late frames in curves
        self._sig_time_from = params.get("sig_time_from_min", 0.0)
        if self.mode == "dynamic" and self._sig_time_from > 0:
            self._late_mask = np.array(tp) >= self._sig_time_from
        else:
            self._late_mask = None

        # ── Load data files ──
        if self.mode == "dynamic":
            tbr_4d_path = os.path.join(output_dir, "tbr_4d.nii.gz")
            if os.path.exists(tbr_4d_path):
                print("  Loading 4D TBR for dynamic curves...")
                self.tbr_volumes = load_vol(tbr_4d_path)
                self.has_curves = True
            else:
                self.has_curves = False
            required = ["map_slope.nii.gz", "map_tbrmax.nii.gz", "mask_clusters.nii.gz"]
        else:
            self.has_curves = True
            required = ["map_tbr_t20.nii.gz", "map_tbr_t40.nii.gz", "map_tbr_t60.nii.gz",
                        "mask_clusters.nii.gz"]
        optional = ["map_t1.nii.gz", "map_sulmax.nii.gz", "map_slope.nii.gz",
                    "map_tbrmax.nii.gz", "mask_brain.nii.gz"]

        self.files = find_files(output_dir, required + optional)
        missing = [f for f in required if f not in self.files]
        if missing:
            print(f"ERROR: missing: {missing}")
            sys.exit(1)

        print("Loading data...")
        if self.mode == "static":
            tbrs = [load_vol(self.files[k]) for k in
                    ["map_tbr_t20.nii.gz", "map_tbr_t40.nii.gz", "map_tbr_t60.nii.gz"]]
            self.tbr_volumes = np.stack(tbrs, axis=-1)

        # Load underlay
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

        # Load cluster mask
        self.clusters, pet_affine = load_vol_with_affine(self.files["mask_clusters.nii.gz"])
        self.clusters = self.clusters.astype(np.int8)
        self.nx, self.ny, self.nz = self.shape = self.underlay.shape
        print(f"  Volume: {self.shape}  ({self.underlay_label})")

        # Load optional maps for voxel info
        self.sulmax_vol = load_vol(self.files["map_sulmax.nii.gz"]) if "map_sulmax.nii.gz" in self.files else None
        self.tbrmax_vol = load_vol(self.files["map_tbrmax.nii.gz"]) if "map_tbrmax.nii.gz" in self.files else None
        self.slope_vol = load_vol(self.files["map_slope.nii.gz"]) if "map_slope.nii.gz" in self.files else None

        # Report params for voxel info
        self._time_span = params.get("time_span_min", 40.0)
        self._tbr_delta_thr = params.get("tbr_delta_threshold", 0.3)

        # ── Window defaults ──
        self._win_lo = 5
        self._win_hi = 95

        # ── Detect native T1 ──
        self._use_native_t1 = False
        self._disp_underlay = self.underlay
        self._disp_clusters = self.clusters
        self._disp_tbrmax = self.tbrmax_vol if self.tbrmax_vol is not None else self.underlay
        self._z_map = None
        self._disp_nx, self._disp_ny = self.nx, self.ny
        self._orig_nx, self._orig_ny = self.nx, self.ny

        t1_orig_path = os.path.join(output_dir, "t1_orig.nii.gz")
        if os.path.exists(t1_orig_path):
            print("  Found t1_orig.nii.gz — using native T1 resolution for display")
            self._use_native_t1 = True
            t1_native, t1_affine = load_vol_with_affine(t1_orig_path)
            self._t1_affine = t1_affine
            self._pet_affine = pet_affine
            tnx, tny, tnz = t1_native.shape
            self._disp_underlay = t1_native
            self._disp_nx, self._disp_ny = tnx, tny

            # Resample clusters to T1 grid
            clusters_img = nib.Nifti1Image(self.clusters.astype(np.int16), pet_affine)
            t1_img = nib.Nifti1Image(np.zeros((tnx, tny, tnz), dtype=np.int16), t1_affine)
            from nibabel.processing import resample_from_to
            resampled_img = resample_from_to(clusters_img, t1_img, order=0)
            self._disp_clusters = np.asarray(resampled_img.dataobj).astype(np.int8)

            # Resample TBRmax to T1 grid if available
            if self.tbrmax_vol is not None:
                tbrmax_img = nib.Nifti1Image(self.tbrmax_vol.astype(np.float32), pet_affine)
                tbrmax_resampled = resample_from_to(tbrmax_img, t1_img, order=1)
                self._disp_tbrmax = np.asarray(tbrmax_resampled.dataobj).astype(np.float64)

            # Physical z mapping
            z_phys_pet = np.array([nib.affines.apply_affine(pet_affine, [0, 0, z])[2]
                                   for z in range(self.nz)])
            z_phys_t1 = np.array([nib.affines.apply_affine(t1_affine, [0, 0, z])[2]
                                  for z in range(tnz)])
            self._z_map = [int(np.argmin(np.abs(z_phys_t1 - zp))) for zp in z_phys_pet]
            print(f"    T1 native: ({tnx}, {tny}, {tnz})")
        elif self._upsample > 1:
            print(f"  Upsampling {self._upsample}x for display...")
            self._disp_underlay = zoom(self.underlay, (self._upsample, self._upsample, 1), order=3)
            self._disp_clusters = zoom(
                self.clusters.astype(np.float32),
                (self._upsample, self._upsample, 1), order=0
            ).astype(np.int8)
            if self.tbrmax_vol is not None:
                self._disp_tbrmax = zoom(self.tbrmax_vol, (self._upsample, self._upsample, 1), order=1)
            self._disp_nx = self.nx * self._upsample
            self._disp_ny = self.ny * self._upsample

        # ── Build pixmaps (two sets: clusters + tbrmax) ──
        self._overlay_mode = "clusters"  # or "tbrmax"
        self._build_pixmaps()

        # ── UI ──
        self._build_ui()

        self.view.slice_changed.connect(self._on_slice_changed)
        self.view.voxel_clicked.connect(self._on_voxel_clicked)
        self.view.voxel_hovered.connect(self._on_voxel_hovered)
        self._update_info()

    # ── Pixmap builder ──────────────────────────────────────────────────

    def _build_pixmaps(self):
        print("  Building slice pixmaps...")
        self._underlay_pixmaps = []
        self._cluster_pixmaps = []
        self._tbrmax_pixmaps = []

        for z in range(self.nz):
            tz = self._z_map[z] if self._z_map else z
            sl = self._disp_underlay[:, :, tz]
            csl_raw = self._disp_clusters[:, :, tz]
            tsl_raw = self._disp_tbrmax[:, :, tz]

            pos = sl[sl > 0]
            if pos.size > 0:
                vmin = np.percentile(pos, self._win_lo)
                vmax = np.percentile(pos, self._win_hi)
            else:
                vmin, vmax = sl.min(), sl.max()
            if vmax <= vmin:
                vmin, vmax = sl.min(), sl.max()

            disp_sl = np.flipud(np.fliplr(sl.T))
            csl = np.flipud(np.fliplr(csl_raw.T))
            tsl = np.flipud(np.fliplr(tsl_raw.T))

            norm = np.clip((disp_sl - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
            under_rgba = np.zeros((self._disp_ny, self._disp_nx, 4), dtype=np.uint8)
            under_rgba[:, :, 0] = norm
            under_rgba[:, :, 1] = norm
            under_rgba[:, :, 2] = norm
            under_rgba[:, :, 3] = 255
            qimg = QtGui.QImage(under_rgba.tobytes(), self._disp_nx, self._disp_ny,
                                self._disp_nx * 4, QtGui.QImage.Format_RGBA8888)
            self._underlay_pixmaps.append(QtGui.QPixmap.fromImage(qimg))

            # Cluster overlay
            over_rgba = np.zeros((self._disp_ny, self._disp_nx, 4), dtype=np.uint8)
            for label, (r, g, b) in _CLUSTER_RGB.items():
                mask = csl == label
                over_rgba[mask, 0] = r
                over_rgba[mask, 1] = g
                over_rgba[mask, 2] = b
                over_rgba[mask, 3] = 200
            qimg2 = QtGui.QImage(over_rgba.tobytes(), self._disp_nx, self._disp_ny,
                                 self._disp_nx * 4, QtGui.QImage.Format_RGBA8888)
            self._cluster_pixmaps.append(QtGui.QPixmap.fromImage(qimg2))

            # TBRmax overlay (vectorized)
            tbr_rgba = np.zeros((self._disp_ny, self._disp_nx, 4), dtype=np.uint8)
            tmax = max(float(np.max(tsl)) if np.any(tsl > 0) else 5.0, 1.0)
            tnorm = np.clip(tsl / tmax, 0, 1)
            # jet-like: 0→blue, .25→cyan, .5→green, .75→yellow, 1→red
            r = np.where(tnorm < 0.75, np.clip(tnorm / 0.75 * 255, 0, 255), 255)
            g = np.where(tnorm < 0.5, 255, np.where(tnorm < 0.75, 255, np.clip((1 - (tnorm - 0.75) / 0.25) * 255, 0, 255)))
            b = np.where(tnorm < 0.25, 255, np.where(tnorm < 0.5, np.clip((1 - (tnorm - 0.25) / 0.25) * 255, 0, 255), 0))
            a = np.where(tsl > 0, 200, 0)
            tbr_rgba[:, :, 0] = r.astype(np.uint8)
            tbr_rgba[:, :, 1] = g.astype(np.uint8)
            tbr_rgba[:, :, 2] = b.astype(np.uint8)
            tbr_rgba[:, :, 3] = a.astype(np.uint8)
            qimg3 = QtGui.QImage(tbr_rgba.tobytes(), self._disp_nx, self._disp_ny,
                                 self._disp_nx * 4, QtGui.QImage.Format_RGBA8888)
            self._tbrmax_pixmaps.append(QtGui.QPixmap.fromImage(qimg3))

        self._cur_z = self.nz // 2

    def _apply_window(self):
        self._build_pixmaps()
        self._refresh_overlay()

    def _refresh_overlay(self):
        """Swap overlay pixmaps based on current tab."""
        ov = self._tbrmax_pixmaps if self._overlay_mode == "tbrmax" else self._cluster_pixmaps
        self.view.swap_overlay(ov)

    # ── UI builder ───────────────────────────────────────────────────────

    def _build_ui(self):
        self.setWindowTitle(f"FET Viewer — {os.path.basename(self.output_dir)}")
        self.setMinimumSize(1200, 750)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)

        # ── Left: tabbed SliceView ──
        self.tabs = QtWidgets.QTabWidget()
        self.view = SliceView()
        self.view.set_data(self._underlay_pixmaps, self._cluster_pixmaps,
                           self.nx, self.ny, self.nz)
        tab1 = QtWidgets.QWidget()
        tab1_l = QtWidgets.QVBoxLayout(tab1)
        tab1_l.setContentsMargins(0, 0, 0, 0)
        tab1_l.addWidget(self.view)
        self.tabs.addTab(tab1, "T1 + Clusters")

        # Tab 2 (same view, swapped overlay)
        tab2 = QtWidgets.QWidget()
        tab2_l = QtWidgets.QVBoxLayout(tab2)
        tab2_l.setContentsMargins(0, 0, 0, 0)
        tab2_l.addWidget(self.view)  # same widget, will be re-parented
        self.tabs.addTab(tab2, "T1 + TBRmax")

        self.tabs.currentChanged.connect(self._on_tab_changed)
        layout.addWidget(self.tabs, 2)

        # ── Right panel ──
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Info
        self.info_label = QtWidgets.QLabel()
        self.info_label.setAlignment(QtCore.Qt.AlignCenter)
        self.info_label.setStyleSheet("font-size: 13px; padding: 6px;")
        right_layout.addWidget(self.info_label)

        # TBR curve
        if self.has_curves:
            self.fig = Figure(figsize=(5, 3.5), dpi=100)
            self.canvas = FigureCanvas(self.fig)
            self.ax = self.fig.add_subplot(111)
            self.ax.set_title("TBR(t) — click a voxel")
            self.ax.set_xlabel("Time (min)")
            self.ax.set_ylabel("TBR")
            self.ax.grid(True, alpha=0.3)
            self.curve_line, = self.ax.plot([], [], "o-", color="#e74c3c", lw=2.5, ms=10,
                                             markeredgecolor="#c0392b", markeredgewidth=1.5)
            self.trend_line, = self.ax.plot([], [], "--", color="#555", lw=1.5, alpha=0.8)
            self.ax.set_xlim(min(self.time_points) - 5, max(self.time_points) + 5)
            self.ax.set_ylim(0, 5)
            self.fig.tight_layout()
            right_layout.addWidget(self.canvas, 1)
        else:
            no_curve = QtWidgets.QLabel("No TBR curves available.")
            no_curve.setAlignment(QtCore.Qt.AlignCenter)
            no_curve.setStyleSheet("color: #888; font-size: 13px;")
            right_layout.addWidget(no_curve, 1)

        # Voxel info panel
        self.voxel_info = QtWidgets.QLabel("Click a voxel for details.")
        self.voxel_info.setStyleSheet(
            "font-size: 12px; padding: 8px; background: #f5f5f5; "
            "border: 1px solid #ddd; border-radius: 4px;")
        self.voxel_info.setWordWrap(True)
        right_layout.addWidget(self.voxel_info)

        # ── T1 Window controls ──
        if self.has_t1 or self._use_native_t1:
            win_group = QtWidgets.QGroupBox("T1 Window")
            win_layout = QtWidgets.QVBoxLayout(win_group)
            lo_row = QtWidgets.QHBoxLayout()
            lo_row.addWidget(QtWidgets.QLabel("Low%:"))
            self._win_lo_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self._win_lo_slider.setRange(0, 40)
            self._win_lo_slider.setValue(self._win_lo)
            self._win_lo_label = QtWidgets.QLabel(f"{self._win_lo}%")
            lo_row.addWidget(self._win_lo_slider)
            lo_row.addWidget(self._win_lo_label)
            win_layout.addLayout(lo_row)
            hi_row = QtWidgets.QHBoxLayout()
            hi_row.addWidget(QtWidgets.QLabel("High%:"))
            self._win_hi_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self._win_hi_slider.setRange(60, 100)
            self._win_hi_slider.setValue(self._win_hi)
            self._win_hi_label = QtWidgets.QLabel(f"{self._win_hi}%")
            hi_row.addWidget(self._win_hi_slider)
            hi_row.addWidget(self._win_hi_label)
            win_layout.addLayout(hi_row)
            right_layout.addWidget(win_group)
            self._win_lo_slider.valueChanged.connect(self._on_window_changed)
            self._win_hi_slider.valueChanged.connect(self._on_window_changed)

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
        self.statusBar().showMessage(f"z={self._cur_z}/{self.nz - 1}  |  {self.underlay_label}")

    # ── Slots ──

    def _on_tab_changed(self, idx: int):
        self._overlay_mode = "tbrmax" if idx == 1 else "clusters"
        self._refresh_overlay()

    def _on_window_changed(self):
        self._win_lo = self._win_lo_slider.value()
        self._win_hi = self._win_hi_slider.value()
        if self._win_lo >= self._win_hi:
            return
        self._win_lo_label.setText(f"{self._win_lo}%")
        self._win_hi_label.setText(f"{self._win_hi}%")
        self._apply_window()

    def _on_opacity_changed(self, value: int):
        self.view.set_overlay_opacity(value / 100.0)
        self.opacity_label.setText(f"{value}%")

    def _on_slice_changed(self, z: int):
        self._cur_z = z
        self._update_info()

    def _pet_coords_from_display(self, dx: int, dy: int, vz: int) -> tuple:
        """Map display coords to PET voxel coords."""
        if self._use_native_t1:
            t1_x = self._disp_nx - 1 - dx
            t1_y = self._disp_ny - 1 - dy
            tz = self._z_map[vz] if self._z_map else vz
            phys = nib.affines.apply_affine(self._t1_affine, [t1_x, t1_y, tz])
            pet_float = np.linalg.inv(self._pet_affine) @ [phys[0], phys[1], phys[2], 1.0]
            px = max(0, min(self.nx - 1, int(round(pet_float[0]))))
            py = max(0, min(self.ny - 1, int(round(pet_float[1]))))
            pz = max(0, min(self.nz - 1, int(round(pet_float[2]))))
            return px, py, pz
        return None, None, None

    def _on_voxel_clicked(self, vx: int, vy: int, vz: int, dx: int, dy: int):
        if not self.has_curves:
            return

        # Resolve PET coords + cluster
        if self._use_native_t1:
            px, py, pz = self._pet_coords_from_display(dx, dy, vz)
            tbr_vals = self.tbr_volumes[px, py, pz, :]
            cluster = int(self.clusters[px, py, pz])
        else:
            px, py, pz = vx, vy, vz
            tbr_vals = self.tbr_volumes[vx, vy, vz, :]
            cluster = int(self.clusters[vx, vy, vz])

        label_map = {1: "Rising", 2: "Falling", 3: "Plateau", 0: "Background"}

        # Apply late-frame mask
        if self._late_mask is not None and len(tbr_vals) == len(self._late_mask):
            show_tp = list(np.array(self.time_points)[self._late_mask])
            show_tbr = list(np.array(tbr_vals)[self._late_mask])
        else:
            show_tp = self.time_points
            show_tbr = list(tbr_vals)

        # Plot curve + trend line
        for txt in list(self.ax.texts):
            txt.remove()
        self.curve_line.set_data(show_tp, show_tbr)

        # Linear regression trend
        tp_arr = np.array(show_tp, dtype=float)
        tbr_arr = np.array(show_tbr, dtype=float)
        if len(tp_arr) >= 2:
            tc = tp_arr - tp_arr.mean()
            denom = np.sum(tc ** 2)
            slope = np.sum(tc * (tbr_arr - tbr_arr.mean())) / denom if denom > 0 else 0.0
            intercept = tbr_arr.mean() - slope * tp_arr.mean()
            trend_y = slope * tp_arr + intercept
            self.trend_line.set_data(tp_arr, trend_y)
        else:
            slope = 0.0
            self.trend_line.set_data([], [])

        ymax = max(float(np.max(show_tbr)) * 1.3, 2.0)
        self.ax.set_ylim(0, ymax)

        # Annotate key points
        idxs = [0, 1, 2] if len(show_tbr) <= 3 else [0, len(show_tbr) // 2, -1]
        for i in idxs:
            t, v = show_tp[i], show_tbr[i]
            self.ax.annotate(f"{v:.2f}", (t, v),
                             textcoords="offset points", xytext=(0, 12),
                             ha="center", fontsize=10, fontweight="bold",
                             color="#c0392b")

        # Curve title
        tbr_preview = " → ".join(f"{v:.2f}" for v in show_tbr[:3])
        if len(show_tbr) > 3:
            tbr_preview += f" … {show_tbr[-1]:.2f}"
        self.ax.set_title(
            f"({px}, {py}, {pz})  {label_map.get(cluster, '?')}\n"
            f"TBR: {tbr_preview}")
        self.fig.tight_layout()
        self.canvas.draw_idle()

        # ── Voxel info panel ──
        sulmax_v = float(self.sulmax_vol[px, py, pz]) if self.sulmax_vol is not None else float(np.max(tbr_vals))
        tbrmax_v = float(self.tbrmax_vol[px, py, pz]) if self.tbrmax_vol is not None else float(np.max(show_tbr))
        slope_v = float(self.slope_vol[px, py, pz]) if self.slope_vol is not None else slope
        delta_tbr = slope_v * self._time_span

        cls_color = {1: "#e33", 2: "#33e", 3: "#3c3", 0: "#888"}.get(cluster, "#888")
        self.voxel_info.setText(
            f"<b>Voxel</b> ({px}, {py}, {pz})<br>"
            f"<b>Cluster:</b> <span style='color:{cls_color};'>{label_map.get(cluster, '?')}</span><br>"
            f"<b>SULmax:</b> {sulmax_v:.3f}<br>"
            f"<b>TBRmax:</b> {tbrmax_v:.3f}<br>"
            f"<b>Slope:</b> {slope_v:.6f} TBR/min<br>"
            f"<b>ΔTBR:</b> {delta_tbr:+.4f} over {self._time_span:.1f} min")

    def _on_voxel_hovered(self, vx: int, vy: int, vz: int, dx: int, dy: int):
        if self._use_native_t1:
            t1_x = self._disp_nx - 1 - dx
            t1_y = self._disp_ny - 1 - dy
            tz = self._z_map[vz] if self._z_map else vz
            cls = int(self._disp_clusters[t1_x, t1_y, tz])
        else:
            cls = int(self.clusters[vx, vy, vz])
        label = {1: "R", 2: "F", 3: "P", 0: "-"}.get(cls, "?")
        val = self.underlay[max(0, min(self.nx-1, vx)), max(0, min(self.ny-1, vy)), vz]
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
    parser.add_argument("--upsample", type=int, default=2,
                        help="Upsampling factor (default: 2, ignored if t1_orig exists)")
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        print(f"ERROR: not a directory: {args.output_dir}")
        sys.exit(1)

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    viewer = FETQtViewer(args.output_dir, upsample=args.upsample)
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
