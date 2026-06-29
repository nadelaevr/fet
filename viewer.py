#!/usr/bin/env python
"""
Interactive FET curve viewer.

Usage:
    python viewer.py <output_dir>

Shows axial slices of the FET analysis output. Click any voxel
to see its TBR(t) curve (TBR vs time).

Controls:
    Up/Down arrows or mouse wheel — scroll slices
    Click on image — show TBR curve for that voxel
    'r' — reset crosshair
    'q' — quit
"""

import argparse
import json
import os
import sys

import numpy as np
import nibabel as nib

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ── Cluster colours (RGBA LUT for vectorised indexing) ──────────────────
_COLOUR_LUT = np.array([
    [0.0, 0.0, 0.0, 0.0],    # 0: background — transparent
    [1.0, 0.2, 0.2, 0.5],    # 1: rising — red
    [0.2, 0.4, 1.0, 0.5],    # 2: falling — blue
    [0.2, 0.8, 0.2, 0.5],    # 3: plateau — green
], dtype=np.float32)


def load_vol(path: str) -> np.ndarray:
    """Load a NIfTI file and reorient to canonical RAS."""
    img = nib.load(path)
    canonical = nib.as_closest_canonical(img)
    return canonical.get_fdata().astype(np.float64)


def find_files(output_dir: str, names: list[str]) -> dict[str, str]:
    """Find files in output directory. Returns {name: full_path}."""
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


# ── Main viewer class ────────────────────────────────────────────────────

class FETViewer:
    """Interactive FET viewer with click -> TBR curve."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir

        # ── Load report ──
        report_path = os.path.join(output_dir, "report.json")
        if os.path.exists(report_path):
            with open(report_path) as f:
                self.report = json.load(f)
        else:
            self.report = {}

        # ── Determine mode ──
        params = self.report.get("parameters", {})
        self.mode = params.get("mode", self.report.get("mode", "static"))

        # ── Time points (safely ensure 1D array) ──
        if self.mode == "dynamic":
            tp = self.report.get("time_points_min",
                                 self.report.get("parameters", {}).get("time_points_min", []))
        else:
            tp = params.get("time_points_min", [20.0, 40.0, 60.0])
        tp = np.asarray(tp, dtype=float).ravel()
        if tp.ndim == 0 or tp.size == 0:
            tp = np.array([20.0, 40.0, 60.0])
        self.time_points = tp

        # ── Files ──
        if self.mode == "dynamic":
            self.has_curves = False
            required = ["map_slope.nii.gz", "map_tbrmax.nii.gz", "mask_clusters.nii.gz"]
        else:
            self.has_curves = True
            required = [
                "map_tbr_t20.nii.gz", "map_tbr_t40.nii.gz", "map_tbr_t60.nii.gz",
                "mask_clusters.nii.gz",
            ]

        optional = ["map_t1.nii.gz", "map_sulmax.nii.gz", "mask_brain.nii.gz"]

        self.files = find_files(output_dir, required + optional)
        missing = [f for f in required if f not in self.files]
        if missing:
            print(f"ERROR: missing required files in {output_dir}:")
            for f in missing:
                print(f"  - {f}")
            sys.exit(1)

        # ── Load volumes ──
        print("Loading data (canonical RAS)...")

        if self.mode == "static":
            tbrs = []
            for key in ["map_tbr_t20.nii.gz", "map_tbr_t40.nii.gz", "map_tbr_t60.nii.gz"]:
                tbrs.append(load_vol(self.files[key]))
            self.tbr_volumes = np.stack(tbrs, axis=-1)  # (nx, ny, nz, 3)

        # Underlay
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

        # ── Shape ──
        self.shape = self.underlay.shape
        self.nx, self.ny, self.nz = self.shape
        print(f"  Underlay: {self.shape}  ({self.underlay_label})")
        print(f"  Clusters: {self.clusters.shape}")
        if self.mode == "static":
            print(f"  TBR vols: {self.tbr_volumes.shape}")

        # Sanity checks
        if self.mode == "static" and self.tbr_volumes.shape[:3] != self.shape:
            print(f"  WARNING: TBR shape {self.tbr_volumes.shape[:3]} != underlay {self.shape}")
        if self.clusters.shape != self.shape:
            print(f"  WARNING: clusters shape {self.clusters.shape} != underlay {self.shape}")

        # ── Precompute ALL display-ready slices upfront ──
        print("  Precomputing display slices...")
        # Underlay: each slice = fliplr(data[:,:,z].T) → (ny, nx)
        self._disp_underlay = np.zeros((self.nz, self.ny, self.nx), dtype=np.float64)
        # Overlay: each slice = fliplr(LUT[clusters[:,:,z]].transpose(1,0,2)) → (ny, nx, 4)
        self._disp_overlay = np.zeros((self.nz, self.ny, self.nx, 4), dtype=np.float32)
        self._vmin = np.zeros(self.nz, dtype=np.float64)
        self._vmax = np.zeros(self.nz, dtype=np.float64)

        for z in range(self.nz):
            # Underlay
            sl = self.underlay[:, :, z]
            self._disp_underlay[z] = np.fliplr(sl.T)

            # Percentiles
            pos = sl[sl > 0]
            if pos.size > 0:
                self._vmin[z] = np.percentile(pos, 2)
                self._vmax[z] = np.percentile(pos, 98)
            else:
                self._vmin[z] = sl.min()
                self._vmax[z] = sl.max()
            if self._vmax[z] <= self._vmin[z]:
                self._vmin[z] = sl.min()
                self._vmax[z] = sl.max()

            # Overlay via vectorised LUT (no Python loops)
            rgba = _COLOUR_LUT[self.clusters[:, :, z]]      # (nx, ny, 4)
            self._disp_overlay[z] = np.fliplr(rgba.transpose(1, 0, 2))

        # ── Current slice ──
        self.cur_z = self.nz // 2

        # ── Setup figure ──
        title = f"FET Viewer — {os.path.basename(output_dir)}"
        self.fig = plt.figure(title, figsize=(14, 7))
        self.fig.canvas.manager.set_window_title(title)

        # Left: axial slice
        self.ax_img = self.fig.add_axes([0.05, 0.1, 0.50, 0.80])
        self.ax_img.set_title("")
        self.ax_img.set_xlabel("X (R-L)")
        self.ax_img.set_ylabel("Y (P-A)")

        # imshow objects created ONCE
        self.img_display = self.ax_img.imshow(
            self._disp_underlay[self.cur_z], cmap="gray",
            vmin=self._vmin[self.cur_z], vmax=self._vmax[self.cur_z],
            origin="lower", aspect="equal"
        )
        self.overlay_display = self.ax_img.imshow(
            self._disp_overlay[self.cur_z], origin="lower", aspect="equal"
        )

        # Persistent crosshair
        self.cross_vline = self.ax_img.axvline(0, color="yellow", lw=0.8, alpha=0.6, visible=False)
        self.cross_hline = self.ax_img.axhline(0, color="yellow", lw=0.8, alpha=0.6, visible=False)

        # Right: TBR curve
        if self.has_curves:
            self.ax_curve = self.fig.add_axes([0.62, 0.30, 0.33, 0.55])
            self.ax_curve.set_title("TBR(t) at clicked voxel")
            self.ax_curve.set_xlabel("Time (min)")
            self.ax_curve.set_ylabel("TBR")
            self.ax_curve.grid(True, alpha=0.3)
            self.curve_line, = self.ax_curve.plot(
                [], [], "o-", color="#e74c3c", linewidth=2, markersize=8
            )
            self.curve_info = self.ax_curve.text(
                0.5, 0.95, "", transform=self.ax_curve.transAxes,
                ha="center", va="top", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8)
            )
            self.ax_curve.set_xlim(self.time_points[0] - 5, self.time_points[-1] + 5)
            self.ax_curve.set_ylim(0, 5)
        else:
            self.ax_curve = None
            self.ax_info = self.fig.add_axes([0.62, 0.30, 0.33, 0.55])
            self.ax_info.text(
                0.5, 0.5,
                "Dynamic mode:\nmulti-frame TBR not saved.\n\n"
                "Use static mode for\ninteractive TBR plots.",
                transform=self.ax_info.transAxes,
                ha="center", va="center", fontsize=12,
            )
            self.ax_info.axis("off")

        # ── Colour legend ──
        self.ax_legend = self.fig.add_axes([0.05, 0.02, 0.50, 0.04])
        self.ax_legend.axis("off")
        legend_items = [
            (1, "Rising", "#e33"),
            (2, "Falling", "#33e"),
            (3, "Plateau", "#3c3"),
        ]
        for i, (label, name, color) in enumerate(legend_items):
            x0 = 0.05 + i * 0.18
            self.ax_legend.add_patch(
                Rectangle((x0, 0.1), 0.04, 0.8, color=color, alpha=0.6)
            )
            self.ax_legend.text(x0 + 0.05, 0.5, name, va="center", fontsize=10)
        self.ax_legend.text(0.62, 0.5, f"Underlay: {self.underlay_label}",
                            va="center", fontsize=9, color="gray")

        # ── Connect events ──
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_hover)

        # ── Initial draw ──
        self._update_title()
        self.fig.canvas.draw_idle()

    # ── Drawing ──────────────────────────────────────────────────────────

    def _update_title(self):
        self.ax_img.set_title(
            f"Axial slice  z={self.cur_z}/{self.nz - 1}  "
            f"[{self.underlay_label}]  (click for TBR curve)"
        )

    def _update_slice(self):
        """Update axial slice — fast path using precomputed caches.

        No numpy operations, no per-slice RGBA construction —
        just set_data + set_clim on existing imshow objects.
        """
        z = self.cur_z
        self.img_display.set_data(self._disp_underlay[z])
        self.img_display.set_clim(self._vmin[z], self._vmax[z])
        self.overlay_display.set_data(self._disp_overlay[z])
        self._update_title()
        self.fig.canvas.draw_idle()

    def _update_curve(self, x: int, y: int, z: int):
        """Update the TBR curve plot for a given voxel."""
        if not self.has_curves:
            return
        if not (0 <= x < self.nx and 0 <= y < self.ny and 0 <= z < self.nz):
            return

        tbr_vals = np.asarray(self.tbr_volumes[x, y, z, :]).ravel()
        cluster = int(self.clusters[x, y, z])

        label_map = {1: "Rising", 2: "Falling", 3: "Plateau", 0: "Background"}

        self.curve_line.set_data(self.time_points, tbr_vals)
        self.ax_curve.set_ylim(0, max(float(tbr_vals.max()) * 1.3, 2.0))
        self.curve_info.set_text(
            f"Voxel ({x}, {y}, {z})  "
            f"Cluster: {label_map.get(cluster, '?')}\n"
            f"TBR: {tbr_vals[0]:.3f} -> {tbr_vals[1]:.3f} -> {tbr_vals[2]:.3f}"
        )
        self.fig.canvas.draw_idle()

    def _show_crosshair(self, x: int, y: int):
        """Position persistent crosshair at display coords."""
        if x < 0 or y < 0:
            self.cross_vline.set_visible(False)
            self.cross_hline.set_visible(False)
        else:
            self.cross_vline.set_xdata(x)
            self.cross_hline.set_ydata(y)
            self.cross_vline.set_visible(True)
            self.cross_hline.set_visible(True)
        self.fig.canvas.draw_idle()

    # ── Events ───────────────────────────────────────────────────────────

    def on_scroll(self, event):
        if event.inaxes != self.ax_img:
            return
        dz = -1 if event.button == "up" else 1
        self.cur_z = np.clip(self.cur_z + dz, 0, self.nz - 1)
        self._update_slice()

    def on_key(self, event):
        if event.key == "up":
            self.cur_z = np.clip(self.cur_z + 1, 0, self.nz - 1)
            self._update_slice()
        elif event.key == "down":
            self.cur_z = np.clip(self.cur_z - 1, 0, self.nz - 1)
            self._update_slice()
        elif event.key == "r":
            self._show_crosshair(-1, -1)
        elif event.key == "q":
            plt.close(self.fig)

    def on_click(self, event):
        if event.inaxes != self.ax_img:
            return
        if event.xdata is None or event.ydata is None:
            return

        dx = int(round(event.xdata))
        dy = int(round(event.ydata))
        vx, vy, vz = self.nx - 1 - dx, dy, self.cur_z

        print(f"  Click: display=({dx},{dy}) -> voxel=({vx},{vy},{vz})  "
              f"nx={self.nx} ny={self.ny}")

        if not (0 <= vx < self.nx and 0 <= vy < self.ny):
            print("  -> OUT OF BOUNDS")
            return

        cluster = int(self.clusters[vx, vy, vz])
        print(f"  -> cluster={cluster}")
        if self.has_curves:
            tbr_vals = self.tbr_volumes[vx, vy, vz, :]
            print(f"  -> TBR=({tbr_vals[0]:.4f}, {tbr_vals[1]:.4f}, {tbr_vals[2]:.4f})")

        self._show_crosshair(dx, dy)
        self._update_curve(vx, vy, vz)

    def on_hover(self, event):
        """Show voxel info in status bar."""
        if not hasattr(self, "ax_img") or event.inaxes != self.ax_img:
            return
        if event.xdata is not None and event.ydata is not None:
            dx, dy = int(round(event.xdata)), int(round(event.ydata))
            vx, vy = self.nx - 1 - dx, dy
            if 0 <= vx < self.nx and 0 <= vy < self.ny:
                val = self.underlay[vx, vy, self.cur_z]
                cls = self.clusters[vx, vy, self.cur_z]
                label = {1: "R", 2: "F", 3: "P", 0: "-"}.get(int(cls), "?")
                self.fig.canvas.manager.set_window_title(
                    f"FET Viewer  |  ({vx}, {vy}, {self.cur_z})  "
                    f"val={val:.3f}  cluster={label}"
                )

    def show(self):
        plt.show()


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Interactive FET curve viewer"
    )
    parser.add_argument("output_dir",
                        help="Pipeline output directory with NIfTI maps")
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        print(f"ERROR: not a directory: {args.output_dir}")
        sys.exit(1)

    viewer = FETViewer(args.output_dir)
    viewer.show()


if __name__ == "__main__":
    main()
