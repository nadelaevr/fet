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
    'r' — reset view
    'q' — quit
"""

import argparse
import json
import os
import sys

import numpy as np
import nibabel as nib

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Rectangle


# ── Cluster colours ──────────────────────────────────────────────────────
CLUSTER_CMAP = {
    0: (0.0, 0.0, 0.0, 0.0),       # background — transparent
    1: (1.0, 0.2, 0.2, 0.5),       # rising — red
    2: (0.2, 0.4, 1.0, 0.5),       # falling — blue
    3: (0.2, 0.8, 0.2, 0.5),       # plateau — green
}


def load_nifti(path: str) -> np.ndarray:
    """Load a NIfTI file and return the 3D data array."""
    img = nib.load(path)
    return img.get_fdata().astype(np.float64)


def find_files(output_dir: str, required: list[str]) -> dict[str, str]:
    """Find required files in output directory. Returns {key: path}."""
    found = {}
    for key in required:
        path = os.path.join(output_dir, key)
        if os.path.exists(path):
            found[key] = path
        else:
            # try .nii (not gzipped)
            alt = path.replace(".nii.gz", ".nii")
            if os.path.exists(alt):
                found[key] = alt
    return found


def build_rgba_overlay(mask: np.ndarray, cmap: dict) -> np.ndarray:
    """
    Build an RGBA overlay from an integer label mask.
    mask: 3D int array with class labels
    cmap: dict {label: (r, g, b, a)}
    Returns: (Z, Y, X, 4) float array in [0, 1]
    """
    rgba = np.zeros(mask.shape + (4,), dtype=np.float32)
    for label, colour in cmap.items():
        if label == 0:
            continue
        idx = mask == label
        for c in range(4):
            rgba[..., c][idx] = colour[c]
    return rgba


class FETViewer:
    """Interactive FET viewer with click → TBR curve."""

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

        # ── Time points ──
        if self.mode == "dynamic":
            tp = self.report.get("time_points_min",
                                 self.report.get("parameters", {}).get("time_points_min", []))
            self.time_points = np.array(tp, dtype=float)
        else:
            tp = params.get("time_points_min", [20.0, 40.0, 60.0])
            self.time_points = np.array(tp, dtype=float)

        # ── Required files ──
        if self.mode == "dynamic":
            # Dynamic: we have slope + tbrmax, but no per-frame TBR saved yet
            # Show what we can
            self.has_curves = False
            required = ["map_slope.nii.gz", "map_tbrmax.nii.gz",
                        "mask_clusters.nii.gz"]
        else:
            self.has_curves = True
            required = [
                "map_tbr_t20.nii.gz",
                "map_tbr_t40.nii.gz",
                "map_tbr_t60.nii.gz",
                "mask_clusters.nii.gz",
            ]

        optional = ["map_sulmax.nii.gz", "mask_brain.nii.gz"]

        self.files = find_files(output_dir, required + optional)
        missing = [f for f in required if f not in self.files]
        if missing:
            print(f"ERROR: missing required files in {output_dir}:")
            for f in missing:
                print(f"  - {f}")
            sys.exit(1)

        # ── Load volumes ──
        print("Loading data...")
        if self.mode == "dynamic":
            self.underlay = load_nifti(self.files["map_tbrmax.nii.gz"])
        else:
            # Use mean TBR as underlay
            tbrs = []
            for key in ["map_tbr_t20.nii.gz", "map_tbr_t40.nii.gz", "map_tbr_t60.nii.gz"]:
                tbrs.append(load_nifti(self.files[key]))
            self.tbr_volumes = np.stack(tbrs, axis=-1)  # (Z, Y, X, 3)
            self.underlay = np.mean(self.tbr_volumes, axis=-1)

        self.clusters = load_nifti(self.files["mask_clusters.nii.gz"]).astype(np.int8)

        # Optional overlay
        self.brain_mask = None
        if "mask_brain.nii.gz" in self.files:
            self.brain_mask = load_nifti(
                self.files["mask_brain.nii.gz"]
            ).astype(bool)

        self.shape = self.underlay.shape
        self.nz, self.ny, self.nx = self.shape

        # ── Build RGBA overlay ──
        self.overlay = build_rgba_overlay(self.clusters, CLUSTER_CMAP)

        # ── Current slice index ──
        self.cur_z = self.nz // 2

        # ── Setup figure ──
        self.fig = plt.figure(f"FET Viewer — {os.path.basename(output_dir)}",
                              figsize=(14, 7))
        self.fig.canvas.manager.set_window_title(
            f"FET Viewer — {os.path.basename(output_dir)}"
        )

        # Left: axial slice
        self.ax_img = self.fig.add_axes([0.05, 0.1, 0.50, 0.80])
        self.ax_img.set_title("Axial slice (click for TBR curve)")
        self.img_display = None
        self.overlay_display = None
        self.crosshair = None

        # Right: TBR curve (hidden if no curves)
        if self.has_curves:
            self.ax_curve = self.fig.add_axes([0.62, 0.30, 0.33, 0.55])
            self.ax_curve.set_title("TBR(t) at clicked voxel")
            self.ax_curve.set_xlabel("Time (min)")
            self.ax_curve.set_ylabel("TBR")
            self.ax_curve.grid(True, alpha=0.3)
            self.curve_line, = self.ax_curve.plot([], [], "o-",
                                                   color="#e74c3c",
                                                   linewidth=2,
                                                   markersize=8)
            self.curve_info = self.ax_curve.text(
                0.5, 0.95, "", transform=self.ax_curve.transAxes,
                ha="center", va="top", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8)
            )
            self.ax_curve.set_xlim(self.time_points[0] - 5,
                                   self.time_points[-1] + 5)
            self.ax_curve.set_ylim(0, 5)
        else:
            self.ax_curve = None
            # Show info panel instead
            self.ax_info = self.fig.add_axes([0.62, 0.30, 0.33, 0.55])
            self.ax_info.text(0.5, 0.5,
                              "Dynamic mode:\nmulti-frame TBR not saved.\n\n"
                              "Run with curves enabled\nor use static mode\n"
                              "for interactive TBR plots.",
                              transform=self.ax_info.transAxes,
                              ha="center", va="center", fontsize=12)
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
            self.ax_legend.text(x0 + 0.05, 0.5, name,
                                va="center", fontsize=10)

        # ── Connect events ──
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_hover)

        # ── Initial draw ──
        self._draw_slice()

    # ── Drawing ──────────────────────────────────────────────────────────

    def _draw_slice(self):
        """Redraw the axial slice at cur_z."""
        self.ax_img.clear()
        self.ax_img.set_title(
            f"Axial slice z={self.cur_z}/{self.nz - 1}  "
            f"(click for TBR curve)"
        )

        # Underlay
        under = self.underlay[self.cur_z, :, :]
        vmin = np.percentile(under[under > 0], 2) if np.any(under > 0) else 0
        vmax = np.percentile(under[under > 0], 98) if np.any(under > 0) else under.max()
        if vmax <= vmin:
            vmin, vmax = under.min(), under.max()
        self.img_display = self.ax_img.imshow(
            under, cmap="gray", vmin=vmin, vmax=vmax,
            origin="lower", aspect="equal"
        )

        # Overlay
        ov = self.overlay[self.cur_z, :, :]
        self.overlay_display = self.ax_img.imshow(
            ov, origin="lower", aspect="equal"
        )

        # Crosshair (reset on redraw)
        self.crosshair = None

        self.ax_img.set_xlabel(f"X ({self.nx})")
        self.ax_img.set_ylabel(f"Y ({self.ny})")
        self.fig.canvas.draw_idle()

    def _update_curve(self, x: int, y: int, z: int):
        """Update the TBR curve plot for a given voxel."""
        if not self.has_curves:
            return

        if z < 0 or z >= self.nz or y < 0 or y >= self.ny or x < 0 or x >= self.nx:
            return

        # Get TBR values at this voxel across time
        tbr_vals = self.tbr_volumes[z, y, x, :]
        cluster = self.clusters[z, y, x]

        label_map = {1: "Rising", 2: "Falling", 3: "Plateau", 0: "Background"}

        # Update plot
        self.curve_line.set_data(self.time_points, tbr_vals)
        self.ax_curve.set_ylim(
            0, max(tbr_vals.max() * 1.3, 2.0)
        )
        self.curve_info.set_text(
            f"Voxel ({x}, {y}, {z})  "
            f"Cluster: {label_map.get(cluster, '?')}\n"
            f"TBR: {tbr_vals[0]:.3f} → {tbr_vals[1]:.3f} → {tbr_vals[2]:.3f}"
        )
        self.fig.canvas.draw_idle()

    def _update_crosshair(self, x: int, y: int):
        """Draw a crosshair at the clicked position."""
        # Remove old crosshair
        for artist in self.ax_img.lines + self.ax_img.patches:
            artist.remove()

        # Don't draw if we just cleared
        if x < 0 or y < 0:
            return

        # Crosshair lines
        lw = 0.8
        self.ax_img.axvline(x, color="yellow", lw=lw, alpha=0.6)
        self.ax_img.axhline(y, color="yellow", lw=lw, alpha=0.6)

        self.fig.canvas.draw_idle()

    # ── Events ───────────────────────────────────────────────────────────

    def on_scroll(self, event):
        if event.inaxes != self.ax_img:
            return
        dz = -1 if event.button == "up" else 1
        self.cur_z = np.clip(self.cur_z + dz, 0, self.nz - 1)
        self._draw_slice()

    def on_key(self, event):
        if event.key == "up":
            self.cur_z = np.clip(self.cur_z + 1, 0, self.nz - 1)
            self._draw_slice()
        elif event.key == "down":
            self.cur_z = np.clip(self.cur_z - 1, 0, self.nz - 1)
            self._draw_slice()
        elif event.key == "r":
            self._draw_slice()
        elif event.key == "q":
            plt.close(self.fig)

    def on_click(self, event):
        if event.inaxes != self.ax_img:
            return
        if event.xdata is None or event.ydata is None:
            return

        x = int(round(event.xdata))
        y = int(round(event.ydata))
        self._update_crosshair(x, y)
        self._update_curve(x, y, self.cur_z)

    def on_hover(self, event):
        """Show voxel coordinates in status bar."""
        if not hasattr(self, "ax_img") or event.inaxes != self.ax_img:
            return
        if event.xdata is not None and event.ydata is not None:
            x, y = int(round(event.xdata)), int(round(event.ydata))
            if 0 <= x < self.nx and 0 <= y < self.ny:
                sul_val = self.underlay[self.cur_z, y, x]
                cls = self.clusters[self.cur_z, y, x]
                self.fig.canvas.manager.set_window_title(
                    f"FET Viewer — ({x}, {y}, {self.cur_z})  "
                    f"value={sul_val:.3f}  cluster={cls}"
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
