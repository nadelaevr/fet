#!/usr/bin/env python
"""
Synthetic test: create fake DICOM-like NIfTI volumes and run the pipeline
to verify correctness end-to-end.
"""

import os
import sys
import numpy as np
import nibabel as nib

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis import (
    trimmed_mean_numpy_only,
    compute_tbr_map,
    compute_slope_map,
    classify_curves,
    compute_sulmax,
    compute_tbrmax,
    build_significant_mask,
)
from output import save_all_outputs


def test_pipeline():
    print("Running synthetic test...\n")

    np.random.seed(42)
    shape = (30, 64, 64)  # small volume for speed

    # Background: low uniform SUL ~1.0 (normal tissue)
    background = np.ones(shape) * 1.0 + np.random.normal(0, 0.1, shape)

    # Create 3 distinct regions:
    # Region 1: RISING TBR (SUL increases over time)
    #   At t=20: SUL=3, t=40: SUL=4.5, t=60: SUL=6
    # Region 2: FALLING TBR (SUL decreases)
    #   At t=20: SUL=6, t=40: SUL=4.5, t=60: SUL=3
    # Region 3: PLATEAU TBR (SUL stable)
    #   At t=20: SUL=4, t=40: SUL=4.2, t=60: SUL=4.1

    sul_t20 = background.copy()
    sul_t40 = background.copy()
    sul_t60 = background.copy()

    # Region 1: slice 10, rows 20:35, cols 10:25 -> RISING
    r1 = (slice(10, 15), slice(20, 35), slice(10, 25))
    sul_t20[r1] = 3.0
    sul_t40[r1] = 4.5
    sul_t60[r1] = 6.0

    # Region 2: slice 10, rows 20:35, cols 35:50 -> FALLING
    r2 = (slice(10, 15), slice(20, 35), slice(35, 50))
    sul_t20[r2] = 6.0
    sul_t40[r2] = 4.5
    sul_t60[r2] = 3.0

    # Region 3: slice 10, rows 40:55, cols 10:25 -> PLATEAU
    r3 = (slice(10, 15), slice(40, 55), slice(10, 25))
    sul_t20[r3] = 4.0
    sul_t40[r3] = 4.2
    sul_t60[r3] = 4.1

    # Low-uptake region (should NOT appear in masks): slice 20 -> SUL=0.5
    r4 = (slice(20, 25), slice(20, 35), slice(10, 25))
    sul_t20[r4] = 0.5
    sul_t40[r4] = 0.6
    sul_t60[r4] = 0.5

    # ---- Run analysis ----
    time_points = (20.0, 40.0, 60.0)
    trim_percent = 2.5
    sul_threshold = 2.0
    tbr_threshold = 2.0
    tbr_delta_threshold = 0.3
    time_span = 40.0

    # SULmean per time point
    tbr_vols = {}
    report_tp = {}
    for label, sul, t_min in [
        ("t20", sul_t20, 20.0),
        ("t40", sul_t40, 40.0),
        ("t60", sul_t60, 60.0),
    ]:
        sul_mean, sul_std, ci_lo, ci_hi = trimmed_mean_numpy_only(sul, trim_percent)
        tbr = compute_tbr_map(sul, sul_mean)
        tbr_vols[label] = tbr
        report_tp[label] = {
            "sulmean": sul_mean,
            "ci": (ci_lo, ci_hi),
        }
        print(f"  {label}: SULmean={sul_mean:.4f}, 95%CI=[{ci_lo:.4f}, {ci_hi:.4f}]")

    tbr_t20 = tbr_vols["t20"]
    tbr_t40 = tbr_vols["t40"]
    tbr_t60 = tbr_vols["t60"]

    # Slope
    slope_map = compute_slope_map(tbr_t20, tbr_t40, tbr_t60, time_points)

    # Classification
    classes = classify_curves(slope_map, tbr_delta_threshold, time_span)

    # SULmax and TBRmax
    sulmax = compute_sulmax(sul_t20, sul_t40, sul_t60)
    tbrmax = compute_tbrmax(tbr_t20, tbr_t40, tbr_t60)

    # Significant mask
    sig_mask = build_significant_mask(
        sulmax, tbr_t20, tbr_t40, tbr_t60,
        sul_threshold, tbr_threshold,
    )

    # Cluster mask
    mask_clusters = np.zeros_like(classes, dtype=np.int8)
    mask_clusters[sig_mask] = classes[sig_mask]

    # ---- Verify ----
    print(f"\nSULmean(t20)={report_tp['t20']['sulmean']:.4f}")
    print(f"SULmean(t40)={report_tp['t40']['sulmean']:.4f}")
    print(f"SULmean(t60)={report_tp['t60']['sulmean']:.4f}")

    # Check regions
    slope_r1 = slope_map[r1]
    slope_r2 = slope_map[r2]
    slope_r3 = slope_map[r3]

    print(f"\nRegion 1 (rising):  mean slope = {slope_r1.mean():.6f} TBR/min")
    print(f"Region 2 (falling): mean slope = {slope_r2.mean():.6f} TBR/min")
    print(f"Region 3 (plateau): mean slope = {slope_r3.mean():.6f} TBR/min")

    class_r1 = classes[r1]
    class_r2 = classes[r2]
    class_r3 = classes[r3]

    print(f"\nRegion 1 classification: {dict(zip(*np.unique(class_r1, return_counts=True)))}")
    print(f"Region 2 classification: {dict(zip(*np.unique(class_r2, return_counts=True)))}")
    print(f"Region 3 classification: {dict(zip(*np.unique(class_r3, return_counts=True)))}")

    n_rising = int(np.sum(mask_clusters == 1))
    n_falling = int(np.sum(mask_clusters == 2))
    n_plateau = int(np.sum(mask_clusters == 3))

    print(f"\nClusters in significant mask:")
    print(f"  Rising:  {n_rising}")
    print(f"  Falling: {n_falling}")
    print(f"  Plateau: {n_plateau}")

    # Assert basic correctness
    assert n_rising > 0, "No rising voxels found!"
    assert n_falling > 0, "No falling voxels found!"
    assert n_plateau > 0, "No plateau voxels found!"

    # Check that low-uptake region is NOT in significant mask
    assert not np.any(sig_mask[r4]), "Low-uptake region should not be significant!"

    # Save test output
    affine = np.eye(4)
    output_dir = os.path.join(os.path.dirname(__file__), "test_output")

    # Build TBR masks
    mask_tbr_gt2_t20 = (tbr_t20 > tbr_threshold).astype(np.uint8)
    mask_tbr_gt2_t40 = (tbr_t40 > tbr_threshold).astype(np.uint8)
    mask_tbr_gt2_t60 = (tbr_t60 > tbr_threshold).astype(np.uint8)
    mask_tbr_gt2 = ((tbr_t20 > tbr_threshold) |
                    (tbr_t40 > tbr_threshold) |
                    (tbr_t60 > tbr_threshold)).astype(np.uint8)
    mask_sulmax_gt2 = (sulmax > sul_threshold).astype(np.uint8)

    report = {
        "test": True,
        "per_timepoint": {
            k: {"sulmean": round(v["sulmean"], 4)} for k, v in report_tp.items()
        },
        "results": {
            "n_rising": n_rising,
            "n_falling": n_falling,
            "n_plateau": n_plateau,
        },
    }

    save_all_outputs(
        output_dir=output_dir,
        affine=affine,
        mask_clusters=mask_clusters,
        map_slope=slope_map,
        map_sulmax=sulmax,
        map_tbrmax=tbrmax,
        mask_sulmax_gt2=mask_sulmax_gt2,
        mask_tbr_gt2=mask_tbr_gt2,
        mask_tbr_gt2_t20=mask_tbr_gt2_t20,
        mask_tbr_gt2_t40=mask_tbr_gt2_t40,
        mask_tbr_gt2_t60=mask_tbr_gt2_t60,
        sul_t20=sul_t20,
        sul_t40=sul_t40,
        sul_t60=sul_t60,
        tbr_t20=tbr_t20,
        tbr_t40=tbr_t40,
        tbr_t60=tbr_t60,
        report=report,
    )

    print("\nALL TESTS PASSED!")
    return True


if __name__ == "__main__":
    test_pipeline()
