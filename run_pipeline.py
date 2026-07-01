#!/usr/bin/env python
"""
FET-PET/MRI Tyrosine Kinetics Analyzer
=======================================

Usage (static — 3 timepoints):
    python run_pipeline.py \\
        --t20 <folder_20min> --t40 <folder_40min> --t60 <folder_60min> \\
        [--t1 <folder_T1>] \\
        --output <output_folder>

Usage (dynamic — 3 DICOM folders with 4D series):
    python run_pipeline.py --dynamic \\
        --dyn1 <folder_series1> --dyn2 <folder_series2> --dyn3 <folder_series3> \\
        [--static-ref <folder_static>] \\
        [--t1 <folder_T1>] \\
        --output <output_folder>
"""

import argparse
import os
import time
import numpy as np

from dicom_reader import dicom_to_nifti, convert_bqml_to_sul, _read_dicom_metadata
from dicom_reader_dynamic import (
    dynamic_dicom_to_4d,
    convert_4d_bqml_to_sul,
    build_dynamic_time_schedule,
    trim_frames,
)
from preprocess import preprocess_volumes, preprocess_4d
from analysis import (
    trimmed_mean_numpy_only,
    compute_tbr_map,
    compute_slope_map,
    classify_curves,
    compute_sulmax,
    compute_tbrmax,
    build_significant_mask,
    filter_small_clusters,
)
from analysis_dynamic import (
    compute_tbr_4d,
    compute_slope_map_dynamic,
    classify_curves_dynamic,
    compute_tbrmax_dynamic,
)
from output import save_all_outputs, save_dynamic_outputs


def parse_args():
    parser = argparse.ArgumentParser(
        description="FET-PET/MRI Tyrosine Kinetics Analyzer"
    )

    # Mode
    parser.add_argument("--dynamic", action="store_true",
                        help="Dynamic mode: 3 DICOM folders with 4D series")

    # Static inputs
    parser.add_argument("--t20", default=None, help="DICOM folder: 20-min static")
    parser.add_argument("--t40", default=None, help="DICOM folder: 40-min static")
    parser.add_argument("--t60", default=None, help="DICOM folder: 60-min static")

    # Dynamic inputs
    parser.add_argument("--dyn1", default=None, help="DICOM folder: dynamic series 1")
    parser.add_argument("--dyn2", default=None, help="DICOM folder: dynamic series 2")
    parser.add_argument("--dyn3", default=None, help="DICOM folder: dynamic series 3")

    # Static reference for dynamic mode (tags may be incomplete in 4D series)
    parser.add_argument("--static-ref", default=None,
                        help="Static DICOM folder with complete patient/dose tags "
                             "(used as fallback when dynamic series tags are missing)")

    # Common
    parser.add_argument("--t1", default=None, help="DICOM folder: T1-weighted MR")
    parser.add_argument("--output", "-o", required=True, help="Output folder")

    # Thresholds
    parser.add_argument("--sul-threshold", type=float, default=2.0)
    parser.add_argument("--tbr-threshold", type=float, default=2.0)
    parser.add_argument("--tbr-delta-threshold", type=float, default=0.3,
                        help="Min TBR change over time span to classify as trend")
    parser.add_argument("--time-span", type=float, default=40.0,
                        help="Time span first->last point (min)")
    parser.add_argument("--time-points", type=float, nargs=3,
                        default=[20.0, 40.0, 60.0])
    parser.add_argument("--trim-percent", type=float, default=2.5)

    # Cluster filtering
    parser.add_argument("--min-cluster-size", type=int, default=45,
                        help="Minimum voxels per connected cluster "
                             "(default: 45, set 0 to disable)")

    # Preprocessing
    parser.add_argument("--no-skull-strip", action="store_true",
                        help="Disable skull-stripping")
    parser.add_argument("--no-smoothing", action="store_true",
                        help="Disable smoothing")
    parser.add_argument("--smooth-sigma", type=float, default=1.0,
                        help="Smoothing sigma in voxels (default: 1.0)")

    # Dynamic: frame trimming
    parser.add_argument("--no-frame", type=int, default=0,
                        help="Skip first N frames in dynamic mode (default: 0)")

    args = parser.parse_args()

    if args.dynamic:
        if not (args.dyn1 and args.dyn2 and args.dyn3):
            parser.error("--dynamic requires --dyn1, --dyn2, --dyn3")
    else:
        if not (args.t20 and args.t40 and args.t60):
            parser.error("Static mode requires --t20, --t40, --t60")
        if args.static_ref:
            print("WARNING: --static-ref has no effect in static mode (ignored)")

    return args


def nib_aff2axcodes(affine):
    """Helper to get axis codes without full nibabel import at top."""
    import nibabel as nib
    return nib.aff2axcodes(affine)


def _fill_meta_from_static(meta: dict, static_meta: dict) -> dict:
    """Fill in missing patient/dose meta fields from a static reference.

    Dynamic 4D series often have incomplete DICOM tags (no PatientSize,
    PatientWeight). If a static series from the same scan session is
    available, use it as a fallback for any zero/missing values.

    Returns the same meta dict (mutated in-place) for convenience.
    """
    _FALLBACK_KEYS = [
        "patient_weight_kg", "patient_height_m", "patient_height_cm",
        "patient_sex", "injected_dose_bq",
    ]
    for key in _FALLBACK_KEYS:
        current = meta.get(key)
        if current is None or current == "" or (isinstance(current, (int, float)) and current <= 0):
            fallback = static_meta.get(key)
            if fallback is not None and fallback != "" and not (isinstance(fallback, (int, float)) and fallback <= 0):
                meta[key] = fallback
                print(f"  [static-ref] {key} <- {fallback}")
    return meta


# ===========================================================================
# STATIC PIPELINE (unchanged logic)
# ===========================================================================

def run_static(args):
    print("=" * 60)
    print("FET-PET/MRI Tyrosine Kinetics Analyzer  [STATIC]")
    print("=" * 60)

    # ---- Step 1: Read DICOM via dcm2niix and convert to SUL ----
    time_folders = {"t20": args.t20, "t40": args.t40, "t60": args.t60}
    sul_volumes = {}
    meta = None
    affine = None

    for label, folder in time_folders.items():
        print(f"\nReading {label}: {folder}")
        t0 = time.time()

        volume_bqml, aff, m = dicom_to_nifti(folder)
        print(f"  dcm2niix: shape={volume_bqml.shape}, {time.time()-t0:.1f}s")
        print(f"  Orientation: {nib_aff2axcodes(aff)}")

        if meta is None:
            meta = m
            affine = aff

        if affine is not None and volume_bqml.shape != sul_volumes.get("t20", volume_bqml).shape:
            print(f"  WARNING: shape mismatch — resampling to reference grid")

        t0 = time.time()
        sul_vol = convert_bqml_to_sul(volume_bqml, m)
        sul_volumes[label] = sul_vol

        nz = sul_vol[sul_vol > 0]
        print(f"  SUL: [{sul_vol.min():.2f} .. {sul_vol.max():.2f}], "
              f"mean(nz)={nz.mean():.3f}, P95={np.percentile(nz, 95):.2f}")

    # ---- Read T1 if provided ----
    t1_volume = None
    t1_affine = None
    if args.t1:
        print(f"\nReading T1: {args.t1}")
        t0 = time.time()
        t1_raw, t1_affine, _ = dicom_to_nifti(args.t1)
        t1_volume = t1_raw
        print(f"  dcm2niix: shape={t1_volume.shape}, {time.time()-t0:.1f}s")
        print(f"  Orientation: {nib_aff2axcodes(t1_affine)}")

    # ---- Step 2: Preprocessing ----
    print("\n" + "-" * 60)
    print("Preprocessing (skull-strip + smoothing)")
    print("-" * 60)

    sul_list = [sul_volumes["t20"], sul_volumes["t40"], sul_volumes["t60"]]
    processed, brain_mask, t1_resampled = preprocess_volumes(
        sul_list,
        affine=affine,
        t1_volume=t1_volume,
        t1_affine=t1_affine,
        apply_skull_strip=not args.no_skull_strip,
        apply_smoothing=not args.no_smoothing,
        smooth_sigma=args.smooth_sigma,
    )
    sul_t20, sul_t40, sul_t60 = processed

    print(f"\n  After preprocessing:")
    for label, vol in [("t20", sul_t20), ("t40", sul_t40), ("t60", sul_t60)]:
        nz = vol[vol > 0]
        if len(nz) > 0:
            print(f"    {label}: [{vol.min():.2f} .. {vol.max():.2f}], "
                  f"mean(nz)={nz.mean():.3f}, P95={np.percentile(nz, 95):.2f}")

    # ---- Step 3: SULmean + TBR ----
    print("\n" + "-" * 60)
    print("Computing SULmean and TBR maps")
    print("-" * 60)

    report = {
        "mode": "static",
        "parameters": {
            "sul_threshold": args.sul_threshold,
            "tbr_threshold": args.tbr_threshold,
            "tbr_delta_threshold": args.tbr_delta_threshold,
            "time_span_min": args.time_span,
            "time_points_min": args.time_points,
            "trim_percent": args.trim_percent,
            "min_cluster_size": args.min_cluster_size,
            "skull_strip": not args.no_skull_strip,
            "t1_used": args.t1 is not None,
            "smoothing": not args.no_smoothing,
            "smooth_sigma": args.smooth_sigma,
            "patient_weight_kg": meta["patient_weight_kg"],
            "patient_height_cm": meta["patient_height_cm"],
            "patient_sex": meta["patient_sex"],
            "injected_dose_bq": meta["injected_dose_bq"],
        },
        "per_timepoint": {},
    }

    tbr_volumes = {}
    for label, sul, t_min in [
        ("t20", sul_t20, args.time_points[0]),
        ("t40", sul_t40, args.time_points[1]),
        ("t60", sul_t60, args.time_points[2]),
    ]:
        print(f"\n  {label} ({t_min} min)")
        sul_mean, sul_std, ci_lo, ci_hi = trimmed_mean_numpy_only(sul, args.trim_percent)
        print(f"    SULmean={sul_mean:.4f}  std={sul_std:.4f}  95%CI=[{ci_lo:.4f}, {ci_hi:.4f}]")

        tbr = compute_tbr_map(sul, sul_mean)
        tbr_volumes[label] = tbr
        n_tbr = int(np.sum(tbr > args.tbr_threshold))
        print(f"    TBR: [{tbr.min():.2f} .. {tbr.max():.2f}]  voxels TBR>{args.tbr_threshold}: {n_tbr}")

        report["per_timepoint"][label] = {
            "time_min": t_min,
            "sulmean": round(sul_mean, 4),
            "sul_std": round(sul_std, 4),
            "ci_lower": round(ci_lo, 4),
            "ci_upper": round(ci_hi, 4),
            "n_voxels_tbr_gt_threshold": n_tbr,
        }

    tbr_t20, tbr_t40, tbr_t60 = tbr_volumes["t20"], tbr_volumes["t40"], tbr_volumes["t60"]

    # ---- Step 4: Slope + classification ----
    print("\n" + "-" * 60)
    print("Slope map and curve classification")
    print("-" * 60)

    slope_map = compute_slope_map(tbr_t20, tbr_t40, tbr_t60, tuple(args.time_points))
    sulmax = compute_sulmax(sul_t20, sul_t40, sul_t60)
    tbrmax = compute_tbrmax(tbr_t20, tbr_t40, tbr_t60)

    sig_mask = build_significant_mask(
        sulmax, tbr_t20, tbr_t40, tbr_t60,
        args.sul_threshold, args.tbr_threshold,
    )

    slope_map_display = slope_map  # full map, no masking

    classes = classify_curves(slope_map, args.tbr_delta_threshold, args.time_span)
    mask_clusters = np.zeros_like(classes, dtype=np.int8)
    mask_clusters[sig_mask] = classes[sig_mask]

    # Filter small clusters
    if args.min_cluster_size > 0:
        mask_clusters_raw = mask_clusters.copy()
        mask_clusters = filter_small_clusters(mask_clusters, args.min_cluster_size)
        n_removed = int(np.sum(mask_clusters_raw != 0)) - int(np.sum(mask_clusters != 0))
        if n_removed > 0:
            print(f"  Cluster filter (min {args.min_cluster_size} voxels): removed {n_removed} voxels")

    mask_tbr_gt2_t20 = (tbr_t20 > args.tbr_threshold).astype(np.uint8)
    mask_tbr_gt2_t40 = (tbr_t40 > args.tbr_threshold).astype(np.uint8)
    mask_tbr_gt2_t60 = (tbr_t60 > args.tbr_threshold).astype(np.uint8)
    mask_tbr_gt2 = ((tbr_t20 > args.tbr_threshold) |
                    (tbr_t40 > args.tbr_threshold) |
                    (tbr_t60 > args.tbr_threshold)).astype(np.uint8)
    mask_sulmax_gt2 = (sulmax > args.sul_threshold).astype(np.uint8)

    n_rising  = int(np.sum(mask_clusters == 1))
    n_falling = int(np.sum(mask_clusters == 2))
    n_plateau = int(np.sum(mask_clusters == 3))
    n_sig     = int(np.sum(sig_mask))
    slope_thr = args.tbr_delta_threshold / args.time_span

    print(f"\nSlope threshold: {slope_thr:.6f} TBR/min")
    print(f"Significant voxels: {n_sig}")
    print(f"  Rising:  {n_rising}")
    print(f"  Falling: {n_falling}")
    print(f"  Plateau: {n_plateau}")

    report["results"] = {
        "slope_threshold_tbr_per_min": round(slope_thr, 6),
        "n_significant_voxels": n_sig,
        "n_rising": n_rising,
        "n_falling": n_falling,
        "n_plateau": n_plateau,
        "n_sulmax_gt2": int(np.sum(mask_sulmax_gt2)),
        "n_tbr_gt2": int(np.sum(mask_tbr_gt2)),
    }
    for cid, cname in [(1, "rising"), (2, "falling"), (3, "plateau")]:
        vox = mask_clusters == cid
        if np.any(vox):
            report["results"][f"mean_slope_{cname}"] = round(float(np.mean(slope_map[vox])), 6)

    # ---- Step 5: Save ----
    print("\n" + "-" * 60)
    print("Saving outputs")
    print("-" * 60)

    save_all_outputs(
        output_dir=args.output,
        affine=affine,
        mask_clusters=mask_clusters,
        map_slope=slope_map_display, map_sulmax=sulmax, map_tbrmax=tbrmax,
        mask_sulmax_gt2=mask_sulmax_gt2,
        mask_tbr_gt2=mask_tbr_gt2,
        mask_tbr_gt2_t20=mask_tbr_gt2_t20,
        mask_tbr_gt2_t40=mask_tbr_gt2_t40,
        mask_tbr_gt2_t60=mask_tbr_gt2_t60,
        sul_t20=sul_t20, sul_t40=sul_t40, sul_t60=sul_t60,
        tbr_t20=tbr_t20, tbr_t40=tbr_t40, tbr_t60=tbr_t60,
        report=report,
        brain_mask=brain_mask,
    )

    # Save T1 underlay if available
    if t1_resampled is not None:
        from output import save_nifti
        save_nifti(t1_resampled, affine,
                   os.path.join(args.output, "map_t1.nii.gz"),
                   dtype=np.float32)
        print("  map_t1.nii.gz        T1 underlay (resampled to PET space)")

        # Also save original T1 at native resolution for viewer
        save_nifti(t1_volume, t1_affine,
                   os.path.join(args.output, "t1_orig.nii.gz"),
                   dtype=np.float32)
        print("  t1_orig.nii.gz       T1 original resolution (for viewer)")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


# ===========================================================================
# DYNAMIC PIPELINE
# ===========================================================================

def run_dynamic(args):
    print("=" * 60)
    print("FET-PET/MRI Tyrosine Kinetics Analyzer  [DYNAMIC]")
    print("=" * 60)

    # ---- Read static reference tags if provided (fallback for incomplete 4D tags) ----
    static_ref_meta = None
    if args.static_ref:
        print(f"\nReading static-ref patient tags: {args.static_ref}")
        static_ref_meta = _read_dicom_metadata(args.static_ref)
        print(f"  weight={static_ref_meta.get('patient_weight_kg')} kg, "
              f"height={static_ref_meta.get('patient_height_cm')} cm, "
              f"dose={static_ref_meta.get('injected_dose_bq'):.2e} Bq")

    # ---- Step 1: Read 3 dynamic DICOM series ----
    dyn_folders = {"dyn1": args.dyn1, "dyn2": args.dyn2, "dyn3": args.dyn3}
    dyn_4d_volumes = []
    dyn_affines = []
    meta = None
    affine = None

    for label, folder in dyn_folders.items():
        print(f"\nReading {label}: {folder}")
        t0 = time.time()

        vol_4d, aff, m = dynamic_dicom_to_4d(folder)
        print(f"  dcm2niix: shape={vol_4d.shape}, {time.time()-t0:.1f}s")
        print(f"  Orientation: {nib_aff2axcodes(aff)}")
        print(f"  N frames: {vol_4d.shape[3] if vol_4d.ndim == 4 else 1}")

        # Fill in missing tags from static reference (dynamic 4D series often
        # omit patient size/weight)
        if static_ref_meta is not None:
            has_issues = (
                m.get("patient_weight_kg", 0) <= 0
                or m.get("patient_height_cm", 0) <= 0
                or m.get("injected_dose_bq", 0) <= 0
            )
            if has_issues:
                print(f"  Missing tags in {label}, patching from --static-ref...")
                _fill_meta_from_static(m, static_ref_meta)

        if meta is None:
            meta = m
            affine = aff

        # Convert Bq/ml -> SUL
        t0 = time.time()
        sul_4d = convert_4d_bqml_to_sul(vol_4d, m)
        dyn_4d_volumes.append(sul_4d)
        dyn_affines.append(aff)

        nz = sul_4d[sul_4d > 0]
        print(f"  SUL: [{sul_4d.min():.2f} .. {sul_4d.max():.2f}], "
              f"mean(nz)={nz.mean():.3f}, P95={np.percentile(nz, 95):.2f}")

    # ---- Concatenate 3 series into one 4D volume ----
    print("\nConcatenating 3 series into single 4D volume...")
    sul_4d_full = np.concatenate(dyn_4d_volumes, axis=3)
    n_total_frames = sul_4d_full.shape[3]
    print(f"  Combined shape: {sul_4d_full.shape} ({n_total_frames} frames)")

    # Build time schedule
    time_points_sec = np.array(build_dynamic_time_schedule())
    time_points_min = time_points_sec / 60.0
    print(f"  Time range: {time_points_sec[0]:.1f}s — {time_points_sec[-1]:.1f}s "
          f"({time_points_min[-1]:.1f} min)")

    # Trim first N frames if requested
    if args.no_frame > 0:
        sul_4d_full, time_points_sec = trim_frames(
            sul_4d_full, time_points_sec.tolist(), args.no_frame
        )
        time_points_sec = np.array(time_points_sec)
        time_points_min = time_points_sec / 60.0
        n_total_frames = sul_4d_full.shape[3]
        print(f"  After trimming: {n_total_frames} frames, "
              f"time range: {time_points_sec[0]:.1f}s — {time_points_sec[-1]:.1f}s")

    # ---- Read T1 if provided ----
    t1_volume = None
    t1_affine = None
    if args.t1:
        print(f"\nReading T1: {args.t1}")
        t0 = time.time()
        t1_raw, t1_affine, _ = dicom_to_nifti(args.t1)
        t1_volume = t1_raw
        print(f"  dcm2niix: shape={t1_volume.shape}, {time.time()-t0:.1f}s")
        print(f"  Orientation: {nib_aff2axcodes(t1_affine)}")

    # ---- Step 2: Preprocessing (4D) ----
    print("\n" + "-" * 60)
    print("Preprocessing 4D (skull-strip + smoothing)")
    print("-" * 60)

    sul_4d_proc, brain_mask, t1_resampled = preprocess_4d(
        sul_4d_full,
        affine=affine,
        t1_volume=t1_volume,
        t1_affine=t1_affine,
        apply_skull_strip=not args.no_skull_strip,
        apply_smoothing=not args.no_smoothing,
        smooth_sigma=args.smooth_sigma,
    )

    print(f"\n  After preprocessing:")
    nz = sul_4d_proc[sul_4d_proc > 0]
    if len(nz) > 0:
        print(f"    SUL: [{sul_4d_proc.min():.2f} .. {sul_4d_proc.max():.2f}], "
              f"mean(nz)={nz.mean():.3f}, P95={np.percentile(nz, 95):.2f}")

    # ---- Step 3: SULmean(t) + TBR(t) ----
    print("\n" + "-" * 60)
    print("Computing SULmean(t) and TBR(t)")
    print("-" * 60)

    tbr_4d, sul_means = compute_tbr_4d(sul_4d_proc, brain_mask, args.trim_percent)

    for t in range(n_total_frames):
        tbr_frame = tbr_4d[:, :, :, t]
        n_tbr = int(np.sum(tbr_frame > args.tbr_threshold))
        print(f"  Frame {t:2d}  t={time_points_sec[t]:7.1f}s  "
              f"SULmean={sul_means[t]:.4f}  "
              f"TBR: [{tbr_frame.min():.2f} .. {tbr_frame.max():.2f}]  "
              f"voxels TBR>{args.tbr_threshold}: {n_tbr}")

    # ---- Step 4: Slope + classification ----
    print("\n" + "-" * 60)
    print("Slope map and curve classification")
    print("-" * 60)

    slope_map = compute_slope_map_dynamic(tbr_4d, time_points_min)
    tbrmax = compute_tbrmax_dynamic(tbr_4d)

    # For dynamic: time_span is total duration in minutes
    time_span_min = time_points_min[-1] - time_points_min[0]
    slope_thr = args.tbr_delta_threshold / time_span_min

    classes = classify_curves_dynamic(slope_map, args.tbr_delta_threshold, time_span_min)

    # Build significant mask (matching static logic: max SUL, any TBR > threshold)
    sul_max = np.max(sul_4d_proc, axis=3)
    tbr_any_gt = np.any(tbr_4d > args.tbr_threshold, axis=3)
    sig_mask = (sul_max > args.sul_threshold) & tbr_any_gt

    # Apply brain mask
    sig_mask = sig_mask & brain_mask

    mask_clusters = np.zeros_like(classes, dtype=np.int8)
    mask_clusters[sig_mask] = classes[sig_mask]

    # Filter small clusters
    if args.min_cluster_size > 0:
        mask_clusters_raw = mask_clusters.copy()
        mask_clusters = filter_small_clusters(mask_clusters, args.min_cluster_size)
        n_removed = int(np.sum(mask_clusters_raw != 0)) - int(np.sum(mask_clusters != 0))
        if n_removed > 0:
            print(f"  Cluster filter (min {args.min_cluster_size} voxels): removed {n_removed} voxels")

    # Slope map: full, no masking
    slope_map_display = slope_map

    n_rising  = int(np.sum(mask_clusters == 1))
    n_falling = int(np.sum(mask_clusters == 2))
    n_plateau = int(np.sum(mask_clusters == 3))
    n_sig     = int(np.sum(sig_mask))

    print(f"\nSlope threshold: {slope_thr:.6f} TBR/min "
          f"(delta={args.tbr_delta_threshold} over {time_span_min:.1f} min)")
    print(f"Significant voxels: {n_sig}")
    print(f"  Rising:  {n_rising}")
    print(f"  Falling: {n_falling}")
    print(f"  Plateau: {n_plateau}")

    report = {
        "mode": "dynamic",
        "parameters": {
            "tbr_delta_threshold": args.tbr_delta_threshold,
            "time_span_min": round(time_span_min, 2),
            "time_span_sec": round(time_points_sec[-1] - time_points_sec[0], 1),
            "trim_percent": args.trim_percent,
            "min_cluster_size": args.min_cluster_size,
            "skull_strip": not args.no_skull_strip,
            "t1_used": args.t1 is not None,
            "smoothing": not args.no_smoothing,
            "smooth_sigma": args.smooth_sigma,
            "no_frame": args.no_frame,
            "sul_threshold": args.sul_threshold,
            "tbr_threshold": args.tbr_threshold,
            "patient_weight_kg": meta["patient_weight_kg"],
            "patient_height_cm": meta["patient_height_cm"],
            "patient_sex": meta["patient_sex"],
            "injected_dose_bq": meta["injected_dose_bq"],
        },
        "results": {
            "slope_threshold_tbr_per_min": round(slope_thr, 6),
            "n_significant_voxels": n_sig,
            "n_rising": n_rising,
            "n_falling": n_falling,
            "n_plateau": n_plateau,
            "n_sulmax_gt2": int(np.sum(sul_max > args.sul_threshold)),
            "n_tbr_gt2": int(np.sum(tbr_any_gt)),
        },
    }
    for cid, cname in [(1, "rising"), (2, "falling"), (3, "plateau")]:
        vox = mask_clusters == cid
        if np.any(vox):
            report["results"][f"mean_slope_{cname}"] = round(float(np.mean(slope_map[vox])), 6)

    # ---- Step 5: Save ----
    print("\n" + "-" * 60)
    print("Saving outputs")
    print("-" * 60)

    save_dynamic_outputs(
        output_dir=args.output,
        affine=affine,
        mask_clusters=mask_clusters,
        map_slope=slope_map_display,
        map_tbrmax=tbrmax,
        report=report,
        brain_mask=brain_mask,
        sul_means=sul_means,
        time_points_min=time_points_min,
        tbr_4d=tbr_4d,
    )

    # Save T1 underlay if available
    if t1_resampled is not None:
        from output import save_nifti
        save_nifti(t1_resampled, affine,
                   os.path.join(args.output, "map_t1.nii.gz"),
                   dtype=np.float32)
        print("  map_t1.nii.gz        T1 underlay (resampled to PET space)")

        # Also save original T1 at native resolution for viewer
        save_nifti(t1_volume, t1_affine,
                   os.path.join(args.output, "t1_orig.nii.gz"),
                   dtype=np.float32)
        print("  t1_orig.nii.gz       T1 original resolution (for viewer)")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    args = parse_args()

    if args.dynamic:
        run_dynamic(args)
    else:
        run_static(args)


if __name__ == "__main__":
    main()
