"""
Output: save all masks, maps, and report as NIfTI + JSON.
"""

import os
import json
import numpy as np
import nibabel as nib


def save_nifti(
    data: np.ndarray,
    affine: np.ndarray,
    filepath: str,
    dtype=None,
):
    """Save a 3D numpy array as NIfTI. Orientation is defined by affine."""
    if dtype is not None:
        data = data.astype(dtype)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, filepath)


def save_all_outputs(
    output_dir: str,
    affine: np.ndarray,
    # Cluster mask
    mask_clusters: np.ndarray,
    # Maps
    map_slope: np.ndarray,
    map_sulmax: np.ndarray,
    map_tbrmax: np.ndarray,
    # Binary masks
    mask_sulmax_gt2: np.ndarray,
    mask_tbr_gt2: np.ndarray,
    mask_tbr_gt2_t20: np.ndarray,
    mask_tbr_gt2_t40: np.ndarray,
    mask_tbr_gt2_t60: np.ndarray,
    # Per-timepoint maps
    sul_t20: np.ndarray,
    sul_t40: np.ndarray,
    sul_t60: np.ndarray,
    tbr_t20: np.ndarray,
    tbr_t40: np.ndarray,
    tbr_t60: np.ndarray,
    # Report data
    report: dict,
    # Optional
    brain_mask: np.ndarray = None,
):
    """Save all outputs to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    # Cluster mask
    save_nifti(mask_clusters, affine, os.path.join(output_dir, "mask_clusters.nii.gz"), dtype=np.int8)

    # Parametric maps
    save_nifti(map_slope, affine, os.path.join(output_dir, "map_slope.nii.gz"), dtype=np.float32)
    save_nifti(map_sulmax, affine, os.path.join(output_dir, "map_sulmax.nii.gz"), dtype=np.float32)
    save_nifti(map_tbrmax, affine, os.path.join(output_dir, "map_tbrmax.nii.gz"), dtype=np.float32)

    # Binary masks
    for name, m in [
        ("mask_sulmax_gt2", mask_sulmax_gt2),
        ("mask_tbr_gt2", mask_tbr_gt2),
        ("mask_tbr_gt2_t20", mask_tbr_gt2_t20),
        ("mask_tbr_gt2_t40", mask_tbr_gt2_t40),
        ("mask_tbr_gt2_t60", mask_tbr_gt2_t60),
    ]:
        save_nifti(m.astype(np.uint8), affine, os.path.join(output_dir, f"{name}.nii.gz"), dtype=np.uint8)

    # Per-timepoint SUL maps
    for name, m in [
        ("map_sul_t20", sul_t20),
        ("map_sul_t40", sul_t40),
        ("map_sul_t60", sul_t60),
    ]:
        save_nifti(m, affine, os.path.join(output_dir, f"{name}.nii.gz"), dtype=np.float32)

    # Per-timepoint TBR maps
    for name, m in [
        ("map_tbr_t20", tbr_t20),
        ("map_tbr_t40", tbr_t40),
        ("map_tbr_t60", tbr_t60),
    ]:
        save_nifti(m, affine, os.path.join(output_dir, f"{name}.nii.gz"), dtype=np.float32)

    # Brain mask
    if brain_mask is not None:
        save_nifti(
            brain_mask.astype(np.uint8), affine,
            os.path.join(output_dir, "mask_brain.nii.gz"), dtype=np.uint8,
        )

    # Report
    with open(os.path.join(output_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"All outputs saved to: {output_dir}")


def save_dynamic_outputs(
    output_dir: str,
    affine: np.ndarray,
    mask_clusters: np.ndarray,
    map_slope: np.ndarray,
    map_tbrmax: np.ndarray,
    report: dict,
    brain_mask: np.ndarray = None,
    sul_means: np.ndarray = None,
    time_points_min: np.ndarray = None,
):
    """
    Save outputs for dynamic (4D) pipeline.

    Saves cluster mask, slope map (full, no masking), TBRmax map, and report.

    Args:
        output_dir: output folder
        affine: 4x4 spatial affine
        mask_clusters: 3D int8 array (1=rising, 2=falling, 3=plateau)
        map_slope: 3D float32 slope map (TBR/min) — full, not masked
        map_tbrmax: 3D float32 max TBR map
        report: dict with parameters and results
        brain_mask: optional 3D boolean brain mask
        sul_means: optional 1D array of SULmean per frame
        time_points_min: optional 1D array of time points in minutes
    """
    os.makedirs(output_dir, exist_ok=True)

    # Cluster mask
    save_nifti(mask_clusters, affine, os.path.join(output_dir, "mask_clusters.nii.gz"), dtype=np.int8)

    # Slope map (full, no masking)
    save_nifti(map_slope, affine, os.path.join(output_dir, "map_slope.nii.gz"), dtype=np.float32)

    # TBRmax map
    save_nifti(map_tbrmax, affine, os.path.join(output_dir, "map_tbrmax.nii.gz"), dtype=np.float32)

    # Brain mask
    if brain_mask is not None:
        save_nifti(
            brain_mask.astype(np.uint8), affine,
            os.path.join(output_dir, "mask_brain.nii.gz"), dtype=np.uint8,
        )

    # Enrich report with dynamic info
    if sul_means is not None:
        report["sul_means_per_frame"] = [round(float(s), 4) for s in sul_means]
    if time_points_min is not None:
        report["time_points_min"] = [round(float(t), 2) for t in time_points_min]
        report["n_frames"] = len(time_points_min)

    with open(os.path.join(output_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"All outputs saved to: {output_dir}")
