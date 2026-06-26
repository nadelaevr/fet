"""
Read dynamic (4D) DICOM series via dcm2niix, return 4D volume + per-volume SUL conversion.

A dynamic series contains multiple time frames (volumes) in one DICOM folder.
dcm2niix handles this natively and outputs a 4D NIfTI file.
"""

import os
import glob
import json
import tempfile
import subprocess
import numpy as np
import nibabel as nib
import pydicom

from dicom_reader import _get_dcm2niix_bin, _read_dicom_metadata, compute_lbm_james


def dynamic_dicom_to_4d(dicom_folder: str, output_dir: str = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Convert a dynamic DICOM series to 4D NIfTI using dcm2niix.

    dcm2niix automatically detects 4D series and outputs a single .nii.gz
    with shape (X, Y, Z, T) where T = number of time frames.

    Returns:
        volume_4d: 4D numpy array, shape (X, Y, Z, T) in Bq/ml
        affine: 4x4 affine matrix (spatial, from NIfTI)
        meta: dict with patient/dose info + dcm2niix sidecar
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="dcm2niix_dyn_")

    cmd = [
        _get_dcm2niix_bin(),
        "-z", "y",          # gzip
        "-f", "%s_%p",      # filename = series_protocol
        "-o", output_dir,
        dicom_folder,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"dcm2niix failed: {result.stderr}")

    # Find 4D NIfTI (may also produce 3D if dcm2niix splits — take the largest)
    nii_files = glob.glob(os.path.join(output_dir, "*.nii.gz"))
    if not nii_files:
        raise FileNotFoundError(f"No NIfTI output from dcm2niix in {output_dir}")

    # Pick the 4D file (ndim == 4), or the largest file if none is 4D
    best = None
    best_ndim = 0
    for f in nii_files:
        try:
            img = nib.load(f)
            if img.ndim > best_ndim:
                best_ndim = img.ndim
                best = f
        except Exception:
            continue

    if best is None:
        raise FileNotFoundError(f"Cannot load any NIfTI from {output_dir}")

    nii_path = best
    json_path = nii_path.replace(".nii.gz", ".json")

    img = nib.load(nii_path)
    volume_4d = img.get_fdata().astype(np.float64)
    affine = img.affine.copy()

    meta = {}
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            meta["sidecar"] = json.load(f)

    dcm_meta = _read_dicom_metadata(dicom_folder)
    meta.update(dcm_meta)

    meta["_nii_path"] = nii_path
    meta["_json_path"] = json_path if os.path.exists(json_path) else None
    meta["_output_dir"] = output_dir
    meta["n_frames"] = volume_4d.shape[3] if volume_4d.ndim == 4 else 1

    return volume_4d, affine, meta


def convert_4d_bqml_to_sul(volume_4d_bqml: np.ndarray, meta: dict) -> np.ndarray:
    """
    Convert 4D Bq/ml volume to SUL (SUVlbm) — same James formula per voxel,
    applied to every time frame.

    SUL(t) = Bq/ml(t) * LBM[g] / InjectedDose[Bq]

    Args:
        volume_4d_bqml: 4D array (X, Y, Z, T) in Bq/ml
        meta: metadata dict with patient_weight_kg, patient_height_cm, patient_sex, injected_dose_bq

    Returns:
        sul_4d: 4D array (X, Y, Z, T) in SUL
    """
    weight = meta["patient_weight_kg"]
    height_cm = meta["patient_height_cm"]
    sex = meta["patient_sex"]
    dose = meta["injected_dose_bq"]

    if dose <= 0:
        raise ValueError("Injected dose is zero or missing")
    if weight <= 0:
        raise ValueError("Patient weight is zero or missing")

    lbm_kg = compute_lbm_james(weight, height_cm, sex)
    lbm_g = lbm_kg * 1000.0

    sul_4d = volume_4d_bqml * lbm_g / dose
    return sul_4d


def build_dynamic_time_schedule() -> list[float]:
    """
    Build the time schedule (in seconds) for all 38 dynamic frames
    across 3 series, measured from the start of series 1.

    Series 1 (30 volumes):
      12 volumes x 5s  -> centers at 2.5, 7.5, ..., 57.5  (0-60s)
       6 volumes x 10s -> centers at 65, 75, ..., 115      (60-120s)
       3 volumes x 20s -> centers at 130, 150, 170         (120-180s)
       5 volumes x 60s -> centers at 210, 270, 330, 390, 450 (180-480s)
       4 volumes x 180s-> centers at 570, 750, 930, 1110   (480-1200s)

    Series 2 (4 volumes):
       4 volumes x 300s-> centers at 1350, 1650, 1950, 2250 (1200-2400s)

    Series 3 (4 volumes):
       4 volumes x 300s-> centers at 2550, 2850, 3150, 3450 (2400-3600s)

    Returns:
        times_sec: list of 38 floats, time in seconds from start of series 1
    """
    times = []

    # Series 1
    t = 0.0
    # 12 x 5s
    for _ in range(12):
        times.append(t + 2.5)
        t += 5.0
    # 6 x 10s
    for _ in range(6):
        times.append(t + 5.0)
        t += 10.0
    # 3 x 20s
    for _ in range(3):
        times.append(t + 10.0)
        t += 20.0
    # 5 x 60s
    for _ in range(5):
        times.append(t + 30.0)
        t += 60.0
    # 4 x 180s
    for _ in range(4):
        times.append(t + 90.0)
        t += 180.0

    # Series 2 starts at 1200s
    t = 1200.0
    for _ in range(4):
        times.append(t + 150.0)
        t += 300.0

    # Series 3 starts at 2400s
    t = 2400.0
    for _ in range(4):
        times.append(t + 150.0)
        t += 300.0

    assert len(times) == 38, f"Expected 38 time points, got {len(times)}"
    return times


def trim_frames(sul_4d: np.ndarray, time_points_sec: list[float],
                no_frame: int) -> tuple[np.ndarray, list[float]]:
    """
    Trim first N frames from 4D volume and time schedule.

    Args:
        sul_4d: 4D SUL array (X, Y, Z, T)
        time_points_sec: time points in seconds
        no_frame: number of frames to skip from the beginning

    Returns:
        sul_4d_trimmed: cropped 4D array (X, Y, Z, T-N)
        time_points_sec_trimmed: cropped time points
    """
    if no_frame <= 0:
        return sul_4d, time_points_sec
    if no_frame >= sul_4d.shape[3]:
        raise ValueError(
            f"--no_frame ({no_frame}) >= number of frames ({sul_4d.shape[3]})"
        )
    print(f"  Trimming first {no_frame} frames "
          f"(keeping frames {no_frame}..{sul_4d.shape[3]-1})")
    return sul_4d[:, :, :, no_frame:], time_points_sec[no_frame:]
