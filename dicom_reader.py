"""
Read DICOM series via dcm2niix for correct geometry/orientation,
then apply SUL (SUVlbm) conversion.
"""

import os
import glob
import json
import tempfile
import shutil
import numpy as np
import nibabel as nib
import pydicom


def dicom_to_nifti(dicom_folder: str, output_dir: str = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Convert a DICOM series to NIfTI using dcm2niix.

    Returns:
        volume: 3D numpy array (as stored by dcm2niix, in Bq/ml or rescaled units)
        affine: 4x4 affine matrix (NIfTI convention)
        meta: dict with patient/dose info extracted from DICOM sidecar JSON
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="dcm2niix_")

    # Run dcm2niix
    import subprocess
    cmd = [
        _get_dcm2niix_bin(),
        "-z", "y",       # gzip
        "-f", "%s",      # filename format
        "-o", output_dir,
        dicom_folder,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"dcm2niix failed: {result.stderr}")

    # Find output NIfTI
    nii_files = glob.glob(os.path.join(output_dir, "*.nii.gz"))
    if not nii_files:
        raise FileNotFoundError(f"No NIfTI output from dcm2niix in {output_dir}")

    nii_path = nii_files[0]
    json_path = nii_path.replace(".nii.gz", ".json")

    # Load NIfTI
    img = nib.load(nii_path)
    volume = img.get_fdata().astype(np.float64)
    affine = img.affine.copy()

    # Load sidecar JSON for metadata
    meta = {}
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            sidecar = json.load(f)
        meta["sidecar"] = sidecar

    # Also read DICOM directly for dose/weight/height (sidecar may miss some)
    dcm_meta = _read_dicom_metadata(dicom_folder)
    meta.update(dcm_meta)

    # Store paths for cleanup
    meta["_nii_path"] = nii_path
    meta["_json_path"] = json_path if os.path.exists(json_path) else None
    meta["_output_dir"] = output_dir

    return volume, affine, meta


def _get_dcm2niix_bin() -> str:
    """Find dcm2niix binary."""
    # Try Python module first
    try:
        import dcm2niix
        return dcm2niix.bin
    except (ImportError, AttributeError):
        pass

    # Try system PATH
    import shutil
    path = shutil.which("dcm2niix")
    if path:
        return path

    raise RuntimeError("dcm2niix not found. Install: uv pip install dcm2niix")


def _read_dicom_metadata(dicom_folder: str) -> dict:
    """Read key metadata from first DICOM file in folder."""
    files = glob.glob(os.path.join(dicom_folder, "*.dcm"))
    if not files:
        files = [
            os.path.join(dicom_folder, f)
            for f in os.listdir(dicom_folder)
            if not f.startswith(".") and os.path.isfile(os.path.join(dicom_folder, f))
        ]
    if not files:
        return {}

    ds = pydicom.dcmread(files[0], force=True)
    meta = {}

    # Patient
    raw_weight = float(getattr(ds, "PatientWeight", 0.0))
    try:
        raw_height_m = float(ds.PatientSize)
    except (AttributeError, ValueError, TypeError):
        raw_height_m = 0.0

    meta["patient_weight_kg"] = raw_weight
    meta["patient_height_m"] = raw_height_m
    meta["patient_height_cm"] = raw_height_m * 100.0

    # Auto-detect swapped weight/height (common scanner bug: height in
    # PatientWeight and weight in PatientSize). Heuristic:
    #   weight > 150 kg AND height < 100 cm → swap.
    if raw_weight > 150.0 and meta["patient_height_cm"] < 100.0:
        meta["patient_weight_kg"] = meta["patient_height_cm"]   # was height
        meta["patient_height_m"] = raw_weight / 100.0            # was weight
        meta["patient_height_cm"] = raw_weight
        print(f"  ⚠ Swapped weight/height: PatientWeight={raw_weight} → height, "
              f"PatientSize={raw_height_m} → weight "
              f"(now: {meta['patient_weight_kg']:.0f} kg, {meta['patient_height_cm']:.0f} cm)")
    try:
        meta["patient_sex"] = str(ds.PatientSex).upper()
    except AttributeError:
        meta["patient_sex"] = "M"

    # Dose from RPI sequence (0054,0016)
    injected_dose = 0.0
    half_life = 0.0
    positron_fraction = 1.0
    rph_start_time = ""

    try:
        rpi_seq = ds[0x0054, 0x0016].value
        for item in rpi_seq:
            try:
                injected_dose = float(item[0x0018, 0x1074].value)
            except (KeyError, AttributeError, ValueError, TypeError):
                pass
            try:
                rph_start_time = str(item[0x0018, 0x1072].value)
            except (KeyError, AttributeError, ValueError, TypeError):
                pass
            try:
                half_life = float(item[0x0018, 0x1075].value)
            except (KeyError, AttributeError, ValueError, TypeError):
                pass
            try:
                positron_fraction = float(item[0x0018, 0x1076].value)
            except (KeyError, AttributeError, ValueError, TypeError):
                pass
    except (KeyError, AttributeError):
        pass

    if injected_dose <= 0:
        try:
            injected_dose = float(ds.RadionuclideTotalDose)
        except (AttributeError, ValueError, TypeError):
            pass

    meta["injected_dose_bq"] = injected_dose
    meta["rph_start_time"] = rph_start_time
    meta["radionuclide_half_life"] = half_life
    meta["radionuclide_positron_fraction"] = positron_fraction

    # Series info
    try:
        meta["series_description"] = str(ds.SeriesDescription)
    except AttributeError:
        meta["series_description"] = ""
    try:
        meta["series_time"] = str(ds.SeriesTime)
    except AttributeError:
        meta["series_time"] = ""

    return meta


def compute_lbm_james(weight_kg: float, height_cm: float, sex: str) -> float:
    """
    Lean Body Mass via James (1988), H in cm.

    Male:   LBM = 1.10 * W - 128 * (W/H)²
    Female: LBM = 1.07 * W - 148 * (W/H)²
    """
    if weight_kg <= 0 or height_cm <= 0:
        raise ValueError(f"Invalid weight/height: {weight_kg} kg, {height_cm} cm")

    wh = weight_kg / height_cm
    if sex.startswith("F"):
        lbm = 1.07 * weight_kg - 148.0 * wh ** 2
    else:
        lbm = 1.10 * weight_kg - 128.0 * wh ** 2

    if lbm <= 0:
        lbm = 0.80 * weight_kg
        print(f"  WARNING: James LBM <= 0, fallback 0.80*W = {lbm:.1f} kg")

    return lbm


def convert_bqml_to_sul(volume_bqml: np.ndarray, meta: dict) -> np.ndarray:
    """
    Convert Bq/ml volume to SUL (SUVlbm).

    SUL = Bq/ml * LBM[g] / InjectedDose[Bq]

    Note: dcm2niix already applies RescaleSlope/Intercept,
    so the volume is in Bq/ml (or the scanner's calibrated units).
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

    sul = volume_bqml * lbm_g / dose
    return sul
