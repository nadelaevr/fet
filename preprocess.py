"""
Preprocessing: antspynet skull-stripping, T1-to-PET resampling, Gaussian smoothing.
"""

import os
import subprocess
import sys
import tempfile
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter, binary_dilation, binary_closing, generate_binary_structure, label


def resample_to_pet(
    t1_volume: np.ndarray,
    t1_affine: np.ndarray,
    pet_shape: tuple,
    pet_affine: np.ndarray,
) -> np.ndarray:
    """
    Resample T1 onto the PET voxel grid.

    T1 and PET from the same session already share scanner space, so this is
    interpolation onto the PET grid, not a registration. Done with nibabel:
    ants.image_read() access-violates on Windows while reading a NIfTI header
    (antsImageHeaderInfo), and that native crash bypasses try/except.
    """
    from nibabel.processing import resample_from_to

    pet_shape = tuple(int(s) for s in pet_shape)
    print(f"  Resampling T1 {t1_volume.shape} -> PET {pet_shape}...", flush=True)

    t1_img = nib.Nifti1Image(
        np.ascontiguousarray(t1_volume, dtype=np.float32),
        np.asarray(t1_affine, dtype=np.float64),
    )
    resampled = resample_from_to(
        t1_img,
        (pet_shape, np.asarray(pet_affine, dtype=np.float64)),
        order=1,
        cval=0.0,
    )
    result = np.asanyarray(resampled.dataobj, dtype=np.float64)
    if result.shape != pet_shape:
        raise RuntimeError(
            f"resampled T1 shape {result.shape} != PET shape {pet_shape}"
        )
    return result


# antspynet.brain_extraction() calls ants.image_read() on its template.
# On Windows that native call access-violates and kills the interpreter,
# so the extraction runs in a child process. A crash becomes a normal
# exception and the caller falls back to the threshold mask.
_BRAIN_WORKER = """
import sys
import numpy as np
import nibabel as nib
import ants
from antspynet.utilities import brain_extraction

src, dst, modality = sys.argv[1], sys.argv[2], sys.argv[3]
ants_img = ants.image_read(src)
print("ANTs image: %s spacing=%s" % (ants_img.shape, ants_img.spacing), flush=True)
prob = brain_extraction(ants_img, modality=modality)
mask = np.asarray(prob.numpy()) > 0.5
ref_shape = nib.load(src).shape
if tuple(mask.shape) != tuple(ref_shape):
    raise SystemExit("mask shape %s != volume %s" % (mask.shape, ref_shape))
np.save(dst, mask)
"""


def _brain_mask_antspynet_subprocess(
    t1_volume: np.ndarray,
    affine: np.ndarray,
    modality: str,
) -> np.ndarray:
    tmp_dir = tempfile.mkdtemp(prefix="antsbrain_")
    src = os.path.join(tmp_dir, "t1.nii.gz")
    dst = os.path.join(tmp_dir, "mask.npy")
    nib.save(
        nib.Nifti1Image(np.ascontiguousarray(t1_volume, dtype=np.float32), affine),
        src,
    )
    print("  Skull-stripping: antspynet (separate process)...", flush=True)
    proc = subprocess.run(
        [sys.executable, "-c", _BRAIN_WORKER, src, dst, modality],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not os.path.exists(dst):
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        last = detail[-1] if detail else ""
        raise RuntimeError(
            f"antspynet process exit {proc.returncode}"
            + (f": {last}" if last else "")
        )
    if proc.stdout:
        for line in proc.stdout.splitlines():
            print(f"  {line}", flush=True)
    return np.load(dst).astype(bool)


def brain_mask_antspynet(
    t1_volume: np.ndarray,
    affine: np.ndarray,
    modality: str = "t1",
) -> np.ndarray:
    """
    Skull-stripping via antspynet deep learning brain extraction.
    """
    if sys.platform == "win32":
        brain_mask = _brain_mask_antspynet_subprocess(t1_volume, affine, modality)
    else:
        import ants
        from antspynet.utilities import brain_extraction

        tmp_dir = tempfile.mkdtemp(prefix="antsbrain_")
        tmp_nii = os.path.join(tmp_dir, "t1_tmp.nii.gz")

        img = nib.Nifti1Image(t1_volume.astype(np.float32), affine)
        nib.save(img, tmp_nii)

        ants_img = ants.image_read(tmp_nii)
        print(f"  ANTs image: {ants_img.shape}, spacing={ants_img.spacing}")

        print(f"  Running antspynet brain_extraction (modality={modality})...")
        brain_prob = brain_extraction(ants_img, modality=modality)

        prob_arr = brain_prob.numpy()
        brain_mask = prob_arr > 0.5

        try:
            os.remove(tmp_nii)
        except OSError:
            pass

    n_brain = int(np.sum(brain_mask))
    n_total = brain_mask.size
    print(f"  Brain mask: {n_brain} / {n_total} voxels ({100*n_brain/n_total:.1f}%)")

    return brain_mask


def brain_mask_from_t1_threshold(
    t1_volume: np.ndarray,
    threshold_fraction: float = 0.10,
    closing_radius: int = 3,
    dilation_radius: int = 2,
    smooth_sigma: float = 3.0,
) -> np.ndarray:
    """Fallback: threshold-based brain mask from T1."""
    smoothed = gaussian_filter(t1_volume.astype(np.float64), sigma=smooth_sigma)
    nonzero = smoothed[smoothed > 0]
    if len(nonzero) == 0:
        return np.ones_like(t1_volume, dtype=bool)

    p95 = np.percentile(nonzero, 95)
    threshold = p95 * threshold_fraction
    binary_mask = smoothed > threshold

    brain_mask = _largest_component(binary_mask, min_frac=0.005)
    struct = generate_binary_structure(3, 2)
    brain_mask = binary_closing(brain_mask, structure=struct, iterations=closing_radius)
    brain_mask = binary_dilation(brain_mask, structure=struct, iterations=dilation_radius)

    return brain_mask.astype(bool)


def brain_mask_from_pet_fallback(
    pet_volume: np.ndarray,
    threshold_fraction: float = 0.15,
    closing_radius: int = 2,
    dilation_radius: int = 1,
    smooth_sigma: float = 4.0,
) -> np.ndarray:
    """Fallback: brain mask from PET when no T1 available."""
    smoothed = gaussian_filter(pet_volume.astype(np.float64), sigma=smooth_sigma)
    nonzero = smoothed[smoothed > 0]
    if len(nonzero) == 0:
        return np.ones_like(pet_volume, dtype=bool)

    p95 = np.percentile(nonzero, 95)
    threshold = p95 * threshold_fraction
    binary_mask = smoothed > threshold

    brain_mask = _largest_component(binary_mask, min_frac=0.001)
    struct = generate_binary_structure(3, 2)
    brain_mask = binary_closing(brain_mask, structure=struct, iterations=closing_radius)
    brain_mask = binary_dilation(brain_mask, structure=struct, iterations=dilation_radius)

    return brain_mask.astype(bool)


def _largest_component(binary_mask: np.ndarray, min_frac: float) -> np.ndarray:
    """Find the largest connected component."""
    struct = generate_binary_structure(3, 2)
    labeled_arr, num_features = label(binary_mask, structure=struct)

    if num_features == 0:
        return binary_mask

    component_sizes = np.bincount(labeled_arr.ravel())
    component_sizes[0] = 0
    min_size = min_frac * np.sum(binary_mask)
    component_sizes[component_sizes < min_size] = 0

    if np.max(component_sizes) == 0:
        return binary_mask

    largest = np.argmax(component_sizes)
    return labeled_arr == largest


def _voxel_spacing(affine: np.ndarray, axis: int) -> float:
    return float(np.linalg.norm(np.asarray(affine, dtype=np.float64)[:3, axis]))


def _mask_orientation(affine: np.ndarray):
    """Superior-inferior, left-right and anterior-posterior axes.

    Returns index of each axis, the index step that moves inferiorly,
    and the index step that moves anteriorly. Falls back to RAS if the
    affine is too oblique for a single letter code.
    """
    affine = np.asarray(affine, dtype=np.float64)
    codes = nib.aff2axcodes(affine)
    find = {c: i for i, c in enumerate(codes)}
    if "S" in find or "I" in find:
        si = find.get("S", find.get("I"))
        inf_sign = -1 if codes[si] == "S" else 1
    else:
        si, inf_sign = 2, -1 if affine[2, 2] >= 0 else 1
    if "R" in find or "L" in find:
        lr = find.get("R", find.get("L"))
    else:
        lr = 0
    if "A" in find or "P" in find:
        ap = find.get("A", find.get("P"))
        ant_sign = 1 if codes[ap] == "A" else -1
    else:
        ap, ant_sign = 1, 1 if affine[1, 1] >= 0 else -1
    return si, lr, ap, inf_sign, ant_sign


def extend_brain_mask_caudal(
    mask: np.ndarray,
    affine: np.ndarray,
    extend_mm: float,
    midline_mm: float = 12.0,
    margin_mm: float = 6.0,
    radius_min_mm: float = 8.0,
    radius_max_mm: float = 18.0,
) -> np.ndarray:
    """Continue the brain mask down the cervical cord.

    antspynet stops at the foramen magnum, so a tumor leaving the medulla
    into the cord is erased. From the anterior midline stump (brainstem,
    not the cerebellar tonsils) a narrow tube is drawn inferiorly for
    ``extend_mm``. Scalp and neck muscle stay outside SULmean.
    """
    if extend_mm <= 0 or mask is None or not np.any(mask):
        return mask

    mask = np.asarray(mask, dtype=bool)
    si, lr, ap, inf_sign, ant_sign = _mask_orientation(affine)
    spacing = [_voxel_spacing(affine, i) for i in range(3)]
    coords = np.argwhere(mask)
    lr_mid = float(np.median(coords[:, lr]))
    lat_mm = np.abs(coords[:, lr] - lr_mid) * spacing[lr]
    mid = coords[lat_mm <= midline_mm]
    if len(mid) < 8:
        print("  Cord extension: no midline stump, mask unchanged")
        return mask

    inf_pos = mid[:, si] * inf_sign
    tip_inf = float(inf_pos.max())
    # Most inferior slice that still has a solid midline cross-section.
    si_order = np.unique(mid[:, si])
    si_order = si_order[np.argsort(-(si_order * inf_sign))]
    tip_si = None
    for sidx in si_order:
        depth_mm = (tip_inf - float(sidx) * inf_sign) * spacing[si]
        if depth_mm > 20.0:
            break
        if int(np.sum(mid[:, si] == sidx)) >= 8:
            tip_si = int(sidx)
            break
    if tip_si is None:
        print("  Cord extension: no midline stump, mask unchanged")
        return mask

    pts = mid[mid[:, si] == tip_si]
    anterior = pts[:, ap] * ant_sign
    front = pts[(float(anterior.max()) - anterior) * spacing[ap] <= 10.0]
    if len(front) < 4:
        front = pts
    center_lr = float(np.mean(front[:, lr]))
    center_ap = float(np.mean(front[:, ap]))
    d_lr = (front[:, lr] - center_lr) * spacing[lr]
    d_ap = (front[:, ap] - center_ap) * spacing[ap]
    rms = float(np.sqrt(np.mean(d_lr ** 2 + d_ap ** 2)))
    radius = float(np.clip(rms + margin_mm, radius_min_mm, radius_max_mm))

    n_steps = int(np.ceil(extend_mm / spacing[si]))
    lr_coords = np.arange(mask.shape[lr])
    ap_coords = np.arange(mask.shape[ap])
    LR, AP = np.meshgrid(lr_coords, ap_coords, indexing="ij")
    disk = (
        ((LR - center_lr) * spacing[lr]) ** 2
        + ((AP - center_ap) * spacing[ap]) ** 2
    ) <= radius ** 2
    plane_axes = [a for a in range(3) if a != si]
    disk_plane = disk if plane_axes[0] == lr else disk.T

    out = mask.copy()
    added = 0
    for step in range(0, n_steps + 1):
        sidx = int(tip_si + inf_sign * step)
        if sidx < 0 or sidx >= mask.shape[si]:
            break
        sl = [slice(None)] * 3
        sl[si] = sidx
        plane = out[tuple(sl)]
        before = int(np.count_nonzero(plane))
        plane |= disk_plane
        added += int(np.count_nonzero(plane)) - before

    print(
        f"  Cord extension: {extend_mm:.0f} mm inferior from the brainstem, "
        f"radius {radius:.1f} mm, +{added} voxels"
    )
    return out


def gaussian_smooth_volume(
    volume: np.ndarray,
    sigma: float = 1.0,
    mask: np.ndarray = None,
) -> np.ndarray:
    """Gaussian smoothing with optional mask constraint."""
    smoothed = gaussian_filter(volume.astype(np.float64), sigma=sigma)
    if mask is not None:
        smoothed[~mask] = 0.0
    return smoothed


def preprocess_volumes(
    sul_volumes: list[np.ndarray],
    affine: np.ndarray,
    t1_volume: np.ndarray = None,
    t1_affine: np.ndarray = None,
    use_antspynet: bool = True,
    apply_skull_strip: bool = True,
    apply_smoothing: bool = True,
    smooth_sigma: float = 1.0,
    mask_out_zero_voxels: bool = True,
    cord_extend_mm: float = 0.0,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray | None]:
    """
    Preprocessing: skull-stripping (antspynet or fallback) + smoothing.

    If T1 shape != PET shape, resamples T1 to PET grid first.

    Returns:
        processed: list of 3 smoothed/skull-stripped SUL volumes
        brain_mask: 3D boolean brain mask
        t1_resampled: T1 resampled to PET space (or None if no T1)
    """
    assert len(sul_volumes) == 3

    if apply_skull_strip:
        # Resample T1 to PET space if needed
        t1_resampled = t1_volume  # may be None
        if t1_volume is not None and t1_volume.shape != sul_volumes[0].shape:
            print(f"  T1 shape {t1_volume.shape} != PET shape {sul_volumes[0].shape}")
            t1_resampled = resample_to_pet(
                t1_volume, t1_affine,
                pet_shape=sul_volumes[0].shape,
                pet_affine=affine,
            )
            print(f"  Resampled T1 shape: {t1_resampled.shape}")
        elif t1_volume is not None:
            t1_resampled = t1_volume.copy()

        if t1_resampled is not None and use_antspynet:
            print("  Skull-stripping: antspynet (T1-based)...")
            try:
                brain_mask = brain_mask_antspynet(t1_resampled, affine, modality="t1")
            except Exception as e:
                print(f"  antspynet failed ({e}), falling back to threshold")
                brain_mask = brain_mask_from_t1_threshold(t1_resampled)
        elif t1_resampled is not None:
            print("  Skull-stripping: threshold (T1)...")
            brain_mask = brain_mask_from_t1_threshold(t1_resampled)
        else:
            print("  No T1 — skull-stripping from 20-min PET...")
            brain_mask = brain_mask_from_pet_fallback(sul_volumes[0])

        n_brain = int(np.sum(brain_mask))
        n_total = brain_mask.size
        print(f"  Brain mask: {n_brain} / {n_total} ({100*n_brain/n_total:.1f}%)")
    else:
        brain_mask = np.ones_like(sul_volumes[0], dtype=bool)
        t1_resampled = t1_volume.copy() if t1_volume is not None else None

    if cord_extend_mm > 0:
        if apply_skull_strip:
            brain_mask = extend_brain_mask_caudal(brain_mask, affine, cord_extend_mm)
        else:
            print("  Cord extension skipped: skull-stripping is off")

    if mask_out_zero_voxels:
        any_zero = (sul_volumes[0] == 0) | (sul_volumes[1] == 0) | (sul_volumes[2] == 0)
        brain_mask = brain_mask & ~any_zero

    processed = []
    for vol in sul_volumes:
        v = vol.copy()
        v[~brain_mask] = 0.0
        if apply_smoothing and smooth_sigma > 0:
            v = gaussian_smooth_volume(v, sigma=smooth_sigma, mask=brain_mask)
        processed.append(v)

    return processed, brain_mask, t1_resampled


# ---------------------------------------------------------------------------
# Dynamic (4D) preprocessing
# ---------------------------------------------------------------------------

def preprocess_4d(
    sul_4d: np.ndarray,
    affine: np.ndarray,
    t1_volume: np.ndarray = None,
    t1_affine: np.ndarray = None,
    apply_skull_strip: bool = True,
    apply_smoothing: bool = True,
    smooth_sigma: float = 1.0,
    cord_extend_mm: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Preprocessing for 4D (dynamic) data: skull-stripping + smoothing.

    Brain mask is computed from the temporal mean (or T1 if provided),
    then applied to all frames. Smoothing is spatial only (per-frame).

    Args:
        sul_4d: 4D SUL array (X, Y, Z, T)
        affine: 4x4 spatial affine
        t1_volume: optional 3D T1 for skull-stripping
        t1_affine: T1 affine
        apply_skull_strip: whether to skull-strip
        apply_smoothing: whether to apply Gaussian smoothing
        smooth_sigma: Gaussian sigma in voxels

    Returns:
        processed_4d: 4D SUL array after preprocessing
        brain_mask: 3D boolean brain mask
        t1_resampled: T1 resampled to PET space (or None if no T1)
    """
    spatial_shape = sul_4d.shape[:3]

    if apply_skull_strip:
        if t1_volume is not None:
            # Resample T1 to PET space if needed
            if t1_volume.shape != spatial_shape:
                print(f"  T1 shape {t1_volume.shape} != PET shape {spatial_shape}")
                t1_volume = resample_to_pet(
                    t1_volume, t1_affine,
                    pet_shape=spatial_shape,
                    pet_affine=affine,
                )
                print(f"  Resampled T1 shape: {t1_volume.shape}")

            print("  Skull-stripping: antspynet (T1-based)...")
            try:
                brain_mask = brain_mask_antspynet(t1_volume, affine, modality="t1")
            except Exception as e:
                print(f"  antspynet failed ({e}), falling back to threshold")
                brain_mask = brain_mask_from_t1_threshold(t1_volume)
        else:
            print("  No T1 — skull-stripping from temporal-mean PET...")
            # Use temporal mean as a reference volume for brain extraction
            pet_mean = np.mean(sul_4d, axis=3)
            brain_mask = brain_mask_from_pet_fallback(pet_mean)

        n_brain = int(np.sum(brain_mask))
        n_total = brain_mask.size
        print(f"  Brain mask: {n_brain} / {n_total} ({100*n_brain/n_total:.1f}%)")
    else:
        brain_mask = np.ones(spatial_shape, dtype=bool)

    if cord_extend_mm > 0:
        if apply_skull_strip:
            brain_mask = extend_brain_mask_caudal(brain_mask, affine, cord_extend_mm)
        else:
            print("  Cord extension skipped: skull-stripping is off")

    # Capture resampled T1 for output (t1_volume may have been reassigned above)
    t1_resampled = t1_volume.copy() if t1_volume is not None else None
    processed = sul_4d.copy()
    for t in range(sul_4d.shape[3]):
        frame = processed[:, :, :, t]
        frame[~brain_mask] = 0.0
        if apply_smoothing and smooth_sigma > 0:
            frame_smoothed = gaussian_smooth_volume(frame, sigma=smooth_sigma, mask=brain_mask)
            processed[:, :, :, t] = frame_smoothed

    return processed, brain_mask, t1_resampled
