"""
Preprocessing: antspynet skull-stripping, T1-to-PET resampling, Gaussian smoothing.
"""

import os
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
    Resample T1 volume into PET space using ANTs registration (rigid/affine).

    Since T1 and PET are already in the same space (same scanner session),
    we just need to resample — no registration needed, only interpolation
    to match PET grid.

    Args:
        t1_volume: T1 3D array
        t1_affine: T1 4x4 affine
        pet_shape: target PET shape
        pet_affine: target PET 4x4 affine

    Returns:
        t1_resampled: T1 resampled to PET grid (shape = pet_shape)
    """
    import ants

    # Save T1 as temp NIfTI
    tmp_dir = tempfile.mkdtemp(prefix="resample_")
    t1_path = os.path.join(tmp_dir, "t1.nii.gz")
    pet_path = os.path.join(tmp_dir, "pet.nii.gz")

    nib.save(nib.Nifti1Image(t1_volume.astype(np.float32), t1_affine), t1_path)

    # Create a dummy PET NIfTI just for geometry
    pet_dummy = np.zeros(pet_shape, dtype=np.float32)
    nib.save(nib.Nifti1Image(pet_dummy, pet_affine), pet_path)

    # Load into ANTs
    t1_ants = ants.image_read(t1_path)
    pet_ants = ants.image_read(pet_path)

    # Resample T1 to PET grid using apply_transforms with identity
    # (they're already in the same space)
    print(f"  Resampling T1 {t1_ants.shape} -> PET {pet_ants.shape}...")
    t1_resampled = ants.resample_image_to_target(
        t1_ants, pet_ants, interp_type=1  # linear interpolation
    )

    result = t1_resampled.numpy().astype(np.float64)

    # Cleanup
    for p in [t1_path, pet_path]:
        try:
            os.remove(p)
        except OSError:
            pass

    return result


def brain_mask_antspynet(
    t1_volume: np.ndarray,
    affine: np.ndarray,
    modality: str = "t1",
) -> np.ndarray:
    """
    Skull-stripping via antspynet deep learning brain extraction.
    """
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

        if t1_volume is not None and use_antspynet:
            print("  Skull-stripping: antspynet (T1-based)...")
            try:
                brain_mask = brain_mask_antspynet(t1_volume, affine, modality="t1")
            except Exception as e:
                print(f"  antspynet failed ({e}), falling back to threshold")
                brain_mask = brain_mask_from_t1_threshold(t1_volume)
        elif t1_volume is not None:
            print("  Skull-stripping: threshold (T1)...")
            brain_mask = brain_mask_from_t1_threshold(t1_volume)
        else:
            print("  No T1 — skull-stripping from 20-min PET...")
            brain_mask = brain_mask_from_pet_fallback(sul_volumes[0])

        n_brain = int(np.sum(brain_mask))
        n_total = brain_mask.size
        print(f"  Brain mask: {n_brain} / {n_total} ({100*n_brain/n_total:.1f}%)")
    else:
        brain_mask = np.ones_like(sul_volumes[0], dtype=bool)

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

    # Apply mask and optional smoothing per frame
    t1_resampled = t1_volume.copy() if t1_volume is not None else None
    processed = sul_4d.copy()
    for t in range(sul_4d.shape[3]):
        frame = processed[:, :, :, t]
        frame[~brain_mask] = 0.0
        if apply_smoothing and smooth_sigma > 0:
            frame_smoothed = gaussian_smooth_volume(frame, sigma=smooth_sigma, mask=brain_mask)
            processed[:, :, :, t] = frame_smoothed

    return processed, brain_mask, t1_resampled
