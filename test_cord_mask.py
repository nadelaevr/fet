"""Brain-mask cord extension: brainstem stump continues, tonsils and neck do not."""

import numpy as np
import nibabel as nib

from preprocess import extend_brain_mask_caudal, _mask_orientation, _voxel_spacing


def _affine_for(axis_codes):
    """Diagonal affine whose voxel axes are the given world directions."""
    direction = {
        "R": np.array([1.0, 0.0, 0.0]),
        "L": np.array([-1.0, 0.0, 0.0]),
        "A": np.array([0.0, 1.0, 0.0]),
        "P": np.array([0.0, -1.0, 0.0]),
        "S": np.array([0.0, 0.0, 1.0]),
        "I": np.array([0.0, 0.0, -1.0]),
    }
    affine = np.eye(4)
    for i, code in enumerate(axis_codes):
        affine[:3, i] = direction[code] * 2.0
    return affine


def _paint_disk(mask, affine, si_index, center_lr, center_ap, radius_mm):
    si, lr, ap, _, _ = _mask_orientation(affine)
    spacing = [_voxel_spacing(affine, i) for i in range(3)]
    sl = [slice(None)] * 3
    sl[si] = si_index
    plane_axes = [a for a in range(3) if a != si]
    lr_coords = np.arange(mask.shape[lr])
    ap_coords = np.arange(mask.shape[ap])
    LR, AP = np.meshgrid(lr_coords, ap_coords, indexing="ij")
    disk = (
        ((LR - center_lr) * spacing[lr]) ** 2
        + ((AP - center_ap) * spacing[ap]) ** 2
    ) <= radius_mm ** 2
    plane = mask[tuple(sl)]
    plane |= disk if plane_axes[0] == lr else disk.T


def _phantom(axis_codes):
    affine = _affine_for(axis_codes)
    shape = (48, 56, 64)
    mask = np.zeros(shape, dtype=bool)
    si, lr, ap, inf_sign, _ = _mask_orientation(affine)
    assert nib.aff2axcodes(affine) == tuple(axis_codes)

    center = [s // 2 for s in shape]
    # Inferior end of the medulla stump, then 8 slices (16 mm) of cord-sized tissue.
    stump = center[si]
    for step in range(0, 8):
        _paint_disk(mask, affine, stump - inf_sign * step, center[lr], center[ap], 8.0)
    # Cerebellar tonsils: further inferior, 24 mm off midline, posterior.
    tonsil_si = stump + inf_sign * 4
    for side in (-12, 12):
        _paint_disk(
            mask, affine, tonsil_si,
            center[lr] + side, center[ap] - 8, 6.0,
        )
    return mask, affine, center, stump, inf_sign, si, lr, ap


def _index(center, si, lr, ap, si_i, lr_i=None, ap_i=None):
    idx = [0, 0, 0]
    idx[si] = si_i
    idx[lr] = center[lr] if lr_i is None else lr_i
    idx[ap] = center[ap] if ap_i is None else ap_i
    return tuple(idx)


def _check(axis_codes):
    mask, affine, center, stump, inf_sign, si, lr, ap = _phantom(axis_codes)
    spacing = _voxel_spacing(affine, si)
    out = extend_brain_mask_caudal(mask, affine, extend_mm=20.0)

    # 16 mm below the stump, on the midline: inside the tube (20 mm reach).
    steps = int(round(16.0 / spacing))
    below = _index(center, si, lr, ap, stump + inf_sign * steps)
    assert out[below], axis_codes

    # Same distance, 24 mm off midline: neck, must stay empty.
    lateral_lr = center[lr] + int(round(24.0 / _voxel_spacing(affine, lr)))
    lateral = _index(center, si, lr, ap, below[si], lr_i=lateral_lr)
    assert not out[lateral], axis_codes

    # Farther than the requested length: still empty.
    too_far_si = stump + inf_sign * int(round(28.0 / spacing))
    assert 0 <= too_far_si < mask.shape[si]
    assert not out[_index(center, si, lr, ap, too_far_si)], axis_codes

    # The stump itself stays, and tissue superior to it is not rewritten.
    assert out[_index(center, si, lr, ap, stump)]
    above = _index(center, si, lr, ap, stump - inf_sign * 4)
    assert out[above] and mask[above], axis_codes

    same = extend_brain_mask_caudal(mask, affine, extend_mm=0.0)
    assert np.array_equal(same, mask)


def test_cord_extension_follows_brainstem_not_tonsils():
    for codes in (("R", "A", "S"), ("L", "A", "I"), ("A", "R", "S")):
        _check(codes)
    print("cord extension ok")


if __name__ == "__main__":
    test_cord_extension_follows_brainstem_not_tonsils()
