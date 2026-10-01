"""
Optional rigid head-motion correction for FET-PET frames.

Inter-frame pose is estimated with ANTs rigid + Mattes mutual information.
Short early frames (under 60 s) share one pose estimated on a ~60 s sum.
A transform is kept only when masked normalized correlation improves, and it
is applied only when the cortical shift is at least 1 mm. With a T1 on the
same grid, one extra rigid step puts the corrected PET into the T1 pose so
the brain mask matches the series.
"""

import json
import os

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, label


RIM_RADIUS_MM = 70.0
MIN_APPLY_RIM_MM = 1.0
WARN_RIM_MM = 3.0
MAX_T1_RIM_MM = 25.0
NCC_GAIN = 0.002
LOW_COUNT_FRACTION = 0.05
LONG_FRAME_SEC = 60.0
STABLE_FRAME_SEC = 180.0
BLOCK_SEC = 60.0


def save_motion_report(output_dir: str, report: dict) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "motion.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return path


def motion_report_summary(report: dict) -> dict:
    """Short block stored in report.json. Frame detail stays in motion.json."""
    summary = {
        "n_corrected": report["n_corrected"],
        "n_frames": report["n_frames"],
        "max_rim_mm": report["max_rim_mm"],
        "motion_warning": report["motion_warning"],
        "n_rejected": report["n_rejected"],
        "n_low_count": report["n_low_count"],
        "n_sub_mm": report["n_sub_mm"],
    }
    t1 = report.get("t1")
    if t1:
        summary["t1_applied"] = bool(t1.get("applied"))
        summary["t1_rim_mm"] = t1.get("rim_mm")
        summary["t1_reason"] = t1.get("reason")
        summary["t1_com_before_mm"] = t1.get("com_before_mm")
        summary["t1_com_after_mm"] = t1.get("com_after_mm")
    return summary


def print_motion_summary(report: dict) -> None:
    n = report["n_frames"]
    print(
        f"  Corrected {report['n_corrected']}/{n} frames, "
        f"max cortical shift {report['max_rim_mm']:.1f} mm"
    )
    print(
        f"  Rejected {report['n_rejected']} (no similarity gain), "
        f"low-count {report['n_low_count']}, "
        f"sub-mm left unresampled {report['n_sub_mm']}"
    )
    if report["motion_warning"]:
        print(f"  WARNING: cortical shift exceeds {WARN_RIM_MM:.0f} mm")
    t1 = report.get("t1")
    if t1:
        if t1.get("applied"):
            print(
                f"  T1 pose: applied, cortical shift {t1['rim_mm']:.1f} mm, "
                f"mask COM {t1['com_before_mm']:.1f} -> {t1['com_after_mm']:.1f} mm"
            )
        else:
            print(f"  T1 pose: not applied ({t1.get('reason', '')})")


def correct_volumes(
    volumes: list[np.ndarray],
    affine: np.ndarray,
    durations_sec: list[float],
    reference_indices: list[int],
    labels: list[str] | None = None,
    time_sec: list[float] | None = None,
) -> tuple[list[np.ndarray], dict]:
    """Rigid-align volumes to a reference pose. Inputs are not modified.

    Two passes. Pass 1 targets the mean of ``reference_indices``. Pass 2
    rebuilds the target from accepted stable frames and registers the
    originals again. A frame falls back to the pass-1 result when pass 2
    does not improve similarity.
    """
    n = len(volumes)
    if n == 0:
        raise ValueError("No volumes to align")
    if len(durations_sec) != n:
        raise ValueError("durations_sec length does not match volumes")
    if not reference_indices:
        raise ValueError("reference_indices is empty")

    labels = labels or [f"f{i:02d}" for i in range(n)]
    durations = [float(d) for d in durations_sec]
    groups = _frame_groups(durations)
    originals = []
    for vol in volumes:
        arr = np.array(vol, dtype=np.float64, copy=True)
        arr[~np.isfinite(arr)] = 0.0
        originals.append(arr)

    ref = _weighted_mean(
        [originals[i] for i in reference_indices],
        [durations[i] for i in reference_indices],
    )
    if _p95(ref) <= 0:
        print("  Motion correction skipped: reference volume is empty")
        report = _empty_report(n, "empty_reference")
        return [v.copy() for v in originals], report

    ncc_mask = _head_mask(ref)
    reg_mask = binary_dilation(ncc_mask, iterations=4)

    print(f"  Pass 1: {len(groups)} groups -> reference frames {reference_indices}")
    pass1 = _run_pass(originals, durations, groups, ref, affine, reg_mask, ncc_mask, pass_id=1)

    # Pass-2 target: stable frames that actually aligned, plus the reference
    # itself. A reference frame that tried to move by >= 1 mm and lost
    # similarity is an outlier and stays out of the new target.
    by1 = {i: dec for dec in pass1 for i in dec["indices"]}
    ref_ids = [
        i for dec in pass1 for i in dec["indices"]
        if dec["accepted"] and durations[i] >= STABLE_FRAME_SEC
    ]
    for i in reference_indices:
        dec = by1[i]
        outlier = dec["reason"] == "no_ncc_gain" and dec["rim_mm"] >= MIN_APPLY_RIM_MM
        if not outlier:
            ref_ids.append(i)
    ref_ids = sorted(set(ref_ids))
    aligned1 = _apply_decisions(originals, pass1)
    if len(ref_ids) >= 2:
        ref2 = _weighted_mean(
            [aligned1[i] for i in sorted(ref_ids)],
            [durations[i] for i in sorted(ref_ids)],
        )
        ncc_mask2 = _head_mask(ref2)
        reg_mask2 = binary_dilation(ncc_mask2, iterations=4)
        print(f"  Pass 2: target rebuilt from {len(ref_ids)} frames")
        pass2 = _run_pass(originals, durations, groups, ref2, affine, reg_mask2, ncc_mask2, pass_id=2)
        final, frame_meta = _merge_passes(originals, pass1, pass2, durations, labels, time_sec)
    else:
        print("  Pass 2 skipped: fewer than 2 stable frames")
        final = aligned1
        frame_meta = _frames_from_pass(pass1, durations, labels, time_sec, applied_pass=1)

    max_rim = max((fr["rim_mm"] for fr in frame_meta if fr["applied"]), default=0.0)
    report = {
        "method": "rigid_mattes",
        "n_frames": n,
        "n_groups": len(groups),
        "reference_indices": list(reference_indices),
        "n_corrected": int(sum(1 for fr in frame_meta if fr["applied"])),
        "n_rejected": int(sum(1 for fr in frame_meta if fr["reason"] == "no_ncc_gain")),
        "n_low_count": int(sum(1 for fr in frame_meta if fr["reason"] == "low_counts")),
        "n_sub_mm": int(sum(1 for fr in frame_meta if fr["reason"] == "sub_mm")),
        "max_rim_mm": round(float(max_rim), 2),
        "warn_rim_mm": WARN_RIM_MM,
        "motion_warning": bool(max_rim > WARN_RIM_MM),
        "frames": frame_meta,
    }
    print_motion_summary(report)
    return final, report


def align_volumes_to_target(
    volumes: list[np.ndarray],
    affine: np.ndarray,
    target: np.ndarray,
    estimate_indices: list[int],
    durations_sec: list[float],
) -> tuple[list[np.ndarray], dict]:
    """Apply one rigid transform that takes the PET reference pose onto ``target``.

    ``target`` must already sit on the same voxel grid as ``volumes`` (T1
    resampled to PET). Cross-modality NCC is not used. The step is kept when
    the head-mask centers move closer and the cortical shift is between 1 and
    25 mm.
    """
    info = {
        "applied": False,
        "reason": "",
        "rim_mm": 0.0,
        "center_mm": 0.0,
        "rotation_deg": 0.0,
        "translation_mm": [0.0, 0.0, 0.0],
        "com_before_mm": None,
        "com_after_mm": None,
    }
    moving = _weighted_mean(
        [volumes[i] for i in estimate_indices],
        [durations_sec[i] for i in estimate_indices],
    )
    pet_mask = _head_mask(moving)
    t1_mask = _head_mask(target, fraction=0.10)
    com_before = _com_distance_mm(pet_mask, t1_mask, affine)
    info["com_before_mm"] = None if com_before is None else round(com_before, 2)

    fixed_mask = binary_dilation(t1_mask, iterations=2)
    try:
        dec = _try_register(
            target, moving, affine, fixed_mask, pet_mask,
            accept_test="geometry",
            check_counts=False,
        )
    except Exception as exc:
        info["reason"] = f"registration_failed: {exc}"
        print(f"  T1 alignment failed: {exc}")
        return [v.copy() for v in volumes], info

    info["rim_mm"] = round(dec["rim_mm"], 2)
    info["center_mm"] = round(dec["center_mm"], 2)
    info["rotation_deg"] = round(dec["rotation_deg"], 2)
    info["translation_mm"] = [round(float(x), 2) for x in dec["translation_mm"]]

    if dec["reason"] != "aligned":
        info["reason"] = dec["reason"]
        return [v.copy() for v in volumes], info
    if dec["rim_mm"] > MAX_T1_RIM_MM:
        info["reason"] = "shift_above_25mm"
        print(f"  T1 alignment rejected: cortical shift {dec['rim_mm']:.1f} mm")
        return [v.copy() for v in volumes], info
    if dec["rim_mm"] < MIN_APPLY_RIM_MM:
        info["reason"] = "already_aligned"
        return [v.copy() for v in volumes], info

    warped_mean = _apply(target, moving, affine, dec["transforms"])
    warped_mask = _head_mask(warped_mean)
    com_after = _com_distance_mm(warped_mask, t1_mask, affine)
    info["com_after_mm"] = None if com_after is None else round(com_after, 2)
    if com_before is None or com_after is None or com_after > com_before - 0.3:
        info["reason"] = "com_not_improved"
        print(
            f"  T1 alignment rejected: mask COM "
            f"{com_before} -> {com_after} mm"
        )
        return [v.copy() for v in volumes], info

    aligned = [
        _apply(target, volumes[i], affine, dec["transforms"])
        for i in range(len(volumes))
    ]
    info["applied"] = True
    info["reason"] = "aligned"
    return aligned, info


def _run_pass(volumes, durations, groups, ref, affine, reg_mask, ncc_mask, pass_id):
    decisions = []
    for group in groups:
        weights = [durations[i] for i in group]
        moving = _weighted_mean([volumes[i] for i in group], weights)
        dur = float(sum(weights))
        try:
            dec = _try_register(ref, moving, affine, reg_mask, ncc_mask)
        except Exception as exc:
            print(f"  pass {pass_id} frames {group[0]}-{group[-1]} failed: {exc}")
            dec = _blank_decision("registration_failed")
        dec["indices"] = list(group)
        dec["pass"] = pass_id
        if dec["accepted"] and dec["rim_mm"] >= MIN_APPLY_RIM_MM:
            dec["warped"] = [
                _apply(ref, volumes[i], affine, dec["transforms"]) for i in group
            ]
        else:
            dec["warped"] = None
        if dec["accepted"] and dec["rim_mm"] >= MIN_APPLY_RIM_MM:
            tag = "keep"
        elif dec["accepted"]:
            tag = "sub_mm"
        else:
            tag = dec["reason"]
        print(
            f"  pass {pass_id} f{group[0]:02d}-f{group[-1]:02d} {dur:6.0f}s  "
            f"ncc {dec['ncc_before']:.3f}->{dec['ncc_after']:.3f}  "
            f"center {dec['center_mm']:.1f}  rim {dec['rim_mm']:.1f} mm  {tag}",
            flush=True,
        )
        decisions.append(dec)
    return decisions


def _try_register(fixed, moving, affine, reg_mask, ncc_mask, accept_test="ncc", check_counts=True):
    before = _ncc(fixed, moving, ncc_mask)
    if check_counts and _p95(moving) < LOW_COUNT_FRACTION * max(_p95(fixed), 1e-6):
        dec = _blank_decision("low_counts")
        dec["ncc_before"] = before
        dec["ncc_after"] = before
        return dec

    fixed_s = gaussian_filter(fixed, sigma=1.0)
    moving_s = gaussian_filter(moving, sigma=1.0)
    import ants

    fi = _as_ants(fixed_s, affine)
    mi = _as_ants(moving_s, affine)
    mask_img = _as_ants(reg_mask.astype(np.float32), affine)
    reg = ants.registration(
        fixed=fi,
        moving=mi,
        type_of_transform="Rigid",
        mask=mask_img,
        aff_metric="mattes",
        aff_sampling=32,
        aff_iterations=(60, 30, 10),
        aff_shrink_factors=(4, 2, 1),
        aff_smoothing_sigmas=(2, 1, 0),
        verbose=False,
    )
    center_mm, rim_mm = _displacements(reg["fwdtransforms"], fi)
    translation, rotation_deg = _transform_components(reg["fwdtransforms"][0])
    warped = _apply(fixed, moving, affine, reg["fwdtransforms"])
    after = _ncc(fixed, warped, ncc_mask)
    accepted = after > before + NCC_GAIN
    reason = "aligned" if accepted else "no_ncc_gain"
    if accept_test == "geometry":
        # Caller decides using mask-center distance. Hand back the pose.
        accepted = True
        reason = "aligned"
    return {
        "accepted": accepted,
        "reason": reason,
        "ncc_before": float(before),
        "ncc_after": float(after),
        "center_mm": float(center_mm),
        "rim_mm": float(rim_mm),
        "translation_mm": translation,
        "rotation_deg": float(rotation_deg),
        "transforms": reg["fwdtransforms"],
        "warped": None,
        "indices": [],
        "pass": 0,
    }


def _blank_decision(reason):
    return {
        "accepted": False,
        "reason": reason,
        "ncc_before": 0.0,
        "ncc_after": 0.0,
        "center_mm": 0.0,
        "rim_mm": 0.0,
        "translation_mm": [0.0, 0.0, 0.0],
        "rotation_deg": 0.0,
        "transforms": None,
        "warped": None,
        "indices": [],
        "pass": 0,
    }


def _apply_decisions(volumes, decisions):
    out = [v.copy() for v in volumes]
    for dec in decisions:
        if dec.get("warped") is None:
            continue
        for local_i, frame_i in enumerate(dec["indices"]):
            out[frame_i] = dec["warped"][local_i]
    return out


def _merge_passes(originals, pass1, pass2, durations, labels, time_sec):
    """Prefer pass 2. Fall back to pass 1 only when pass 2 found no similarity gain."""
    by1 = {i: dec for dec in pass1 for i in dec["indices"]}
    by2 = {i: dec for dec in pass2 for i in dec["indices"]}
    final = [v.copy() for v in originals]
    frames = []
    for i in range(len(originals)):
        second = by2[i]
        first = by1[i]
        chosen = None
        reason = second["reason"]
        if second.get("warped") is not None:
            chosen = second
            reason = "aligned"
        elif second["reason"] in ("no_ncc_gain", "registration_failed") and first.get("warped") is not None:
            chosen = first
            reason = "kept_pass1"
        elif second["accepted"] and second["rim_mm"] < MIN_APPLY_RIM_MM:
            reason = "sub_mm"
        applied = chosen is not None
        src = chosen if chosen is not None else second
        if applied:
            local = chosen["indices"].index(i)
            final[i] = chosen["warped"][local]
        frames.append(_frame_record(i, src, reason, applied, durations, labels, time_sec))
    return final, frames


def _frames_from_pass(decisions, durations, labels, time_sec, applied_pass):
    frames = []
    for dec in decisions:
        for i in dec["indices"]:
            applied = dec.get("warped") is not None
            reason = "aligned" if applied else dec["reason"]
            if dec["accepted"] and dec["rim_mm"] < MIN_APPLY_RIM_MM:
                reason = "sub_mm"
                applied = False
            frames.append(_frame_record(i, dec, reason, applied, durations, labels, time_sec))
    frames.sort(key=lambda fr: fr["index"])
    return frames


def _frame_record(index, src, reason, applied, durations, labels, time_sec):
    rec = {
        "index": int(index),
        "label": labels[index],
        "duration_sec": round(float(durations[index]), 1),
        "applied": bool(applied),
        "reason": reason,
        "pass": int(src.get("pass", 0)),
        "ncc_before": round(float(src["ncc_before"]), 4),
        "ncc_after": round(float(src["ncc_after"]), 4),
        "center_mm": round(float(src["center_mm"]), 2),
        "rim_mm": round(float(src["rim_mm"]), 2),
        "translation_mm": [round(float(x), 2) for x in src["translation_mm"]],
        "rotation_deg": round(float(src["rotation_deg"]), 2),
    }
    if time_sec is not None:
        rec["time_sec"] = round(float(time_sec[index]), 1)
    return rec


def _empty_report(n, reason):
    return {
        "method": "rigid_mattes",
        "n_frames": n,
        "n_groups": 0,
        "reference_indices": [],
        "n_corrected": 0,
        "n_rejected": 0,
        "n_low_count": 0,
        "n_sub_mm": 0,
        "max_rim_mm": 0.0,
        "warn_rim_mm": WARN_RIM_MM,
        "motion_warning": False,
        "skipped": reason,
        "frames": [],
    }


def _frame_groups(durations: list[float]) -> list[list[int]]:
    """Group sub-minute frames into ~60 s blocks. Longer frames stay alone."""
    groups = []
    buf = []
    acc = 0.0
    for i, dur in enumerate(durations):
        if dur >= LONG_FRAME_SEC:
            if buf:
                groups.append(buf)
                buf, acc = [], 0.0
            groups.append([i])
            continue
        buf.append(i)
        acc += dur
        if acc >= BLOCK_SEC:
            groups.append(buf)
            buf, acc = [], 0.0
    if buf:
        groups.append(buf)
    return groups


def _weighted_mean(volumes, weights) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    w = w / max(w.sum(), 1e-8)
    acc = np.zeros(volumes[0].shape, dtype=np.float64)
    for vol, wi in zip(volumes, w):
        acc += wi * vol
    return acc


def _p95(vol: np.ndarray) -> float:
    nz = vol[np.isfinite(vol) & (vol > 0)]
    if nz.size < 50:
        return 0.0
    return float(np.percentile(nz, 95))


def _head_mask(vol: np.ndarray, fraction: float = 0.20) -> np.ndarray:
    peak = _p95(vol)
    if peak <= 0:
        return np.zeros(vol.shape, dtype=bool)
    mask = vol > peak * fraction
    return _largest_component(mask)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, n = label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return labeled == int(np.argmax(sizes))


def _ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    if mask is None or int(np.count_nonzero(mask)) < 50:
        return 0.0
    x = a[mask].astype(np.float64)
    y = b[mask].astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    if denom < 1e-8:
        return 0.0
    return float((x * y).sum() / denom)


def _geometry(affine: np.ndarray):
    matrix = np.array(affine[:3, :3], dtype=np.float64)
    spacing = np.linalg.norm(matrix, axis=0)
    spacing[spacing == 0] = 1.0
    direction = matrix / spacing
    origin = np.array(affine[:3, 3], dtype=np.float64)
    return tuple(float(v) for v in spacing), tuple(float(v) for v in origin), direction


def _as_ants(vol: np.ndarray, affine: np.ndarray):
    import ants
    spacing, origin, direction = _geometry(affine)
    data = np.ascontiguousarray(vol, dtype=np.float32)
    return ants.from_numpy(data, origin=origin, spacing=spacing, direction=direction)


def _apply(fixed_vol, moving_vol, affine, transforms) -> np.ndarray:
    import ants
    fixed = _as_ants(fixed_vol, affine)
    moving = _as_ants(moving_vol, affine)
    warped = ants.apply_transforms(
        fixed, moving, transformlist=transforms, interpolator="linear",
    )
    arr = np.array(warped.numpy(), dtype=np.float64, copy=True)
    if arr.shape != np.shape(moving_vol):
        raise RuntimeError(f"Warped shape {arr.shape} != {np.shape(moving_vol)}")
    arr[~np.isfinite(arr)] = 0.0
    arr[arr < 0] = 0.0
    return arr


def _displacements(transforms, ref_img) -> tuple[float, float]:
    import ants
    import pandas as pd

    origin = np.array(ref_img.origin, dtype=np.float64)
    spacing = np.array(ref_img.spacing, dtype=np.float64)
    direction = np.array(ref_img.direction, dtype=np.float64).reshape(3, 3)
    center = origin + direction @ ((np.array(ref_img.shape, dtype=np.float64) / 2.0) * spacing)
    points = [center]
    for axis in range(3):
        for sign in (-RIM_RADIUS_MM, RIM_RADIUS_MM):
            point = center.copy()
            point[axis] += sign
            points.append(point)
    table = pd.DataFrame(points, columns=["x", "y", "z"])
    moved = ants.apply_transforms_to_points(3, table, transformlist=transforms)
    disp = np.linalg.norm(moved[["x", "y", "z"]].to_numpy() - np.asarray(points), axis=1)
    return float(disp[0]), float(np.max(disp[1:]))


def _transform_components(path: str) -> tuple[list[float], float]:
    import ants
    params = np.array(ants.read_transform(path).parameters, dtype=np.float64)
    if params.size >= 12:
        matrix = params[:9].reshape(3, 3)
        translation = params[9:12]
    elif params.size >= 6:
        translation = params[3:6]
        matrix = np.eye(3)
    else:
        return [0.0, 0.0, 0.0], 0.0
    trace = float(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))
    rotation_deg = float(np.degrees(np.arccos(trace)))
    return [float(v) for v in translation], rotation_deg


def _com_distance_mm(mask_a, mask_b, affine) -> float | None:
    ca = _mask_com_world(mask_a, affine)
    cb = _mask_com_world(mask_b, affine)
    if ca is None or cb is None:
        return None
    return float(np.linalg.norm(ca - cb))


def _mask_com_world(mask, affine) -> np.ndarray | None:
    idx = np.argwhere(mask)
    if idx.shape[0] < 20:
        return None
    hom = np.c_[idx.astype(np.float64), np.ones(idx.shape[0])]
    world = (np.asarray(affine, dtype=np.float64) @ hom.T).T[:, :3]
    return world.mean(axis=0)
