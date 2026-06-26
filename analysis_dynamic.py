"""
Dynamic analysis: TBR(t) per time frame, slope map from linear regression
over all time points, curve classification.

For dynamic data we have N time frames (e.g. 38). For each frame t:
    TBR(x,y,z,t) = SUL(x,y,z,t) / SULmean(t)
where SULmean(t) is the trimmed mean of SUL in the brain mask at time t.

Then per-voxel slope is computed by linear regression of TBR(t) vs time.
"""

import numpy as np
from typing import Tuple

from analysis import trimmed_mean_numpy_only


def compute_tbr_4d(sul_4d: np.ndarray, brain_mask: np.ndarray,
                   trim_percent: float = 2.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute TBR for each time frame.

    For each frame t:
        SULmean(t) = trimmed_mean(SUL[brain_mask, t])
        TBR[:,:,:,t] = SUL[:,:,:,t] / SULmean(t)

    Args:
        sul_4d: 4D SUL array (X, Y, Z, T)
        brain_mask: 3D boolean brain mask (X, Y, Z)
        trim_percent: trim percent for SULmean

    Returns:
        tbr_4d: 4D TBR array (X, Y, Z, T)
        sul_means: 1D array of SULmean per frame, shape (T,)
    """
    n_frames = sul_4d.shape[3]
    tbr_4d = np.zeros_like(sul_4d)
    sul_means = np.zeros(n_frames)

    for t in range(n_frames):
        frame = sul_4d[:, :, :, t]
        brain_vals = frame[brain_mask]

        if len(brain_vals) == 0:
            sul_means[t] = 0.0
            continue

        sul_mean, _, _, _ = trimmed_mean_numpy_only(
            frame * brain_mask, trim_percent
        )
        sul_means[t] = sul_mean

        if sul_mean > 0:
            tbr_4d[:, :, :, t] = frame / sul_mean
        # else leave as zeros

    return tbr_4d, sul_means


def compute_slope_map_dynamic(tbr_4d: np.ndarray,
                              time_points_min: np.ndarray) -> np.ndarray:
    """
    Compute per-voxel slope of TBR vs time using linear regression
    over ALL time points (not just 3).

    slope = sum((t - t_mean) * (TBR(t) - TBR_mean)) / sum((t - t_mean)^2)

    This is the natural generalization of compute_slope_map from analysis.py
    to N time points instead of 3.

    Args:
        tbr_4d: 4D TBR array (X, Y, Z, T)
        time_points_min: 1D array of time points in minutes, shape (T,)

    Returns:
        slope: 3D array (X, Y, Z) in TBR per minute
    """
    t = time_points_min.astype(np.float64)
    t_mean = np.mean(t)
    t_centered = t - t_mean
    denom = np.sum(t_centered ** 2)

    if denom == 0:
        raise ValueError("All time points are identical — cannot compute slope")

    # TBR_mean per voxel across time
    tbr_mean = np.mean(tbr_4d, axis=3)  # (X, Y, Z)

    # numerator = sum_t (t - t_mean) * (TBR(t) - TBR_mean)
    numerator = np.zeros(tbr_4d.shape[:3], dtype=np.float64)
    for i, tc in enumerate(t_centered):
        numerator += tc * (tbr_4d[:, :, :, i] - tbr_mean)

    slope = numerator / denom
    return slope


def classify_curves_dynamic(
    slope_map: np.ndarray,
    tbr_delta_threshold: float = 0.3,
    time_span_min: float = 40.0,
) -> np.ndarray:
    """
    Classify each voxel's TBR curve into:
        1 = rising
        2 = falling
        3 = plateau

    Same logic as classify_curves from analysis.py, but time_span_min
    is the total duration of the dynamic acquisition (default 40 min
    for 0-2400s range).

    Args:
        slope_map: 3D array of slopes (TBR/min)
        tbr_delta_threshold: minimum TBR change over time_span to count as trend
        time_span_min: total time span in minutes

    Returns:
        classes: 3D int8 array (0=unclassified, 1=rising, 2=falling, 3=plateau)
    """
    slope_threshold = tbr_delta_threshold / time_span_min

    classes = np.zeros_like(slope_map, dtype=np.int8)
    classes[slope_map > slope_threshold] = 1    # rising
    classes[slope_map < -slope_threshold] = 2   # falling
    classes[np.abs(slope_map) <= slope_threshold] = 3  # plateau

    return classes


def compute_tbrmax_dynamic(tbr_4d: np.ndarray) -> np.ndarray:
    """
    Compute max TBR across all time frames for each voxel.

    Args:
        tbr_4d: 4D TBR array (X, Y, Z, T)

    Returns:
        tbrmax: 3D array (X, Y, Z) — max TBR over time axis
    """
    return np.max(tbr_4d, axis=3)
