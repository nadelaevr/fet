"""
Core analysis: trimmed mean, TBR computation, slope, curve classification.
"""

import numpy as np
from typing import Tuple


def trimmed_mean_with_ci(
    volume: np.ndarray,
    trim_percent: float = 2.5,
    ci_percent: float = 95.0,
) -> Tuple[float, float, float, float]:
    """
    Compute trimmed mean of non-zero voxels with confidence interval.

    Args:
        volume: 3D array (e.g., SUL values)
        trim_percent: percent to trim from each tail (default 2.5)
        ci_percent: confidence interval level (default 95%)

    Returns:
        mean_trimmed: trimmed mean
        std_trimmed: standard deviation of trimmed data
        ci_lower: lower bound of CI
        ci_upper: upper bound of CI
    """
    # Use non-zero voxels only (exclude background/air)
    vals = volume[volume > 0].ravel()

    if len(vals) == 0:
        return 0.0, 0.0, 0.0, 0.0

    # Sort and trim
    vals_sorted = np.sort(vals)
    n = len(vals_sorted)
    trim_n = int(np.round(n * trim_percent / 100.0))

    if trim_n * 2 >= n:
        # Too much trim, use all data
        trimmed = vals_sorted
    else:
        trimmed = vals_sorted[trim_n : n - trim_n]

    mean_val = float(np.mean(trimmed))
    std_val = float(np.std(trimmed, ddof=1))

    # CI
    n_trimmed = len(trimmed)
    se = std_val / np.sqrt(n_trimmed)

    # z-value for CI
    from scipy.stats import norm as _norm  # lazy import

    alpha = 1.0 - ci_percent / 100.0
    z = _norm.ppf(1.0 - alpha / 2.0)

    ci_lower = mean_val - z * se
    ci_upper = mean_val + z * se

    return mean_val, std_val, ci_lower, ci_upper


def trimmed_mean_numpy_only(
    volume: np.ndarray,
    trim_percent: float = 2.5,
    ci_percent: float = 95.0,
) -> Tuple[float, float, float, float]:
    """
    Same as trimmed_mean_with_ci but uses numpy-only z-approximation
    (avoids scipy dependency).
    """
    vals = volume[volume > 0].ravel()

    if len(vals) == 0:
        return 0.0, 0.0, 0.0, 0.0

    vals_sorted = np.sort(vals)
    n = len(vals_sorted)
    trim_n = int(np.round(n * trim_percent / 100.0))

    if trim_n * 2 >= n:
        trimmed = vals_sorted
    else:
        trimmed = vals_sorted[trim_n : n - trim_n]

    mean_val = float(np.mean(trimmed))
    std_val = float(np.std(trimmed, ddof=1))

    n_trimmed = len(trimmed)
    se = std_val / np.sqrt(n_trimmed)

    # z = 1.96 for 95% CI (exact for normal)
    # For other CI levels, use rational approximation
    if abs(ci_percent - 95.0) < 0.1:
        z = 1.96
    elif abs(ci_percent - 99.0) < 0.1:
        z = 2.576
    elif abs(ci_percent - 90.0) < 0.1:
        z = 1.645
    else:
        # Rational approximation of inverse normal CDF
        p = 1.0 - (1.0 - ci_percent / 100.0) / 2.0
        t = np.sqrt(-2.0 * np.log(1.0 - p))
        z = t - (2.515517 + 0.802853 * t + 0.010328 * t * t) / (
            1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t
        )

    ci_lower = mean_val - z * se
    ci_upper = mean_val + z * se

    return mean_val, std_val, ci_lower, ci_upper


def compute_tbr_map(sul_volume: np.ndarray, sul_mean: float) -> np.ndarray:
    """
    Compute TBR = SUL / SULmean for every voxel.

    Args:
        sul_volume: 3D SUL array
        sul_mean: trimmed mean SUL value

    Returns:
        tbr: 3D TBR array
    """
    if sul_mean <= 0:
        raise ValueError(f"SULmean must be > 0, got {sul_mean}")
    return sul_volume / sul_mean


def compute_slope_map(
    tbr_t20: np.ndarray,
    tbr_t40: np.ndarray,
    tbr_t60: np.ndarray,
    time_points: Tuple[float, float, float] = (20.0, 40.0, 60.0),
) -> np.ndarray:
    """
    Compute per-voxel slope of TBR vs time using linear regression.

    slope = sum((t - t_mean) * (TBR - TBR_mean)) / sum((t - t_mean)^2)

    Args:
        tbr_t20, tbr_t40, tbr_t60: 3D TBR arrays at each time point
        time_points: time values in minutes

    Returns:
        slope: 3D array of slopes (TBR per minute)
    """
    t = np.array(time_points, dtype=np.float64)
    t_mean = np.mean(t)
    t_centered = t - t_mean
    denom = np.sum(t_centered ** 2) # noqa: E226

    # Stack TBRs: shape (3, Z, Y, X)
    tbr_stack = np.stack([tbr_t20, tbr_t40, tbr_t60], axis=0)
    tbr_mean = np.mean(tbr_stack, axis=0)

    numerator = (
        t_centered[0] * (tbr_t20 - tbr_mean)
        + t_centered[1] * (tbr_t40 - tbr_mean)
        + t_centered[2] * (tbr_t60 - tbr_mean)
    )

    slope = numerator / denom
    return slope


def classify_curves(
    slope_map: np.ndarray,
    tbr_delta_threshold: float = 0.3,
    time_span: float = 40.0,
) -> np.ndarray:
    """
    Classify each voxel's TBR curve into:
        1 = rising
        2 = falling
        3 = plateau

    Based on slope: the threshold on slope is tbr_delta_threshold / time_span.

    A slope > threshold means TBR rises by more than tbr_delta_threshold
    over the time_span (40 min by default).

    Args:
        slope_map: 3D array of slopes (TBR/min)
        tbr_delta_threshold: minimum TBR change over time_span to count as trend
        time_span: time interval between first and last point (min)

    Returns:
        classes: 3D int8 array (0=unclassified, 1=rising, 2=falling, 3=plateau)
    """
    slope_threshold = tbr_delta_threshold / time_span

    classes = np.zeros_like(slope_map, dtype=np.int8)
    classes[slope_map > slope_threshold] = 1   # rising
    classes[slope_map < -slope_threshold] = 2   # falling
    classes[np.abs(slope_map) <= slope_threshold] = 3  # plateau

    return classes


def compute_sulmax(
    sul_t20: np.ndarray,
    sul_t40: np.ndarray,
    sul_t60: np.ndarray,
) -> np.ndarray:
    """Max SUL across three time points per voxel."""
    return np.maximum(np.maximum(sul_t20, sul_t40), sul_t60)


def compute_tbrmax(
    tbr_t20: np.ndarray,
    tbr_t40: np.ndarray,
    tbr_t60: np.ndarray,
) -> np.ndarray:
    """Max TBR across three time points per voxel."""
    return np.maximum(np.maximum(tbr_t20, tbr_t40), tbr_t60)


def build_significant_mask(
    sulmax: np.ndarray,
    tbr_t20: np.ndarray,
    tbr_t40: np.ndarray,
    tbr_t60: np.ndarray,
    sul_threshold: float = 2.0,
    tbr_threshold: float = 2.0,
) -> np.ndarray:
    """
    Significant mask: SULmax > threshold AND at least one TBR > threshold.

    Args:
        sulmax: 3D SULmax map
        tbr_t20, tbr_t40, tbr_t60: 3D TBR maps at each time point
        sul_threshold: minimum SULmax (default 2)
        tbr_threshold: minimum TBR at any time point (default 2)

    Returns:
        mask: boolean 3D array
    """
    sul_ok = sulmax > sul_threshold
    tbr_any_ok = (tbr_t20 > tbr_threshold) | (tbr_t40 > tbr_threshold) | (tbr_t60 > tbr_threshold)
    return sul_ok & tbr_any_ok
