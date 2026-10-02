"""Clean FastF1 laps, align them to the track centerline and average them into one raceline."""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from f1init.geometry import arc_length, lateral_offsets

POS_SCALE = 0.1  # FastF1 X/Y are in 1/10 m
ROT_STEP = 1.1
ROT_REFINE = (0.7, 0.3, 0.1)
TRANS_RANGE = 20.0
TRANS_STEP = 0.5
MAX_ITERS = 5
MIN_IMPROVE = 0.002
N_POINTS = 3000
RAY_LENGTH = 20.0


def load_lap(path) -> np.ndarray:
    df = pd.read_csv(path)
    if not {"X", "Y"} <= set(df.columns):
        raise ValueError(f"{path}: no X/Y columns")
    if "Status" in df:
        df = df[df["Status"] == "OnTrack"]
    xy = df[["X", "Y"]].dropna().to_numpy(float) * POS_SCALE
    if len(xy) > 1:
        xy = xy[np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-6]]
    if len(xy) < 10:
        raise ValueError(f"{path}: only {len(xy)} valid points")
    return xy


def _rot(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])


def apply(xy: np.ndarray, R: np.ndarray, T: np.ndarray) -> np.ndarray:
    return xy @ R + T


def align(laps: list[np.ndarray], center: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Grid search for the rotation R and translation T (xy @ R + T) that bring the laps onto the centerline."""
    pts = np.vstack(laps)
    pts = pts[np.linspace(0, len(pts) - 1, min(N_POINTS, len(pts))).astype(int)]
    mean = pts.mean(axis=0)
    src = pts - mean
    tree = cKDTree(center)

    def cost(q: np.ndarray) -> float:
        return float(tree.query(q)[0].mean())

    def best_rotation(angles, T0):
        best = (np.inf, 0.0, T0)
        for deg in angles:
            q = src @ _rot(deg)
            idx = tree.query(q + T0)[1]
            T = T0 + np.median(center[idx] - (q + T0), axis=0)
            c = cost(q + T)
            if c < best[0]:
                best = (c, deg, T)
        return best

    err, deg, T = best_rotation(np.arange(-180.0, 180.0, ROT_STEP), center.mean(axis=0))
    shifts = np.arange(-TRANS_RANGE, TRANS_RANGE + 1e-9, TRANS_STEP)
    prev = np.inf
    for it in range(MAX_ITERS):
        span = ROT_STEP if it == 0 else 5.0
        for step in ROT_REFINE:
            candidate = best_rotation(np.arange(deg - span, deg + span + 1e-9, step), T)
            if candidate[0] < err:
                err, deg, T = candidate
            span = step
        q = src @ _rot(deg)
        base = T
        for dx in shifts:
            for dy in shifts:
                c = cost(q + base + (dx, dy))
                if c < err:
                    err, T = c, base + (dx, dy)
        if prev - err < MIN_IMPROVE:
            break
        prev = err

    R = _rot(deg)
    return R, T - mean @ R, err


def mean_offsets(laps: list[np.ndarray], center: np.ndarray) -> np.ndarray:
    per_lap = np.array([lateral_offsets(center, lap, RAY_LENGTH) for lap in laps])
    count = np.sum(~np.isnan(per_lap), axis=0)
    d = np.where(count > 0, np.nansum(per_lap, axis=0) / np.maximum(count, 1), np.nan)
    missing = np.isnan(d)
    if missing.mean() > 0.05:
        raise ValueError(f"{missing.mean():.0%} of the track points have no lap crossing; alignment failed?")
    if missing.any():
        s, length = arc_length(center)
        d[missing] = np.interp(s[missing], s[~missing], d[~missing], period=length)
    return d
