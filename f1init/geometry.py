"""Geometry of closed tracks given as (N, 2) centerline points in driving direction."""
import numpy as np
from scipy.interpolate import CubicSpline


def drop_closing_point(xy: np.ndarray) -> np.ndarray:
    if len(xy) > 1 and np.linalg.norm(xy[0] - xy[-1]) < 1e-6:
        return xy[:-1]
    return xy


def arc_length(xy: np.ndarray) -> tuple[np.ndarray, float]:
    seg = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg[:-1])))
    return s, float(seg.sum())


def resample(xy: np.ndarray, columns: dict[str, np.ndarray], ds: float = 2.0):
    s, length = arc_length(xy)
    s_closed = np.append(s, length)
    s_new = np.arange(0.0, length, ds)

    def spline(values: np.ndarray) -> np.ndarray:
        closed = np.concatenate((values, values[:1]), axis=0)
        return CubicSpline(s_closed, closed, bc_type="periodic")(s_new)

    return spline(xy), {name: spline(values) for name, values in columns.items()}


def tangents_normals(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.roll(xy, -1, axis=0) - np.roll(xy, 1, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    n = np.column_stack((t[:, 1], -t[:, 0]))
    return t, n


def curvature(xy: np.ndarray) -> np.ndarray:
    """Positive when the track turns left."""
    t, _ = tangents_normals(xy)
    psi = np.arctan2(t[:, 1], t[:, 0])
    seg = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)
    dpsi = np.angle(np.exp(1j * (np.roll(psi, -1) - np.roll(psi, 1))))
    return dpsi / (seg + np.roll(seg, 1))


def lateral_offsets(center: np.ndarray, line: np.ndarray, max_dist: float = 20.0) -> np.ndarray:
    """Distance along each right normal to the nearest crossing of the closed line; positive = right, NaN = none."""
    _, n = tangents_normals(center)
    ab = np.roll(line, -1, axis=0) - line
    offsets = np.full(len(center), np.nan)
    for start in range(0, len(center), 512):
        p = center[start:start + 512, None, :]
        nn = n[start:start + 512, None, :]
        ap = line[None, :, :] - p
        denom = nn[..., 0] * ab[None, :, 1] - nn[..., 1] * ab[None, :, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (ap[..., 0] * ab[None, :, 1] - ap[..., 1] * ab[None, :, 0]) / denom
            u = (ap[..., 0] * nn[..., 1] - ap[..., 1] * nn[..., 0]) / denom
        valid = (u >= 0.0) & (u <= 1.0) & (np.abs(t) <= max_dist)
        t = np.where(valid, t, np.inf)
        best = np.argmin(np.abs(t), axis=1)
        chosen = t[np.arange(len(best)), best]
        offsets[start:start + 512] = np.where(np.isfinite(chosen), chosen, np.nan)
    return offsets
