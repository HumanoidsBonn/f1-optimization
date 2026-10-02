"""Track CSVs and the sliding windows the network is trained on."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from f1init.geometry import curvature, drop_closing_point, resample

HISTORY = 100
FUTURE = 300
TARGET = 10
REQUIRED = ["x_m", "y_m", "w_tr_right_m", "w_tr_left_m"]


def read_track(path, need_d: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip("# ") for c in df.columns]
    missing = [c for c in REQUIRED + (["d_m"] if need_d else []) if c not in df]
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    if df[REQUIRED].isna().any().any():
        raise ValueError(f"{path} contains empty or non-numeric values")
    xy = drop_closing_point(df[["x_m", "y_m"]].to_numpy(float))
    values = {c: df[c].to_numpy(float)[:len(xy)] for c in REQUIRED[2:] + (["d_m"] if "d_m" in df else [])}
    if len(xy) < 50:
        raise ValueError(f"{path} has only {len(xy)} points; at least 50 are needed")
    spacing = np.median(np.linalg.norm(np.diff(xy, axis=0), axis=1))
    if abs(spacing - 2.0) > 0.05:
        print(f"resampling {path} from {spacing:.2f} m to 2.00 m spacing")
        xy, values = resample(xy, values, ds=2.0)
    return pd.DataFrame({"x_m": xy[:, 0], "y_m": xy[:, 1], **values})


def track_features(df: pd.DataFrame) -> np.ndarray:
    """[curvature, -left width, right width] per point."""
    kappa = curvature(df[["x_m", "y_m"]].to_numpy(float))
    return np.column_stack((kappa, -df["w_tr_left_m"], df["w_tr_right_m"])).astype(np.float32)


def window_indices(start: int, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    history = (start + np.arange(HISTORY)) % n
    future = (start + HISTORY + np.arange(FUTURE)) % n
    return history, future, future[:TARGET]


def history_input(geo: np.ndarray, d: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """[curvature, offset, -left width, right width] per history point."""
    return np.column_stack((geo[idx, 0], d[idx], geo[idx, 1], geo[idx, 2])).astype(np.float32)


class TrackWindows(Dataset):
    def __init__(self, folder: str):
        files = sorted(Path(folder).glob("*.csv"))
        if not files:
            raise ValueError(f"no track CSVs found in {folder}")
        self.tracks = []
        for f in files:
            df = read_track(f, need_d=True)
            self.tracks.append((track_features(df), df["d_m"].to_numpy(np.float32)))
        self.ends = np.cumsum([len(d) for _, d in self.tracks])

    def __len__(self) -> int:
        return int(self.ends[-1])

    def __getitem__(self, i: int):
        track = int(np.searchsorted(self.ends, i, side="right"))
        start = i - (self.ends[track - 1] if track else 0)
        geo, d = self.tracks[track]
        h, f, t = window_indices(start, len(d))
        return (torch.from_numpy(history_input(geo, d, h)),
                torch.from_numpy(geo[f]),
                torch.from_numpy(d[t]))
