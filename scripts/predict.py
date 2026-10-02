"""Predict the raceline of a track, one lap, recursive or non-recursive.

    python scripts/predict.py --track data/split_01/test/barcelona.csv --mode recursive --plot
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from f1init.dataset import TARGET, history_input, read_track, track_features, window_indices  # noqa: E402
from f1init.geometry import arc_length, tangents_normals  # noqa: E402
from f1init.model import load_model  # noqa: E402


def rollout(model, geo: np.ndarray, d_true: np.ndarray | None) -> np.ndarray:
    """Recursive (history = own predictions) if d_true is None, otherwise history = d_true."""
    n = len(geo)
    d_pred = np.zeros(n, dtype=np.float32)
    with torch.no_grad():
        for k in range(math.ceil(n / TARGET)):
            h, f, t = window_indices(k * TARGET, n)
            past = history_input(geo, d_pred if d_true is None else d_true, h)
            out = model(torch.from_numpy(past)[None], torch.from_numpy(geo[f])[None])
            d_pred[t] = out[0].numpy()
    return d_pred


def closed(a: np.ndarray) -> np.ndarray:
    return np.vstack((a, a[:1]))


def plot(df: pd.DataFrame, n: np.ndarray, title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xy = df[["x_m", "y_m"]].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 8))
    for w, sign in (("w_tr_right_m", 1), ("w_tr_left_m", -1)):
        ax.plot(*closed(xy + sign * df[w].to_numpy()[:, None] * n).T, color="black", lw=0.8)
    ax.plot(*closed(xy).T, color="grey", lw=0.5, label="centerline")
    if "d_m" in df:
        ax.plot(*closed(xy + df["d_m"].to_numpy()[:, None] * n).T, "b--", lw=1.0, label="F1 raceline")
    ax.plot(*closed(df[["raceline_x_m", "raceline_y_m"]].to_numpy()).T, "r", lw=1.0, label="predicted")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="best")
    ax.set_title(title)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--track", required=True,
                   help="CSV with x_m, y_m, w_tr_right_m, w_tr_left_m (points in driving direction)")
    p.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "f1nn_split01_ep1500.pt"),
                   help="trained model (default: the released split_01 model)")
    p.add_argument("--mode", choices=["recursive", "non-recursive"], default="recursive")
    p.add_argument("--out", help="output CSV (default: <track name>_nn.csv)")
    p.add_argument("--plot", action="store_true", help="also save a PNG next to the output CSV")
    args = p.parse_args()

    try:
        df = read_track(args.track, need_d=args.mode == "non-recursive")
    except ValueError as e:
        raise SystemExit(e)
    geo = track_features(df)
    model = load_model(args.checkpoint)
    d_true = df["d_m"].to_numpy(np.float32) if args.mode == "non-recursive" else None
    d_pred = rollout(model, geo, d_true)

    xy = df[["x_m", "y_m"]].to_numpy()
    _, n = tangents_normals(xy)
    out = pd.DataFrame({
        "s_m": arc_length(xy)[0],
        "x_m": xy[:, 0], "y_m": xy[:, 1],
        "w_tr_right_m": df["w_tr_right_m"], "w_tr_left_m": df["w_tr_left_m"],
        "kappa_radpm": geo[:, 0],
        "d_pred_m": d_pred,
        "raceline_x_m": xy[:, 0] + d_pred * n[:, 0],
        "raceline_y_m": xy[:, 1] + d_pred * n[:, 1],
    })
    out_path = Path(args.out or f"{Path(args.track).stem}_nn.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, float_format="%.6f")
    print(f"wrote {out_path} ({len(out)} points, {args.mode})")
    if "d_m" in df:
        print(f"mean |d_pred - d_m| = {np.mean(np.abs(d_pred - df['d_m'].to_numpy())):.2f} m")
    if args.plot:
        png = out_path.with_suffix(".png")
        plot(pd.concat([df, out[["raceline_x_m", "raceline_y_m"]]], axis=1), n,
             f"{Path(args.track).stem} ({args.mode})", png)
        print(f"wrote {png}")


if __name__ == "__main__":
    main()
