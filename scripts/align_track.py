"""Align downloaded laps to a track centerline and average them into a raceline.

    python scripts/align_track.py --track data/split_01/train/monza.csv --telemetry telemetry/monza \\
        --out monza_aligned.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from f1init.align import align, apply, load_lap, mean_offsets  # noqa: E402
from f1init.dataset import read_track  # noqa: E402
from f1init.geometry import arc_length, curvature, tangents_normals  # noqa: E402


def plot(center, widths, raw_laps, aligned_laps, d, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, n = tangents_normals(center)

    def closed(a: np.ndarray) -> np.ndarray:
        return np.vstack((a, a[:1]))

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    raw_shift = center.mean(axis=0) - np.vstack(raw_laps).mean(axis=0)
    for ax, laps, title in ((axes[0], [lap + raw_shift for lap in raw_laps], "before alignment"),
                            (axes[1], aligned_laps, "after alignment")):
        for w, sign in (("w_tr_right_m", 1), ("w_tr_left_m", -1)):
            ax.plot(*closed(center + sign * widths[w][:, None] * n).T, color="black", lw=0.8)
        for lap in laps:
            ax.plot(*lap.T, ".", ms=0.8, color="tab:orange", alpha=0.5)
        if ax is axes[1]:
            ax.plot(*closed(center + d[:, None] * n).T, "r", lw=1.0, label="raceline")
            ax.legend(loc="best")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--track", required=True,
                   help="CSV with x_m, y_m, w_tr_right_m, w_tr_left_m (points in driving direction)")
    p.add_argument("--telemetry", required=True, help="folder with lap*.csv from download_telemetry.py")
    p.add_argument("--out", required=True, help="output track CSV with the raceline offset d_m")
    p.add_argument("--plot", action="store_true", help="also save a before/after PNG next to the output CSV")
    args = p.parse_args()

    try:
        track = read_track(args.track)
    except ValueError as e:
        raise SystemExit(e)
    center = track[["x_m", "y_m"]].to_numpy()
    widths = {c: track[c].to_numpy() for c in ("w_tr_right_m", "w_tr_left_m")}
    laps = []
    for f in sorted(Path(args.telemetry).glob("lap*.csv")):
        try:
            laps.append(load_lap(f))
        except ValueError as e:
            print(f"skipping {f.name}: {e}")
    if not laps:
        raise SystemExit(f"no usable lap*.csv in {args.telemetry}")

    R, T, err = align(laps, center)
    angle = np.degrees(np.arctan2(R[0, 1], R[0, 0]))
    print(f"{len(laps)} laps aligned: rotation {angle:.2f} deg, translation ({T[0]:.1f}, {T[1]:.1f}) m, "
          f"mean distance to centerline {err:.2f} m")
    aligned = [apply(lap, R, T) for lap in laps]
    try:
        d = mean_offsets(aligned, center)
    except ValueError as e:
        raise SystemExit(e)

    out = pd.DataFrame({
        "s_m": arc_length(center)[0],
        "x_m": center[:, 0], "y_m": center[:, 1],
        "kappa_radpm": curvature(center),
        "w_tr_right_m": widths["w_tr_right_m"], "w_tr_left_m": widths["w_tr_left_m"],
        "d_m": d,
    })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, float_format="%.6f")
    print(f"wrote {args.out} ({len(out)} points)")
    if args.plot:
        png = Path(args.out).with_suffix(".png")
        plot(center, widths, laps, aligned, d, png)
        print(f"wrote {png}")


if __name__ == "__main__":
    main()
