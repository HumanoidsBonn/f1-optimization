# f1init

The code shared by the scripts.

| module | contents | used by |
|---|---|---|
| `geometry.py` | arc length, 2 m resampling, right normals, curvature, lateral offset of a line along the normals | `dataset.py`, `align.py`, `predict.py`, `align_track.py` |
| `align.py` | cleaning FastF1 laps, aligning them to the centerline, averaging them into one raceline | `align_track.py` |
| `dataset.py` | the input features and the sliding windows (100 points history, 300 future, 10 target) | `train.py`, `predict.py`, `align_track.py` |
| `model.py` | `RacelineNet` and `load_model` | `train.py`, `predict.py` |

The scripts add the repository root to the Python path, so they run from any folder without installing a package.
