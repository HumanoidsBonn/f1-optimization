# Configs

`train.yaml` is read by `scripts/train.py`. Paths are relative to the repository root.

| field | value | meaning |
|---|---|---|
| `data` | `data/split_01/train` | folder of track CSVs to train on |
| `epochs` | 1500 | passes over the training windows (the released model was trained for 1500) |
| `batch_size` | 64 | windows per optimizer step |
| `learning_rate` | 1e-5 | Adam learning rate |
| `val_fraction` | 0.2 | share of windows held out for validation, drawn at random |
| `seed` | 0 | seed of that random split |
| `out` | `runs/split01` | output folder; the final model is saved as `model_final.pt` next to a copy of the config |

The network size is fixed in [`f1init/model.py`](../f1init/model.py) and needs no configuration.
