# Checkpoints

`f1nn_split01_ep1500.pt` is a trained [`RacelineNet`](../f1init/model.py) (2.8 M parameters, 11 MB), trained for
1500 epochs on the 14 tracks in [`data/split_01/train/`](../data/README.md). It has never seen the three tracks in
`data/split_01/test/`.

The file is a plain PyTorch `state_dict` and runs on CPU. From the repository root:

```python
from f1init.model import load_model
model = load_model("checkpoints/f1nn_split01_ep1500.pt")
```

`scripts/predict.py` uses it by default.
