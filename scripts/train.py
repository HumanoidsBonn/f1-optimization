"""Train RacelineNet with teacher forcing.

    python scripts/train.py --config configs/train.yaml [--epochs 2]
"""
import argparse
import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, random_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from f1init.dataset import TrackWindows  # noqa: E402
from f1init.model import RacelineNet  # noqa: E402


def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mae = (pred - target).abs().mean()
    cos = F.cosine_similarity(pred, target, dim=1).mean()
    return 0.5 * (1.0 - cos) + 0.5 * mae, mae


def run_epoch(model, loader, device, optimizer=None) -> tuple[float, float]:
    model.train(optimizer is not None)
    total_loss = total_mae = 0.0
    with torch.set_grad_enabled(optimizer is not None):
        for past, future, target in loader:
            past, future, target = past.to(device), future.to(device), target.to(device)
            loss, mae = loss_fn(model(past, future), target)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(target)
            total_mae += mae.item() * len(target)
    return total_loss / len(loader.dataset), total_mae / len(loader.dataset)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--epochs", type=int, help="override the number of epochs in the config")
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    epochs = cfg["epochs"] if args.epochs is None else args.epochs

    try:
        data = TrackWindows(ROOT / cfg["data"])
    except ValueError as e:
        raise SystemExit(e)
    n_val = int(len(data) * cfg["val_fraction"])
    train_set, val_set = random_split(data, [len(data) - n_val, n_val],
                                      generator=torch.Generator().manual_seed(cfg["seed"]))
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"], num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RacelineNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    print(f"{len(train_set)} training and {len(val_set)} validation windows, device {device}")

    for epoch in range(1, epochs + 1):
        train_loss, _ = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_mae = run_epoch(model, val_loader, device)
        print(f"epoch {epoch:4d}  train loss {train_loss:.4f}  val loss {val_loss:.4f}  val MAE {val_mae:.3f} m")

    out = ROOT / cfg["out"]
    out.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out / "model_final.pt")
    shutil.copy(args.config, out / "train.yaml")
    print(f"saved {out / 'model_final.pt'}")


if __name__ == "__main__":
    main()
