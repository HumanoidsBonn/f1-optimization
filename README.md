# Efficient Trajectory Optimization for Autonomous Racing via Formula-1 Data-Driven Initialization

This repository contains the official implementation for the paper:

**Efficient Trajectory Optimization for Autonomous Racing via Formula-1 Data-Driven Initialization**  
Samir Shehadeh, Lukas Kutsch, Nils Dengler, Sicong Pan, Maren Bennewitz  
[arXiv:2603.07126](https://arxiv.org/abs/2603.07126)

![Overview of the method](assets/overview.png)

*Formula 1 photograph from Pexels.*

## Overview

Trajectory optimization is a key component of autonomous racing, but practical minimum-time optimization pipelines can be highly sensitive to the initial trajectory. Poor initialization, such as using the track centerline or purely geometric baselines, may lead to slow convergence or suboptimal local solutions.

This work proposes a learning-informed initialization strategy that uses expert Formula 1 telemetry as a prior for autonomous racing trajectory optimization. The method reconstructs and aligns real-world Formula 1 racelines across multiple tracks, trains a neural network to predict expert-like raceline offsets from local track geometry, and uses the predicted raceline as an initialization seed for a minimum-time optimal control solver.

This repository contains the dataset of 17 tracks, the network with a trained model, and the scripts to download
telemetry, build a raceline from it, train, and predict. The predicted raceline (`predict.py` output) is the
initialization handed to the optimizer.

## Key Ideas

- Reconstruction of expert Formula 1 racelines from telemetry data.
- Standardized track-centered representation using centerline arc length, curvature, track boundaries, and lateral raceline offsets.
- Neural raceline prediction from local track geometry.
- Learned initialization for minimum-time trajectory optimization.
- Evaluation across 17 Formula 1 tracks.
- Hardware validation on a 1:10 RoboRacer platform.

## Method Summary

The full pipeline consists of three main stages:

1. **Formula 1 telemetry processing**  
   Real-world Formula 1 telemetry is reconstructed, aligned to track geometry, and converted into a Frenet-frame representation.

2. **Learning-based raceline prediction**  
   A neural network predicts expert-like raceline offsets from local track features such as curvature and track boundary offsets.

3. **Minimum-time trajectory optimization**  
   The predicted raceline is used as an informed initialization for a physics-based minimum-time optimal control solver.

![Network architecture](assets/architecture.png)

## Install

Tested with Python 3.12. A GPU is optional.

```bash
pip install -r requirements.txt
```

## Quick start

Predict the raceline of a track the released model has never seen, and plot it:

```bash
python scripts/predict.py --track data/split_01/test/barcelona.csv --mode recursive --plot
python scripts/predict.py --track data/split_01/test/barcelona.csv --mode non-recursive --out barcelona_nonrec_nn.csv --plot
```

Train a model on the 14 training tracks:

```bash
python scripts/train.py --config configs/train.yaml
```

Build a raceline from F1 telemetry yourself (download a driver's laps, align them to the track, average them):

```bash
python scripts/download_telemetry.py --year 2025 --event "Italian Grand Prix" --driver VER --laps 1 4-12 --out telemetry/monza
python scripts/align_track.py --track data/split_01/train/monza.csv --telemetry telemetry/monza --out monza_aligned.csv --plot
```

## Repository

| folder | contents |
|---|---|
| [`data/`](data/README.md) | 17 tracks with the expert F1 raceline, split into 14 training and 3 test tracks; sources and disclaimer |
| [`checkpoints/`](checkpoints/README.md) | the trained model for this split |
| [`scripts/`](scripts/README.md) | `predict.py`, `train.py`, `download_telemetry.py`, `align_track.py` |
| [`f1init/`](f1init/README.md) | geometry, alignment, dataset and model code used by the scripts |
| [`configs/`](configs/README.md) | training configuration |
| [`assets/`](assets/README.md) | images used in the READMEs |

## Not included yet

- Minimum-time optimization initialization interface. The optimizer is the minimum-time planner of TUM's
  [global_racetrajectory_optimization](https://github.com/TUMFTM/global_racetrajectory_optimization); we added a
  small patch that lets it start from any given line instead of only the centerline.
- RoboRacer-related validation utilities

## Citation

If you use this work, please cite the paper:

```bibtex
@inproceedings{shehadeh2026efficient,
  title     = {Efficient Trajectory Optimization for Autonomous Racing via Formula-1 Data-Driven Initialization},
  author    = {Shehadeh, Samir and Kutsch, Lukas and Dengler, Nils and Pan, Sicong and Bennewitz, Maren},
  booktitle = {IEEE International Conference on Intelligent Transportation Systems (ITSC)},
  year      = {2026}
}
```

## License

The code is released under the [MIT License](LICENSE). For the data sources and their terms, see
[`data/README.md`](data/README.md#data-sources-and-disclaimer).

## Contact

For questions, please open an issue or contact the authors directly.
