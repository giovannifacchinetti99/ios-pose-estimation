"""
Train the keypoint detector on the synthetic images fetched by
fetch_data.py.

The images are split as in SPEED+, by whether their label is in
train.json or in validation.json. The training sources are prepared once
and cached in data/cache; the weights with the lowest validation error
and one line of metrics per epoch are written to data/models. On a
laptop processor an epoch over about 6,700 images takes some ten
minutes.

Run from the root of the repository:

    python examples/03_keypoint_detector/train.py --epochs 20
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from ios_pose_estimation.detector import KeypointNet, TrainingSources, prepare_sources, train
from ios_pose_estimation.speedplus import SpeedPlus

ROOT = Path(__file__).resolve().parents[2] / "data"


def labeled_images(dataset, split):
    """Labels of a synthetic split whose image is in the local folder."""
    folder = dataset.root / "synthetic" / "images"
    return [label for label in dataset.labels("synthetic", split) if (folder / label.filename).exists()]


def sources_for(dataset, camera, split):
    cache = ROOT / "cache" / f"synthetic_{split}.npz"
    if cache.exists():
        return TrainingSources.load(cache)
    labels = labeled_images(dataset, split)
    start = time.time()
    sources = prepare_sources(dataset, camera, labels,
                              progress=lambda n: print(f"  {split}: {n}/{len(labels)} images", flush=True))
    print(f"{split}: {len(labels)} images prepared in {time.time() - start:.0f} s", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    sources.save(cache)
    return sources


def merge(first, second):
    return TrainingSources(
        filenames=first.filenames + second.filenames,
        keypoints=np.concatenate([first.keypoints, second.keypoints]),
        fine=np.concatenate([first.fine, second.fine]),
        fine_transforms=np.concatenate([first.fine_transforms, second.fine_transforms]),
        coarse=np.concatenate([first.coarse, second.coarse]),
        coarse_transform=first.coarse_transform,
        distances=np.concatenate([first.distances, second.distances]),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--validation", type=int, default=500,
                        help="validation images measured after every epoch")
    arguments = parser.parse_args()

    dataset = SpeedPlus(ROOT / "speedplus", remote=False)
    camera = dataset.camera()
    training = sources_for(dataset, camera, "train")
    validation = sources_for(dataset, camera, "validation")
    sources = merge(training, validation)
    train_indices = np.arange(len(training))
    validation_indices = len(training) + np.arange(min(arguments.validation, len(validation)))
    print(f"training on {len(train_indices)} images, validating on {len(validation_indices)}", flush=True)

    models = ROOT / "models"
    models.mkdir(parents=True, exist_ok=True)
    log = models / "detector_log.jsonl"
    log.unlink(missing_ok=True)
    torch.manual_seed(0)
    history = train(KeypointNet(pretrained=True), sources, train_indices, validation_indices,
                    epochs=arguments.epochs, batch_size=arguments.batch, log_path=log,
                    checkpoint_path=models / "detector.pt")
    for entry in history:
        print(entry, flush=True)
