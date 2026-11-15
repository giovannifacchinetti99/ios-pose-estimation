"""
Fetch the part of SPEED+ used to train and evaluate the keypoint detector.

The images are taken as runs of the archive, which are random samples of
each domain: 8,400 synthetic images (about 2 GB, split by the labels into
training and validation), all 2,791 sunlamp images (90 MB) and 2,000
lightbox images (0.7 GB). With four requests at a time it took six
minutes on the connection where it was written; images already in the
local folder are not fetched again.

Run from the root of the repository:

    python examples/03_keypoint_detector/fetch_data.py
"""

import time
from pathlib import Path

from ios_pose_estimation.speedplus import SpeedPlus

ROOT = Path(__file__).resolve().parents[2] / "data" / "speedplus"
RUNS = {"synthetic": 8400, "sunlamp": 2791, "lightbox": 2000}

if __name__ == "__main__":
    dataset = SpeedPlus(ROOT)
    for domain, count in RUNS.items():
        start = time.time()
        names = dataset.prefetch(domain, count)
        print(f"{domain}: {len(names)} images in {time.time() - start:.0f} s", flush=True)
