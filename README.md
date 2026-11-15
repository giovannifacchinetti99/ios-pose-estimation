# ios-pose-estimation

Monocular **pose estimation of a target spacecraft** for rendezvous and docking: from an image of a known target, recover its position and orientation with respect to the camera, with an honest estimate of the error.

**[>> Read the documentation](https://giovannifacchinetti99.github.io/ios-pose-estimation/)**

The project is the vision part of a docking sensor subsystem made of a camera, an illuminator and this software. It grew out of [ios-mission-planner](https://github.com/giovannifacchinetti99/ios-mission-planner), whose closed-loop simulations showed why it is needed: with a sensor that measures only where the target is, a docking fails in half of the cases as soon as the target does not hold its attitude. The book starts from that problem and builds the software one layer at a time.

## Data

The methods are developed and measured on [SPEED+](https://purl.stanford.edu/wv398fc4383) (Park, Märtens, Lecuyer, Izzo, D'Amico, 2021), the dataset of ESA's Satellite Pose Estimation Competition 2021, distributed under the CC BY-NC-SA 4.0 license. It is not included in this repository. `ios_pose_estimation.speedplus.SpeedPlus` fetches the files it needs from the public archive on demand, one at a time, into a local `data/` folder, so the 17 GB archive does not have to be downloaded.

## Installation

```bash
git clone https://github.com/giovannifacchinetti99/ios-pose-estimation.git
cd ios-pose-estimation

conda env create -f environment.yml
conda activate ios-pose-estimation
```

or, with any Python 3.11 or later, `pip install -e ".[test]"`. The keypoint detector needs PyTorch, which the conda environment installs in its build for processors; with pip, add the `train` extra.

## The keypoint detector

The detector is a small heatmap network, trained on the synthetic images of SPEED+ on a laptop processor in a few hours. Two scripts reproduce it from the root of the repository:

```bash
python examples/03_keypoint_detector/fetch_data.py   # about 3 GB of images, a few minutes
python examples/03_keypoint_detector/train.py        # writes data/models/detector.pt
```

The trained weights are not distributed, since they derive from a dataset licensed for non-commercial use with share-alike terms.

## Testing

```bash
pytest
```

The tests do not access the network; those of the detector are skipped when PyTorch is not installed.

## Usage

```python
from ios_pose_estimation.speedplus import SpeedPlus, TANGO_KEYPOINTS
from ios_pose_estimation.camera import project_pose

dataset = SpeedPlus("data/speedplus")        # files are fetched on first use
camera = dataset.camera()
label = dataset.labels("synthetic", "train")[0]

pixels, depth = project_pose(camera, label.quaternion, label.translation, TANGO_KEYPOINTS)

# with a trained detector: keypoints, then pose with covariance by RANSAC
from ios_pose_estimation.detector import KeypointDetector

detector = KeypointDetector.load("data/models/detector.pt")
estimate, detection = detector.pose(camera, dataset.image("synthetic", label.filename))
```
