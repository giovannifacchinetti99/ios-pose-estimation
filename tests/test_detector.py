import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ios_pose_estimation.camera import PinholeCamera, project_pose  # noqa: E402
from ios_pose_estimation.crops import apply_transform  # noqa: E402
from ios_pose_estimation.detector import (  # noqa: E402
    HEATMAP_SIZE,
    INPUT_SIZE,
    KeypointNet,
    decode_batch,
    heatmap_loss,
    make_batch,
    prepare_sources,
)
from ios_pose_estimation.speedplus import TANGO_KEYPOINTS, PoseLabel  # noqa: E402

CAMERA = PinholeCamera(fx=2988.58, fy=2988.34, cx=960.0, cy=600.0, width=1920, height=1200,
                       distortion=np.array([-0.2238, 0.5141, -0.000665, -0.000214, -0.1312]))


class SpotDataset:
    """Images with a bright square on every keypoint of the label."""

    def __init__(self, labels):
        self.labels = {label.filename: label for label in labels}

    def image(self, domain, filename):
        label = self.labels[filename]
        pixels, _ = project_pose(CAMERA, label.quaternion, label.translation, TANGO_KEYPOINTS)
        image = np.zeros((1200, 1920), dtype=np.uint8)
        for x, y in np.round(pixels).astype(int):
            image[max(y - 4, 0):y + 5, max(x - 4, 0):x + 5] = 255
        return image


def _labels(n, seed=0):
    rng = np.random.default_rng(seed)
    labels = []
    for i in range(n):
        q = rng.standard_normal(4)
        direction = np.array([*rng.uniform(-0.08, 0.08, 2), 1.0])
        t = rng.uniform(4.0, 9.0) * direction / np.linalg.norm(direction)
        labels.append(PoseLabel(f"img{i:06d}.jpg", q / np.linalg.norm(q), t))
    return labels


@pytest.fixture(scope="module")
def sources():
    labels = _labels(6)
    return prepare_sources(SpotDataset(labels), CAMERA, labels, fine_size=384)


def test_targets_decode_back_to_the_keypoints(sources):
    rng = np.random.default_rng(1)
    chosen = np.array([4, 1, 5, 2])
    _, targets, weights, transforms = make_batch(sources, chosen, rng, torch.Generator().manual_seed(0),
                                                 coarse_fraction=0.5)

    points, peaks = decode_batch(targets, transforms)

    inside = weights.numpy() > 0
    error = np.linalg.norm(points - sources.keypoints[chosen], axis=2)
    assert inside.mean() > 0.9
    assert error[inside].max() < 1e-3
    assert np.all(peaks[inside] > 0.5)


def test_crops_show_the_keypoints_where_the_transform_puts_them(sources):
    chosen = np.array([3, 0, 5])
    images, _, _, transforms = make_batch(sources, chosen, np.random.default_rng(2), augment=False)

    for image, transform, keypoints in zip(images[:, 0].numpy(), transforms, sources.keypoints[chosen]):
        for x, y in np.round(apply_transform(transform, keypoints)).astype(int):
            assert image[y, x] > 0.3


def test_network_shapes_and_loss():
    model = KeypointNet(pretrained=False)
    output = model(torch.zeros(2, 1, INPUT_SIZE, INPUT_SIZE))
    assert output.shape == (2, len(TANGO_KEYPOINTS), HEATMAP_SIZE, HEATMAP_SIZE)

    targets = torch.rand(2, 11, HEATMAP_SIZE, HEATMAP_SIZE)
    weights = torch.ones(2, 11)
    weights[0, 3] = 0.0
    assert heatmap_loss(targets, targets, weights).item() == 0.0
    wrong = targets.clone()
    wrong[0, 3] += 1.0                    # an excluded keypoint does not count
    assert heatmap_loss(wrong, targets, weights).item() == 0.0
