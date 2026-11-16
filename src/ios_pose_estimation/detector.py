"""
A keypoint detector for the Tango spacecraft: a convolutional network
that predicts one heatmap per keypoint, its training on the synthetic
images of SPEED+, and its use in two passes, a coarse one on the whole
image that finds the target and a fine one on a crop around it.

The network is small enough to be trained on a laptop processor in a few
hours: the first three stages of a ResNet-18 pretrained on ImageNet,
which turn the image into features at one sixteenth of its resolution,
followed by a decoder that brings them back to one quarter of it,
combining them with the finer features of the earlier stages, as in a
feature pyramid. The output is 11 heatmaps of 64 x 64 for an input of
256 x 256.

Requires PyTorch and torchvision (the "train" extra of the package).
"""

import json
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ios_pose_estimation.camera import project_pose
from ios_pose_estimation.crops import (
    apply_transform,
    crop_transform,
    decode_heatmaps,
    keypoint_box,
    warp,
)
from ios_pose_estimation.pnp import ransac_pnp
from ios_pose_estimation.speedplus import TANGO_KEYPOINTS

INPUT_SIZE = 256
STRIDE = 4
HEATMAP_SIZE = INPUT_SIZE // STRIDE
SIGMA = 1.5                     # of the target bumps [px of the heatmap]
FINE_SCALE = 1.4                # side of the fine crop over the side of the keypoint box
FINE_SCALE_RANGE = (1.2, 1.6)   # in training
IMAGE_SIZE = (1920, 1200)
IMAGE_CENTER = np.array([0.5 * (IMAGE_SIZE[0] - 1), 0.5 * (IMAGE_SIZE[1] - 1)])
COARSE_SIDE = float(IMAGE_SIZE[0])
FOREGROUND_WEIGHT = 20.0        # extra weight of the pixels near a keypoint in the loss


def heatmap_to_input(points):
    """Pixels of the heatmap to pixels of the input: pixel h covers inputs 4h ... 4h + 3."""
    return STRIDE * np.asarray(points) + 0.5 * (STRIDE - 1)


def input_to_heatmap(points):
    return (np.asarray(points) - 0.5 * (STRIDE - 1)) / STRIDE


def _conv_bn_relu(inputs, outputs):
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
                         nn.BatchNorm2d(outputs), nn.ReLU(inplace=True))


class KeypointNet(nn.Module):
    """
    Heatmap network: ResNet-18 encoder truncated after its third stage and
    a two-step upsampling decoder with lateral connections.

    Parameters
    ----------
    keypoints : int
        Number of heatmaps.
    pretrained : bool
        Whether to start the encoder from the ImageNet weights of
        torchvision.

    Author: Giovanni Facchinetti, 2026
    Reference: He, Zhang, Ren, Sun, "Deep Residual Learning for Image
    Recognition," CVPR 2016; Lin et al., "Feature Pyramid Networks for
    Object Detection," CVPR 2017
    """

    def __init__(self, keypoints=len(TANGO_KEYPOINTS), pretrained=True):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1, self.layer2, self.layer3 = backbone.layer1, backbone.layer2, backbone.layer3
        self.lateral3 = nn.Conv2d(256, 128, 1)
        self.lateral2 = nn.Conv2d(128, 128, 1)
        self.lateral1 = nn.Conv2d(64, 128, 1)
        self.smooth2 = _conv_bn_relu(128, 128)
        self.smooth1 = _conv_bn_relu(128, 64)
        self.head = nn.Sequential(_conv_bn_relu(64, 64), nn.Conv2d(64, keypoints, 1))

    def forward(self, images):
        """images: (batch, 1, 256, 256) gray levels in [0, 1]; returns (batch, keypoints, 64, 64)."""

        x = ((images - 0.449) / 0.226).expand(-1, 3, -1, -1)
        f1 = self.layer1(self.stem(x))
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        p = F.interpolate(self.lateral3(f3), scale_factor=2.0, mode="bilinear", align_corners=False)
        p = self.smooth2(p + self.lateral2(f2))
        p = F.interpolate(p, scale_factor=2.0, mode="bilinear", align_corners=False)
        p = self.smooth1(p + self.lateral1(f1))
        return self.head(p)


# ---------------------------------------------------------------------------
# training data

@dataclass
class TrainingSources:
    """
    Images prepared for training, kept in memory.

    For every labeled image there is a fine source, a square of
    fine_margin times the keypoint box around the target resampled to
    fine_size pixels, and a coarse source, the whole image reduced to a
    thumbnail. The crops seen by the network are cut from these at random
    scales, positions and rotations, which is cheap enough to do for
    every batch.

    Attributes
    ----------
    filenames : list of str
    keypoints : numpy.ndarray, shape (n, 11, 2)
        Keypoints in the pixels of the original image, with distortion.
    fine : numpy.ndarray, shape (n, fine_size, fine_size), uint8
    fine_transforms : numpy.ndarray, shape (n, 3, 3)
        From the original image to each fine source.
    coarse : numpy.ndarray, shape (n, height, width), uint8
    coarse_transform : numpy.ndarray, shape (3, 3)
        From the original image to the thumbnails.
    distances : numpy.ndarray, shape (n,)
        Distance of the target [m].
    """

    filenames: list
    keypoints: np.ndarray
    fine: np.ndarray
    fine_transforms: np.ndarray
    coarse: np.ndarray
    coarse_transform: np.ndarray
    distances: np.ndarray

    def __len__(self):
        return len(self.filenames)

    def save(self, path):
        np.savez(path, filenames=np.array(self.filenames), keypoints=self.keypoints, fine=self.fine,
                 fine_transforms=self.fine_transforms, coarse=self.coarse,
                 coarse_transform=self.coarse_transform, distances=self.distances)

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            return cls(filenames=list(data["filenames"]), keypoints=data["keypoints"], fine=data["fine"],
                       fine_transforms=data["fine_transforms"], coarse=data["coarse"],
                       coarse_transform=data["coarse_transform"], distances=data["distances"])


def prepare_sources(dataset, camera, labels, domain="synthetic", fine_size=512, fine_margin=2.5,
                    coarse_width=320, progress=None):
    """
    Build the training sources of a list of labels.

    Parameters
    ----------
    dataset : ios_pose_estimation.speedplus.SpeedPlus
    camera : ios_pose_estimation.camera.PinholeCamera
    labels : list of PoseLabel
    progress : callable or None
        Called with the number of images done, every 500 images.
    """

    scale = coarse_width / IMAGE_SIZE[0]
    coarse_transform = np.array([[scale, 0.0, 0.5 * scale - 0.5],
                                 [0.0, scale, 0.5 * scale - 0.5],
                                 [0.0, 0.0, 1.0]])
    coarse_size = (coarse_width, int(round(IMAGE_SIZE[1] * scale)))

    n = len(labels)
    keypoints = np.empty((n, len(TANGO_KEYPOINTS), 2))
    fine = np.empty((n, fine_size, fine_size), dtype=np.uint8)
    fine_transforms = np.empty((n, 3, 3))
    coarse = np.empty((n, coarse_size[1], coarse_size[0]), dtype=np.uint8)
    for i, label in enumerate(labels):
        image = dataset.image(domain, label.filename)
        pixels, _ = project_pose(camera, label.quaternion, label.translation, TANGO_KEYPOINTS)
        center, side = keypoint_box(pixels)
        transform = crop_transform(center, fine_margin * side, fine_size)
        keypoints[i] = pixels
        fine[i] = warp(image, transform, fine_size)
        fine_transforms[i] = transform
        coarse[i] = warp(image, coarse_transform, coarse_size)
        if progress is not None and (i + 1) % 500 == 0:
            progress(i + 1)

    return TrainingSources(filenames=[label.filename for label in labels], keypoints=keypoints, fine=fine,
                           fine_transforms=fine_transforms, coarse=coarse, coarse_transform=coarse_transform,
                           distances=np.array([label.distance for label in labels]))


def sample_crops(sources, indices, rng, coarse_fraction=0.25, augment=True):
    """
    Choose the crop of each sample of a batch.

    A fine crop is centered on the keypoint box, shifted at random by up
    to a tenth of its side, with a side between 1.2 and 1.6 times the box,
    which covers the error of the box found by the coarse pass. A coarse
    crop is the whole image, slightly zoomed and shifted. Both are turned
    by a random angle: a turn about the line of sight gives an image of
    another valid pose. Without augmentation every crop is the one used
    at inference.

    Returns
    -------
    transforms : numpy.ndarray, shape (batch, 3, 3)
        From the original image to the input of the network.
    coarse : numpy.ndarray, shape (batch,), bool
    """

    batch = len(indices)
    transforms = np.empty((batch, 3, 3))
    coarse = rng.random(batch) < coarse_fraction if augment else np.zeros(batch, dtype=bool)
    for b, index in enumerate(indices):
        if coarse[b]:
            center = IMAGE_CENTER + rng.uniform(-100.0, 100.0, 2)
            side = COARSE_SIDE * rng.uniform(0.95, 1.2)
        else:
            center, side0 = keypoint_box(sources.keypoints[index])
            if augment:
                center = center + side0 * rng.uniform(-0.1, 0.1, 2)
                side = side0 * rng.uniform(*FINE_SCALE_RANGE)
            else:
                side = side0 * FINE_SCALE
        angle = rng.uniform(-np.pi, np.pi) if augment else 0.0
        transforms[b] = crop_transform(center, side, INPUT_SIZE, angle)
    return transforms, coarse


def _normalizer(width, height):
    """From pixels to the [-1, 1] coordinates of grid_sample with align_corners=True."""
    return np.array([[2.0 / (width - 1), 0.0, -1.0], [0.0, 2.0 / (height - 1), -1.0], [0.0, 0.0, 1.0]])


def _resample(stack, source_transforms, transforms):
    """Cut the crops of a batch from their sources with grid_sample."""

    sources = torch.from_numpy(stack).float().unsqueeze(1) / 255.0
    height, width = stack.shape[1:]
    output = np.linalg.inv(_normalizer(INPUT_SIZE, INPUT_SIZE))
    theta = _normalizer(width, height) @ source_transforms @ np.linalg.inv(transforms) @ output
    grid = F.affine_grid(torch.from_numpy(theta[:, :2]).float(), (len(stack), 1, INPUT_SIZE, INPUT_SIZE),
                         align_corners=True)
    return F.grid_sample(sources, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def _photometric(images, generator):
    """
    Random changes of exposure, contrast, gamma, blur and noise.

    They make the network rely on the shape of the target rather than on
    the exact gray levels of the renderer, the first defence against the
    gap between rendered and real images.
    """

    batch = images.shape[0]

    def uniform(low, high):
        return low + (high - low) * torch.rand(batch, 1, 1, 1, generator=generator)

    images = images.clamp(0.0, 1.0) ** uniform(0.6, 1.6)
    images = images * uniform(0.5, 1.5) + uniform(-0.05, 0.1)

    sigma = uniform(0.4, 1.6).view(batch)
    blurred = torch.rand(batch, generator=generator) < 0.4
    if blurred.any():
        offsets = torch.arange(-3, 4, dtype=torch.float32)
        kernels = torch.exp(-offsets[None, :] ** 2 / (2.0 * sigma[blurred, None] ** 2))
        kernels = kernels / kernels.sum(dim=1, keepdim=True)
        chosen = images[blurred].transpose(0, 1)                      # (1, m, H, W)
        m = kernels.shape[0]
        chosen = F.conv2d(chosen, kernels.view(m, 1, 1, 7), padding=(0, 3), groups=m)
        chosen = F.conv2d(chosen, kernels.view(m, 1, 7, 1), padding=(3, 0), groups=m)
        images = images.clone()
        images[blurred] = chosen.transpose(0, 1)

    images = images + uniform(0.0, 0.05) * torch.randn(images.shape, generator=generator)
    return images.clamp(0.0, 1.0)


def make_batch(sources, indices, rng, generator=None, coarse_fraction=0.25, augment=True):
    """
    Inputs, target heatmaps and target weights of a batch.

    The weight of a keypoint is zero when it falls outside the crop, so
    that the network is not asked to place it.

    Returns
    -------
    images : torch.Tensor, shape (batch, 1, 256, 256)
    targets : torch.Tensor, shape (batch, 11, 64, 64)
    weights : torch.Tensor, shape (batch, 11)
    transforms : numpy.ndarray, shape (batch, 3, 3)
    """

    indices = np.asarray(indices)
    transforms, coarse = sample_crops(sources, indices, rng, coarse_fraction, augment)
    images = torch.empty(len(indices), 1, INPUT_SIZE, INPUT_SIZE)
    for chosen, stack, source_transforms in [
        (~coarse, sources.fine, sources.fine_transforms[indices]),
        (coarse, sources.coarse, np.broadcast_to(sources.coarse_transform, (len(indices), 3, 3))),
    ]:
        if chosen.any():
            images[torch.from_numpy(chosen)] = _resample(stack[indices[chosen]], source_transforms[chosen],
                                                         transforms[chosen])
    if augment:
        images = _photometric(images, generator)

    points = np.stack([input_to_heatmap(apply_transform(t, k))
                       for t, k in zip(transforms, sources.keypoints[indices])])
    grid = torch.arange(HEATMAP_SIZE, dtype=torch.float32)
    p = torch.from_numpy(points).float()
    dx = grid[None, None, None, :] - p[:, :, 0, None, None]
    dy = grid[None, None, :, None] - p[:, :, 1, None, None]
    targets = torch.exp(-(dx ** 2 + dy ** 2) / (2.0 * SIGMA ** 2))
    inside = (p >= -0.5).all(dim=2) & (p <= HEATMAP_SIZE - 0.5).all(dim=2)
    return images, targets, inside.float(), transforms


def heatmap_loss(predicted, targets, weights, foreground=FOREGROUND_WEIGHT):
    """
    Mean squared difference of the heatmaps, keypoints outside the crop
    excluded.

    The bump of a keypoint covers a dozen of the 4096 pixels of its
    heatmap, so a plain mean squared difference is already small for a
    network that predicts zero everywhere, and training tends to settle
    there first. Each pixel is therefore weighted by 1 + foreground times
    its target value, which makes the pixels near the keypoint count as
    much as the whole background.
    """

    squared = ((predicted - targets) ** 2 * (1.0 + foreground * targets)).mean(dim=(2, 3))
    return 1000.0 * (squared * weights).sum() / weights.sum().clamp(min=1.0)


def decode_batch(heatmaps, transforms):
    """
    Keypoints of a batch in the pixels of the original images.

    Returns
    -------
    points : numpy.ndarray, shape (batch, 11, 2)
    peaks : numpy.ndarray, shape (batch, 11)
    """

    heatmaps = heatmaps.detach().cpu().numpy()
    points, peaks = [], []
    for maps, transform in zip(heatmaps, transforms):
        decoded, peak = decode_heatmaps(maps)
        points.append(apply_transform(np.linalg.inv(transform), heatmap_to_input(decoded)))
        peaks.append(peak)
    return np.array(points), np.array(peaks)


@torch.no_grad()
def keypoint_errors(model, sources, indices, batch_size=64, coarse=False):
    """
    Distance [px of the original image] between the detected and the true
    keypoints, on the crops used at inference with the true box (fine) or
    on the whole image (coarse).

    Returns
    -------
    numpy.ndarray, shape (len(indices), 11)
    """

    model.eval()
    rng = np.random.default_rng(0)
    errors = []
    for start in range(0, len(indices), batch_size):
        chunk = np.asarray(indices[start:start + batch_size])
        if coarse:
            transforms = np.broadcast_to(crop_transform(IMAGE_CENTER, COARSE_SIDE, INPUT_SIZE), (len(chunk), 3, 3))
            images = _resample(sources.coarse[chunk], np.broadcast_to(sources.coarse_transform, transforms.shape),
                               transforms)
        else:
            images, _, _, transforms = make_batch(sources, chunk, rng, augment=False)
        points, _ = decode_batch(model(images), transforms)
        errors.append(np.linalg.norm(points - sources.keypoints[chunk], axis=2))
    return np.concatenate(errors)


def train(model, sources, train_indices, validation_indices, epochs=20, batch_size=32, learning_rate=1e-3,
          weight_decay=1e-4, coarse_fraction=0.25, log_path=None, checkpoint_path=None, seed=0):
    """
    Train the network on random crops of the training sources.

    The learning rate follows one cycle, a warm-up over the first tenth of
    the steps and a cosine decay to zero. After every epoch the median
    keypoint error on the validation images is measured, on the fine
    crops of the true box and on the whole image, written to the log, one
    JSON line per epoch, and the weights with the lowest fine error are
    saved.

    Returns
    -------
    list of dict
        The log.

    Author: Giovanni Facchinetti, 2026
    """

    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    steps_per_epoch = len(train_indices) // batch_size
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, pct_start=0.1,
                                                    total_steps=epochs * steps_per_epoch)
    history, best = [], np.inf
    for epoch in range(epochs):
        model.train()
        start, losses = time.time(), []
        order = rng.permutation(train_indices)
        for step in range(steps_per_epoch):
            images, targets, weights, _ = make_batch(sources, order[step * batch_size:(step + 1) * batch_size],
                                                     rng, generator, coarse_fraction)
            loss = heatmap_loss(model(images), targets, weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

        fine = keypoint_errors(model, sources, validation_indices)
        coarse = keypoint_errors(model, sources, validation_indices, coarse=True)
        entry = {"epoch": epoch + 1, "loss": float(np.mean(losses)), "seconds": time.time() - start,
                 "fine_median_px": float(np.median(fine)), "fine_p90_px": float(np.percentile(fine, 90)),
                 "coarse_median_px": float(np.median(coarse))}
        history.append(entry)
        if log_path is not None:
            with open(log_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry) + "\n")
        if checkpoint_path is not None and entry["fine_median_px"] < best:
            best = entry["fine_median_px"]
            torch.save({"state_dict": model.state_dict(), "epoch": epoch + 1, "metrics": entry}, checkpoint_path)
    return history


# ---------------------------------------------------------------------------
# inference

@dataclass
class Detection:
    """
    Keypoints found in one image.

    Attributes
    ----------
    pixels : numpy.ndarray, shape (11, 2)
        In the pixels of the original image.
    confidence : numpy.ndarray, shape (11,)
        Heights of the heatmap peaks, near 1 for a clear detection.
    coarse_pixels : numpy.ndarray, shape (11, 2)
        Keypoints of the coarse pass, from which the crop was chosen.
    box : tuple
        Center [px] and side [px] of the fine crop.
    """

    pixels: np.ndarray
    confidence: np.ndarray
    coarse_pixels: np.ndarray = None
    box: tuple = field(default=None)


class KeypointDetector:
    """
    The trained network applied to whole images in two passes.

    The coarse pass sees the whole image reduced to 256 pixels across,
    where the target is a few tens of pixels wide; its keypoints are
    rough but enough to place a box around the target. The fine pass
    sees the crop of that box, with the margin used in training, and
    gives the keypoints that are used for the pose.

    Parameters
    ----------
    model : KeypointNet
    passes : int
        Number of fine passes; each one recenters the crop on the
        keypoints of the previous one.

    Author: Giovanni Facchinetti, 2026
    """

    def __init__(self, model, passes=1):
        self.model = model.eval()
        self.passes = passes

    @classmethod
    def load(cls, path, passes=1):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = KeypointNet(pretrained=False)
        model.load_state_dict(checkpoint["state_dict"])
        return cls(model, passes)

    @torch.no_grad()
    def _run(self, image, transform):
        crop = warp(image, transform, INPUT_SIZE)
        tensor = torch.from_numpy(np.array(crop, dtype=np.float32))[None, None] / 255.0
        points, peaks = decode_batch(self.model(tensor), transform[None])
        return points[0], peaks[0]

    def detect(self, image, box=None):
        """
        Keypoints of an image.

        Parameters
        ----------
        image : numpy.ndarray, shape (1200, 1920), uint8
        box : tuple or None
            Center and side of the keypoint box, to skip the coarse pass
            (for example the true one, to measure the fine pass alone).
        """

        coarse_pixels = None
        if box is None:
            coarse_pixels, _ = self._run(image, crop_transform(IMAGE_CENTER, COARSE_SIDE, INPUT_SIZE))
            box = keypoint_box(coarse_pixels)
        center, side = box
        for _ in range(self.passes):
            crop = (np.asarray(center, dtype=float), FINE_SCALE * side)
            pixels, confidence = self._run(image, crop_transform(*crop, INPUT_SIZE))
            center, side = keypoint_box(pixels)
        return Detection(pixels=pixels, confidence=confidence, coarse_pixels=coarse_pixels, box=crop)

    def pose(self, camera, image, threshold=4.0, pixel_std=None, min_confidence=0.0, box=None, rng=None):
        """
        Pose of the target in an image: detection followed by RANSAC PnP on
        the keypoints whose confidence is at least min_confidence.

        The threshold of RANSAC and the pixel error are given in pixels of
        the input of the fine pass, in which the error of the detector does
        not depend on the distance, and converted to pixels of the image
        with the scale of the crop.

        Parameters
        ----------
        threshold : float
            Largest reprojection error of an inlier [px of the input].
        pixel_std : float or None
            Standard deviation of a keypoint coordinate [px of the input],
            for the covariance.

        Returns
        -------
        estimate : ios_pose_estimation.pnp.PoseEstimate or None
            None when fewer than six keypoints are left.
        detection : Detection
        """

        detection = self.detect(image, box)
        used = detection.confidence >= min_confidence
        if used.sum() < 6:
            return None, detection
        scale = detection.box[1] / INPUT_SIZE
        estimate = ransac_pnp(camera, TANGO_KEYPOINTS[used], detection.pixels[used], threshold=threshold * scale,
                              pixel_std=None if pixel_std is None else pixel_std * scale, rng=rng)
        inliers = np.zeros(len(used), dtype=bool)
        inliers[np.flatnonzero(used)[estimate.inliers]] = True
        estimate.inliers = inliers
        return estimate, detection
