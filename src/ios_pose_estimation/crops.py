"""
Crops around the target and heatmaps of its keypoints.

A keypoint detector does not look at the whole image. The target may
cover a few percent of it at 10 m and all of it at 2 m, and a network
sees a fixed number of pixels. The image is therefore cut around the
target and resampled to a square of fixed size, so that the target
always appears at about the same scale. The cut is a similarity
transform, a scale, a rotation and a shift, from the pixels of the image
to the pixels of the crop, and the keypoints found in the crop are taken
back to the image by its inverse.

Coordinates follow the convention of arrays: the center of the pixel in
row i and column j is at (x, y) = (j, i).

The network gives, for each keypoint, a heatmap: an image whose value
is high where the keypoint is likely to be. It is trained to reproduce a
Gaussian bump centered on the true position, and the position is read
back from the peak of the bump with sub-pixel precision.
"""

import numpy as np


def keypoint_box(pixels):
    """
    Square box around a set of points.

    Returns
    -------
    center : numpy.ndarray, shape (2,)
        Center of the bounding rectangle of the points [px].
    side : float
        Larger side of the rectangle [px].
    """

    pixels = np.asarray(pixels, dtype=float)
    low, high = pixels.min(axis=0), pixels.max(axis=0)
    return 0.5 * (low + high), float(np.max(high - low))


def crop_transform(center, side, size, angle=0.0):
    """
    Transform from the pixels of an image to the pixels of a crop.

    The square of the given side centered on `center`, turned by `angle`,
    is mapped onto a crop of size x size pixels.

    Parameters
    ----------
    center : array_like, shape (2,)
        Center of the square in the image [px].
    side : float
        Side of the square in the image [px].
    size : int
        Side of the crop [px].
    angle : float
        Rotation of the content of the crop [rad]: a point of the image
        appears in the crop turned by this angle about the center.

    Returns
    -------
    numpy.ndarray, shape (3, 3)
        Homogeneous transform: crop = T @ [x, y, 1].
    """

    scale = size / side
    c, s = np.cos(angle), np.sin(angle)
    linear = scale * np.array([[c, -s], [s, c]])
    transform = np.eye(3)
    transform[:2, :2] = linear
    transform[:2, 2] = 0.5 * (size - 1) - linear @ np.asarray(center, dtype=float)
    return transform


def apply_transform(transform, points):
    """Points (n, 2) mapped by a homogeneous transform (3, 3)."""

    points = np.asarray(points, dtype=float)
    return points @ transform[:2, :2].T + transform[:2, 2]


def warp(image, transform, size, antialias=True):
    """
    Resample an image through a transform.

    Parameters
    ----------
    image : numpy.ndarray, shape (height, width), dtype uint8
    transform : numpy.ndarray, shape (3, 3)
        From the pixels of `image` to the pixels of the output.
    size : int or (int, int)
        Side of the output, or (width, height).
    antialias : bool
        When the output is much smaller than the region it covers, the
        image is first reduced by box averaging, so that the bilinear
        resampling does not alias fine detail.

    Returns
    -------
    numpy.ndarray, shape (height, width), dtype uint8
        Pixels outside the image are zero.
    """

    from PIL import Image

    width, height = (size, size) if np.isscalar(size) else size
    transform = np.asarray(transform, dtype=float)
    picture = Image.fromarray(np.asarray(image, dtype=np.uint8))

    scale = np.sqrt(abs(np.linalg.det(transform[:2, :2])))
    factor = int(np.floor(1.0 / scale)) if antialias else 1
    if factor >= 2:
        # pixel j of the reduced image averages pixels factor*j ... factor*j + factor - 1
        picture = picture.reduce(factor)
        reduction = np.array([[1.0 / factor, 0.0, 0.5 / factor - 0.5],
                              [0.0, 1.0 / factor, 0.5 / factor - 0.5],
                              [0.0, 0.0, 1.0]])
        transform = transform @ np.linalg.inv(reduction)

    # PIL maps the continuous coordinates of the output, in which the center
    # of pixel j is at j + 0.5, to those of the input
    inverse = np.linalg.inv(transform)
    shift = np.array([[1.0, 0.0, -0.5], [0.0, 1.0, -0.5], [0.0, 0.0, 1.0]])
    pil = np.linalg.inv(shift) @ inverse @ shift
    coefficients = tuple(pil[:2].ravel())
    output = picture.transform((width, height), Image.AFFINE, coefficients, resample=Image.BILINEAR)
    return np.asarray(output)


def gaussian_heatmaps(points, size, sigma):
    """
    Target heatmaps: one Gaussian bump per point.

    Parameters
    ----------
    points : numpy.ndarray, shape (n, 2)
        Positions in the pixels of the heatmap.
    size : int
        Side of the heatmaps.
    sigma : float
        Standard deviation of the bumps [px of the heatmap].

    Returns
    -------
    numpy.ndarray, shape (n, size, size)
    """

    grid = np.arange(size, dtype=float)
    dx = grid[None, None, :] - np.asarray(points)[:, 0, None, None]
    dy = grid[None, :, None] - np.asarray(points)[:, 1, None, None]
    return np.exp(-(dx ** 2 + dy ** 2) / (2.0 * sigma ** 2))


def decode_heatmaps(heatmaps):
    """
    Position and height of the peak of each heatmap.

    The peak is the pixel of largest value, refined to sub-pixel
    precision by fitting a parabola, along each axis, to the logarithm of
    the values at the peak and at its two neighbours. For a Gaussian bump
    the logarithm is exactly a parabola, so the refinement recovers the
    center of the bump between pixels.

    Parameters
    ----------
    heatmaps : numpy.ndarray, shape (n, height, width)

    Returns
    -------
    points : numpy.ndarray, shape (n, 2)
        Positions of the peaks [px of the heatmap].
    peaks : numpy.ndarray, shape (n,)
        Values at the peaks, a measure of confidence.
    """

    heatmaps = np.asarray(heatmaps, dtype=float)
    n, height, width = heatmaps.shape
    flat = heatmaps.reshape(n, -1).argmax(axis=1)
    rows, cols = np.divmod(flat, width)
    peaks = heatmaps[np.arange(n), rows, cols]

    points = np.column_stack([cols, rows]).astype(float)
    floor = 1e-6
    for k in range(n):
        r, c = rows[k], cols[k]
        if 0 < c < width - 1:
            left, mid, right = np.log(np.maximum(heatmaps[k, r, c - 1:c + 2], floor))
            curvature = left - 2.0 * mid + right
            if curvature < 0.0:
                points[k, 0] += np.clip(0.5 * (left - right) / curvature, -0.5, 0.5)
        if 0 < r < height - 1:
            up, mid, down = np.log(np.maximum(heatmaps[k, r - 1:r + 2, c], floor))
            curvature = up - 2.0 * mid + down
            if curvature < 0.0:
                points[k, 1] += np.clip(0.5 * (up - down) / curvature, -0.5, 0.5)
    return points, peaks
