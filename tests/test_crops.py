import numpy as np

from ios_pose_estimation.crops import (
    apply_transform,
    crop_transform,
    decode_heatmaps,
    gaussian_heatmaps,
    keypoint_box,
    warp,
)


def test_box_and_transform_map_the_square_onto_the_crop():
    points = np.array([[100.0, 50.0], [300.0, 90.0], [180.0, 250.0]])
    center, side = keypoint_box(points)
    assert np.allclose(center, [200.0, 150.0]) and side == 200.0

    transform = crop_transform(center, side, 64)
    corners = np.array([[100.0, 50.0], [300.0, 250.0]])
    # the corners of the square land on the edges of the crop
    assert np.allclose(apply_transform(transform, corners), [[-0.5, -0.5], [63.5, 63.5]])

    turned = crop_transform(center, side, 64, angle=np.pi / 2)
    # a point to the right of the center appears below it after a quarter turn
    assert np.allclose(apply_transform(turned, [[250.0, 150.0]]), [[31.5, 47.5]])


def test_warp_follows_the_transform():
    image = np.zeros((400, 600), dtype=np.uint8)
    image[120:124, 340:344] = 255                 # a bright spot centered at (341.5, 121.5)
    for angle, size in [(0.0, 128), (0.7, 96), (2.0, 40)]:
        transform = crop_transform([300.0, 150.0], 180.0, size, angle)
        crop = warp(image, transform, size).astype(float)
        rows, cols = np.indices(crop.shape)
        found = np.array([(cols * crop).sum(), (rows * crop).sum()]) / crop.sum()
        expected = apply_transform(transform, [[341.5, 121.5]])[0]
        assert np.allclose(found, expected, atol=0.35)


def test_heatmaps_are_decoded_to_sub_pixel_precision():
    rng = np.random.default_rng(0)
    points = rng.uniform(3.0, 60.0, size=(11, 2))
    heatmaps = gaussian_heatmaps(points, 64, sigma=1.5)

    decoded, peaks = decode_heatmaps(heatmaps)

    assert np.abs(decoded - points).max() < 1e-6
    assert np.all(peaks > 0.75)
