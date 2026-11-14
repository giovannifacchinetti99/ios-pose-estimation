import numpy as np
import pytest

from ios_pose_estimation.camera import PinholeCamera, project_pose
from ios_pose_estimation.pnp import (
    dlt_pose,
    pose_errors,
    ransac_pnp,
    solve_pnp,
    speed_score,
    undistort,
)
from ios_pose_estimation.rotations import quaternion_to_rotation_matrix, rotation_matrix_to_quaternion
from ios_pose_estimation.speedplus import TANGO_KEYPOINTS

CAMERA = PinholeCamera(fx=2988.58, fy=2988.34, cx=960.0, cy=600.0, width=1920, height=1200,
                       distortion=np.array([-0.2238, 0.5141, -0.000665, -0.000214, -0.1312]))


def _random_poses(n, seed=0):
    rng = np.random.default_rng(seed)
    poses = []
    for _ in range(n):
        q = rng.standard_normal(4)
        q /= np.linalg.norm(q)
        distance = rng.uniform(2.5, 10.0)
        direction = np.array([*rng.uniform(-0.12, 0.12, 2), 1.0])
        poses.append((q, distance * direction / np.linalg.norm(direction)))
    return poses


def test_undistort_inverts_the_distortion():
    rng = np.random.default_rng(0)
    normalized = rng.uniform([-0.3, -0.19], [0.3, 0.19], size=(50, 2))
    pixels = CAMERA.project(np.column_stack([normalized, np.ones(50)]))

    assert np.allclose(undistort(CAMERA, pixels), normalized, atol=1e-10)


def test_dlt_recovers_an_exact_pose():
    q, t = _random_poses(1, seed=1)[0]
    rotation = quaternion_to_rotation_matrix(q).T
    camera_points = TANGO_KEYPOINTS @ rotation.T + t
    normalized = camera_points[:, :2] / camera_points[:, 2:]

    estimated_rotation, estimated_translation = dlt_pose(TANGO_KEYPOINTS, normalized)

    assert np.allclose(estimated_rotation, rotation, atol=1e-9)
    assert np.allclose(estimated_translation, t, atol=1e-9)
    with pytest.raises(ValueError):
        dlt_pose(TANGO_KEYPOINTS[:5], normalized[:5])


def test_solve_pnp_is_exact_without_noise():
    for q, t in _random_poses(30, seed=2):
        pixels, _ = project_pose(CAMERA, q, t, TANGO_KEYPOINTS)
        estimate = solve_pnp(CAMERA, TANGO_KEYPOINTS, pixels)
        orientation, translation, _ = pose_errors(estimate.quaternion, estimate.translation, q, t)

        assert orientation < 1e-6 and translation < 1e-8
        assert estimate.rms_reprojection < 1e-6


def test_covariance_matches_the_actual_errors():
    rng = np.random.default_rng(3)
    sigma = 2.0
    q, t = _random_poses(1, seed=4)[0]
    rotation = quaternion_to_rotation_matrix(q).T
    pixels, _ = project_pose(CAMERA, q, t, TANGO_KEYPOINTS)

    errors, covariances = [], []
    for _ in range(300):
        estimate = solve_pnp(CAMERA, TANGO_KEYPOINTS, pixels + sigma * rng.standard_normal(pixels.shape), sigma)
        d = quaternion_to_rotation_matrix(estimate.quaternion).T @ rotation.T
        small_rotation = 0.5 * np.array([d[2, 1] - d[1, 2], d[0, 2] - d[2, 0], d[1, 0] - d[0, 1]])
        errors.append(np.concatenate([small_rotation, estimate.translation - t]))
        covariances.append(estimate.covariance)
    errors = np.array(errors)

    # the spread of the errors is the predicted one, axis by axis
    predicted = np.sqrt(np.diag(np.mean(covariances, axis=0)))
    assert np.allclose(errors.std(axis=0), predicted, rtol=0.15)
    # and the normalized squared error averages to the number of parameters
    nees = [e @ np.linalg.solve(c, e) for e, c in zip(errors, covariances)]
    assert 5.0 < np.mean(nees) < 7.0


def test_ransac_rejects_a_misplaced_keypoint():
    rng = np.random.default_rng(5)
    q, t = _random_poses(1, seed=6)[0]
    pixels, _ = project_pose(CAMERA, q, t, TANGO_KEYPOINTS)
    noisy = pixels + 0.5 * rng.standard_normal(pixels.shape)
    noisy[4] += [70.0, 50.0]

    plain = solve_pnp(CAMERA, TANGO_KEYPOINTS, noisy)
    robust = ransac_pnp(CAMERA, TANGO_KEYPOINTS, noisy, threshold=8.0)

    assert not robust.inliers[4] and robust.inliers.sum() == 10
    plain_error = pose_errors(plain.quaternion, plain.translation, q, t)[0]
    robust_error = pose_errors(robust.quaternion, robust.translation, q, t)[0]
    assert robust_error < 0.25 * plain_error


def test_errors_and_score():
    q, t = _random_poses(1, seed=7)[0]
    turn = rotation_matrix_to_quaternion(quaternion_to_rotation_matrix(q) @ np.array(
        [[np.cos(0.02), np.sin(0.02), 0.0], [-np.sin(0.02), np.cos(0.02), 0.0], [0.0, 0.0, 1.0]]))
    shifted = t + np.array([0.0, 0.0, 0.01 * np.linalg.norm(t)])

    orientation, translation, normalized = pose_errors(turn, shifted, q, t)

    assert np.isclose(orientation, 0.02) and np.isclose(normalized, 0.01)
    assert np.isclose(speed_score(turn, shifted, q, t), 0.03)
    # within the thresholds both terms vanish
    assert speed_score(q, t * 1.001, -q, t) == 0.0
    assert speed_score(q, t * 1.001, -q, t, apply_threshold=False) > 0.0
