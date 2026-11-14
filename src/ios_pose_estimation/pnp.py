"""
Pose from points: the Perspective-n-Point problem.

Given n points of a body, known in its own frame, and the pixels where they
appear in an image, find the pose of the body with respect to the camera.
The solution here has the three stages described in the book: a linear
estimate by the direct linear transformation, a refinement that minimizes
the reprojection error in pixels, which is the maximum likelihood estimate
for Gaussian pixel noise, and the covariance of the result from the
Jacobian at the solution. A RANSAC loop around the first two stages
rejects keypoints that are grossly wrong.
"""

from dataclasses import dataclass, field

import numpy as np

from ios_pose_estimation.rotations import (
    quaternion_to_rotation_matrix,
    rotation_angle_between,
    rotation_matrix_to_quaternion,
    rotation_vector_to_matrix,
)


@dataclass
class PoseEstimate:
    """
    A pose recovered from an image.

    Attributes
    ----------
    quaternion : numpy.ndarray, shape (4,)
        Rotation from the camera frame to the body frame, scalar first.
    translation : numpy.ndarray, shape (3,)
        Position of the body origin in the camera frame [m].
    covariance : numpy.ndarray, shape (6, 6) or None
        Covariance of the error, ordered as a small rotation of the body
        expressed in the camera frame [rad] and the translation [m]. None
        when no pixel noise was given.
    rms_reprojection : float
        Root mean square distance between the observed pixels and the
        pixels predicted by the pose, over the points used [px].
    inliers : numpy.ndarray of bool
        Which points were used.
    iterations : int
        Iterations of the refinement.
    """

    quaternion: np.ndarray
    translation: np.ndarray
    covariance: np.ndarray = None
    rms_reprojection: float = np.nan
    inliers: np.ndarray = field(default=None)
    iterations: int = 0

    @property
    def orientation_std(self):
        """Standard deviation of the orientation error, per axis [rad]."""

        return None if self.covariance is None else np.sqrt(np.diag(self.covariance)[:3])

    @property
    def translation_std(self):
        """Standard deviation of the translation error, per axis [m]."""

        return None if self.covariance is None else np.sqrt(np.diag(self.covariance)[3:])


def undistort(camera, pixels, iterations=20):
    """
    Normalized coordinates (X / Z, Y / Z) of the rays that produced the
    given pixels, removing the lens distortion. The distortion model has
    no closed-form inverse; the fixed-point iteration used here converges
    in a few steps for the moderate distortion of a real lens.

    Author: Giovanni Facchinetti, 2026
    """

    pixels = np.atleast_2d(np.asarray(pixels, dtype=float))
    distorted = np.column_stack([(pixels[:, 0] - camera.cx) / camera.fx, (pixels[:, 1] - camera.cy) / camera.fy])
    if not np.any(camera.distortion != 0.0):
        return distorted

    k1, k2, p1, p2, k3 = camera.distortion
    normalized = distorted.copy()
    for _ in range(iterations):
        x, y = normalized[:, 0], normalized[:, 1]
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        normalized = np.column_stack([(distorted[:, 0] - dx) / radial, (distorted[:, 1] - dy) / radial])

    return normalized


def _nearest_rotation(m):
    """Rotation matrix closest to m in the Frobenius norm."""

    u, _, vt = np.linalg.svd(m)
    return u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt


def dlt_pose(points_body, normalized):
    """
    Linear estimate of a pose by the direct linear transformation.

    Each correspondence between a body point and its normalized image
    coordinates (x, y) gives two equations that are linear in the twelve
    entries of the 3x4 matrix [M | t], with M the rotation from body to
    camera: x (m3 p + t3) = m1 p + t1 and y (m3 p + t3) = m2 p + t2, where
    m_i are the rows of M. Six points or more determine the matrix up to
    a scale, as the right singular vector of the smallest singular value.
    The scale follows from M being a rotation, the sign from the points
    being in front of the camera, and the nearest true rotation replaces
    the estimated M. The 3D points are centered and scaled first, which
    keeps the linear system well conditioned.

    The estimate minimizes an algebraic error, not the error in pixels,
    and ignores that M must be a rotation until the end; it is a starting
    point, not the answer.

    Parameters
    ----------
    points_body : array_like, shape (n, 3)
        Points in the body frame, n >= 6, not all on one plane.
    normalized : array_like, shape (n, 2)
        Their normalized image coordinates, see undistort.

    Returns
    -------
    rotation : numpy.ndarray, shape (3, 3)
        Body-to-camera rotation M.
    translation : numpy.ndarray, shape (3,)

    Author: Giovanni Facchinetti, 2026
    Reference: Hartley, Zisserman, Multiple View Geometry in Computer
    Vision, chapter 7
    """

    points = np.asarray(points_body, dtype=float)
    image = np.asarray(normalized, dtype=float)
    n = len(points)
    if n < 6:
        raise ValueError("the direct linear transformation needs at least six points")

    center = points.mean(axis=0)
    scale = np.sqrt(3.0) / np.mean(np.linalg.norm(points - center, axis=1))
    scaled = (points - center) * scale

    a = np.zeros((2 * n, 12))
    homogeneous = np.column_stack([scaled, np.ones(n)])
    x, y = image[:, 0], image[:, 1]
    a[0::2, 0:4] = homogeneous
    a[0::2, 8:12] = -x[:, None] * homogeneous
    a[1::2, 4:8] = homogeneous
    a[1::2, 8:12] = -y[:, None] * homogeneous

    _, _, vt = np.linalg.svd(a)
    p = vt[-1].reshape(3, 4)

    # undo the normalization of the points: p acts on (scale (X - c), 1)
    m = p[:, :3] * scale
    t = p[:, 3] - m @ center

    # the scale of a rotation's rows is one; points in front have positive depth
    norm = np.cbrt(np.linalg.det(m)) if abs(np.linalg.det(m)) > 0 else np.linalg.norm(m[2])
    m, t = m / norm, t / norm
    if np.median(points @ m[2] + t[2]) < 0:
        m, t = -m, -t

    rotation = _nearest_rotation(m)
    # with the rotation fixed, the translation is linear in the equations
    b = np.zeros((2 * n, 3))
    rhs = np.zeros(2 * n)
    rotated = points @ rotation.T
    b[0::2] = [1.0, 0.0, 0.0] - np.outer(x, [0.0, 0.0, 1.0])
    b[1::2] = [0.0, 1.0, 0.0] - np.outer(y, [0.0, 0.0, 1.0])
    rhs[0::2] = x * rotated[:, 2] - rotated[:, 0]
    rhs[1::2] = y * rotated[:, 2] - rotated[:, 1]
    translation = np.linalg.lstsq(b, rhs, rcond=None)[0]

    return rotation, translation


def _project(camera, rotation, translation, points):
    return camera.project(points @ rotation.T + translation)


def _apply(rotation, translation, delta):
    """Perturb a pose: a small rotation of the body in the camera frame, and a shift."""

    # rotation_vector_to_matrix is passive; its transpose is the active rotation
    return rotation_vector_to_matrix(delta[:3]).T @ rotation, translation + delta[3:]


def _jacobian(camera, rotation, translation, points, step=1e-7):
    base = _project(camera, rotation, translation, points).ravel()
    jac = np.zeros((base.size, 6))
    for k in range(6):
        delta = np.zeros(6)
        delta[k] = step
        moved = _project(camera, *_apply(rotation, translation, delta), points).ravel()
        jac[:, k] = (moved - base) / step
    return jac


def refine_pose(camera, points_body, pixels, rotation, translation, max_iterations=50, tolerance=1e-10):
    """
    Refine a pose by minimizing the reprojection error, the sum over the
    points of the squared distance in pixels between where each point is
    observed and where the pose projects it, lens distortion included.
    For independent Gaussian errors of the same size on every pixel
    coordinate this is the maximum likelihood estimate of the pose.

    The minimization is a Levenberg-Marquardt iteration on six parameters:
    a small rotation of the body, expressed in the camera frame, and a
    shift of the translation. After each step the small rotation is folded
    into the rotation matrix, so the parameters always stay near zero and
    the rotation stays exactly a rotation.

    Returns
    -------
    rotation, translation : the refined body-to-camera rotation and translation
    jacobian : numpy.ndarray, shape (2n, 6), at the solution
    residual : numpy.ndarray, shape (2n,), observed minus predicted pixels
    iterations : int

    Author: Giovanni Facchinetti, 2026
    Reference: Hartley, Zisserman, Multiple View Geometry in Computer
    Vision, appendix 6
    """

    points = np.asarray(points_body, dtype=float)
    observed = np.asarray(pixels, dtype=float).ravel()

    residual = observed - _project(camera, rotation, translation, points).ravel()
    cost = residual @ residual
    damping = 1e-3
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        jac = _jacobian(camera, rotation, translation, points)
        normal = jac.T @ jac
        gradient = jac.T @ residual
        improved = False
        while damping < 1e12:
            step = np.linalg.solve(normal + damping * np.diag(np.diag(normal) + 1e-12), gradient)
            candidate = _apply(rotation, translation, step)
            new_residual = observed - _project(camera, *candidate, points).ravel()
            new_cost = new_residual @ new_residual
            if new_cost < cost:
                rotation, translation = candidate
                residual, improvement, cost = new_residual, cost - new_cost, new_cost
                damping = max(damping / 10.0, 1e-12)
                improved = True
                break
            damping *= 10.0
        if not improved or improvement < tolerance * max(cost, 1e-12) or np.abs(step).max() < 1e-12:
            break

    return rotation, translation, _jacobian(camera, rotation, translation, points), residual, iterations


def solve_pnp(camera, points_body, pixels, pixel_std=None):
    """
    Pose of a body from the pixels of n >= 6 of its points: linear
    estimate, refinement on the reprojection error and, if the pixel noise
    is given, covariance of the result.

    The covariance is the first-order propagation of the pixel noise
    through the solution: with J the Jacobian of the projected pixels with
    respect to the six pose parameters at the solution, it is
    pixel_std^2 (J' J)^-1. It is the Cramér-Rao bound of the problem, the
    smallest covariance that any unbiased estimator can have with that
    noise, and the maximum likelihood estimate reaches it when the noise
    is small.

    Parameters
    ----------
    camera : ios_pose_estimation.camera.PinholeCamera
    points_body : array_like, shape (n, 3)
    pixels : array_like, shape (n, 2)
    pixel_std : float, optional
        Standard deviation of the error of each pixel coordinate [px].

    Returns
    -------
    PoseEstimate

    Author: Giovanni Facchinetti, 2026
    """

    points = np.asarray(points_body, dtype=float)
    pixels = np.asarray(pixels, dtype=float)

    rotation, translation = dlt_pose(points, undistort(camera, pixels))
    rotation, translation, jac, residual, iterations = refine_pose(camera, points, pixels, rotation, translation)

    covariance = None
    if pixel_std is not None:
        covariance = pixel_std**2 * np.linalg.inv(jac.T @ jac)

    return PoseEstimate(
        quaternion=rotation_matrix_to_quaternion(rotation.T),
        translation=translation,
        covariance=covariance,
        rms_reprojection=float(np.sqrt(np.mean(residual.reshape(-1, 2) ** 2) * 2.0)),
        inliers=np.ones(len(points), dtype=bool),
        iterations=iterations,
    )


def ransac_pnp(camera, points_body, pixels, threshold=10.0, pixel_std=None, trials=200, sample_size=6, rng=None):
    """
    Pose from points some of which may be grossly wrong.

    Random samples of sample_size points give candidate poses; each
    candidate is scored by how many points it projects within threshold
    pixels of where they were observed. The pose is then refined on the
    largest consistent set. A single keypoint misplaced by tens of pixels,
    the typical failure of a detector that confuses two similar corners of
    the target, would otherwise pull the whole solution away.

    Parameters
    ----------
    camera, points_body, pixels, pixel_std : see solve_pnp
    threshold : float
        Largest reprojection error of a point that agrees with a pose [px].
    trials : int
        Number of random samples.
    sample_size : int
        Points per sample, at least six.
    rng : numpy.random.Generator, optional

    Returns
    -------
    PoseEstimate
        With the points used marked in inliers.

    Author: Giovanni Facchinetti, 2026
    Reference: Fischler, Bolles, "Random Sample Consensus," Communications
    of the ACM, 24(6), 1981
    """

    points = np.asarray(points_body, dtype=float)
    pixels = np.asarray(pixels, dtype=float)
    rng = np.random.default_rng(0) if rng is None else rng
    n = len(points)

    best = np.ones(n, dtype=bool)
    best_count, best_error = -1, np.inf
    if n > sample_size:
        normalized = undistort(camera, pixels)
        for _ in range(trials):
            sample = rng.choice(n, sample_size, replace=False)
            try:
                rotation, translation = dlt_pose(points[sample], normalized[sample])
            except np.linalg.LinAlgError:
                continue
            errors = np.linalg.norm(_project(camera, rotation, translation, points) - pixels, axis=1)
            agree = errors < threshold
            count, spread = agree.sum(), errors[agree].sum()
            if count > best_count or (count == best_count and spread < best_error):
                best, best_count, best_error = agree, count, spread

    if best.sum() < sample_size:
        best = np.ones(n, dtype=bool)

    estimate = solve_pnp(camera, points[best], pixels[best], pixel_std)
    # points left out at first that the refined pose explains are taken back
    rotation = quaternion_to_rotation_matrix(estimate.quaternion).T
    errors = np.linalg.norm(_project(camera, rotation, estimate.translation, points) - pixels, axis=1)
    final = errors < threshold
    if final.sum() >= sample_size and not np.array_equal(final, best):
        estimate = solve_pnp(camera, points[final], pixels[final], pixel_std)
        best = final
    estimate.inliers = best

    return estimate


def pose_errors(estimate_q, estimate_t, true_q, true_t):
    """
    Errors of an estimated pose in the terms of the SPEED+ competition.

    Returns
    -------
    orientation : float
        Angle between the estimated and the true attitude [rad].
    translation : float
        Distance between the estimated and the true position [m].
    normalized_translation : float
        The same divided by the true distance.

    Author: Giovanni Facchinetti, 2026
    """

    orientation = rotation_angle_between(estimate_q, true_q)
    translation = float(np.linalg.norm(np.asarray(estimate_t) - np.asarray(true_t)))

    return orientation, translation, translation / float(np.linalg.norm(true_t))


def speed_score(estimate_q, estimate_t, true_q, true_t, rotation_threshold_deg=0.5, position_threshold=0.005,
                apply_threshold=True):
    """
    Score of one image in the SPEED+ competitions: the orientation error
    in radians plus the position error divided by the distance; lower is
    better. Errors smaller than the thresholds count as zero, because they
    are below the accuracy with which the labels themselves are known. The
    default thresholds are those of the reference implementation of the
    SPEED+ baseline (speedplusbaseline, src/utils/metrics.py).

    Author: Giovanni Facchinetti, 2026
    Reference: Kisantal, Sharma, Park, Izzo, Märtens, D'Amico, "Satellite
    Pose Estimation Challenge: Dataset, Competition Design and Results,"
    IEEE Transactions on Aerospace and Electronic Systems, 56(5), 2020
    """

    orientation, _, normalized = pose_errors(estimate_q, estimate_t, true_q, true_t)
    if apply_threshold:
        orientation = 0.0 if np.degrees(orientation) < rotation_threshold_deg else orientation
        normalized = 0.0 if normalized < position_threshold else normalized

    return orientation + normalized
