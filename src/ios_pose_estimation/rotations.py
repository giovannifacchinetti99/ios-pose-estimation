"""
Rotations and quaternions.

A quaternion is written (q0, q1, q2, q3) with the scalar part first, in the
Hamilton convention, and describes the rotation "from frame A to frame B"
in the passive sense: its rotation matrix turns the components of a vector
in frame A into its components in frame B, v_B = R v_A. This is the
convention of the SPEED+ labels, where A is the camera frame and B the body
frame of the target, and of the ios-mission-planner package.
"""

import numpy as np


def quaternion_to_rotation_matrix(q):
    """
    Rotation matrix of a quaternion: v_B = R @ v_A.

    Parameters
    ----------
    q : array_like, shape (4,)
        (q0, q1, q2, q3), not required to be exactly unit norm.

    Returns
    -------
    numpy.ndarray, shape (3, 3)

    Author: Giovanni Facchinetti, 2026
    Reference: Markley, Crassidis, Fundamentals of Spacecraft Attitude
    Determination and Control
    """

    q = np.asarray(q, dtype=float)
    q0, q1, q2, q3 = q / np.sqrt(q @ q)

    return np.array([
        [1 - 2 * (q2**2 + q3**2), 2 * (q1 * q2 + q0 * q3), 2 * (q1 * q3 - q0 * q2)],
        [2 * (q1 * q2 - q0 * q3), 1 - 2 * (q1**2 + q3**2), 2 * (q2 * q3 + q0 * q1)],
        [2 * (q1 * q3 + q0 * q2), 2 * (q2 * q3 - q0 * q1), 1 - 2 * (q1**2 + q2**2)],
    ])


def rotation_matrix_to_quaternion(r):
    """
    Quaternion of a rotation matrix, inverse of quaternion_to_rotation_matrix,
    by Shepperd's method, which picks the best conditioned of four
    equivalent formulas. The scalar part of the result is not negative.

    Author: Giovanni Facchinetti, 2026
    Reference: Shepperd, "Quaternion from Rotation Matrix," Journal of
    Guidance and Control, 1(3), 1978
    """

    r = np.asarray(r, dtype=float)
    trace = r[0, 0] + r[1, 1] + r[2, 2]

    # passive convention: the antisymmetric terms are R[j,i] - R[i,j]
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q = [0.25 * s, (r[1, 2] - r[2, 1]) / s, (r[2, 0] - r[0, 2]) / s, (r[0, 1] - r[1, 0]) / s]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        q = [(r[1, 2] - r[2, 1]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s]
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        q = [(r[2, 0] - r[0, 2]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        q = [(r[0, 1] - r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s]

    q = np.array(q)
    q = q / np.sqrt(q @ q)
    return -q if q[0] < 0.0 else q


def quaternion_multiply(q, p):
    """Hamilton product q (x) p."""

    q0, q1, q2, q3 = q
    p0, p1, p2, p3 = p

    return np.array([
        q0 * p0 - q1 * p1 - q2 * p2 - q3 * p3,
        q0 * p1 + q1 * p0 + q2 * p3 - q3 * p2,
        q0 * p2 - q1 * p3 + q2 * p0 + q3 * p1,
        q0 * p3 + q1 * p2 - q2 * p1 + q3 * p0,
    ])


def quaternion_conjugate(q):
    """Conjugate, the inverse rotation for a unit quaternion."""

    return np.array([q[0], -q[1], -q[2], -q[3]])


def rotation_vector_to_matrix(rotation_vector):
    """
    Rotation matrix, in the passive convention of this module, of the
    rotation by |rotation_vector| radians about its direction: the frame
    B is frame A turned by that rotation.

    Author: Giovanni Facchinetti, 2026
    """

    w = np.asarray(rotation_vector, dtype=float)
    angle = np.sqrt(w @ w)
    k = skew(w)
    if angle < 1e-8:
        return np.eye(3) - k + 0.5 * k @ k
    k = k / angle
    # the active rotation is I + sin K + (1 - cos) K^2; the passive one is its transpose
    return np.eye(3) - np.sin(angle) * k + (1.0 - np.cos(angle)) * k @ k


def skew(v):
    """Matrix of the cross product: skew(a) @ b = a x b."""

    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def rotation_angle_between(q_a, q_b):
    """
    Angle of the rotation that takes one attitude to the other [rad],
    the orientation error used by the SPEED+ score. q and -q give the
    same angle.
    """

    q_a = np.asarray(q_a, dtype=float) / np.linalg.norm(q_a)
    q_b = np.asarray(q_b, dtype=float) / np.linalg.norm(q_b)

    return 2.0 * np.arccos(min(1.0, abs(float(q_a @ q_b))))
