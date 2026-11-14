import numpy as np

from ios_pose_estimation.rotations import (
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_to_rotation_matrix,
    rotation_angle_between,
    rotation_matrix_to_quaternion,
    rotation_vector_to_matrix,
    skew,
)


def _random_quaternions(n, seed=0):
    q = np.random.default_rng(seed).standard_normal((n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def test_rotation_matrix_is_proper_and_round_trips_in_every_branch():
    quaternions = list(_random_quaternions(50))
    # one quaternion dominated by each component exercises the four branches of Shepperd's method
    quaternions += [np.array(v) / np.linalg.norm(v) for v in
                    ([0.9, 0.1, 0.1, 0.1], [0.05, 0.9, 0.2, 0.1], [0.05, 0.1, 0.9, 0.2], [0.05, 0.2, 0.1, 0.9])]

    for q in quaternions:
        r = quaternion_to_rotation_matrix(q)
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(r), 1.0)
        back = rotation_matrix_to_quaternion(r)
        assert back[0] >= 0.0
        assert np.allclose(back, q if q[0] >= 0 else -q, atol=1e-12)


def test_product_composes_in_the_order_of_the_frames():
    q_ab, q_bc = _random_quaternions(2, seed=1)
    # A to B, then B to C: the matrices multiply in the opposite order
    q_ac = quaternion_multiply(q_ab, q_bc)

    assert np.allclose(quaternion_to_rotation_matrix(q_ac),
                       quaternion_to_rotation_matrix(q_bc) @ quaternion_to_rotation_matrix(q_ab))
    assert np.allclose(quaternion_multiply(q_ab, quaternion_conjugate(q_ab)), [1, 0, 0, 0], atol=1e-12)


def test_rotation_vector_matrix_matches_the_quaternion():
    rng = np.random.default_rng(2)
    for _ in range(20):
        w = rng.standard_normal(3)
        angle = np.linalg.norm(w)
        q = np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * w / angle])
        assert np.allclose(rotation_vector_to_matrix(w), quaternion_to_rotation_matrix(q), atol=1e-12)
    assert np.allclose(rotation_vector_to_matrix(np.zeros(3)), np.eye(3))
    tiny = np.array([1e-10, -2e-10, 3e-10])
    assert np.allclose(rotation_vector_to_matrix(tiny), np.eye(3) - skew(tiny), atol=1e-18)


def test_angle_between_attitudes():
    q = _random_quaternions(1, seed=3)[0]
    turn = np.array([np.cos(0.1), np.sin(0.1), 0.0, 0.0])          # 0.2 rad about x
    other = quaternion_multiply(q, turn)

    assert np.isclose(rotation_angle_between(q, other), 0.2)
    assert np.isclose(rotation_angle_between(q, -other), 0.2)
    assert rotation_angle_between(q, q) < 1e-6


def test_skew_is_the_cross_product():
    a, b = np.array([1.0, -2.0, 0.5]), np.array([0.3, 0.7, -1.1])

    assert np.allclose(skew(a) @ b, np.cross(a, b))
