import numpy as np

from ios_pose_estimation.rotations import rotation_vector_to_matrix, rotation_matrix_to_quaternion
from ios_pose_estimation.camera import PinholeCamera, body_to_camera, project_pose


def _camera(distortion=None):
    return PinholeCamera(fx=3000.0, fy=2990.0, cx=960.0, cy=600.0, width=1920, height=1200,
                         distortion=np.zeros(5) if distortion is None else np.asarray(distortion, dtype=float))


def test_point_on_the_optical_axis_projects_to_the_principal_point():
    camera = _camera()

    assert np.allclose(camera.project([[0.0, 0.0, 4.0]]), [[960.0, 600.0]])
    assert np.allclose(camera.matrix, [[3000.0, 0.0, 960.0], [0.0, 2990.0, 600.0], [0.0, 0.0, 1.0]])


def test_projection_divides_by_depth():
    camera = _camera()
    near = camera.project([[0.5, -0.2, 5.0]])[0]
    far = camera.project([[0.5, -0.2, 10.0]])[0]

    # x to the right, y downward: a point above the axis has a smaller v
    assert np.allclose(near, [960.0 + 3000.0 * 0.1, 600.0 - 2990.0 * 0.04])
    # twice as far, half the offset from the principal point
    assert np.allclose(far - [960.0, 600.0], 0.5 * (near - [960.0, 600.0]))
    # every point of a ray through the optical center lands on the same pixel
    assert np.allclose(camera.project([[1.0, -0.4, 10.0]])[0], near)


def test_distortion_follows_the_brown_model():
    k1, k2, p1, p2, k3 = -0.22, 0.51, -0.0007, -0.0002, -0.13
    camera = _camera([k1, k2, p1, p2, k3])
    x, y = 0.12, -0.07
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    expected = [x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
                y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y]

    assert np.allclose(camera.distort([[x, y]]), [expected])
    assert np.allclose(camera.distort([[0.0, 0.0]]), [[0.0, 0.0]])       # the axis is not bent

    distorted = camera.project([[x * 5.0, y * 5.0, 5.0]])
    ideal = camera.project([[x * 5.0, y * 5.0, 5.0]], distort=False)
    assert np.allclose(distorted, [[3000.0 * expected[0] + 960.0, 2990.0 * expected[1] + 600.0]])
    assert np.linalg.norm(distorted - ideal) > 0.5                      # a visible effect, in pixels
    assert np.allclose(_camera().project([[x, y, 1.0]]), _camera().project([[x, y, 1.0]], distort=False))


def test_in_image_mask():
    camera = _camera()
    pixels = np.array([[0.0, 0.0], [1919.0, 1199.0], [1920.0, 600.0], [-1.0, 10.0], [960.0, 600.0]])

    assert list(camera.in_image(pixels)) == [True, True, False, False, True]
    assert list(camera.in_image(pixels, margin=5.0)) == [False, False, False, False, True]


def test_identity_pose_only_translates():
    points = np.array([[0.1, 0.2, 0.3], [-0.4, 0.0, 0.5]])
    translation = np.array([0.5, -0.3, 6.0])

    assert np.allclose(body_to_camera(points, [1.0, 0.0, 0.0, 0.0], translation), points + translation)


def test_quaternion_is_the_rotation_from_camera_to_body():
    # a body turned by +90 degrees about the optical axis: its x axis is
    # along the y axis of the camera, its y axis along minus x
    quaternion = rotation_matrix_to_quaternion(rotation_vector_to_matrix([0.0, 0.0, np.pi / 2]))
    axes_body = np.eye(3)

    axes_camera = body_to_camera(axes_body, quaternion, np.zeros(3))

    assert np.allclose(axes_camera, [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], atol=1e-12)


def test_rigid_transformation_preserves_distances_and_handedness():
    rng = np.random.default_rng(0)
    quaternion = rng.standard_normal(4)
    quaternion /= np.linalg.norm(quaternion)
    points = rng.standard_normal((6, 3))

    moved = body_to_camera(points, quaternion, [0.2, -0.1, 7.0])
    distances = lambda p: np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)

    assert np.allclose(distances(moved), distances(points))
    triple = lambda p: np.dot(np.cross(p[1] - p[0], p[2] - p[0]), p[3] - p[0])
    assert np.isclose(triple(moved), triple(points))
    # q and -q are the same attitude
    assert np.allclose(body_to_camera(points, -quaternion, [0.2, -0.1, 7.0]), moved)


def test_project_pose_of_a_body_in_front_of_the_camera():
    camera = _camera()
    points = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.5]])

    pixels, depth = project_pose(camera, [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 5.0], points)

    assert np.allclose(depth, [5.0, 5.0, 5.5])
    assert np.allclose(pixels[0], [960.0, 600.0])
    assert np.allclose(pixels[1], [960.0 + 3000.0 * 0.1, 600.0])
    assert np.allclose(pixels[2], [960.0, 600.0])       # a point further along the axis hides behind the origin
