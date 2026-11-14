from dataclasses import dataclass, field

import numpy as np

from ios_pose_estimation.rotations import quaternion_to_rotation_matrix


@dataclass
class PinholeCamera:
    """
    Pinhole camera with lens distortion.

    The camera frame has its origin at the optical center, the z axis
    along the optical axis, pointing at the scene, x to the right of the
    image and y downward. A point (X, Y, Z) in that frame is first
    reduced to normalized coordinates x = X / Z, y = Y / Z, which removes
    the depth, and then mapped to pixels,

        u = fx * x + cx,    v = fy * y + cy,

    where fx and fy are the focal length divided by the size of a pixel
    and (cx, cy) is the pixel where the optical axis meets the image.

    A real lens bends the rays slightly. The standard model corrects the
    normalized coordinates before the mapping to pixels, with a radial
    term that depends on r^2 = x^2 + y^2 and a tangential term due to
    the lens not being perfectly parallel to the detector:

        x_d = x (1 + k1 r^2 + k2 r^4 + k3 r^6) + 2 p1 x y + p2 (r^2 + 2 x^2)
        y_d = y (1 + k1 r^2 + k2 r^4 + k3 r^6) + p1 (r^2 + 2 y^2) + 2 p2 x y

    Attributes
    ----------
    fx, fy : float
        Focal length in pixels along the two image axes.
    cx, cy : float
        Principal point [pixel].
    width, height : int
        Size of the image [pixel].
    distortion : numpy.ndarray, shape (5,)
        Distortion coefficients (k1, k2, p1, p2, k3), the order used by
        OpenCV. All zero for an ideal pinhole.

    Author: Giovanni Facchinetti, 2026
    Reference: Hartley, Zisserman, Multiple View Geometry in Computer
    Vision; Brown, "Decentering Distortion of Lenses," Photogrammetric
    Engineering, 32(3), 1966
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion: np.ndarray = field(default_factory=lambda: np.zeros(5))

    @property
    def matrix(self):
        """The 3x3 matrix of the intrinsic parameters."""

        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    def distort(self, normalized):
        """
        Apply the lens distortion to normalized image coordinates.

        Parameters
        ----------
        normalized : array_like, shape (n, 2)
            Ideal normalized coordinates (x, y) = (X / Z, Y / Z).

        Returns
        -------
        numpy.ndarray, shape (n, 2)
        """

        normalized = np.atleast_2d(np.asarray(normalized, dtype=float))
        k1, k2, p1, p2, k3 = self.distortion
        x, y = normalized[:, 0], normalized[:, 1]
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2**2 + k3 * r2**3

        return np.column_stack([
            x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x),
            y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y,
        ])

    def project(self, points_camera, distort=True):
        """
        Project points given in the camera frame to pixel coordinates.

        Parameters
        ----------
        points_camera : array_like, shape (n, 3)
            Points in the camera frame; only those with positive z are
            in front of the camera.
        distort : bool
            Whether to apply the lens distortion.

        Returns
        -------
        numpy.ndarray, shape (n, 2)
            Pixel coordinates (u, v).
        """

        points_camera = np.atleast_2d(np.asarray(points_camera, dtype=float))
        normalized = points_camera[:, :2] / points_camera[:, 2:3]
        if distort and np.any(self.distortion != 0.0):
            normalized = self.distort(normalized)

        return np.column_stack([self.fx * normalized[:, 0] + self.cx, self.fy * normalized[:, 1] + self.cy])

    def in_image(self, pixels, margin=0.0):
        """Which pixel coordinates fall inside the image, with an optional margin."""

        pixels = np.atleast_2d(pixels)
        return (
            (pixels[:, 0] >= margin) & (pixels[:, 0] <= self.width - 1 - margin)
            & (pixels[:, 1] >= margin) & (pixels[:, 1] <= self.height - 1 - margin)
        )


def body_to_camera(points_body, quaternion, translation):
    """
    Express points of a body in the camera frame, given the pose of the
    body.

    The pose is the pair used throughout the literature on spacecraft
    pose estimation and in the SPEED+ dataset: the quaternion of the
    rotation from the camera frame to the body frame, and the position
    of the body origin in the camera frame. With R the rotation matrix of
    the quaternion, which turns camera components into body components,
    a point p_b of the body is seen by the camera at

        p_c = R' p_b + t.

    The quaternion has the convention of ios_pose_estimation.rotations,
    scalar part first and rotation "from frame A to frame B", with the
    camera as frame A and the body as frame B.

    Parameters
    ----------
    points_body : array_like, shape (n, 3)
        Points in the body frame [m].
    quaternion : array_like, shape (4,)
        Rotation from the camera frame to the body frame, scalar first.
    translation : array_like, shape (3,)
        Position of the body origin in the camera frame [m].

    Returns
    -------
    numpy.ndarray, shape (n, 3)

    Author: Giovanni Facchinetti, 2026
    """

    points_body = np.atleast_2d(np.asarray(points_body, dtype=float))
    camera_to_body = quaternion_to_rotation_matrix(quaternion)

    # row vectors: (R' p)' = p' R
    return points_body @ camera_to_body + np.asarray(translation, dtype=float)


def project_pose(camera, quaternion, translation, points_body, distort=True):
    """
    Pixel coordinates of points of a body seen by a camera, for a given
    pose of the body.

    Parameters
    ----------
    camera : PinholeCamera
    quaternion, translation :
        Pose of the body, see body_to_camera.
    points_body : array_like, shape (n, 3)
        Points in the body frame [m].
    distort : bool
        Whether to apply the lens distortion.

    Returns
    -------
    pixels : numpy.ndarray, shape (n, 2)
    depth : numpy.ndarray, shape (n,)
        Distance of each point along the optical axis [m]; positive for
        points in front of the camera.

    Author: Giovanni Facchinetti, 2026
    """

    points_camera = body_to_camera(points_body, quaternion, translation)

    return camera.project(points_camera, distort=distort), points_camera[:, 2]
