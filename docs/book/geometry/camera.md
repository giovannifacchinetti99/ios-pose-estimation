# The camera and the pose

Part I set the goal: from an image of a known target, recover its position
and orientation. Before any method can do that, the reverse has to be
clear: how a target in a given pose becomes an image. This chapter builds
that forward model, from the geometry of an ideal camera to the distortion
of a real lens and the pose of the target, and introduces the camera and
the conventions of SPEED+, on which every later chapter relies. It uses the
pixel coordinates of {doc}`../foundations/images` and the rotations and
poses of {doc}`../foundations/rotations`.

## The pinhole camera

An ideal camera lets light reach the detector through a single point, the
optical center. Every point of the scene sends one ray through it, and the
ray continues until it meets the detector. The image is therefore a central
projection of the scene.

The camera frame has its origin at the optical center and its $z$ axis
along the optical axis, toward the scene; $x$ points to the right of the
image and $y$ downward, so that the axes of the image and of the frame
agree. A point at $(X, Y, Z)$, with $Z > 0$, meets the plane at unit
distance from the optical center at the normalized coordinates

$$
x = \frac{X}{Z}, \qquad y = \frac{Y}{Z}.
$$

The division by $Z$ is the essence of a camera, and its main limitation:
every point of a ray through the optical center lands on the same image
point, so the depth of a single point is lost. An object of size $L$ at
distance $d$ subtends an angle of about $L/d$, the same as an object twice
as large at twice the distance. Only the knowledge of the size and shape of
the target lets a single image recover its distance.

The detector is a grid of pixels. The normalized coordinates become pixel
coordinates through the intrinsic parameters of the camera,

$$
u = f_x\,x + c_x, \qquad v = f_y\,y + c_y,
$$

where $f_x$ and $f_y$ are the focal length divided by the width and the
height of a pixel, and $(c_x, c_y)$ is the pixel where the optical axis meets
the detector. The two steps can be written as one with homogeneous
coordinates, which describe a point of the image by three numbers $(\lambda
u, \lambda v, \lambda)$ defined up to a common factor: the pixel is recovered
by dividing the first two by the third. In this form the whole projection
is a matrix product followed by that division,

$$
\lambda \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = K \begin{bmatrix} X \\ Y
\\ Z \end{bmatrix}, \qquad K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y
\\ 0 & 0 & 1 \end{bmatrix},
$$

with $\lambda = Z$. The matrix $K$ is the camera matrix.

The focal length in pixels fixes the angular resolution. One pixel covers
an angle of $1/f_x$ radians, so a feature of size $\ell$ at distance $d$
spans

$$
\ell_{px} \simeq f_x\,\frac{\ell}{d}
$$

pixels. This simple relation, the apparent size, will return in every
estimate of accuracy: the error of a pose is a pixel error multiplied by
the distance and divided by the focal length.

## The lens

A real lens does not send every ray exactly through one point. The rays far
from the axis are bent slightly more or less than they should be, and the
image of a straight line becomes a curve, bowed outward or inward. The
standard model, due to Brown, corrects the normalized coordinates before
they are turned into pixels. With $r^2 = x^2 + y^2$, a radial term that
grows with the distance from the axis and a tangential term that accounts
for the lens not being exactly parallel to the detector give

$$
\begin{aligned}
x_d &= x\left(1 + k_1 r^2 + k_2 r^4 + k_3 r^6\right) + 2 p_1 x y + p_2\left(r^2 + 2x^2\right), \\
y_d &= y\left(1 + k_1 r^2 + k_2 r^4 + k_3 r^6\right) + p_1\left(r^2 + 2y^2\right) + 2 p_2 x y .
\end{aligned}
$$

The coefficients are measured once, by calibrating the camera on a known
pattern. Their effect is zero on the optical axis and grows toward the
corners of the image; the notebook of this chapter measures it for the
camera of SPEED+.

## The pose

The target is a rigid body with its own frame, fixed to it, in which its
geometry is known: the coordinates $\vect p_b$ of any point of it are
constants. Its pose with respect to the camera, as defined in
{doc}`../foundations/rotations`, is the position $\vect t$ of its origin in
the camera frame together with the quaternion $\vect q$ of the rotation from
the camera frame to the body frame, in the passive convention of SPEED+.
With $R$ the matrix of that quaternion, a point of the target is seen by the
camera at

$$
\vect p_c = R^{\mathsf T}\,\vect p_b + \vect t .
$$

The complete forward model of an image point is then this rigid
transformation, the central projection, the distortion and the intrinsic
matrix, in this order. Six numbers define a pose, three for the position and
three for the orientation; the quaternion uses four, with the constraint of
unit norm, and $\vect q$ and $-\vect q$ describe the same rotation.

A convention like this one is easy to state and easy to get wrong by a
transposition. A transposed rotation still produces a perfectly plausible
image of the target, just not the right one. The notebook checks the
convention against the images rather than trusting the documentation.

## The camera of SPEED+ and the reference points of Tango

The images of SPEED+ were rendered and taken with the same camera model: a
detector of 1920 by 1200 pixels with a focal length of about 2988 pixels,
which gives a field of view of 35.6 by 22.7 degrees, and a calibrated lens
distortion. At 6 m, the median distance of the dataset, one pixel covers
2 mm.

The target is Tango, a satellite of the PRISMA mission whose body is a box
of 0.74 by 0.77 by 0.32 m with a solar panel on top and three antennas. Its
geometry is described by eleven reference points in its body frame: the
four corners of the top face, four points of the bottom face, and the tips
of the three antennas. These keypoints are the link between the image and
the pose. Every method of the next chapters works by finding them in the
image.

## In the code

`ios_pose_estimation.camera` holds the forward model. `PinholeCamera`
stores the intrinsic parameters and the distortion coefficients; its method
`project` turns points in the camera frame into pixels, with or without the
distortion, and `distort` applies the lens model to normalized coordinates.
`body_to_camera` applies a pose to points of the body, and `project_pose`
does both steps at once.

`ios_pose_estimation.speedplus` gives access to the dataset. `SpeedPlus`
reads the camera, the labels and the images from a local folder and
fetches, from the public archive, only the files that are missing: the
archive is a single file of 17 GB, but its server accepts requests for a
range of bytes, and a zip archive keeps its index at the end, so the index
and then any single member can be read on their own. `TANGO_KEYPOINTS`
holds the eleven reference points of Tango.

`ios_pose_estimation.rotations` holds the quaternion and rotation functions,
in the convention described above.

## In the notebook

{doc}`../examples/01_camera_and_dataset/camera_and_dataset` reads the camera
and the labels of SPEED+, measures the distortion of the lens across the
image, verifies the convention of the poses against the images, and
describes the dataset from the point of view of pose estimation: the
distances, the apparent size of the target, how many keypoints are visible,
and where in the image the target appears.

## References

1. R. Hartley, A. Zisserman, *Multiple View Geometry in Computer Vision*,
   2nd edition, Cambridge University Press, 2004.
2. D. C. Brown, "Decentering Distortion of Lenses," *Photogrammetric
   Engineering*, 32(3), 1966.
3. T. H. Park, M. Märtens, G. Lecuyer, D. Izzo, S. D'Amico, "SPEED+:
   Next-Generation Dataset for Spacecraft Pose Estimation across Domain
   Gap," IEEE Aerospace Conference, 2022.
