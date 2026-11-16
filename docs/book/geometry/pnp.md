# From keypoints to a pose

The previous chapter described how a target in a given pose becomes an
image. This chapter inverts it. Suppose the pixels where some known points
of the target appear have been found, by a method the next part will
build: which pose put them there, and how accurately can it be known? The
answer has three parts, a direct estimate, its refinement, and the error
that remains, and a fourth for the case in which some of the pixels are
simply wrong. Each part applies a tool of
{doc}`../foundations/estimation` to the camera model of the previous
chapter.

## The problem

The target has $n$ keypoints whose coordinates $\vect p_i$ in its body
frame are known. The image shows them at pixels $\vect u_i$. The pose, a
rotation $M$ from the body to the camera frame and a translation $\vect t$,
must satisfy, for every point,

$$
\vect u_i \simeq \pi\left(M\,\vect p_i + \vect t\right),
$$

where $\pi$ is the projection of the camera, distortion included. This is
the Perspective-n-Point problem. The pose has six unknowns and each point
gives two equations, so three points are the minimum; with three, however,
the equations have up to four solutions, and four or more points are needed
for the solution to be unique in general. With the eleven keypoints of
Tango the problem is overdetermined, which is what allows the pixel errors
to average out.

## A linear estimate

The projection is nonlinear because of the division by the depth, but the
division can be moved to the other side. With $(x_i, y_i)$ the normalized
coordinates of the pixel, obtained by removing the intrinsic matrix and the
distortion, and $\vect m_1, \vect m_2, \vect m_3$ the rows of $M$, the two
equations of a point become

$$
x_i\left(\vect m_3\cdot\vect p_i + t_3\right) = \vect m_1\cdot\vect p_i + t_1,
\qquad
y_i\left(\vect m_3\cdot\vect p_i + t_3\right) = \vect m_2\cdot\vect p_i + t_2 ,
$$

which are linear in the twelve entries of $[M \mid \vect t]$. Stacking them
for all points gives a homogeneous system $A\,\vect h = 0$. With six points
or more, not all on one plane, its solution is determined up to a scale, as
the singular vector of $A$ with the smallest singular value, the
construction explained in the section on homogeneous systems of the
estimation chapter. This is the direct linear transformation.

The solution does not know that $M$ is a rotation. Its scale is fixed by
requiring the determinant of $M$ to be one, its sign by requiring the
points to be in front of the camera, and the nearest rotation, obtained
from the singular value decomposition of $M$ as shown in the same section,
replaces it. The translation
is then solved again, linearly, with the rotation fixed.

This estimate has two weaknesses. It minimizes an algebraic quantity, the
residual of the linear equations, which weighs the points by their depth
and has no direct meaning in pixels. And it enforces the rotation only at
the end. With exact pixels it gives the exact pose; with noisy pixels it is
a good starting point, not the best answer.

## The best estimate: minimizing the reprojection error

If the pixel errors are independent and Gaussian with the same standard
deviation $\sigma$, the most likely pose is the one that minimizes the sum
of the squared distances, in pixels, between where the points are observed
and where the pose projects them,

$$
E(M, \vect t) = \sum_{i=1}^{n} \left\| \vect u_i - \pi\left(M\,\vect p_i +
\vect t\right)\right\|^2 .
$$

This reprojection error is the quantity that the noise actually affects, so
its minimum is the maximum likelihood estimate.

The minimization is the nonlinear least squares of the estimation chapter.
A rotation cannot be updated by adding increments to its entries without
ceasing to be a rotation, so the unknowns of each iteration are six small
corrections: a rotation vector $\boldsymbol\delta$, which turns the body by
the angle $|\boldsymbol\delta|$ about its direction, as in the section on
small rotations of {doc}`../foundations/rotations`, and a shift of the
translation.
Around the current pose the projected pixels change linearly with them,
through the Jacobian $J$, the $2n \times 6$ matrix of the derivatives of the
pixels with respect to the corrections. The Gauss-Newton step solves

$$
\left(J^{\mathsf T} J\right)\Delta = J^{\mathsf T}\vect r ,
$$

with $\vect r$ the vector of the residuals. The Levenberg-Marquardt variant
adds a damping term to the diagonal, which shortens the step when the
linear approximation is poor and lets it grow as the solution is
approached. After each step the small rotation is folded into the rotation
matrix, so that the corrections always stay near zero and the matrix stays
an exact rotation. Starting from the linear estimate, a few iterations
suffice.

## How large the error is

At the solution, the Jacobian also gives the uncertainty. To first order the
error of the pose is the pixel noise mapped back through $J$, and its
covariance is

$$
P = \sigma^2\left(J^{\mathsf T} J\right)^{-1}.
$$

This is the Cramér-Rao bound of the problem, the smallest covariance that
any unbiased estimator can have with that noise, and the minimization of
the reprojection error reaches it when the noise is small. The covariance
comes with every pose, at no extra cost, and it is what a navigation filter
needs to weigh the measurement.

The structure of $J$ explains how the error behaves, and a simple argument
gives the same answer. A target of size $L$ at distance $d$ appears about
$f L / d$ pixels wide, with $f$ the focal length in pixels.

The direction of the target is measured directly, by where it is in the
image. An error of $\sigma$ pixels on each of $n$ points shifts the
estimated direction by about $\sigma / (f\sqrt{n})$, so the position across
the line of sight is uncertain by

$$
\delta_{\perp} \sim \frac{d\,\sigma}{f\sqrt{n}} ,
$$

proportional to the distance.

The distance is measured indirectly, from how large the target looks. An
error of $\sigma$ pixels on an apparent size of $f L / d$ is a relative error
of $\sigma d / (f L)$, and the distance inherits it:

$$
\delta_{\parallel} \sim \frac{d^2\,\sigma}{f L\sqrt{n}} ,
$$

proportional to the square of the distance. A camera measures directions
far better than distances.

The orientation is measured by the relative displacements of the points: a
rotation of the body by $\delta\theta$ moves the points by about
$f L\,\delta\theta / d$ pixels, so

$$
\delta\theta \sim \frac{d\,\sigma}{f L\sqrt{n}} ,
$$

proportional to the distance as well. Two conclusions follow. All the errors
are proportional to the pixel error, so the accuracy of the keypoints is
the single quantity that the rest of the chain depends on. And they all
shrink as the deputy closes in, which is the direction in which a docking
needs them to shrink.

## When a keypoint is wrong

The analysis assumes that every pixel is near its true value. A keypoint
detector fails differently: it finds most points to within a few pixels and
occasionally places one far away, on a similar corner, a reflection, an
antenna that looks like another. Least squares spreads the error of such a
point over the whole pose, and one outlier among eleven points can turn an
error of a fraction of a degree into several degrees.

The standard remedy is RANSAC, random sample consensus, introduced in the
estimation chapter. Small random
subsets of the points, six at a time, give candidate poses by the linear
estimate. Each candidate is scored by how many of all the points it
projects within a threshold of where they were observed. The candidate with
the largest consensus identifies the points that agree with one pose; the
refinement is then run on those alone, and the points left out are
reported. Since a wrong point agrees with no pose consistent with the
others, it is left out as soon as one sample avoids it.

The residual of the accepted points is useful beyond the rejection. A pose
whose accepted points still have large residuals, or that was accepted with
few points, is a pose to doubt, and the navigation should be told.

## The score of the competitions

The SPEED+ competitions rank methods by a single number per image, the
orientation error in radians plus the position error divided by the
distance,

$$
S = \theta_{err} + \frac{\|\hat{\vect t} - \vect t\|}{\|\vect t\|} ,
$$

averaged over the test images, with errors below the precision of the
labels counted as zero. The analysis above shows that both terms scale as
$\sigma d / (f L)$, which is why they are of similar size and can be added
without weights.

## In the code

`ios_pose_estimation.pnp` implements the chain. `undistort` removes the
intrinsic matrix and the lens distortion from pixels, `dlt_pose` computes
the linear estimate, `refine_pose` minimizes the reprojection error by
Levenberg-Marquardt, and `solve_pnp` runs the three stages and returns a
`PoseEstimate` with the pose, its covariance and the residual.
`ransac_pnp` wraps them in the RANSAC loop and marks the points it used.
`pose_errors` and `speed_score` measure an estimated pose against the truth
in the terms of the competitions.

## In the notebook

{doc}`../examples/02_pose_from_keypoints/pose_from_keypoints` applies the
chain to the camera of SPEED+, the keypoints of Tango and the poses of the
labels, with known errors added to the pixels. It shows that the chain is
exact without noise, that its errors are proportional to the pixel error
and grow with the distance as derived above, that its covariance is
consistent with the actual errors, how much is lost with fewer keypoints,
and how RANSAC removes a misplaced one. It ends with the requirement on the
keypoint detector that follows from the requirement on the pose.

## References

1. R. Hartley, A. Zisserman, *Multiple View Geometry in Computer Vision*,
   2nd edition, Cambridge University Press, 2004.
2. V. Lepetit, F. Moreno-Noguer, P. Fua, "EPnP: An Accurate O(n) Solution
   to the PnP Problem," *International Journal of Computer Vision*, 81(2),
   2009.
3. M. A. Fischler, R. C. Bolles, "Random Sample Consensus: A Paradigm for
   Model Fitting with Applications to Image Analysis and Automated
   Cartography," *Communications of the ACM*, 24(6), 1981.
4. M. Kisantal, S. Sharma, T. H. Park, D. Izzo, M. Märtens, S. D'Amico,
   "Satellite Pose Estimation Challenge: Dataset, Competition Design, and
   Results," *IEEE Transactions on Aerospace and Electronic Systems*,
   56(5), 2020.
