# Estimating from noisy measurements

A pose is never observed; it is inferred from pixels that are slightly
wrong. This chapter collects the tools of that inference: how to find the
unknowns that best explain a set of measurements, how to solve the
resulting equations when they are linear and when they are not, how to say
how uncertain the answer is, and what to do when some measurements are not
slightly but completely wrong. Part III applies each of them to the pose.

## Measurements, models and noise

An estimation problem has three ingredients. The unknowns, collected in a
vector $\vect x$, are what one wants to know, for example the six numbers
of a pose. The measurements, collected in $\vect y$, are what one has, for
example the pixel coordinates of the keypoints. The model $\vect h$ says
what the measurements would be if the unknowns were known: here, the
projection of the keypoints by the camera. Reality differs from the model
by an error,

$$
\vect y = \vect h(\vect x) + \vect e .
$$

The error is not known, but its behaviour often is. The simplest and most
common assumption is that its components are independent, have zero mean
and follow a Gaussian distribution with standard deviation $\sigma$. About
two thirds of such errors are smaller than $\sigma$ and 95 percent smaller
than $2\sigma$.

## Least squares

The probability of observing $\vect y$ when the unknowns are $\vect x$ is,
for independent Gaussian errors, proportional to

$$
\exp\left(-\frac{1}{2\sigma^2}\sum_i \left(y_i - h_i(\vect x)\right)^2\right).
$$

The most likely unknowns, the maximum likelihood estimate, are the ones
that make this largest, which is the same as making the sum of the squared
residuals $y_i - h_i(\vect x)$ smallest. This is the method of least
squares, and its justification: under Gaussian errors it is not only a
convenient choice but the best one.

When the model is linear, $\vect h(\vect x) = A\vect x$, the minimum is
found by setting the derivative of $\|\vect y - A\vect x\|^2$ to zero,
which gives the normal equations

$$
A^{\mathsf T} A\,\hat{\vect x} = A^{\mathsf T}\vect y .
$$

They have a single solution when the columns of $A$ are independent, that
is, when the measurements contain enough information to separate all the
unknowns. When they nearly fail to be independent, the problem is ill
conditioned: small errors in $\vect y$ produce large errors in $\hat{\vect
x}$. Six keypoints that happen to lie almost on one plane are an example
met in Part III.

## Homogeneous systems and the singular value decomposition

Some problems lead instead to a system $A\vect h = 0$, with no measurement
on the right, whose unknowns are only defined up to a scale factor. The
direct linear transformation of Part III is one. With noise there is no
exact solution, and the sensible question is which unit vector $\vect h$
makes $\|A\vect h\|$ smallest.

The answer comes from the singular value decomposition. Every matrix can
be written as

$$
A = U\,\Sigma\,V^{\mathsf T},
$$

where $U$ and $V$ are orthonormal and $\Sigma$ is diagonal with the
singular values $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$. Read from right to
left, it says that the action of any matrix is a rotation, a stretch along
the axes and another rotation. The unit vector that $A$ shrinks the most is
the last column of $V$, the one stretched by the smallest singular value,
and that is the solution. The ratio of the largest to the smallest
singular value measures how ill conditioned a problem is.

The same decomposition answers another question that returns in Part III:
which rotation matrix is closest to a matrix $M$ that should be a rotation
but, because of noise, is not quite. Writing $M = U\Sigma V^{\mathsf T}$,
the answer is $U V^{\mathsf T}$, with the sign of the last column of $U$
changed if that product has determinant $-1$, which would make it a
reflection: the stretches are discarded and the rotations kept.

## Nonlinear least squares

The projection of a camera is not linear, because it divides by the depth,
so the best pose cannot be found by solving the normal equations once. It
is found by repeating a linear step. Around the current estimate
$\vect x_k$, the model is replaced by its first order approximation,

$$
\vect h(\vect x_k + \Delta) \simeq \vect h(\vect x_k) + J\,\Delta ,
$$

where $J$ is the Jacobian, the matrix of the derivatives of every
measurement with respect to every unknown. The correction $\Delta$ that
minimizes the squared residuals of this linear model solves

$$
J^{\mathsf T} J\,\Delta = J^{\mathsf T}\left(\vect y - \vect h(\vect x_k)\right),
$$

and the estimate becomes $\vect x_{k+1} = \vect x_k + \Delta$. This is the
Gauss-Newton method. Near the solution it converges very fast, roughly
doubling the number of correct digits at each step; far from it, the linear
approximation may be poor and the step may overshoot. The
Levenberg-Marquardt method adds a term $\lambda I$ to $J^{\mathsf T} J$,
which shortens the step and turns it toward the direction of steepest
descent. When a step reduces the residuals, $\lambda$ is decreased and the
method behaves like Gauss-Newton; when it does not, the step is rejected
and $\lambda$ increased.

A nonlinear method needs a starting point. Started too far away, it may
converge to a minimum that is not the best one. This is why Part III first
computes a linear estimate, which is not optimal but needs no start, and
only then refines it.

The derivatives in $J$ can be written by hand or computed numerically, by
changing each unknown by a small amount and measuring how the model
changes. The numerical way is slower but cannot be wrong in the formulas,
and for six unknowns its cost is negligible.

## How uncertain the answer is

The estimate inherits the errors of the measurements. Its uncertainty is
described by its covariance matrix $P$, whose diagonal holds the variances
of each unknown and whose other entries say how their errors are related.
For a Gaussian distribution, the points within a given probability lie in
an ellipsoid whose axes are the eigenvectors of $P$ and whose lengths are
proportional to the square roots of its eigenvalues.

If an error $\vect e$ with covariance $P_e$ passes through a linear map,
$\vect z = M\vect e$, its covariance becomes $M P_e M^{\mathsf T}$. Applied
to the least squares solution, with measurement errors of covariance
$\sigma^2 I$, this gives

$$
P = \sigma^2 \left(J^{\mathsf T} J\right)^{-1} .
$$

The Jacobian at the solution, already computed by the last iteration, gives
the covariance at no extra cost. It is also, for small errors, the
Cramér-Rao bound: no unbiased estimator can do better with the same
measurements, so least squares extracts all the information there is.

A covariance is only useful if it is right. The test is the normalized
estimation error squared,

$$
\varepsilon = \vect e^{\mathsf T} P^{-1} \vect e ,
$$

where $\vect e$ is the actual error of an estimate, known when the truth is
known. If the covariance is right, $\varepsilon$ follows a chi-square
distribution with as many degrees of freedom as unknowns: for a pose, its
average is 6, its median 5.3, and it exceeds 12.6 in 5 percent of the
cases. A larger average means the estimates are worse than they claim, a
smaller one that they are better; a navigation filter that trusts a
covariance that is too small will believe wrong measurements.

## When some measurements are wrong

Least squares assumes that every error is small. A keypoint detector
sometimes places a keypoint on the wrong corner of the target, an error of
a hundred pixels among errors of one. Squaring makes such an outlier
dominate the sum, and least squares bends the whole solution to reduce
it, spoiling an estimate that the other measurements would have made
accurate.

Random sample consensus, RANSAC, deals with this by not trusting any
measurement until it agrees with others. It picks a small random subset,
just large enough to determine the unknowns, computes the solution from it,
and counts how many of all the measurements agree with that solution within
a threshold. After many trials it keeps the solution with the largest
agreement, and refines it by least squares on the agreeing measurements,
the inliers, alone. If a fraction $w$ of the measurements is correct and a
subset has $s$ of them, a subset is free of outliers with probability
$w^s$, and $N$ trials find at least one such subset with probability

$$
1 - \left(1 - w^s\right)^N .
$$

With half of the measurements wrong and subsets of six, 300 trials succeed
99 percent of the time; with one in five wrong, 20 trials suffice.

## Describing errors that are not Gaussian

Real errors have tails: most are small, a few are very large. Their mean and
standard deviation are then dominated by the few, and describe neither. The
book therefore reports medians and percentiles. The median is the value
below which half of the errors lie, the 95th percentile the value below
which 95 percent lie; they are not moved by a handful of disasters, and the
distance between them tells how long the tail is. For the same reason the
spread of an error is estimated from its median absolute value: for a
Gaussian, the standard deviation is 1.4826 times the median of the absolute
deviations from the median, a relation that holds for the bulk of a
distribution even when a few points are wild.

## References

1. R. Hartley, A. Zisserman, *Multiple View Geometry in Computer Vision*,
   2nd edition, Cambridge University Press, 2004.
2. Y. Bar-Shalom, X. R. Li, T. Kirubarajan, *Estimation with Applications
   to Tracking and Navigation*, Wiley, 2001.
3. J. Nocedal, S. J. Wright, *Numerical Optimization*, 2nd edition,
   Springer, 2006.
4. M. A. Fischler, R. C. Bolles, "Random Sample Consensus: A Paradigm for
   Model Fitting with Applications to Image Analysis and Automated
   Cartography," *Communications of the ACM*, 24(6), 1981.
