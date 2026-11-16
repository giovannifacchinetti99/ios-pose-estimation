# Frames, rotations and poses

A pose says where an object is and how it is turned with respect to
something else. The first half is a vector and needs little explanation;
the second half is a rotation, which is less familiar and the source of
most of the mistakes made in this field. This chapter introduces the
ways a rotation is written and the convention that the book follows.

## Vectors and frames

A vector, an arrow from one point to another, exists independently of how
it is described. To describe it by numbers one needs a frame: an origin and
three axes of unit length, perpendicular to one another, ordered so that
the third is the cross product of the first two. The components of the
vector are its projections on the three axes.

The same arrow has different components in different frames. In this book
there are two frames that matter. The camera frame has its origin in the
optical center of the camera and its $z$ axis along the line of sight. The
body frame is fixed to the target, with its origin in a reference point of
the target and its axes along its edges, so that every point of the target
has constant coordinates in it, the ones listed in its drawings. Writing
$\vect v_C$ and $\vect v_B$ for the components of the same vector in the
camera frame and in the body frame, the question of the relative
orientation is: how are $\vect v_C$ and $\vect v_B$ related?

## Rotation matrices

They are related by a matrix,

$$
\vect v_B = R\,\vect v_C ,
$$

whose rows are the axes of the body frame written in the camera frame:
the first component of $\vect v_B$ is the projection of the vector on the
first body axis, which is the dot product of $\vect v_C$ with that axis.
Because the axes have unit length and are perpendicular, the rows of $R$
are orthonormal, and so

$$
R^{\mathsf T} R = I, \qquad R^{-1} = R^{\mathsf T}, \qquad \det R = 1 .
$$

Inverting a rotation is as cheap as transposing it: $\vect v_C =
R^{\mathsf T}\vect v_B$. A matrix with these properties is a rotation
matrix.

The same matrix can be read in two ways. Read as above, it keeps the vector
fixed and changes the frame in which it is described; this is the passive
reading. Read as an operation on vectors, it takes a vector and turns it
into another one, in the same frame; this is the active reading. The two
are transposes of each other: turning the frame one way looks, from the
frame, like turning the vector the other way. Neither is more correct, but
a formula written for one gives wrong results when used with the other, and
a wrong result is still a perfectly good rotation, which is why the mistake
is hard to see. This book, like SPEED+ and ios-mission-planner, uses the
passive reading throughout: the rotation from frame A to frame B is the
matrix that turns components in A into components in B.

Rotations compose by multiplying matrices. If $R_{AB}$ goes from A to B and
$R_{BC}$ from B to C, then

$$
\vect v_C = R_{BC}\,\vect v_B = R_{BC} R_{AB}\,\vect v_A ,
$$

so $R_{AC} = R_{BC} R_{AB}$, with the first rotation on the right. Matrix
products do not commute, and neither do rotations: turning a book by a
quarter turn about one edge and then about another leaves it differently
placed than the same turns in the opposite order.

## Three numbers, not nine

A rotation matrix has nine entries, but the six conditions of
orthonormality leave only three free. Three numbers are therefore enough to
describe an orientation, and the question is which three.

The most familiar choice is three successive angles about chosen axes,
such as yaw, pitch and roll. They are easy to picture, but every such
choice has orientations where two of the three axes line up and one angle
becomes undefined, the gimbal lock, and their formulas are full of
trigonometric functions. Pose estimation avoids them.

A better description comes from a theorem of Euler: any rotation is a
single turn by an angle $\theta$ about some axis $\vect n$. The product
$\boldsymbol\phi = \theta\,\vect n$ is the rotation vector, three numbers
whose direction is the axis and whose length is the angle. For the frame
B obtained by turning A by $\theta$ about $\vect n$, the matrix is given by
the formula of Rodrigues,

$$
R = I - \sin\theta\,[\vect n]_\times + (1 - \cos\theta)\,[\vect n]_\times^2 ,
$$

where $[\vect n]_\times$ is the matrix of the cross product,

$$
[\vect n]_\times = \begin{bmatrix} 0 & -n_3 & n_2 \\ n_3 & 0 & -n_1 \\ -n_2 & n_1 & 0
\end{bmatrix}, \qquad [\vect n]_\times\,\vect w = \vect n \times \vect w .
$$

The minus sign in front of the sine is the passive reading; the active
formula has a plus.

For a small angle the formula simplifies to

$$
R \simeq I - [\boldsymbol\phi]_\times ,
$$

a correction linear in the three components of $\boldsymbol\phi$. This is
how the refinement of a pose in Part III changes an orientation: it
computes a small rotation vector and multiplies it into the current
matrix, which stays an exact rotation, rather than adding numbers to the
nine entries, which would not.

## Quaternions

The rotation vector has one weakness: its components jump when the angle
passes 180 degrees, where a turn by $\theta$ about $\vect n$ becomes a turn
by $360° - \theta$ about $-\vect n$. The quaternion removes it by using four
numbers,

$$
\vect q = \begin{bmatrix} q_0 \\ q_1 \\ q_2 \\ q_3 \end{bmatrix}
= \begin{bmatrix} \cos(\theta/2) \\ \vect n\,\sin(\theta/2) \end{bmatrix},
$$

a scalar part $q_0$ and a vector part, with $q_0^2 + q_1^2 + q_2^2 + q_3^2
= 1$. A quaternion is smooth everywhere, its constraint is easy to enforce
by dividing by its norm, and the rotation matrix follows from it without
trigonometric functions. In the passive convention of this book,

$$
R(\vect q) = \begin{bmatrix}
1 - 2(q_2^2 + q_3^2) & 2(q_1 q_2 + q_0 q_3) & 2(q_1 q_3 - q_0 q_2) \\
2(q_1 q_2 - q_0 q_3) & 1 - 2(q_1^2 + q_3^2) & 2(q_2 q_3 + q_0 q_1) \\
2(q_1 q_3 + q_0 q_2) & 2(q_2 q_3 - q_0 q_1) & 1 - 2(q_1^2 + q_2^2)
\end{bmatrix}.
$$

Two facts about quaternions matter in practice. The first is that $\vect q$
and $-\vect q$ give the same matrix, since every entry is a product of two
components: a turn by $\theta$ about $\vect n$ and a turn by $360° -
\theta$ about $-\vect n$ are the same rotation. Two quaternions can
therefore look completely different and describe the same orientation, and
any comparison has to allow for the sign. SPEED+ does not fix it, and half
of its quaternions have a negative scalar part.

The second is that quaternions compose by a product of their own, the
Hamilton product, which for $\vect q = (q_0, \vect q_v)$ and $\vect p =
(p_0, \vect p_v)$ is

$$
\vect q \otimes \vect p = \begin{bmatrix} q_0 p_0 - \vect q_v \cdot \vect p_v \\
q_0\,\vect p_v + p_0\,\vect q_v + \vect q_v \times \vect p_v \end{bmatrix}.
$$

With the passive matrices of this book, the quaternion of $R_{AC} = R_{BC}
R_{AB}$ is $\vect q_{AC} = \vect q_{AB} \otimes \vect q_{BC}$, the factors
in the order in which the frames are crossed. The inverse rotation is the
conjugate, $(q_0, -\vect q_v)$.

## How far apart two orientations are

The error of an estimated orientation is the angle of the rotation that
takes it to the true one. For two quaternions $\vect q_a$ and $\vect q_b$,
the scalar part of the rotation between them is their dot product, so

$$
\theta_{err} = 2 \arccos \left| \vect q_a \cdot \vect q_b \right| ,
$$

where the absolute value takes care of the sign ambiguity. This single
angle, which does not depend on the axis of the error, is the orientation
error reported everywhere in the book and in the scores of the SPEED+
competitions.

## The pose

A pose adds a position to the orientation. The target is placed by the
vector $\vect t$ from the optical center of the camera to the origin of the
body frame, written in the camera frame. A point of the target with body
coordinates $\vect p_B$ is then seen by the camera at

$$
\vect p_C = R^{\mathsf T}\,\vect p_B + \vect t ,
$$

where $R$ is the rotation from the camera frame to the body frame, as in
the labels of SPEED+: $R^{\mathsf T}$ turns the body components of the
point into camera components, and $\vect t$ moves them to the right place.
This combination of a rotation and a translation is a rigid
transformation, and its six numbers, three for the orientation and three
for the position, are the six unknowns of the pose estimation problem.

## In the code

`ios_pose_estimation.rotations` implements every formula of this chapter
in the passive convention: `quaternion_to_rotation_matrix` and its inverse
`rotation_matrix_to_quaternion`, `quaternion_multiply` and
`quaternion_conjugate`, `rotation_vector_to_matrix` with `skew`, the matrix
of the cross product, and `rotation_angle_between`, the error angle. The
tests of the package check each of them against the others, which is how
a convention mistake is caught: a round trip from a matrix to a quaternion
and back only closes if both directions use the same convention.

## References

1. F. L. Markley, J. L. Crassidis, *Fundamentals of Spacecraft Attitude
   Determination and Control*, Springer, 2014.
2. M. D. Shuster, "A Survey of Attitude Representations," *Journal of the
   Astronautical Sciences*, 41(4), 1993.
3. J. Solà, "Quaternion Kinematics for the Error-State Kalman Filter,"
   arXiv:1711.02508, 2017.
