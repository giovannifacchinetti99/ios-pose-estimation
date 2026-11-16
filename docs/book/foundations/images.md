# What an image is

Everything this book does starts from an image, and most of what can go
wrong with a pose estimate is already decided when the image is taken. This
chapter follows the light from the target to the numbers that a program
reads: how a detector turns light into pixels, what limits their quality,
how positions in an image are written, and what happens when an image is
resized or turned, an operation that the keypoint detector of Part IV
performs on every image it sees.

## From light to numbers

A camera is a lens in front of a detector. The lens collects the light that
leaves each point of the scene in many directions and brings it back
together on one point of the detector, so that the detector receives a
small, inverted copy of the scene. The detector is a grid of light
sensitive cells, the pixels, each a few micrometres wide. During the
exposure, the time for which the camera lets light in, each pixel collects
the photons that fall on it and turns a fraction of them into electrons.
At the end of the exposure the charge of each pixel is measured, amplified
and converted into an integer by an analog to digital converter.

The image is the grid of these integers. For a gray level image with
$H$ rows and $W$ columns it is a matrix of $H \times W$ numbers, usually
between 0, no light, and 255, the most the pixel can record, when each value
is stored on eight bits. The camera of SPEED+ gives images of 1200 rows and
1920 columns, two million pixels, each with its gray level. A color camera
puts a mosaic of red, green and blue filters on the pixels and stores three
numbers per pixel; the images of SPEED+ are in gray levels, which is all a
pose needs.

## What limits the quality of a pixel

Three effects stand between the scene and the numbers.

The first is noise. Light arrives in photons, at random times, so even a
perfectly steady scene gives a pixel a count that fluctuates from one image
to the next. If a pixel receives $N$ photons on average, the fluctuation is
about $\sqrt N$, the shot noise of light; the electronics that read the
charge add a noise of their own, small but fixed. A brightly lit pixel
therefore has a good ratio of signal to noise and a dim one a poor ratio.
This is one of the reasons a sensor that brings its own light, as proposed
in this project, sees better than one that waits for the Sun.

The second is the range of light the pixel can record. A pixel holds a
limited number of electrons; beyond that it saturates and records the
largest value, whatever the true brightness. Below a certain level, the
signal is lost in the noise. Scenes in space strain both ends at once: the
sunlit insulation of a spacecraft can be thousands of times brighter than a
face in shadow, which gets no light from an atmosphere, as it would on the
ground. An exposure that shows the shadow saturates the sunlit side, and
one that preserves the sunlit side leaves the shadow black. In either case
some edges of the target, and the keypoints on them, simply disappear from
the image.

The third is blur. The lens does not focus a point to a point but to a
small spot, a little larger when the target is out of focus, and the
target moves during the exposure. An edge is therefore spread over a few
pixels rather than falling sharply between two of them.

Blur sounds like an enemy of accuracy, but it is what makes positions
better than a pixel possible. A feature spread over several pixels shows
up in all of them, and comparing their gray levels tells where its center
lies between pixel centers. The centroid of a small bright spot, the
average of the pixel positions weighted by their gray levels, locates it to
a tenth of a pixel when the noise is low. This is why the requirements of
this book can be stated in fractions of a pixel, and why the keypoint
detector reads its answers with sub-pixel precision.

The images of SPEED+ are also compressed as JPEG files, which store an
approximation of the image that is invisible to the eye but adds small
artifacts around sharp edges, of the order of a gray level or two.

## Positions in an image

A position in an image is written as a pair $(u, v)$: $u$ is the column,
growing to the right, and $v$ the row, growing downward. This is the
reverse of the usual order of a matrix, which gives the row first, and of
the usual direction of a vertical axis, which grows upward; both are
conventions of image processing that the rest of the book follows.

Positions are real numbers, not integers, because a point of the scene
lands anywhere on the detector, not only on the center of a pixel. The
code of this project takes the center of the pixel in row $i$ and column
$j$ at the position $(u, v) = (j, i)$, so that the pixel covers the square
from $j - \tfrac12$ to $j + \tfrac12$. Other software puts the corner of
the pixel there instead. The two conventions differ by half a pixel, which
is irrelevant to the eye and is not irrelevant to a method that aims at an
accuracy of one pixel; a convention has to be chosen and kept everywhere.

## Changing the size or the orientation of an image

A neural network looks at images of a fixed, small size, and Part IV cuts
a square around the target and brings it to 256 by 256 pixels, often turned
by some angle. This is a change of coordinates: each pixel of the new image
corresponds to a position in the original one.

For a crop of side $s$ around a center $\vect c$, turned by an angle
$\alpha$ and brought to $n$ pixels, the position $\vect u$ in the original
image and the position $\vect u'$ in the crop are related by

$$
\vect u' = \frac{n}{s}\,
\begin{bmatrix} \cos\alpha & -\sin\alpha \\ \sin\alpha & \cos\alpha \end{bmatrix}
(\vect u - \vect c) + \vect c',
$$

with $\vect c'$ the center of the crop. A scale, a rotation and a shift: a
similarity transformation, which preserves shapes and angles. Positions
found in the crop are taken back to the original image by inverting it.

To compute the crop, each pixel of the new image is mapped back to a
position in the original, and its gray level is read there. That position
almost never falls on a pixel center, so the gray level has to be
interpolated. The simplest choice takes the nearest pixel, which makes
edges jagged. Bilinear interpolation weighs the four surrounding pixels by
how close the position is to each: at a position a fraction $a$ of the way
from one column to the next and $b$ from one row to the next, the four
neighbours $g_{00}, g_{01}, g_{10}, g_{11}$ give

$$
g = (1-a)(1-b)\,g_{00} + a(1-b)\,g_{01} + (1-a)\,b\,g_{10} + a\,b\,g_{11} ,
$$

which varies smoothly and keeps the position of edges.

Shrinking an image needs one more step. If each new pixel covers ten
original pixels but reads only the four nearest to its center, it ignores
the other ninety six, and fine detail produces patterns that are not in
the scene, an effect called aliasing. The cure is to average the original
pixels over the area each new pixel covers before interpolating, which the
code does by first reducing the image by an integer factor, averaging
blocks of pixels, and then interpolating.

## Rendered and real images

The images of SPEED+ come from two sources. Most are rendered: a program
computes, for a model of the spacecraft with its materials and a model of
the lighting, how much light reaches each pixel of a virtual camera. Their
pose is known exactly, because the program placed the model where it
wanted. The others are photographs of a physical mock-up taken in a
laboratory, whose pose is measured by an external tracking system.

A renderer reproduces what its authors chose to model. A real camera adds
everything this chapter has listed, noise, saturation, blur, compression,
and a real mock-up adds the creases of its foil, the reflections of the
room and the stray light inside the lens. The images look alike to a
person and differ in many small ways to a program trained on one kind and
used on the other. That is the domain gap mentioned in Part I, and the
reason the book measures its methods on both kinds.

## In the code

`ios_pose_estimation.speedplus.SpeedPlus.image` reads an image of the
dataset as a matrix of gray levels. `ios_pose_estimation.crops` holds the
operations of the last sections: `crop_transform` builds the similarity
transformation of a crop, `apply_transform` maps positions through it, and
`warp` computes the crop, with the averaging that prevents aliasing and
bilinear interpolation.

## References

1. R. Szeliski, *Computer Vision: Algorithms and Applications*, 2nd
   edition, Springer, 2022.
2. J. R. Janesick, *Photon Transfer*, SPIE Press, 2007.
