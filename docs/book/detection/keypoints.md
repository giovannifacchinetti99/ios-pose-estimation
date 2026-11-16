# Finding the keypoints

The previous part ended with a requirement. If the eleven keypoints of
Tango are found in the image within about two pixels at 10 m, and within
about three at 5 to 7 m, the pose follows with an orientation error below
one degree in 95 percent of the cases; and if one of them is found
completely wrong, RANSAC can still recover, as long as most of the others
are right. This chapter builds the part of the chain that produces those
pixels from an image: a neural network that looks at the image and says
where each keypoint is.

## Why a learned detector

Image processing offers many detectors of corners and edges, and a
spacecraft has plenty of both. They answer the wrong question. A corner
detector finds hundreds of corners on the body, the solar panel, the
antennas and the shadows, and says nothing about which of them is the
third corner of the top panel. The pose needs the identity of each point,
not only its position, and the identity depends on the whole target: a
corner is the third corner of the top panel because of where the rest of
the body is.

The surface of a spacecraft makes things worse. Multi-layer insulation is
crumpled and specular, so the same face looks different from one degree of
attitude to the next; solar cells are dark mirrors; the shadows are black
and the sunlit faces saturate. Methods that match local texture, which work
well on buildings and on faces, find little stable texture to match.

A convolutional network trained on many labeled images learns both things
at once: what the neighbourhood of each keypoint looks like under many
lightings, and where it must be with respect to the rest of the body. That
second part is what lets it place a keypoint that is hidden behind the
body, from the visible parts that imply it.

## Heatmaps rather than coordinates

The network could be asked for the 22 coordinates of the keypoints
directly. It works better when asked for an image per keypoint, a heatmap,
whose value is high where the keypoint is. Convolution is equivariant to
translation: shifting the input shifts the features by the same amount. A
heatmap output keeps that property all the way to the answer, so that the
network learns one local appearance rather than a separate mapping for
every position in the image. The training target is a Gaussian bump of
standard deviation $\sigma$ centered on the true position $\vect u_k$,

$$
H_k(\vect x) = \exp\left(-\frac{\|\vect x - \vect u_k\|^2}{2\sigma^2}\right),
$$

and the network is trained to reproduce it by minimizing the squared
difference over all pixels and keypoints.

A plain mean of that difference has a trap. With $\sigma = 1.5$ pixels the
bump covers about a dozen of the 4096 pixels of a 64 by 64 heatmap, so a
network that predicts zero everywhere is already almost right on average,
and training drifts there first: the first attempt of this project, after
an epoch of training, had heatmaps worth 0.01 at the true keypoints and a
loss higher than that of an empty map. The remedy is to weigh each pixel by
$1 + w\,H_k(\vect x)$, with $w = 20$, so that the few pixels near the
keypoint count as much as the whole background. The same imbalance between
a rare foreground and a vast background is what focal losses address in
object detection.

The heatmap carries more than the position. The height of its peak says
how sure the network is; two peaks say that it hesitates between two
places, as it may on a nearly symmetric body such as Tango, whose top panel
looks almost the same after a half turn; a flat map says that it found
nothing. These are the signals that let the chain reject a keypoint before
it reaches the pose.

The position is read from the peak. The pixel of largest value is only
the nearest pixel, and the heatmap is coarser than the image, so the peak
is refined: the logarithm of a Gaussian is a parabola, and fitting a
parabola, along each axis, to the logarithms of the peak and of its two
neighbours gives the center of the bump between pixels. For an exact
Gaussian the refinement is exact.

## The network

The network has two halves. The encoder turns the image into features: a
cascade of convolutions, each looking at a three by three neighbourhood of
the previous layer, interleaved with reductions of the resolution. Each
reduction doubles the size of the region that a feature summarizes, its
receptive field, so that after a few of them a feature sees a whole antenna
or a whole face of the body, at the price of a coarser grid. The decoder
brings the coarse, informed features back to a finer grid, and adds to them
the features of the earlier, finer layers of the encoder, which still know
exactly where the edges are. The output combines the context of the coarse
layers with the localization of the fine ones.

The encoder used here is the first part of ResNet-18, a network trained to
classify the photographs of ImageNet. Its first layers have learned to
respond to edges, corners and blobs, which are as useful for a spacecraft
as for a photograph, and starting from them, rather than from random
weights, is what makes a few thousand training images enough. Truncated
after its third stage it reduces the input sixteen times; the decoder
brings it back to one quarter. With an input of 256 by 256 pixels the
network gives eleven heatmaps of 64 by 64 and has 3.1 million parameters,
small enough to be trained on a laptop processor in a few hours and to run
on it in about 50 milliseconds per pass.

## Why the image is cropped

The input of the network is a fixed number of pixels, and the target does
not have a fixed size. The keypoints of Tango span about 1.1 m; at
distance $d$ they span $f L / d$ pixels, some 330 pixels at 10 m and 1,300,
two thirds of the width of the image, at 2.5 m. Reducing the whole 1920 pixel image to the 256 of the input
divides everything by 7.5: at 10 m the target becomes 44 pixels wide, a
pixel of the heatmap becomes 30 pixels of the image, and the two pixel
requirement is out of reach.

The image is therefore cut first: a square around the target, of side
$s = 1.4\,b$ where $b$ is the side of the box enclosing the keypoints, is
resampled to the input. One pixel of the input is then $s/256$ pixels of
the image, about 1.8 at 10 m and 6 at 3 m.

The crop does more than recover resolution: it makes the problem the same
at every distance. The chapter on the pose showed that the orientation
error is about $d\,\sigma / (f L \sqrt n)$ for an error of $\sigma$ pixels
of the image. In the crop, an error of $\sigma_c$ pixels of the input is
$\sigma = \sigma_c\,s/256$ pixels of the image, and since $s$ is
proportional to the apparent size $f L / d$ of the target, the distance
cancels:

$$
\delta\theta \sim \frac{1.4\,\sigma_c}{256\,\sqrt n} .
$$

A detector that is equally good on all crops gives the same orientation
error at every distance, and the requirement of the previous part, about
two pixels of the image at 10 m and three at 5 to 7 m, is in both cases
about one pixel of the input, a quarter of a pixel of the heatmap. This is
the target for the network, independent of the distance.

The box comes from a first, coarse pass of the same network on the whole
image reduced to the input. Its keypoints are a few tens of image pixels
off, which is plenty to place a box with a margin of 40 percent. The fine
pass then runs on the crop. To make one network good at both, a quarter of
its training samples are whole images and the rest are crops.

## Training

The network is trained on the synthetic images of SPEED+, rendered from
the model of Tango with known poses, so that the true pixel of every
keypoint is known exactly from the label and the camera model of the
first part, distortion included. No image has to be annotated by hand.

Each training sample is a crop chosen at random: its center is shifted by
up to a tenth of the box, its side varies between 1.2 and 1.6 times the
box, which covers the error of the coarse pass, and it is turned by a
random angle about its center. The turn is legitimate because a rotation of
the target about the line of sight gives an image of another pose of the
same target, so it multiplies the poses seen by the network at no cost. A
keypoint that falls outside the crop is excluded from the loss, since the
network cannot be asked to place it. The gray levels are then changed at
random, in exposure, contrast and gamma, with blur and with noise. These
changes do not make the synthetic images look like real ones, but they keep
the network from relying on the exact gray levels of the renderer, which is
the first and cheapest defence against the gap between the two.

The weights are updated by AdamW, a gradient method that adapts its step
to each parameter and decays the weights slightly at each step. The
learning rate follows one cycle: it rises over the first tenth of the
training and then decreases to zero along a cosine, which lets the early
steps move far and the last ones settle. After every pass over the
training images, an epoch, the error of the keypoints is measured on
validation images that the network never trains on, and the weights with
the lowest error are kept.

## What to expect

On images of the same kind as those it was trained on, a detector of this
kind is expected to come close to the requirement: the synthetic images are
clean, the target is always fully modelled, and nothing in them was not in
the training set. Not all synthetic images are equally easy, though: half
of those of SPEED+ show the Earth behind the target, which fills the
background with clouds, coastlines and bright edges that compete with the
edges of the spacecraft. On images taken in a laboratory it is expected to do much
worse, because the mock-up, its materials, the lights and the camera all
differ from the renderer in ways the network has never seen. That is the
domain gap that SPEED+ was designed to measure, and the subject of the next
part.

Two limits of this setup are deliberate. The network is small and trained
on about 6,700 images on a processor, where the entries of the competitions
used tens of thousands of images, larger networks and graphics processors;
the code is the same, and only the scale changes. And the detector does
not yet know when it is wrong: the height of the peaks and the residuals of
the pose are the signals that will tell, and their value is measured, not
assumed.

## In the code

`ios_pose_estimation.crops` holds the geometry, independent of the network:
`keypoint_box` and `crop_transform` define the crop, `warp` resamples an
image through it, `gaussian_heatmaps` and `decode_heatmaps` make the
targets and read the peaks. `ios_pose_estimation.detector` holds the
network, `KeypointNet`; the preparation of the training images,
`prepare_sources`, and of each batch with its random crops and changes of
gray levels, `make_batch`; the training loop, `train`; and
`KeypointDetector`, which applies the two passes to a whole image and,
through `pose`, gives the pose by RANSAC on the keypoints found. It needs
PyTorch, installed with the `train` extra of the package.

## In the notebook

{doc}`../examples/03_keypoint_detector/keypoint_detector` follows the
training, measures the error of the keypoints on synthetic validation
images, with the true box and with the box of the coarse pass, compares it
with the requirement distance by distance, and finds where the grossly
wrong keypoints come from. It then carries the keypoints through RANSAC to
the pose and shows how a rule on the residual and the number of inliers
separates the poses to trust from those to reject.

In short: the error of the keypoints is indeed independent of the distance
in pixels of the input, at about 1.4 px, 40 percent above the requirement;
one keypoint in five is grossly wrong, mostly keypoints hidden behind the
body; and after the rejection of doubtful poses, two poses out of three are
delivered with a median orientation error of 0.85 degrees, a 95th
percentile of 2.6 and a consistent covariance.

## References

1. B. Xiao, H. Wu, Y. Wei, "Simple Baselines for Human Pose Estimation and
   Tracking," *European Conference on Computer Vision*, 2018.
2. A. Newell, K. Yang, J. Deng, "Stacked Hourglass Networks for Human Pose
   Estimation," *European Conference on Computer Vision*, 2016.
3. K. He, X. Zhang, S. Ren, J. Sun, "Deep Residual Learning for Image
   Recognition," *IEEE Conference on Computer Vision and Pattern
   Recognition*, 2016.
4. T.-Y. Lin, P. Dollár, R. Girshick, K. He, B. Hariharan, S. Belongie,
   "Feature Pyramid Networks for Object Detection," *IEEE Conference on
   Computer Vision and Pattern Recognition*, 2017.
5. F. Zhang, X. Zhu, H. Dai, M. Ye, C. Zhu, "Distribution-Aware Coordinate
   Representation for Human Pose Estimation," *IEEE Conference on Computer
   Vision and Pattern Recognition*, 2020.
6. B. Chen, J. Cao, A. Parra, T.-J. Chin, "Satellite Pose Estimation with
   Deep Landmark Regression and Nonlinear Pose Refinement," *IEEE
   International Conference on Computer Vision Workshops*, 2019.
7. I. Loshchilov, F. Hutter, "Decoupled Weight Decay Regularization,"
   *International Conference on Learning Representations*, 2019.
8. T. H. Park, M. Märtens, G. Lecuyer, D. Izzo, S. D'Amico, "SPEED+:
   Next-Generation Dataset for Spacecraft Pose Estimation across Domain
   Gap," *IEEE Aerospace Conference*, 2022.
