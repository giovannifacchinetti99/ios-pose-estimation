# Learning from examples

The geometry of Part III turns the pixels of known points of the target
into a pose. Finding those pixels is a different kind of problem: there is
no formula that says where the third corner of the top panel is in an
image. Part IV solves it with a neural network, a function whose behaviour
is not written by hand but learned from many images whose answer is known.
This chapter explains what such a function is, how it learns, why the
networks used for images have the shape they have, and how one can tell
whether what was learned will work on images never seen.

## A function with many parameters

A network is a function $\vect y = f(\vect x; \boldsymbol\theta)$ that takes
an input $\vect x$, here an image, and gives an output $\vect y$, here maps
of where the keypoints are. Its behaviour depends on a large set of numbers
$\boldsymbol\theta$, the parameters or weights. Training means choosing
them so that the function gives the right output on a set of examples, the
training set, for which the right output is known.

The building block is simple. A neuron takes some inputs $x_j$, forms a
weighted sum plus a constant,

$$
z = \sum_j w_j\,x_j + b ,
$$

and passes it through a nonlinear function, most often the rectified
linear unit, $\max(0, z)$, which lets positive values through and blocks
negative ones. A layer is many neurons reading the same inputs, each with
its own weights; a network is many layers, each reading the outputs of the
previous one.

The nonlinearity is essential. Without it, a stack of layers would be a
product of matrices, which is a single matrix, and the network could only
represent linear functions. With it, each layer can bend and fold the
representation produced by the previous one, and a deep enough network can
approximate almost any function. The price is that its parameters can no
longer be computed by a formula, as those of a linear least squares
problem are; they have to be searched for.

## The loss and gradient descent

How wrong the network is on an example is measured by a loss, a number that
is zero when the output is right and grows as it gets worse. Part IV uses
the squared difference between the predicted and the target maps, the same
quantity least squares minimizes. Training minimizes the average loss over
the training set,

$$
L(\boldsymbol\theta) = \frac{1}{N}\sum_{k=1}^{N} \ell\left(f(\vect x_k;
\boldsymbol\theta), \vect y_k\right).
$$

With millions of parameters the only practical way is to go downhill. The
gradient $\nabla L$, the vector of the derivatives of the loss with respect
to every parameter, points in the direction in which the loss grows
fastest, so a small step against it lowers the loss:

$$
\boldsymbol\theta \leftarrow \boldsymbol\theta - \eta\,\nabla L .
$$

The learning rate $\eta$ sets the size of the step. Too small, and
training takes forever; too large, and the steps overshoot the valleys and
the loss jumps around or grows.

Two refinements make this work in practice. The first is to estimate the
gradient on a small random batch of examples, a few tens, rather than on
the whole training set. Each step is then cheap and slightly noisy, and the
noise even helps to escape poor regions. A pass over the whole training
set, batch after batch, is called an epoch. The second is to adapt the
step to each parameter: the Adam method keeps running averages of each
derivative and of its square, and moves each parameter by an amount
normalized by how large and how consistent its derivatives have been. A
small weight decay, which pulls every parameter slightly toward zero at
each step, discourages solutions that rely on a few very large weights.

The learning rate is usually not constant. Large steps at the beginning
move quickly toward a good region, small steps at the end settle into it;
the schedule of Part IV raises the rate for the first tenth of the training
and then lowers it to zero along a cosine.

## Computing the gradient

A network is a long composition of simple functions, and the derivative of
a composition follows from the chain rule: the derivative of $f(g(x))$ is
$f'(g(x))\,g'(x)$. Applied backward, from the loss to the first layer, it
gives the derivatives with respect to all the parameters for about the
cost of two evaluations of the network. This is backpropagation, and
libraries such as PyTorch perform it automatically: the program describes
only how the output is computed, and the library records the operations
and differentiates them.

## Networks for images

An image of 256 by 256 pixels has 65,536 inputs. A layer of neurons each
connected to all of them would need tens of millions of weights for a
modest number of neurons, and would have to learn separately what an edge
looks like in every position of the image. Images have a structure that
makes this unnecessary: what matters at a point depends mostly on its
neighbourhood, and the same pattern means the same thing wherever it
appears.

A convolution uses that structure. A small grid of weights, the kernel,
typically three by three, is slid over the image, and at each position the
weighted sum of the pixels under it is computed. The result is a new image,
a feature map, which is large where the neighbourhood resembles the
pattern of the kernel: an edge of a given orientation, a corner, a bright
spot. A convolutional layer applies many kernels and produces many feature
maps, called channels, and the next layer convolves all of them together.
The weights are shared across positions, so a layer has only a few
thousand of them, and the layer is equivariant to translation: shifting the
input shifts its output by the same amount.

A three by three kernel sees very little. To see larger structures,
networks reduce the resolution between layers, by computing the
convolution only at every second pixel or by keeping the largest value of
each block of two by two. Each halving doubles the region of the original
image that one value of a feature map summarizes, its receptive field.
After a few halvings the features see whole parts of the target, an antenna
or a face, at the price of a coarse grid; the decoder of Part IV then
brings them back to a finer grid to say precisely where things are.

Two further ingredients make deep networks trainable. Batch normalization
rescales the values of each channel, during training, to zero mean and unit
variance over the batch, which keeps the scale of the signals under control
from layer to layer. Residual connections add the input of a group of layers
to its output, so that the group only has to learn a correction; the
gradient then flows directly through the additions, and networks with
dozens of layers train as easily as shallow ones. The ResNet family of
networks, used in Part IV, is built this way.

## What was learned, and whether it generalizes

A network with enough parameters can memorize its training set: it can
give exactly the right output on every training image and nonsense on any
other. What matters is the error on new images, the generalization. It is
measured on a validation set, images that are never used to update the
parameters. A training loss that keeps falling while the validation error
rises is the sign of overfitting.

Three habits keep this measurement honest, and Part IV follows all of
them. The validation images must be representative of the images the
network will meet: a subset chosen by order rather than at random can
differ systematically, which happens in SPEED+, where the images are sorted
by name and the second half of the names has the Earth in the background.
Any choice made by looking at the validation error, such as the epoch whose
weights are kept, makes that error slightly optimistic, so the final
measurement is made on images not used for any choice. And the measurement
reports the whole distribution of errors, not only its average.

Overfitting is fought with data. More examples is the best remedy;
augmentation is the next best: each training image is changed at random,
cropped, shifted, turned, made darker or noisier, in ways that do not
change what the right answer is, so that the network sees a new variant at
every epoch and cannot rely on accidental details.

## Starting from what others learned

The first layers of any network trained on natural images end up detecting
the same things: edges, corners, blobs, textures. Training them from random
values needs a great many images. It is common to start instead from a
network already trained on a large collection of photographs, ImageNet,
with over a million labeled images, and to continue training it on the
images of the task. This transfer learning is what makes a few thousand
images of a spacecraft enough.

## The cost of training

A network is evaluated by a few billion multiplications per image; training
needs about three times as many, forward and backward, for every image of
every epoch. A graphics processor performs tens of trillions of operations
per second and trains a network of this size on tens of thousands of images
in an hour or two. A laptop processor performs a few hundred billion, which
is why Part IV keeps the network small, the images few and the input
small, and trains for four hours. The method is the same at both scales;
the larger scale buys accuracy.

## When the images change

A network learns the images it was shown. If the images it meets in use
differ from those, in lighting, in texture, in noise, in the camera, its
error grows, sometimes a little and sometimes dramatically, and nothing in
its output warns of it. This is the domain gap. It is the reason the
augmentation of Part IV varies the gray levels so much, the reason the
confidence of each answer is measured rather than assumed, and the subject
of the evaluation on images of a real mock-up that follows.

## References

1. I. Goodfellow, Y. Bengio, A. Courville, *Deep Learning*, MIT Press,
   2016.
2. D. P. Kingma, J. Ba, "Adam: A Method for Stochastic Optimization,"
   *International Conference on Learning Representations*, 2015.
3. S. Ioffe, C. Szegedy, "Batch Normalization: Accelerating Deep Network
   Training by Reducing Internal Covariate Shift," *International
   Conference on Machine Learning*, 2015.
4. K. He, X. Zhang, S. Ren, J. Sun, "Deep Residual Learning for Image
   Recognition," *IEEE Conference on Computer Vision and Pattern
   Recognition*, 2016.
5. J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, L. Fei-Fei, "ImageNet: A
   Large-Scale Hierarchical Image Database," *IEEE Conference on Computer
   Vision and Pattern Recognition*, 2009.
