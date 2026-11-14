# The problem: docking to a target you cannot fully see

A docking ends when the port of the servicer, which we call the deputy,
meets the port of the target, the chief, within a few centimetres and a few
degrees, at a few centimetres per second. Getting there is a problem of
navigation: the deputy has to know where the port of the chief is, and how
it is oriented, accurately enough and early enough to steer onto it. This
chapter explains why that knowledge is the weak point of a docking, what
accuracy it must reach, and why the answer proposed here is a camera with a
light source of its own and pose estimation software.

## What the simulation showed

The companion project, ios-mission-planner, simulates the whole approach in
closed loop, from a hold point 5 km behind the chief to the contact of the
ports. A nonlinear model of both spacecraft plays the role of reality; the
deputy sees it only through its sensors and acts on it only through its
thrusters and wheels. Its relative sensor is the kind used by most
approaches flown so far: it measures the distance and the direction of a
target pattern mounted next to the port of the chief.

With that sensor, a Monte Carlo campaign of 200 runs with dispersed
hardware and initial conditions docks in every run. The lateral offset at
contact stays within a few millimetres, the closing speed within a
millimetre per second of the commanded value. The one quantity that uses a
large part of its limit is the angular misalignment between the two ports,
on average 0.4 degrees, up to 0.9, against a limit of 4. And it does not
come from the deputy. It is the attitude error of the chief itself, which
in that campaign was dispersed by 0.3 degrees: the sensor measures where
the target pattern is, not how the port is oriented, so the deputy aligns
with the direction in which the port should be pointing, and arrives with
whatever error the chief happens to have.

The campaign then degraded one element at a time. A relative sensor ten
times noisier changed nothing. A field of view seven times narrower changed
nothing. Thrusters 25 times coarser still docked. An attitude error of the
chief ten times larger, 3 degrees, halved the probability of success, and
every failure was an angular misalignment at contact. No change to the
deputy can fix this, because the deputy does not measure the quantity that
is wrong.

## Cooperative and uncooperative targets

The case is not exotic. A target that holds its attitude within a fraction
of a degree is a target that is working perfectly, carries a docking aid
designed for the purpose, and keeps controlling its attitude during the
approach. That is the situation of a space station receiving a cargo
vehicle. It is not the situation of in-orbit servicing, whose clients are
satellites that need help: out of propellant, with a failed attitude
control, or simply never designed to be approached. Such a target holds its
attitude loosely or not at all, and has no docking aid; at best it has a
known shape.

For these targets the relative attitude has to be measured, and it has to
be measured from the shape of the target itself. This is the problem of
pose estimation: from an image of a known object, recover its position and
its orientation with respect to the camera.

## How accurate is enough

The limits that a docking mechanism accepts at first contact give the
accuracy that the measurement has to reach. For the reference mission of
ios-mission-planner they are a lateral offset of 10 cm, an angular
misalignment of 4 degrees, an axial speed between 5 and 10 cm/s, a lateral
speed of 4 cm/s and an angular rate of 0.2 degrees per second.

The guidance and the control use up part of each budget; the measurement
has to leave the rest. A reasonable allocation leaves about a third of each
limit to the knowledge of the pose. In the last metres this means knowing
the relative orientation to about 1 degree and the lateral position to a
few centimetres. Further away the requirement relaxes, because there is
still time to correct, but the measurement must be available early enough
for the guidance to align the deputy with the port before the final
translation, which in the reference profile starts 20 m from the port.

The measurement must also say how good it is. A pose that is wrong by 30
degrees, because the software has confused two similar faces of the
target, is far more dangerous than a pose that is noisy by 1 degree. The
navigation filter that receives the measurement needs its covariance, and
it needs a way to reject the measurements that are simply wrong.

## The light

The second result of the simulation concerns lighting. An optical sensor
sees the target only when the Sun lights it. The illumination analysis of
the reference mission found that a port looking backward along the orbit is
well lit for about 24 minutes per orbit, right before the chief enters the
shadow of the Earth, that for a few days a year, when the Sun is nearly
perpendicular to the orbit plane, it is never well lit at all, and that a
deputy waiting 20 m from the port for the light to be right spends about
0.8 m/s of propellant per hour.

A sensor that brings its own light removes this constraint. An illuminator
pointed along the optical axis lights what the camera sees, in eclipse as
well as in daylight, and from a direction that does not depend on the
Sun. It does not remove every difficulty: direct sunlight still blinds the
camera and creates hard shadows and reflections. But it turns the lighting
from a constraint of the mission into a property of the sensor.

## Why a camera

Three families of sensors measure a relative pose at short range.

A scanning or flash lidar measures the distance to every point it sees and
returns a cloud of points in three dimensions. The pose follows from
aligning that cloud with a model of the target. It is insensitive to
lighting and gives the distance directly, but it is heavy, consumes power,
is expensive, and has a limited resolution at the shortest ranges.

A radar measures distance and relative speed very well at long range and
in any condition, but it sees a target as a few reflecting centres, not as
a shape, and cannot give an orientation to a degree.

A camera is small, light, cheap and has a resolution far beyond the other
two: at 6 m, one pixel of the camera of SPEED+ covers 2 mm. It measures
only directions, not distances, so it needs a model of the target to
recover the pose, and it needs light. The first need is met by the
methods of this book; the second by the illuminator.

For the last metres of a docking to a target of known shape, a camera with
an illuminator is the most capable sensor for its mass and cost. A lidar or
a radar remains useful further away, where the target is a few pixels wide
and its distance matters more than its orientation.

## The difficulty of learning from images

Recovering a pose from an image of a known object is a solved problem when
the object has been recognized: if the pixels of a few known points of the
object are given, geometry gives the pose, as Part II shows. The hard part
is recognizing those points, on a spacecraft whose surfaces are covered in
reflective foil, against the Earth or the black of space, in sunlight or
under the illuminator. The methods that work best today use neural
networks trained on many images whose true pose is known.

Those images cannot come from orbit, where the true pose of a target is
never known well enough. They are rendered by a computer, and a network
trained on rendered images loses accuracy on real ones. This is the domain
gap, and it is the central difficulty of the field. SPEED+ was built to
measure it: it contains rendered images for training, and images of a
mock-up taken in a laboratory under realistic illumination for testing.

## What this project does

The goal is a pose estimation chain that takes an image and returns a pose
with its uncertainty, validated on the laboratory images of SPEED+, and a
model of its errors that can be flown in the simulation of the mission to
show what it changes. The chain has the structure that the competitions
found most accurate: a network finds a set of known points of the target in
the image, and geometry turns those points into a pose. The book follows
that structure, starting from the geometry, because it is the part that
can be understood completely and that tells how good the network has to be.
