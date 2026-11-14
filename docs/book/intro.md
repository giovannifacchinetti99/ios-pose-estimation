# IOS Pose Estimation

`ios-pose-estimation` is the vision part of a rendezvous and docking system:
the software that looks at a target spacecraft through a camera and returns
where it is and how it is oriented, together with an honest statement of
how wrong that answer may be. This book is its theoretical companion. It
explains the problem, derives the methods from first principles, and
validates each of them in a notebook.

The idea behind the project is a sensor subsystem for the last metres of a
docking: a camera, a light source of its own, and the software of this
repository. The book starts from the reason such a subsystem is needed,
which is not obvious, and then builds the software one layer at a time.

## Where this comes from

The project grew out of
[ios-mission-planner](https://giovannifacchinetti99.github.io/ios-mission-planner/),
a toolkit that plans and simulates the approach and docking of a servicer
to a target in closed loop, with sensors, navigation, guidance, control and
actuators. Its Monte Carlo campaigns showed that the docking succeeds in
every run as long as the target holds its attitude, and that it fails in
half of the runs as soon as the target does not. The reason is that the
sensor of the servicer measures where the target is, not how it is
oriented. {doc}`problem/problem` tells this story in detail, because it is
the requirement from which everything else follows.

## Road map

Part I, *The problem*, explains why measuring the relative attitude decides
the success of a docking, which accuracy is needed, why lighting is a
problem of its own, and why a camera is the sensor of choice for the last
metres.

Part II, *Geometry*, describes how a camera forms an image, how a pose is
represented, and the SPEED+ dataset on which the methods are developed and
measured. It then derives how a pose is recovered from points recognized
in an image, and how accurate it can be.

The parts that follow add the recognition of those points with a neural
network, the evaluation on images taken in a laboratory, and the model of
the sensor that goes back into the simulation of the mission.

## The dataset

The methods are developed on SPEED+, the dataset of the 2021 Satellite Pose
Estimation Competition of ESA's Kelvins platform, built by the Space
Rendezvous Laboratory of Stanford University: images of the Tango
spacecraft of the PRISMA mission, synthetic and taken in a laboratory, each
labeled with the true pose. SPEED+ is distributed under the Creative
Commons Attribution-NonCommercial-ShareAlike 4.0 license. It is used here to
learn and to validate the methods; its images are not reproduced in this
book, which shows only quantities computed from them, and a product would
be trained on images of its own.

## Conventions

Lengths are in metres, angles in radians in the code and in degrees in the
plots and tables. The camera frame has its origin at the optical center,
its $z$ axis along the optical axis toward the scene, $x$ to the right of
the image and $y$ downward. Attitudes are unit quaternions with the scalar
part first, in the Hamilton convention, describing the rotation from one
frame to another in the passive sense, as in SPEED+ and in
ios-mission-planner. Vectors are in bold, $\vect r$, and their magnitude in
italics, $r$.
