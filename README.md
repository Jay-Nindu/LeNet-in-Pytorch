# LeNet-in-Pytorch
Many sources claim to have implemented LeNet, but have actually greatly simplified the algorithm. This project seeks to implement LeNet exactly as specified in "GradientBased Learning Applied to Document  Recognition" by LeCun et al. This is part of a bigger project which examines two papers fundamental to modern day computer vision techniques.

## Current progress
Currently, the complete model described by LeCun et al. has been implemented in pytorch - except for their calculatino of learning rates using the diagonals of a Hessian matrix approximated with the Guass-Newton method. This is because I have run out of free GPU use for google collab.


