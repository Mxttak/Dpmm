This a C++ implementaion of the code for a Gaussian Dirichlet Process Mixture Model with Monte Carlo sampling by Jacob Eisenstein that is available at https://github.com/jacobeisenstein/DPMM

It uses
* Eigen (not included)
* digamma function by Richard Mathar that is available at http://www2.mpia-hd.mpg.de/~mathar/progs/digamma.c

The demo code uses
* two original functions by Jacob (addNewClass and unhideObservations, in /dpmm)
* Gaussian Mixture distribution from the Nonlinear Estimation Toolbox by Jannik Steinbring that is availablee at http://nonlinearestimation.bitbucket.org/ (in /NonlinearEstimationToolbox)
* a function for plotting of Gaussian ellipsoids by Gautam Vallabha (in /misc)

Compilation (if Eigen is in the folder Eigen):
mex gibbsSampling.cpp -IEigen

For a demo, run simmain.

Tested on Win 10 x64.

Copyright (C) 2016 Maxim Dolgov: m<dot>dolgov<at>web<dot>de
No warranty, no commercial use.

Definitely contains errors; use with caution.