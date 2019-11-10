# Numerical computation of the Jacobian using CUDA

## Problem Statement
Computation of a Jacobian comes up as a part of many problems. For example, in
computation fluid dynamics application, where a numerical approximation of a derivative
must be computed at every point on a 1000x1000x1000 grid at every time step, so it is
beneficial to speed up the simulation as a whole.
