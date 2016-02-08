#! /usr/bin/env python
# Alexey Abramov <abramov@physik3.gwdg.de>

import numpy as np

#------------------------------------------------------------------------------------------

def get_dense_disparity_regular_lsq(X, Z):

  Z_fl = Z[Z.nonzero()].flatten()
  Z = np.reshape(Z_fl, ([len(Z_fl),1]))

  # get the linear least squares solution for X*A = Z, A = Z \ X
  A_lsq = np.linalg.lstsq(X,Z)[0]

  return A_lsq

#------------------------------------------------------------------------------------------
