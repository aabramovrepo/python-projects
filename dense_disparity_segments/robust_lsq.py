#! /usr/bin/env python
# Alexey Abramov <abramov@physik3.gwdg.de>

import numpy as np

#------------------------------------------------------------------------------------------

def get_dense_disparity_robust_lsq(X, Z, segment_regr_weights):

  rows, cols = Z.nonzero()  
  Z_fl = Z[Z.nonzero()].flatten()
  Z = np.reshape(Z_fl, ([len(Z_fl),1]))

  # get the linear least squares solution for X*A = Z, A = Z \ X
  A_lsq = np.linalg.lstsq(X,Z)[0]

  # start the robust least squares estimation with the solution
  # given by the non-robust least squares
  A_robust = A_lsq
  n_robust_it = 10
    
  # iterate till the fit converges
  for robust_it in range(n_robust_it):

    # compute absolute value of residuals (fit minus data)
    abs_resid = abs(np.dot(X, A_robust) - Z)

    # compute the scaling factor for the standardization of residuals
    # using the median absolute deviation of the residuals
    # 6.9460 is a tuning constant (4.685/0.6745)
      
    mad = np.median(abs_resid)
    abs_res_scale = 6.9460 * mad
    #abs_res_scale = 4.0 * mad

    # standardize residuals
    abs_res_scale = max(abs_res_scale, 0.001)
    w = abs_resid / abs_res_scale

    # compute the robust bisquare weights excluding outliers
    outliers = (w > 1)*1
    w[ outliers.nonzero() ] = 0
    good_values = (w != 0)*1

    # calculate robust weights for 'good' points
    # Note that if you supply your own regression weight vector,
    # the final weight is the product of the robust weight and the regression weight.
    tmp = 1 - np.power(w[ good_values.nonzero() ], 2)
    w[ good_values.nonzero() ] = np.power(tmp, 2) * segment_regr_weights[ good_values.nonzero() ]

    # get weighted X'es
    #XW = np.tile(w, (1, 5)) * X
    XW = w * X

    a = np.dot(XW.T, X)
    b = np.dot(XW.T, Z)

    # get the least-squares solution to a linear matrix equation
    A_robust = np.linalg.lstsq(a,b)[0]

  return A_robust, abs_res_scale

#------------------------------------------------------------------------------------------
