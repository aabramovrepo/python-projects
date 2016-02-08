#! /usr/bin/env python
# Alexey Abramov <abramov@physik3.gwdg.de>

import numpy as np

#------------------------------------------------------------------------------------------

def estimate_weights(X, Z, A, abs_res_scale, segm_regr_w, disp_shape):

  disp_segm_rob = np.zeros(disp_shape, dtype=float)    # for robust disparity values
  segment_weights = np.zeros(disp_shape, dtype=float)  # for robust weights

  rows, cols = Z.nonzero()  
  Z_fl = Z[Z.nonzero()].flatten()
  Z = np.reshape(Z_fl, ([len(Z_fl),1]))

  # to keep only robust sparse disparity estimations
  Z_robust = np.zeros(Z.shape)
  Z_robust[:] = Z

  # compute absolute value of residuals (fit minus data)
  abs_resid = abs(np.dot(X,A) - Z)
  
  # get new median absolute residual
  mad = np.median(abs_resid)
  
  # weights using original median absolute residual
  w = abs_resid / abs_res_scale

  # compute the robust bisquare weights excluding outliers
  outliers = (w > 1)*1
  w[ outliers.nonzero() ] = 0
  Z_robust[ outliers.nonzero() ] = 0
      
  good_values = (w != 0)*1

  # calculate robust weights for 'good' points
  # Note that if you supply your own regression weight vector,
  # the final weight is the product of the robust weight and the regression weight.
  tmp = 1 - np.power(w[ good_values.nonzero() ], 2)
  w[ good_values.nonzero() ] = np.power(tmp, 2) * segm_regr_w[ good_values.nonzero() ]

  # create segment's disparity and weights without outliers
  Z_robust = np.reshape(Z_robust, ([1,len(Z_robust)]))
  w = np.reshape(w, ([1,len(w)]))
  
  ind = 0
    
  for row, col in zip(rows,cols):
    disp_segm_rob[row,col] = Z_robust[0][ind]
    segment_weights[row,col] = w[0][ind]
    ind += 1

  return mad, segment_weights, disp_segm_rob

#------------------------------------------------------------------------------------------
