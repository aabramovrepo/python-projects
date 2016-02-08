#! /usr/bin/env python
# Alexey Abramov <abramov@physik3.gwdg.de>

import numpy as np
import datetime as dt

# local imports
import robust_lsq
import estimate_weights
import plot_data as plt

#------------------------------------------------------------------------------------------

def surface_fitting_robust_lsq(local_folder, arr_segments, arr_init_disp, arr_edge_disp, disp_min, disp_max):

  # get segment labels
  labels = set(np.unique(arr_segments)).difference((0,))
  #print 'labels = ', labels

  # matrix with regression weights
  arr_regr_weights_init = np.zeros(arr_init_disp.shape, dtype=float)
  
  arr_regr_weights_init[ arr_init_disp.nonzero() ] = 0.3
  arr_regr_weights_init[ arr_edge_disp.nonzero() ] = 1.0

  # initial sparse disparity maps without outliers for a specific model
  disparity_init_const = np.zeros(arr_init_disp.shape, dtype=float)
  disparity_init_planar = np.zeros(arr_init_disp.shape, dtype=float)
  disparity_init_quadric = np.zeros(arr_init_disp.shape, dtype=float)

  # weights for sparse disparity values without outliers
  weights_const = np.zeros(arr_init_disp.shape, dtype=float)
  weights_planar = np.zeros(arr_init_disp.shape, dtype=float)
  weights_quadric = np.zeros(arr_init_disp.shape, dtype=float)

  # dense disparity maps obtained using a specific model
  disparity_const = np.zeros(arr_init_disp.shape, dtype=float)
  disparity_planar = np.zeros(arr_init_disp.shape, dtype=float)
  disparity_quadric = np.zeros(arr_init_disp.shape, dtype=float)

  # final dense disparity map after the model selection
  disparity_final = np.zeros(arr_init_disp.shape, dtype=float)

  t1 = dt.datetime.now()

  # go over all segments and fit sparse disparity input to the model
  for label in labels:
  
    #print ''
    #print 'current label = ', label
  
    # masks of the considered segment and its sparse disparity map
    segment_mask = (arr_segments == label)*1
    disparity_mask = (arr_init_disp > 0)*segment_mask*1

    # disparity of the current segment
    disparity = arr_init_disp * disparity_mask
    
    if disparity.sum() == 0:
      continue
    
    segment_regr_weights = np.zeros(disparity.shape, dtype=float)
    segment_regr_weights = (disparity > 0) * arr_regr_weights_init
 
    # get the robust least squares solution for the current segment
    # and for const, planar, and quadric models
    
    # get indices of given disparity values
    indices = disparity.nonzero()

    x = np.zeros(len(indices[1]))
    y = np.zeros(len(indices[0]))

    x[:] = indices[1]
    y[:] = indices[0]

    z_ones = np.ones([x.size,1])
    Z = disparity

    segment_regr_weights_fl = segment_regr_weights[ segment_regr_weights.nonzero() ].flatten()
    segment_regr_weights = np.reshape(segment_regr_weights_fl, ([len(segment_regr_weights_fl),1]))

    # constant model
    X = z_ones
    a_lsq_const, abs_res_scale_const = robust_lsq.get_dense_disparity_robust_lsq(X,Z,segment_regr_weights)
    mad_const, w_const, d_const = estimate_weights.estimate_weights(X,Z,a_lsq_const,abs_res_scale_const,segment_regr_weights,disparity.shape)
    
    # planar model
    X = np.hstack((np.reshape(x, ([len(x),1])), np.reshape(y, ([len(y),1])), z_ones))
    a_lsq_planar, abs_res_scale_planar = robust_lsq.get_dense_disparity_robust_lsq(X,Z,segment_regr_weights)
    mad_planar, w_planar, d_planar = estimate_weights.estimate_weights(X,Z,a_lsq_planar,abs_res_scale_planar,segment_regr_weights,disparity.shape)
    
    # quadric model
    x2 = x**2
    y2 = y**2
    xy = x*y

    #v1 = np.reshape(x2, ([len(x2),1]))
    #v2 = np.reshape(y2, ([len(y2),1]))
    #v3 = np.reshape(x, ([len(x),1]))
    #v4 = np.reshape(y, ([len(y),1]))
    
    v1 = np.reshape(x2, ([len(x2),1]))
    v2 = np.reshape(y2, ([len(y2),1]))
    v3 = np.reshape(xy, ([len(xy),1]))
    v4 = np.reshape(x, ([len(x),1]))
    v5 = np.reshape(y, ([len(y),1]))

    #X = np.hstack((v1, v2, v3, v4, z_ones))
    X = np.hstack((v1, v2, v3, v4, v5, z_ones))
    a_lsq_quadric, abs_res_scale_quadric = robust_lsq.get_dense_disparity_robust_lsq(X,Z,segment_regr_weights)
    mad_quadric, w_quadric, d_quadric = estimate_weights.estimate_weights(X,Z,a_lsq_quadric,abs_res_scale_quadric,segment_regr_weights,disparity.shape)

    # calculate disparity for the current segment based on the found model
    segment_indices = segment_mask.nonzero()
  
    disparity_init_const[segment_indices] = d_const[segment_indices]
    disparity_init_planar[segment_indices] = d_planar[segment_indices]
    disparity_init_quadric[segment_indices] = d_quadric[segment_indices]

    weights_const[segment_indices] = w_const[segment_indices]
    weights_planar[segment_indices] = w_planar[segment_indices]
    weights_quadric[segment_indices] = w_quadric[segment_indices]
  
    x_s = np.zeros(len(segment_indices[1]))
    y_s = np.zeros(len(segment_indices[0]))

    x_s[:] = segment_indices[1]
    y_s[:] = segment_indices[0]

    # calculate disparity values according to found models
    disparity_const[ segment_indices ] = a_lsq_const[0]
    disparity_planar[ segment_indices ] = a_lsq_planar[0]*x_s + a_lsq_planar[1]*y_s + a_lsq_planar[2]
    #disparity_quadric[ segment_indices ] = a_lsq_quadric[0]*x_s*x_s + a_lsq_quadric[1]*y_s*y_s + a_lsq_quadric[2]*x_s + a_lsq_quadric[3]*y_s + a_lsq_quadric[4]
    disparity_quadric[ segment_indices ] = a_lsq_quadric[0]*x_s*x_s + a_lsq_quadric[1]*y_s*y_s + a_lsq_quadric[2]*x_s*y_s + a_lsq_quadric[3]*x_s + a_lsq_quadric[4]*y_s + a_lsq_quadric[5]
    
    # select the best model    
    winner = np.argmin([mad_const, mad_planar, mad_quadric])
    
    if winner == 0:
      disparity_final[ segment_indices ] = disparity_const[ segment_indices ]
    elif winner == 1:
      disparity_final[ segment_indices ] = disparity_planar[ segment_indices ]
    elif winner == 2:
      disparity_final[ segment_indices ] = disparity_quadric[ segment_indices ]     

  # calculated elapsed time
  t2 = dt.datetime.now()
  ms = (t2 - t1).microseconds / 1e3
  print 'Robust LSQ, elapsed time = ', ms, ' ms'
  print ''


  # replace zeros by NaN values just for displaying
  disparity_init_const[ (disparity_init_const == 0).nonzero() ] = 'nan'
  disparity_init_planar[ (disparity_init_planar == 0).nonzero() ] = 'nan'
  disparity_init_quadric[ (disparity_init_quadric == 0).nonzero() ] = 'nan'
  
  arr_regr_weights_init[ (arr_regr_weights_init == 0).nonzero() ] = 'nan'
  
  weights_const[ (weights_const == 0).nonzero() ] = 'nan'
  weights_planar[ (weights_planar == 0).nonzero() ] = 'nan'
  weights_quadric[ (weights_quadric == 0).nonzero() ] = 'nan'
 
  plt.plot_array('Initial disparity (robust, const)', disparity_init_const, disp_min, disp_max, local_folder + 'init-disparity-rob-const')
  plt.plot_array('Initial disparity (robust, planar)', disparity_init_planar, disp_min, disp_max, local_folder + 'init-disparity-rob-planar')
  plt.plot_array('Initial disparity (robust, quadric)', disparity_init_quadric, disp_min, disp_max, local_folder + 'init-disparity-rob-quadric')
  
  plt.plot_array('Regression weights', arr_regr_weights_init, 0, 1, local_folder + 'regression-weight')

  w_min = 0.0
  w_max = 0.37
  
  plt.plot_array('Weights (robust, const)', weights_const, w_min, w_max, local_folder + 'weights-rob-const')
  plt.plot_array('Weights (robust, planar)', weights_planar, w_min, w_max, local_folder + 'weights-rob-planar')
  plt.plot_array('Weights (robust, quadric)', weights_quadric, w_min, w_max, local_folder + 'weights-rob-quadric')
  
  return disparity_const, disparity_planar, disparity_quadric, disparity_final
 
#------------------------------------------------------------------------------------------
