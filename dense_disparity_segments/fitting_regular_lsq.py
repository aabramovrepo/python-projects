#! /usr/bin/env python
# Alexey Abramov <abramov@physik3.gwdg.de>

import numpy as np
import datetime as dt

# local imports
import regular_lsq

#------------------------------------------------------------------------------------------

def surface_fitting_regular_lsq(arr_segments, arr_init_disp):

  # get segment labels
  labels = set(np.unique(arr_segments)).difference((0,))
  print 'labels = ', labels

  disparity_const = np.zeros(arr_init_disp.shape, dtype=float)
  disparity_planar = np.zeros(arr_init_disp.shape, dtype=float)
  disparity_quadric = np.zeros(arr_init_disp.shape, dtype=float)

  t1 = dt.datetime.now()
  
  # go over all segments and fit sparse disparity input to the model
  for label in labels:

    # masks of the considered segment and its sparse disparity map
    segment_mask = (arr_segments == label)*1
    disparity_mask = (arr_init_disp > 0)*segment_mask*1

    # disparity of the current segment
    disparity = arr_init_disp * disparity_mask
    
    if disparity.sum() == 0:
      continue
    
    # get the least squares solution for the current segment
    # and for both planar and quadric models
    
    indices = disparity.nonzero()
    
    x = np.zeros(len(indices[1]))
    y = np.zeros(len(indices[0]))

    x[:] = indices[1]
    y[:] = indices[0]
    
    z_ones = np.ones([x.size,1])
    Z = disparity
    
    # constant model
    X = z_ones
    a_lsq_const = regular_lsq.get_dense_disparity_regular_lsq(X,Z)
    
    # planar model
    X = np.hstack((np.reshape(x, ([len(x),1])), np.reshape(y, ([len(y),1])), z_ones))
    a_lsq_planar = regular_lsq.get_dense_disparity_regular_lsq(X,Z)

    # quadric model
    x2 = x**2
    y2 = y**2
    xy = x*y

    #v1 = np.reshape(x2, ([len(x2),1]))
    #v2 = np.reshape(y2, ([len(y2),1]))
    #v3 = np.reshape(x, ([len(x),1]))
    #v4 = np.reshape(y, ([len(y),1]))
    #v5 = np.reshape(xy, ([len(xy),1]))

    v1 = np.reshape(x2, ([len(x2),1]))
    v2 = np.reshape(y2, ([len(y2),1]))
    v3 = np.reshape(xy, ([len(xy),1]))
    v4 = np.reshape(x, ([len(x),1]))
    v5 = np.reshape(y, ([len(y),1]))

    #X = np.hstack((v1, v2, v3, v4, z_ones))
    X = np.hstack((v1, v2, v3, v4, v5, z_ones))
    a_lsq_quadric = regular_lsq.get_dense_disparity_regular_lsq(X,Z)

    # calculate disparity for the current segment based on the found model
    segment_indices = segment_mask.nonzero()
  
    x_s = np.zeros(len(segment_indices[1]))
    y_s = np.zeros(len(segment_indices[0]))

    x_s[:] = segment_indices[1]
    y_s[:] = segment_indices[0]

    # insert the found dense map into the final disparity map
    disparity_const[ segment_indices ] = a_lsq_const[0]
    disparity_planar[ segment_indices ] = a_lsq_planar[0]*x_s + a_lsq_planar[1]*y_s + a_lsq_planar[2]
    #disparity_quadric[ segment_indices ] = a_lsq_quadric[0]*x_s*x_s + a_lsq_quadric[1]*y_s*y_s + a_lsq_quadric[2]*x_s + a_lsq_quadric[3]*y_s + a_lsq_quadric[4]
    disparity_quadric[ segment_indices ] = a_lsq_quadric[0]*x_s*x_s + a_lsq_quadric[1]*y_s*y_s + a_lsq_quadric[2]*x_s*y_s + a_lsq_quadric[3]*x_s + a_lsq_quadric[4]*y_s + a_lsq_quadric[5]

  # calculated elapsed time
  t2 = dt.datetime.now()
  ms = (t2 - t1).microseconds / 1e3
  print 'Regular LSQ, elapsed time = ', ms, ' ms'
  print ''


  return disparity_const, disparity_planar, disparity_quadric
    
#------------------------------------------------------------------------------------------
