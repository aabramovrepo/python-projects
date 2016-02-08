#! /usr/bin/env python
# Alexey Abramov <abramov@physik3.gwdg.de>

import numpy as np

#------------------------------------------------------------------------------------------

def complement_initial_disparity(segments, disparity_init, disparity_consist_input):
  
  for label in set(np.unique(segments)).difference((0,)):
  
    segment_mask = (segments == label)
    disparity_mask_init = (disparity_init > 0)*segment_mask
    disparity_mask_consist_input = (disparity_consist_input > 0)*segment_mask

    disparity_rate = float(disparity_mask_init.sum()) / float(segment_mask.sum())

    # for segments with a few estimations in the initial map
    # take estimations from the input consistent map
    if disparity_rate < 0.25:
      disparity_init[ segment_mask.nonzero() ] = disparity_consist_input[ segment_mask.nonzero() ]
    
  return disparity_init

#------------------------------------------------------------------------------------------

def disparity_interpolation(disparity, segments, occlusion_range):

  # do not touch the occlusion area
  disparity[:,0:occlusion_range] = -1

  indices = (disparity == 0).nonzero()
  
  for ind in range( len(indices[0]) ):
    
    y = indices[0][ind]
    x = indices[1][ind]
    
    # for non-border pixels
    if y > 0 and y < (disparity.shape[0]-1) and x > 0 and x < (disparity.shape[1]-1):
      
      w_disparity = np.zeros([3,3], dtype=disparity.dtype)
      w_disparity[:,:] = disparity[y-1:y+2, x-1:x+2]
      disparity[y,x] = float( w_disparity.sum() ) / float( (w_disparity != 0).sum() )
    
    # for the upper border
    if y == 0 and x < (disparity.shape[1]-1):

      w_disparity = np.zeros([2,3], dtype=disparity.dtype)
      w_disparity[:,:] = disparity[y:y+2, x-1:x+2]
      disparity[y,x] = float( w_disparity.sum() ) / float( (w_disparity != 0).sum() )

    # for the bottom border
    if y == (disparity.shape[0]-1) and x < (disparity.shape[1]-1):

      w_disparity = np.zeros([2,3], dtype=disparity.dtype)
      w_disparity[:,:] = disparity[y-1:y+1, x-1:x+2]
      disparity[y,x] = float( w_disparity.sum() ) / float( (w_disparity != 0).sum() )
      
    # for the right border
    if x == (disparity.shape[1]-1) and y > 0 and y < (disparity.shape[0]-1):

      w_disparity = np.zeros([3,2], dtype=disparity.dtype)
      w_disparity[:,:] = disparity[y-1:y+2, x-1:x+1]
      disparity[y,x] = float( w_disparity.sum() ) / float( (w_disparity != 0).sum() )

    # upper right corner
    if y == 0 and x == (disparity.shape[1]-1):

      w_disparity = np.zeros([2,2], dtype=disparity.dtype)
      w_disparity[:,:] = disparity[y:y+2, x-1:x+1]
      disparity[y,x] = float( w_disparity.sum() ) / float( (w_disparity != 0).sum() )

    # bottom right corner
    if y == (disparity.shape[0]-1) and x == (disparity.shape[1]-1):

      w_disparity = np.zeros([2,2], dtype=disparity.dtype)
      w_disparity[:,:] = disparity[y-1:y+1, x-1:x+1]
      disparity[y,x] = float( w_disparity.sum() ) / float( (w_disparity != 0).sum() )

  # fill the occlusion are by zeros again
  disparity[:,0:occlusion_range] = 0
  
  return disparity
  
#------------------------------------------------------------------------------------------
