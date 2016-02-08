#! /usr/bin/env python
# Alexey Abramov <abramov@physik3.gwdg.de>

import numpy as np
import Image as img
import scipy as sp

import plot_data as plt

#---------------------------------------------------------------------------------------------------------------------------------------------------

def get_true_occlusions(disparities):

    disparities = disparities.round()
    occlusions = np.zeros(disparities.shape, dtype=bool)
    height, width = disparities.shape

    for row in range(height):
        for col in range(width):
            if disparities[row, col] > col:
                # Pixel moves out of image (not visible in right image)
                occlusions[row, col] = True

            match = np.arange(width - col) == disparities[row, col:width]
            indexes = match.nonzero()[0] + col

            if len(indexes) > 1:
                # Multiple pixels map to this one (occluded in right image)
                occlusions[row, indexes[:-1]] = True
            elif col > 3 and not occlusions[row, col-1] and occlusions[row, col-2] and not occlusions[row, col-3] and not occlusions[row, col-4]:
                # Left neighbor is not really occluded. These errors happen because
                # the disparity map has integer precision.
                occlusions[row, col-2] = False

    return occlusions

#---------------------------------------------------------------------------------------------------------------------------------------------------

def extend_border_horizontal(arr):

  h = arr.shape[0]
  w = arr.shape[1]

  for i in range(0,h):
    for j in range(0,w):

      if(arr[i,j] != 0 and j > 0):
          arr[i,j-1] = arr[i,j]

      if(arr[i,j] != 0 and (j-1) > 1):
          arr[i,j-2] = arr[i,j]

      if(arr[i,j] != 0 and (j-2) > 1):
          arr[i,j-3] = arr[i,j]

#---------------------------------------------------------------------------------------------------------------------------------------------------

def extend_border_vertical(arr):

  h = arr.shape[0]
  w = arr.shape[1]

  for j in range(0,w):
    for i in range(0,h):

      if(arr[i,j] != 0 and i > 0):
          arr[i-1,j] = arr[i,j]

      if(arr[i,j] != 0 and (i-1) > 0):
          arr[i-2,j] = arr[i,j]

      if(arr[i,j] != 0 and (i-2) > 0):
          arr[i-3,j] = arr[i,j]

#---------------------------------------------------------------------------------------------------------------------------------------------------

def extend_sigma_horizontal(arr,val):

  h = arr.shape[0]
  w = arr.shape[1]

  for i in range(0,h):
    for j in range(0,w):

      if(arr[i,j] == val and j > 0):
          arr[i,j-1] = arr[i,j]

      if(arr[i,j] == val and (j-1) > 1):
          arr[i,j-2] = arr[i,j]

      if(arr[i,j] == val and (j-2) > 1):
          arr[i,j-3] = arr[i,j]

#---------------------------------------------------------------------------------------------------------------------------------------------------

def extend_sigma_vertical(arr,val):

  h = arr.shape[0]
  w = arr.shape[1]

  for j in range(0,w):
    for i in range(0,h):

      if(arr[i,j] == val and i > 0):
          arr[i-1,j] = arr[i,j]

      if(arr[i,j] == val and (i-1) > 0):
          arr[i-2,j] = arr[i,j]

      if(arr[i,j] == val and (i-2) > 0):
          arr[i-3,j] = arr[i,j]

#---------------------------------------------------------------------------------------------------------------------------------------------------

def view_disparity(path, disparity_min, disparity_max, occlusion_range):

  arr_sparse_disparity_incons = abs(np.loadtxt(path+'sparse_disparity_incons.dat'))
  arr_sparse_disparity_cons = abs(np.loadtxt(path+'sparse_disparity_cons.dat'))

  arr_average_disparity = abs(np.loadtxt(path+'average-disparity.dat'))
  arr_average_disparity_lines = abs(np.loadtxt(path+'average-disparity-lines.dat'))
  arr_edge_disparity = abs(np.loadtxt(path+'edge-disparity.dat'))
  arr_edge_disparity_final = abs(np.loadtxt(path+'edge-disparity-final.dat'))
  arr_initial_disparity = abs(np.loadtxt(path+'initial-disparity.dat'))
  arr_sigma_map = np.loadtxt(path+'sigma-map.dat')
  arr_occlusion_map = np.loadtxt(path+'occlusion-map.dat')
  arr_outliers_map = np.loadtxt(path+'outliers-map.dat')
  arr_edge_disparity_mask = np.loadtxt(path+'edge-disparity-mask.dat')
  arr_sparse_filtered = abs(np.loadtxt(path+'sparse-disparity-filtered.dat'))

  # get size
  height = arr_sparse_disparity_cons.shape[0]
  width = arr_sparse_disparity_cons.shape[1]

  print 'height = ', height
  print 'width = ', width

  # plot inconsistent sparse disparity map
  min_val = disparity_min
  max_val = disparity_max

  arr_sparse_disparity_incons[0:height,0:occlusion_range] = 'nan'
  arr_average_disparity[0:height,0:occlusion_range] = 'nan'
  arr_average_disparity_lines[0:height,0:occlusion_range] = 'nan'
  arr_edge_disparity[0:height,0:occlusion_range] = 'nan'
  arr_edge_disparity_final[0:height,0:occlusion_range] = 'nan'
  arr_initial_disparity[0:height,0:occlusion_range] = 'nan'
  arr_occlusion_map[0:height,0:occlusion_range] = 'nan'
  arr_edge_disparity_mask[0:height,0:occlusion_range] = 'nan'
  arr_sigma_map[0:height,0:occlusion_range] = 'nan'
  arr_outliers_map[0:height,0:occlusion_range] = 'nan'

  plt.plot_array('Inconsistent sparse disparity map (phase-based)', arr_sparse_disparity_incons, min_val, max_val, path+'sparse-disparity-incons')

  plt.plot_array('Consistent sparse disparity map (phase-based)', arr_sparse_disparity_cons, min_val, max_val, path+'sparse-disparity-cons')
  plt.plot_array('Filtered sparse disparity map', arr_sparse_filtered, min_val, max_val, path+'sparse-disparity-filtered')
  plt.plot_array('Average disparity', arr_average_disparity, min_val, max_val, path+'average-disparity')
  plt.plot_array('Average disparity (lines)', arr_average_disparity_lines, min_val, max_val, path+'average-disparity-lines')

  # make borders thicker
  extend_border_horizontal(arr_edge_disparity)
  extend_border_vertical(arr_edge_disparity)

  extend_border_horizontal(arr_edge_disparity_final)
  extend_border_vertical(arr_edge_disparity_final)

  extend_sigma_horizontal(arr_sigma_map,0.3)
  extend_sigma_vertical(arr_sigma_map,0.3)

  # replace zeros by NaN values
  for i in range(1,height):
    for j in range(1,width):

      if(arr_edge_disparity_final[i,j] != 0):
        arr_initial_disparity[i,j] = arr_edge_disparity_final[i,j]
    
      if(arr_edge_disparity[i,j] == 0):
        arr_edge_disparity[i,j] = 'nan'

      if(arr_edge_disparity_final[i,j] == 0):
        arr_edge_disparity_final[i,j] = 'nan'

      if(arr_edge_disparity_mask[i,j] == 0):
        arr_edge_disparity_mask[i,j] = 'nan'

      if(arr_occlusion_map[i,j] == 0):
        arr_occlusion_map[i,j] = 'nan'

      if(arr_outliers_map[i,j] == 0):
        arr_outliers_map[i,j] = 'nan'

  plt.plot_array('Edge disparity', arr_edge_disparity, min_val, max_val, path+'edge-disparity')
  plt.plot_array('Final edge disparity', arr_edge_disparity_final, min_val, max_val, path+'edge-disparity-final')

  plt.plot_array('Initial disparity', arr_initial_disparity, min_val, max_val, path+'initial-disparity')

  #plot_array('Sigma map', arr_sigma_map, arr_sigma_map.min(), arr_sigma_map.max(), 'sigma-map')
  plt.plot_array('Sigma map', arr_sigma_map, 0, 1, path+'sigma-map')

  #plot_array('Occlusion map', arr_occlusion_map, arr_occlusion_map.min(), arr_occlusion_map.max(), 'occlusion-map')
  plt.plot_array('Occlusion map', arr_occlusion_map, 0, 1, path+'occlusion-map')
  plt.plot_array('Outliers map', arr_outliers_map, 0, 1, path+'outliers-map')
  plt.plot_array('Edge disparity mask', arr_edge_disparity_mask, 0, 1, path+'edge-disparity-mask')
  
#---------------------------------------------------------------------------------------------------------------------------------------------------
