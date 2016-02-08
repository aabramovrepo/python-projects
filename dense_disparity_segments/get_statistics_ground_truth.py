#! /usr/bin/env python
# Alexey Abramov <abramov@physik3.gwdg.de>

import numpy as np
import Image as img
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys
import os
import math

import plot_data as plt

#------------------------------------------------------------------------------------------

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

#------------------------------------------------------------------------------------------
    
def get_ground_truth_map(path, scale_factor, width, height):
    
  # read ground truth from a grayscale image
  ground_truth_img = img.open(path + 'disparity_left.png')
  img_w = ground_truth_img.size[0]
  
  print 'img_w = ', img_w
  print 'width = ', width
  
  ground_truth_img = ground_truth_img.resize((width, height), img.NEAREST) 
  arr_ground_truth = np.asarray(ground_truth_img, dtype=float)  
  width_scale = float(img_w) / width
  
  print 'width_scale = ', width_scale
  
  arr_ground_truth = arr_ground_truth / (scale_factor*width_scale)
  
  return arr_ground_truth

#------------------------------------------------------------------------------------------

def evaluate_disparity(disparity, ground_truth):
  
  h = disparity.shape[0]
  w = disparity.shape[1]
  
  # percentage of found correspondences
  found_corresp = (disparity > 0)
  indices = found_corresp.nonzero()[0]
  nmb_found_corresp = len(indices)
  rate = (float(nmb_found_corresp) / float(h*w))*100

  # RMS error between the computed and ground truth disparity maps (measured in disparity units)
  diff = abs(disparity - ground_truth) * found_corresp * (ground_truth > 0)
  error_map_pow = pow(diff,2)
  rms = error_map_pow.sum() / (h*w)
  rms = math.sqrt(rms)
  
  return rms, rate

#------------------------------------------------------------------------------------------

def get_statistics_current_disparity(path, ground_truth, disparity, min_val_disp, max_val_disp):

  benchmark_path = path + 'benchmarks/'
  
  rms, rate = evaluate_disparity(disparity, ground_truth)  
  disparity[ (disparity == 0).nonzero() ] = 'nan'
  
  plt.plot_array('Current disparity map', disparity, min_val_disp, max_val_disp, benchmark_path + 'maps/' + 'current-method')
  
  return rms, rate

#------------------------------------------------------------------------------------------

def get_statistics_from_image(path, ground_truth, benchmark_name, height, width, scale_factor, mrf_factor, min_val_disp, max_val_disp, occlusion):

  benchmark_path = path + 'benchmarks/'
  basename = os.path.splitext(benchmark_name)[0]
  
  # read found disparity map by one of benchmarks
  arr_final_disparity_img = img.open(benchmark_path + benchmark_name)
  img_w, img_h = arr_final_disparity_img.size
  width_scale = float(img_w) / width
  
  arr_final_disparity_img = arr_final_disparity_img.resize((width, height), img.NEAREST) 
  arr_final_disparity = np.asarray(arr_final_disparity_img, dtype=float)

  # disparity maps from MRF software need to be rescaled (see the scaling factor in the MRF output)
  arr_final_disparity = arr_final_disparity / (mrf_factor*width_scale)
  arr_final_disparity[:,0:occlusion] = 0
  
  rms, rate = evaluate_disparity(arr_final_disparity, ground_truth)
  
  arr_final_disparity[ (arr_final_disparity == 0).nonzero() ] = 'nan'
  plt.plot_array('Final disparity map (' + basename + ')', arr_final_disparity, min_val_disp, max_val_disp, benchmark_path + 'maps/' + basename)
  
  return rms, rate

#------------------------------------------------------------------------------------------

def get_statistics_from_txt(path, ground_truth, benchmark_name, height, width, scale_factor, mrf_factor, min_val_disp, max_val_disp, occlusion):

  benchmark_path = path + 'benchmarks/'
  basename = os.path.splitext(benchmark_name)[0]
  
  # read the map to be evaluated
  arr_final_disparity = np.loadtxt(benchmark_path + benchmark_name, dtype=float)
  arr_final_disparity = abs(arr_final_disparity)
  w = arr_final_disparity.shape[1]
  width_scale = float(w) / width
  
  im = img.fromarray(arr_final_disparity)
  im = im.resize((width, height), img.NEAREST)
  arr_final_disparity = np.asarray(im, dtype=float)
  
  arr_final_disparity = arr_final_disparity / (mrf_factor*width_scale)
  arr_final_disparity[:,0:occlusion] = 0
  rms, rate = evaluate_disparity(arr_final_disparity, ground_truth)
  
  arr_final_disparity[ (arr_final_disparity == 0).nonzero() ] = 'nan'  
  plt.plot_array('Final disparity map (' + basename + ')', arr_final_disparity, min_val_disp, max_val_disp, benchmark_path + 'maps/' + basename)
  
  return rms, rate

#------------------------------------------------------------------------------------------

def get_statistics(path, scale_factor, disparity, min_val_disp, max_val_disp, occlusion):

  # size of disparity maps that need to be anlysed
  width = disparity.shape[1]
  height = disparity.shape[0]

  # format of the benchmark entry:
  #
  #  < local name of the disparity file >
  #  < MRF factor > WARNING maps from the MRF methods need to be rescaled (see the factor in the output)

  benchmarks = ( ('BM.dat', 1), ('BP-M.png', 4), ('BP-S.png', 4), ('BP.dat', 1),
                 ('CSBP.dat', 1), ('Expansion.png', 4), ('ICM.png', 4), ('phase-based.dat', 1),
                 ('Swap.png', 4), ('TRW-S.png', 4)
               )

  # dictionary: name of the method -> RMS error, percentage of found values
  statistics_output = {}

  # get the ground-truth map of the pre-defined size with the proper scaling
  ground_truth = get_ground_truth_map(path, scale_factor, width, height)

  # evaluate disparity estimated by the current method
  disparity = np.nan_to_num(disparity)
  rms, rate = get_statistics_current_disparity(path, ground_truth, disparity, min_val_disp, max_val_disp)
  
  statistics_output['Current'] = (rms,rate)

  # evaluate maps given by other techniques
  for b in benchmarks:
    
    ext = os.path.splitext(b[0])[1][1:]
    mrf_factor = b[1]
    
    rms = 0.0
    rate = 0.0
    
    if ext == 'png':
      rms, rate = get_statistics_from_image(path, ground_truth, b[0], height, width, scale_factor, mrf_factor, min_val_disp, max_val_disp, occlusion)
      
    elif ext == 'dat':
      rms, rate = get_statistics_from_txt(path, ground_truth, b[0], height, width, scale_factor, mrf_factor, min_val_disp, max_val_disp, occlusion)
      
    basename = os.path.splitext(b[0])[0]
    statistics_output[basename] = (rms,rate)


  ground_truth[ (ground_truth == 0).nonzero() ] = 'nan'  
  plt.plot_array('Ground truth', ground_truth, min_val_disp, max_val_disp, path + 'benchmarks/' + 'maps/' + './ground-truth')

  return statistics_output

#------------------------------------------------------------------------------------------
