#! /usr/bin/env python
# Alexey Abramov <abramov@physik3.gwdg.de>

import numpy as np

# local imports
import fitting_regular_lsq
import fitting_robust_lsq
import find_small_segments
import disparity_operations
import plot_data as plt

print('')
print('***************************************************************************************')
print('***   Surface fitting for found segments based on the sparse disparity estimation   ***')
print('***                  (Alexey Abramov, abramov@physik3.gwdg.de)                      ***')
print('***************************************************************************************')
print('')

#------------------------------------------------------------------------------------------

def surface_fitting(output_path, occlusion_range, disp_min, disp_max):

  #fname_segments = './segments-left.dat'             # input left segments
  #fname_cons_disp = './sparse_disparity_cons.dat'    # consistent sparse disparity estimated by the phase-based method
  #fname_edge_disp = './edge-disparity-final.dat'     # edge disparity map
  #fname_init_disp = './initial-disparity.dat'        # initial disparity map

  fname_segments = output_path + 'segments-left.dat'             # input left segments
  fname_cons_disp = output_path + 'sparse_disparity_cons.dat'    # consistent sparse disparity estimated by the phase-based method
  fname_edge_disp = output_path + 'edge-disparity-final.dat'     # edge disparity map
  fname_init_disp = output_path + 'initial-disparity.dat'        # initial disparity map

  #local_folder = './tmp/'  # folder for final results
  #occlusion_range = 32     # range of a half-occlusion area (for left segments)
  
  # min and max disparity values for color scaling
  #disp_min = 0
  #disp_max = 50#45#48#50
  
  arr_segments = np.loadtxt(fname_segments, dtype=float)
  arr_cons_disp = abs(np.loadtxt(fname_cons_disp, dtype=float))
  arr_edge_disp = np.loadtxt(fname_edge_disp, dtype=float)
  arr_init_disp = np.loadtxt(fname_init_disp, dtype=float)

  # replace 'NaN's by zeros
  arr_init_disp = np.nan_to_num(arr_init_disp)
  arr_cons_disp = np.nan_to_num(arr_cons_disp)

  # clean input segments and find regions remained unfound after segmentation
  arr_segments, arr_segments_gaps, arr_segments_final = find_small_segments.detect_small_segments(arr_segments, occlusion_range)
  
  # save clean segments, segmentation gaps and final segments (clean segments + filled gaps)
  segment_colors = plt.generate_colors()
  find_small_segments.save_segments(output_path + 'segments-left-original', arr_segments, segment_colors)
  find_small_segments.save_segments(output_path + 'segments-left-gaps', arr_segments_gaps, segment_colors)
  find_small_segments.save_segments(output_path + 'segments-left-final', arr_segments_final, segment_colors)
  
  # for segments with a few estimations in the initial map use input consistent disparity
  arr_init_disp = disparity_operations.complement_initial_disparity(arr_segments, arr_init_disp, arr_cons_disp)
  
  # CASE 1: regular least squares (for both segment gaps and segments)
  disparity_gaps_reg_const, disparity_gaps_reg_planar, disparity_gaps_reg_quadric = fitting_regular_lsq.surface_fitting_regular_lsq(arr_segments_gaps, arr_cons_disp)
  disparity_reg_const, disparity_reg_planar, disparity_reg_quadric = fitting_regular_lsq.surface_fitting_regular_lsq(arr_segments, arr_init_disp)
  
  # go over gap labels and fill gaps in all maps by const disparity values
  for lb in set(np.unique(arr_segments_gaps)).difference((0,)):
  
    disparity_reg_const[ (arr_segments_gaps == lb).nonzero() ] = disparity_gaps_reg_const[ (arr_segments_gaps == lb).nonzero() ]
    disparity_reg_planar[ (arr_segments_gaps == lb).nonzero() ] = disparity_gaps_reg_const[ (arr_segments_gaps == lb).nonzero() ]
    disparity_reg_quadric[ (arr_segments_gaps == lb).nonzero() ] = disparity_gaps_reg_const[ (arr_segments_gaps == lb).nonzero() ]
  
  
  # CASE 2: robust least squares (for both segment gaps and segments)
  disparity_gaps_rob_const, disparity_gaps_rob_planar, disparity_gaps_rob_quadric, disparity_gaps_rob_final = fitting_robust_lsq.surface_fitting_robust_lsq(output_path, arr_segments_gaps, arr_cons_disp, arr_edge_disp, disp_min, disp_max)
  disparity_rob_const, disparity_rob_planar, disparity_rob_quadric, disparity_rob_final = fitting_robust_lsq.surface_fitting_robust_lsq(output_path, arr_segments, arr_init_disp, arr_edge_disp, disp_min, disp_max)  

  # go over gap labels and fill gaps in all maps by const disparity values
  for lb in set(np.unique(arr_segments_gaps)).difference((0,)):
  
    disparity_rob_const[ (arr_segments_gaps == lb).nonzero() ] = disparity_gaps_rob_const[ (arr_segments_gaps == lb).nonzero() ]
    disparity_rob_planar[ (arr_segments_gaps == lb).nonzero() ] = disparity_gaps_rob_const[ (arr_segments_gaps == lb).nonzero() ]
    disparity_rob_quadric[ (arr_segments_gaps == lb).nonzero() ] = disparity_gaps_rob_const[ (arr_segments_gaps == lb).nonzero() ]
    disparity_rob_final[ (arr_segments_gaps == lb).nonzero() ] = disparity_gaps_rob_const[ (arr_segments_gaps == lb).nonzero() ]

    
  # perform interpolation of disparity values for empty pixels
  disparity_reg_const = disparity_operations.disparity_interpolation(disparity_reg_const, arr_segments_final, occlusion_range)
  disparity_reg_planar = disparity_operations.disparity_interpolation(disparity_reg_planar, arr_segments_final, occlusion_range)
  disparity_reg_quadric = disparity_operations.disparity_interpolation(disparity_reg_quadric, arr_segments_final, occlusion_range)
  
  disparity_rob_const = disparity_operations.disparity_interpolation(disparity_rob_const, arr_segments_final, occlusion_range)
  disparity_rob_planar = disparity_operations.disparity_interpolation(disparity_rob_planar, arr_segments_final, occlusion_range)
  disparity_rob_quadric = disparity_operations.disparity_interpolation(disparity_rob_quadric, arr_segments_final, occlusion_range)
  disparity_rob_final = disparity_operations.disparity_interpolation(disparity_rob_final, arr_segments_final, occlusion_range)
  
  # PLOT DATA  
  
  # replace zeros by NaN to make them white in plots
  arr_edge_disp[ (arr_edge_disp == 0).nonzero() ] = 'nan'
  arr_init_disp[ (arr_init_disp == 0).nonzero() ] = 'nan'
  arr_cons_disp[ (arr_cons_disp == 0).nonzero() ] = 'nan'

  disparity_reg_const[ (disparity_reg_const == 0).nonzero() ] = 'nan'
  disparity_reg_planar[ (disparity_reg_planar == 0).nonzero() ] = 'nan'
  disparity_reg_quadric[ (disparity_reg_quadric == 0).nonzero() ] = 'nan'

  disparity_gaps_reg_const[ (disparity_gaps_reg_const == 0).nonzero() ] = 'nan'
  disparity_gaps_reg_planar[ (disparity_gaps_reg_planar == 0).nonzero() ] = 'nan'
  disparity_gaps_reg_quadric[ (disparity_gaps_reg_quadric == 0).nonzero() ] = 'nan'

  disparity_gaps_rob_const[ (disparity_gaps_rob_const == 0).nonzero() ] = 'nan'
  disparity_gaps_rob_planar[ (disparity_gaps_rob_planar == 0).nonzero() ] = 'nan'
  disparity_gaps_rob_quadric[ (disparity_gaps_rob_quadric == 0).nonzero() ] = 'nan'

  disparity_rob_const[ (disparity_rob_const == 0).nonzero() ] = 'nan'
  disparity_rob_planar[ (disparity_rob_planar == 0).nonzero() ] = 'nan'
  disparity_rob_quadric[ (disparity_rob_quadric == 0).nonzero() ] = 'nan'
  disparity_rob_final[ (disparity_rob_final == 0).nonzero() ] = 'nan'

  plt.plot_array('Consistent sparse disparity (phase-based)', arr_cons_disp, disp_min, disp_max, output_path + 'sparse-disparity-consistent')
  plt.plot_array('Edge Disparity', arr_edge_disp, disp_min, disp_max, output_path + 'edge-disparity')
  plt.plot_array('Initial Disparity', arr_init_disp, disp_min, disp_max, output_path + 'init-disparity')

  plt.plot_array('Final Disparity (regular LSQ, const)', disparity_reg_const, disp_min, disp_max, output_path + 'dense-disparity-reg-const')
  plt.plot_array('Final Disparity (regular LSQ, planar)', disparity_reg_planar, disp_min, disp_max, output_path + 'dense-disparity-reg-planar')
  plt.plot_array('Final Disparity (regular LSQ, quadric)', disparity_reg_quadric, disp_min, disp_max, output_path + 'dense-disparity-reg-quadric')

  #plt.plot_array('Gaps Disparity (regular LSQ, const)', disparity_gaps_reg_const, disp_min, disp_max, local_folder + 'dense-disparity-gaps-reg-const')
  #plt.plot_array('Gaps Disparity (regular LSQ, planar)', disparity_gaps_reg_planar, disp_min, disp_max, local_folder + 'dense-disparity-gaps-reg-planar')
  #plt.plot_array('Gaps Disparity (regular LSQ, quadric)', disparity_gaps_reg_quadric, disp_min, disp_max, local_folder + 'dense-disparity-gaps-reg-quadric')

  #plt.plot_array('Gaps Disparity (robust LSQ, const)', disparity_gaps_rob_const, disp_min, disp_max, local_folder + 'dense-disparity-gaps-rob-const')
  #plt.plot_array('Gaps Disparity (robust LSQ, planar)', disparity_gaps_rob_planar, disp_min, disp_max, local_folder + 'dense-disparity-gaps-rob-planar')
  #plt.plot_array('Gaps Disparity (robust LSQ, quadric)', disparity_gaps_rob_quadric, disp_min, disp_max, local_folder + 'dense-disparity-gaps-rob-quadric')

  plt.plot_array('Final Disparity (robust LSQ, const)', disparity_rob_const, disp_min, disp_max, output_path + 'dense-disparity-rob-const')
  plt.plot_array('Final Disparity (robust LSQ, planar)', disparity_rob_planar, disp_min, disp_max, output_path + 'dense-disparity-rob-planar')
  plt.plot_array('Final Disparity (robust LSQ, quadric)', disparity_rob_quadric, disp_min, disp_max, output_path + 'dense-disparity-rob-quadric')
  plt.plot_array('Final Disparity (robust, model selection)', disparity_rob_final, disp_min, disp_max, output_path + 'dense-disparity-rob-final')
  
  # save estimated disparities as text files
  np.savetxt(output_path + 'dense-disparity-reg-const.dat', disparity_reg_const)
  np.savetxt(output_path + 'dense-disparity-reg-planar.dat', disparity_reg_planar)
  np.savetxt(output_path + 'dense-disparity-reg-quadric.dat', disparity_reg_quadric)
  
  np.savetxt(output_path + 'dense-disparity-rob-const.dat', disparity_rob_const)
  np.savetxt(output_path + 'dense-disparity-rob-planar.dat', disparity_rob_planar)
  np.savetxt(output_path + 'dense-disparity-rob-quadric.dat', disparity_rob_quadric)
  
  np.savetxt(output_path + 'dense-disparity-rob-final.dat', disparity_rob_final)
  
  return disparity_rob_final

#------------------------------------------------------------------------------------------
