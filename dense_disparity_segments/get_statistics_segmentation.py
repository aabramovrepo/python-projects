#! /usr/bin/env python
# Alexey Abramov <abramov@physik3.gwdg.de>

import numpy as np
import Image as img
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import mahotas

print('')
print('********************************************************************************')
print('*** Statistics for object tracking (Alexey Abramov, abramov@physik3.gwdg.de) ***')
print('********************************************************************************')
print('')

#------------------------------------------------------------------------------------------------------

# plot an array
def plot_array(title_str, data, scale_min, scale_max, fname):

  plt.figure()
  plt.title(title_str)
  plt.imshow(data)
  plt.clim(scale_min, scale_max)
  plt.colorbar()
  plt.savefig(fname + '.png')
#  plt.savefig(fname + '.eps')
  plt.close()

#------------------------------------------------------------------------------------------------------

# compute RMS error fro two adjacent frames (t) and (t+1) from one video stream

def compute_RMS_adjacent_frames(frame_1_fname, frame_2_fname, segments_1, segments_2):

  rms_error_2D = 0

  # original frames
  frame_1 = np.asarray(frame_1_fname, dtype=float)
  frame_2 = np.asarray(frame_2_fname, dtype=float)

  # separate RGB components of the frame (t)
  arr_R_1 = frame_1[:,:,0]
  arr_G_1 = frame_1[:,:,1]
  arr_B_1 = frame_1[:,:,2]

  # separate RGB components of the frame (t+1)
  arr_R_2 = frame_2[:,:,0]
  arr_G_2 = frame_2[:,:,1]
  arr_B_2 = frame_2[:,:,2]

  # get frame size
  height = segments_1.shape[0]
  width = segments_1.shape[1]

  # for labels presented in segmentation results
  for lb in set(np.unique(segments_1)).difference((0,)):

#    arr_segment_mask = (segments_1 == lb)
#    fn = str(lb)
#    plot_array('Segment ' + fn, arr_segment_mask, 0, 1, 'segment-' + fn)

    # compute 2D and 3D RMS errors for one chosen label
    if lb == 221:

      #print 'label = ', lb

      # get masks for segments (t) and (t+1)
      arr_segment_mask_1 = (segments_1 == lb)
      arr_segment_mask_2 = (segments_2 == lb)

      fname = str(lb)
#      plot_array('Mask (t) ' + fname, arr_segment_mask_1, 0, 1, 'mask-t-' + fname)
#      plot_array('Mask (t+1) ' + fname, arr_segment_mask_2, 0, 1, 'mask-t2-' + fname)

      # calculate the center of masses for segments (t) and (t+1)
      y0, x0 = mahotas.center_of_mass(arr_segment_mask_1, labels=None)
      y1, x1 = mahotas.center_of_mass(arr_segment_mask_2, labels=None)

      y0 = round(y0)
      x0 = round(x0)

      y1 = round(y1)
      x1 = round(x1)

      #print 'y0 = ', y0, ' x0 = ', x0
      #print 'y1 = ', y1, ' x1 = ', x1

      delta_x = x1 - x0
      delta_y = y1 - y0

      #print 'delta_x = ', delta_x
      #print 'delta_y = ', delta_y

      # an array for a segment (t) shifted to the center of mass of a segment (t+1)
      arr_segment_mask_1_shift = np.zeros([height,width])

      # color data of the segment (t+1)
      arr_R_seg_2 = np.zeros([height,width])
      arr_G_seg_2 = np.zeros([height,width])
      arr_B_seg_2 = np.zeros([height,width])

      # color data of the segment (t) shifted to (t+1)
      arr_R_seg_1 = np.zeros([height,width])
      arr_G_seg_1 = np.zeros([height,width])
      arr_B_seg_1 = np.zeros([height,width])

      # shift the segment (t) to the center of mass of the segment (t+1)
      for i in range(height):
        for j in range(width):

          cur_i = i + delta_y
          cur_j = j + delta_x

          if cur_i >= 0 and cur_i < height and cur_j >= 0 and cur_j < width:
            arr_segment_mask_1_shift[cur_i][cur_j] = arr_segment_mask_1[i][j]
            arr_R_seg_1[cur_i][cur_j] = arr_R_1[i][j]
            arr_G_seg_1[cur_i][cur_j] = arr_G_1[i][j]
            arr_B_seg_1[cur_i][cur_j] = arr_B_1[i][j]

      # compute overlap of segments (t) and (t+1)
      segments_overlap = arr_segment_mask_1_shift * arr_segment_mask_2
#      plot_array('Segments overlap 2D ' + fname, segments_overlap, 0, 1, 'overlap-2D-' + fname)

      segments_error = arr_segment_mask_1_shift - segments_overlap
#      plot_array('Segments error 2D ' + fname, segments_error, 0, 1, 'error-2D-' + fname)

      # color data of the segment (t) shifted to the center of mass of the segment (t+1)
      arr_R_seg_1 = arr_R_seg_1 * segments_overlap
      arr_G_seg_1 = arr_G_seg_1 * segments_overlap
      arr_B_seg_1 = arr_B_seg_1 * segments_overlap

      # color data of the segment (t+1)
      arr_R_seg_2 = arr_R_2 * segments_overlap
      arr_G_seg_2 = arr_G_2 * segments_overlap
      arr_B_seg_2 = arr_B_2 * segments_overlap

      # store color data of the segment for frames (t) and (t+1) respectively
#      sp.misc.imsave('./segment-1.png',[arr_R_seg_1, arr_G_seg_1, arr_B_seg_1])
#      sp.misc.imsave('./segment-2.png',[arr_R_seg_2, arr_G_seg_2, arr_B_seg_2])

      # calculate RMS error for 2D
      cmp_image = (pow((arr_R_seg_2 - arr_R_seg_1),2) + pow((arr_G_seg_2 - arr_G_seg_1),2) + pow((arr_B_seg_2 - arr_B_seg_1),2)) * segments_overlap
      cmp_image_sum = cmp_image.sum()
      object_size = segments_overlap.sum()

      #print 'Segment size (overlap) for 2D = ', object_size

      rms_error_2D = cmp_image_sum / object_size
      rms_error_2D = math.sqrt(rms_error_2D)

  return rms_error_2D

#------------------------------------------------------------------------------------------------------

# compute RMS error fro two adjacent frames (t) and (t+1) and correspondent left and right frames (t)
# this function was used earlier (deprecated)

def compute_one_stereo_pair(frame_left_1_fname, frame_left_2_fname, frame_right_fname, left_segments_1, left_segments_2, right_segments_1):

  rms_error_2D = 0
  rms_error_3D = 0

  # original frames
  frame_left_1 = np.asarray(frame_left_1_fname, dtype=float)
  frame_left_2 = np.asarray(frame_left_2_fname, dtype=float)
  frame_right = np.asarray(frame_right_fname, dtype=float)

  # separate RGB components of the left frame (t)
  arr_R_left_1 = frame_left_1[:,:,0]
  arr_G_left_1 = frame_left_1[:,:,1]
  arr_B_left_1 = frame_left_1[:,:,2]

  # separate RGB components of the left frame (t+1)
  arr_R_left_2 = frame_left_2[:,:,0]
  arr_G_left_2 = frame_left_2[:,:,1]
  arr_B_left_2 = frame_left_2[:,:,2]

  # separate RGB components of the right frame (t)
  arr_R_right = frame_right[:,:,0]
  arr_G_right = frame_right[:,:,1]
  arr_B_right = frame_right[:,:,2]

  # get frame size
  height = left_segments_1.shape[0]
  width = right_segments_1.shape[1]

  # for labels presented in segmentation results
  for lb in set(np.unique(left_segments_1)).difference((0,)):

#    arr_segment_mask = (left_segments_1 == lb)
#    fn = str(lb)
#    plot_array('Segment ' + fn, arr_segment_mask, 0, 1, 'segment-' + fn)

    # compute 2D and 3D RMS errors for one chosen label
    if lb == 332:

      #print 'label = ', lb

      # get masks for left segments (t), (t+1) and right segments (t)
      arr_segment_mask_left_1 = (left_segments_1 == lb)
      arr_segment_mask_left_2 = (left_segments_2 == lb)
      arr_segment_mask_right = (right_segments_1 == lb)

      fname = str(lb)
#      plot_array('Left mask (t) ' + fname, arr_segment_mask_left_1, 0, 1, 'left-mask-t-' + fname)
#      plot_array('Left mask (t+1) ' + fname, arr_segment_mask_left_2, 0, 1, 'left-mask-t2-' + fname)
#      plot_array('Right mask (t) ' + fname, arr_segment_mask_right, 0, 1, 'right-mask-t-' + fname)

      # calculate the center of masses for left segments (t), (t+1) and right segments (t)
      y0, x0 = mahotas.center_of_mass(arr_segment_mask_left_1, labels=None)
      y1, x1 = mahotas.center_of_mass(arr_segment_mask_left_2, labels=None)
      y2, x2 = mahotas.center_of_mass(arr_segment_mask_right, labels=None)

      y0 = round(y0)
      x0 = round(x0)

      y1 = round(y1)
      x1 = round(x1)

      y2 = round(y2)
      x2 = round(x2)

      #print 'y0 = ', y0, ' x0 = ', x0
      #print 'y1 = ', y1, ' x1 = ', x1
      #print 'y2 = ', y2, ' x2 = ', x2

      delta_x_left = x1 - x0
      delta_y_left = y1 - y0

      delta_x_right = x2 - x0
      delta_y_right = y2 - y0

      #print 'delta_x_left = ', delta_x_left
      #print 'delta_y_left = ', delta_y_left

      #print 'delta_x_right = ', delta_x_right
      #print 'delta_y_right = ', delta_y_right

      # an array for a segment (t) shifted to the center of mass of a segment (t+1)
      arr_segment_mask_left_1_shift = np.zeros([height,width])

      # color data of the left segment (t+1)
      arr_R_left_seg_2 = np.zeros([height,width])
      arr_G_left_seg_2 = np.zeros([height,width])
      arr_B_left_seg_2 = np.zeros([height,width])

      # color data of the left segment (t) shifted to (t+1)
      arr_R_left_seg_1 = np.zeros([height,width])
      arr_G_left_seg_1 = np.zeros([height,width])
      arr_B_left_seg_1 = np.zeros([height,width])

      # shift the left segment (t) to the center of mass of the left segment (t+1)
      for i in range(height):
        for j in range(width):

          cur_i = i + delta_y_left
          cur_j = j + delta_x_left

          if cur_i >= 0 and cur_i < height and cur_j >= 0 and cur_j < width:
            arr_segment_mask_left_1_shift[cur_i][cur_j] = arr_segment_mask_left_1[i][j]
            arr_R_left_seg_1[cur_i][cur_j] = arr_R_left_1[i][j]
            arr_G_left_seg_1[cur_i][cur_j] = arr_G_left_1[i][j]
            arr_B_left_seg_1[cur_i][cur_j] = arr_B_left_1[i][j]

#      plot_array('Left mask t (shifted to (t+1)) ' + fname, arr_segment_mask_left_1_shift, 0, 1, 'left-mask-t-shifted-' + fname)

      # compute overlap of left segments (t) and (t+1)
      segments_overlap = arr_segment_mask_left_1_shift * arr_segment_mask_left_2
#      plot_array('Segments overlap 2D ' + fname, segments_overlap, 0, 1, 'overlap-2D-' + fname)

      segments_error = arr_segment_mask_left_1_shift - segments_overlap
#      plot_array('Segments error 2D ' + fname, segments_error, 0, 1, 'error-2D-' + fname)

      # color data of the left segment (t) shifted to the center of mass of the left segment (t+1)
      arr_R_left_seg_1 = arr_R_left_seg_1 * segments_overlap
      arr_G_left_seg_1 = arr_G_left_seg_1 * segments_overlap
      arr_B_left_seg_1 = arr_B_left_seg_1 * segments_overlap

      # color data of the left segment (t+1)
      arr_R_left_seg_2 = arr_R_left_2 * segments_overlap
      arr_G_left_seg_2 = arr_G_left_2 * segments_overlap
      arr_B_left_seg_2 = arr_B_left_2 * segments_overlap

      # store color data of the left segment for frames (t) and (t+1) respectively
#      sp.misc.imsave('./segment-left-1.png',[arr_R_left_seg_1, arr_G_left_seg_1, arr_B_left_seg_1])
#      sp.misc.imsave('./segment-left-2.png',[arr_R_left_seg_2, arr_G_left_seg_2, arr_B_left_seg_2])

      # calculate RMS error for 2D
      cmp_image = (pow((arr_R_left_seg_2 - arr_R_left_seg_1),2) + pow((arr_G_left_seg_2 - arr_G_left_seg_1),2) + pow((arr_B_left_seg_2 - arr_B_left_seg_1),2)) * segments_overlap
      cmp_image_sum = cmp_image.sum()
      object_size = segments_overlap.sum()

      #print 'Segment size (overlap) for 2D = ', object_size

      rms_error_2D = cmp_image_sum / object_size
      rms_error_2D = math.sqrt(rms_error_2D)

      # an array for the left segment (t) shifted to the center of mass of the right segment (t+1)
      arr_segment_mask_right_shift = np.zeros([height,width])

      # color data of the right segment
      arr_R_right_seg = np.zeros([height,width])
      arr_G_right_seg = np.zeros([height,width])
      arr_B_right_seg = np.zeros([height,width])

      # color data of the left segment shifted to the right
      arr_R_left_seg = np.zeros([height,width])
      arr_G_left_seg = np.zeros([height,width])
      arr_B_left_seg = np.zeros([height,width])

      # shift the left segment (t) to the center of mass of the right segment (t)
      for i in range(height):
        for j in range(width):

          cur_i = i + delta_y_right
          cur_j = j + delta_x_right

          if cur_i >= 0 and cur_i < height and cur_j >= 0 and cur_j < width:
            arr_segment_mask_right_shift[cur_i][cur_j] = arr_segment_mask_left_1[i][j]
            arr_R_left_seg[cur_i][cur_j] = arr_R_left_1[i][j]
            arr_G_left_seg[cur_i][cur_j] = arr_G_left_1[i][j]
            arr_B_left_seg[cur_i][cur_j] = arr_B_left_1[i][j]

      # compute overlap of left segment (t) and right segment (t)
      segments_overlap_right = arr_segment_mask_right_shift * arr_segment_mask_right
#      plot_array('Segments overlap 3D ' + fname, segments_overlap_right, 0, 1, 'overlap-3D-' + fname)
#      plot_array('Left 1 segment (shifted) to the right ' + fname, arr_segment_mask_right_shift, 0, 1, 'left-mask-right-shifted-' + fname)

      segments_error_right = arr_segment_mask_right_shift - segments_overlap_right
#      plot_array('Segments error 3D ' + fname, segments_error_right, 0, 1, 'error-3D-' + fname)

      # color data of the left segment (t) shifted to the center of mass of the right segment (t)
      arr_R_left_seg = arr_R_left_seg * segments_overlap_right
      arr_G_left_seg = arr_G_left_seg * segments_overlap_right
      arr_B_left_seg = arr_B_left_seg * segments_overlap_right

      # color data of the right segment (from the overlapped area)
      arr_R_right_seg = arr_R_right * segments_overlap_right
      arr_G_right_seg = arr_G_right * segments_overlap_right
      arr_B_right_seg = arr_B_right * segments_overlap_right

      # store color data of the left and right segments for frames (t) respectively
#      sp.misc.imsave('./segment-left.png',[arr_R_left_seg, arr_G_left_seg, arr_B_left_seg])
#      sp.misc.imsave('./segment-right.png',[arr_R_right_seg, arr_G_right_seg, arr_B_right_seg])

      # calculate RMS error for 3D
      cmp_image = (pow((arr_R_right_seg - arr_R_left_seg),2) + pow((arr_G_right_seg - arr_G_left_seg),2) + pow((arr_B_right_seg - arr_B_left_seg),2)) * segments_overlap_right
      cmp_image_sum = cmp_image.sum()
      object_size = segments_overlap_right.sum()

      #print 'Segment size (overlap) for 3D = ', object_size

      rms_error_3D = cmp_image_sum / object_size
      rms_error_3D = math.sqrt(rms_error_3D)

  return rms_error_2D, rms_error_3D

#------------------------------------------------------------------------------------------------------

# compute RMS error for adjacent frames (t) and (t+1) taken from the left and right video streams respectively
# left and right segments are considered here independent of each other

def compute_RMS_left_right_2D():

  # number of frames that need to be compared
  frames_nmb = 270#227

  index_1 = 2
  index_2 = 0
  index_3 = 0
  index_4 = 0

  index_2_1 = 3
  index_2_2 = 0
  index_2_3 = 0
  index_2_4 = 0

  cnt = 0
  avg_2D_left = 0
  avg_2D_right = 0

  # loop over the whool stereoscopic sequence
  for ind in range(1,frames_nmb):

    # load original images
    fname = '/home/abramov/Desktop/results_for_paper/sequence_5/input_left_out/img_' + str(index_4) + str(index_3) + str(index_2) + str(index_1) + '-left.png'
    frame_left_1_img = img.open(fname)

    fname = '/home/abramov/Desktop/results_for_paper/sequence_5/input_left_out/img_' + str(index_2_4) + str(index_2_3) + str(index_2_2) + str(index_2_1) + '-left.png'
    frame_left_2_img = img.open(fname)

    fname = '/home/abramov/Desktop/results_for_paper/sequence_5/input_right_out/img_' + str(index_4) + str(index_3) + str(index_2) + str(index_1) + '-right.png'
    frame_right_1_img = img.open(fname)

    fname = '/home/abramov/Desktop/results_for_paper/sequence_5/input_right_out/img_' + str(index_2_4) + str(index_2_3) + str(index_2_2) + str(index_2_1) + '-right.png'
    frame_right_2_img = img.open(fname)

    # load segments
    fname = '/home/abramov/Desktop/results_for_paper/sequence_5/clusters_left_values/img_' + str(index_4) + str(index_3) + str(index_2) + str(index_1) + '_clusters-left.dat'
    left_segments_1 = np.loadtxt(fname)

    fname = '/home/abramov/Desktop/results_for_paper/sequence_5/clusters_left_values/img_' + str(index_2_4) + str(index_2_3) + str(index_2_2) + str(index_2_1) + '_clusters-left.dat'
    left_segments_2 = np.loadtxt(fname)

    fname = '/home/abramov/Desktop/results_for_paper/sequence_5/clusters_right_values/img_' + str(index_4) + str(index_3) + str(index_2) + str(index_1) + '_clusters-right.dat'
    right_segments_1 = np.loadtxt(fname)

    fname = '/home/abramov/Desktop/results_for_paper/sequence_5/clusters_right_values/img_' + str(index_2_4) + str(index_2_3) + str(index_2_2) + str(index_2_1) + '_clusters-right.dat'
    right_segments_2 = np.loadtxt(fname)

    # compute RMS error for adjacent left frames (t) and (t+1) and right frames (t) and (t+1)
    rms_err_2D_left = compute_RMS_adjacent_frames(frame_left_1_img, frame_left_2_img, left_segments_1, left_segments_2)
    rms_err_2D_right = compute_RMS_adjacent_frames(frame_right_1_img, frame_right_2_img, right_segments_1, right_segments_2)

    # accumulate RMS errors to get an average error
    avg_2D_left += rms_err_2D_left
    avg_2D_right += rms_err_2D_right

    cnt += 1

    # compute average RMS errors
    if cnt == 5:

      print '===> ', index_4, index_3, index_2, index_1

      avg_2D_left = avg_2D_left / 5
      avg_2D_right = avg_2D_right / 5

      # in percentage
      avg_2D_left = (avg_2D_left / 433) * 100
      avg_2D_right = (avg_2D_right / 433) * 100

      print 'RMS 2D left (%) = ', avg_2D_left
      print 'RMS 2D right (%) = ', avg_2D_right
      print ''

      cnt = 0
      avg_2D_left = 0
      avg_2D_right = 0

    index_1 += 1

    if index_1 == 10:
      index_1 = 0
      index_2 += 1

    if index_2 == 10:
      index_2 = 0
      index_3 += 1

    if index_3 == 10:
      index_3 = 0
      index_4 += 1


    index_2_1 += 1

    if index_2_1 == 10:
      index_2_1 = 0
      index_2_2 += 1

    if index_2_2 == 10:
      index_2_2 = 0
      index_2_3 += 1

    if index_2_3 == 10:
      index_2_3 = 0
      index_2_4 += 1

#------------------------------------------------------------------------------------------------------

# compute RMS error for left frames (t) (t+1) and correspondent left and right frames (t)
# this function was used earlier (deprecated)

def compute_RMS_2D_3D():

  frames_nmb = 227

  index_1 = 2
  index_2 = 0
  index_3 = 0
  index_4 = 0

  index_2_1 = 3
  index_2_2 = 0
  index_2_3 = 0
  index_2_4 = 0

  cnt = 0
  avg_2D = 0
  avg_3D = 0

  # loop over the whool stereoscopic sequence
  for ind in range(1,frames_nmb):

    fname = '/home/abramov/Desktop/results_for_paper/sequence_1/input_left_out/img_' + str(index_4) + str(index_3) + str(index_2) + str(index_1) + '-left.png'
    frame_left_1_img = img.open(fname)

    fname = '/home/abramov/Desktop/results_for_paper/sequence_1/input_left_out/img_' + str(index_2_4) + str(index_2_3) + str(index_2_2) + str(index_2_1) + '-left.png'
    frame_left_2_img = img.open(fname)

    fname = '/home/abramov/Desktop/results_for_paper/sequence_1/input_right_out/img_' + str(index_4) + str(index_3) + str(index_2) + str(index_1) + '-right.png'
    frame_right_img = img.open(fname)

    fname = '/home/abramov/Desktop/results_for_paper/sequence_1/clusters_left_values/img_' + str(index_4) + str(index_3) + str(index_2) + str(index_1) + '_clusters-left.dat'
    left_segments_1 = np.loadtxt(fname)

    fname = '/home/abramov/Desktop/results_for_paper/sequence_1/clusters_left_values/img_' + str(index_2_4) + str(index_2_3) + str(index_2_2) + str(index_2_1) + '_clusters-left.dat'
    left_segments_2 = np.loadtxt(fname)

    fname = '/home/abramov/Desktop/results_for_paper/sequence_1/clusters_right_values/img_' + str(index_4) + str(index_3) + str(index_2) + str(index_1) + '_clusters-right.dat'
    right_segments_1 = np.loadtxt(fname)

    # compute RMS error for adjacent left frames (t), (t+1) and correspondent left and right frames (t)
    rms_err_2D, rms_err_3D = compute_one_stereo_pair(frame_left_1_img, frame_left_2_img, frame_right_img, left_segments_1, left_segments_2, right_segments_1)

    # accumulate RMS errors to get an average error
    avg_2D += rms_err_2D
    avg_3D += rms_err_3D

    cnt += 1

    # compute average RMS errors
    if cnt == 5:

      print '===> ', index_4, index_3, index_2, index_1
      
      avg_2D = avg_2D / 5
      avg_3D = avg_3D / 5

      # in percentage
      avg_2D = (avg_2D / 433) * 100
      avg_3D = (avg_3D / 433) * 100

      print 'RMS 2D (%) = ', avg_2D
      print 'RMS 3D (%) = ', avg_3D
      print ''

      cnt = 0
      avg_2D = 0
      avg_3D = 0

    index_1 += 1

    if index_1 == 10:
      index_1 = 0
      index_2 += 1

    if index_2 == 10:
      index_2 = 0
      index_3 += 1

    if index_3 == 10:
      index_3 = 0
      index_4 += 1


    index_2_1 += 1

    if index_2_1 == 10:
      index_2_1 = 0
      index_2_2 += 1

    if index_2_2 == 10:
      index_2_2 = 0
      index_2_3 += 1

    if index_2_3 == 10:
      index_2_3 = 0
      index_2_4 += 1

#------------------------------------------------------------------------------------------------------

# get quantitative analysis for left frames (t) and (t+1) and right frame (t)

# input data for comparison: left frames (t), (t+1) and a right frame (t)
#frame_left_1_fn = img.open('/home/abramov/Desktop/results_for_paper/sequence_1/input_left_out/img_0121-left.png')
#frame_left_2_fn = img.open('/home/abramov/Desktop/results_for_paper/sequence_1/input_left_out/img_0122-left.png')
#frame_right_fn = img.open('/home/abramov/Desktop/results_for_paper/sequence_1/input_right_out/img_0121-right.png')

# left segments (t) and (t+1) and right segments (t)
#left_segments_1_fn = np.loadtxt('/home/abramov/Desktop/results_for_paper/sequence_1/clusters_left_values/img_0121_clusters-left.dat')
#left_segments_2_fn = np.loadtxt('/home/abramov/Desktop/results_for_paper/sequence_1/clusters_left_values/img_0122_clusters-left.dat')
#right_segments_1_fn = np.loadtxt('/home/abramov/Desktop/results_for_paper/sequence_1/clusters_right_values/img_0121_clusters-right.dat')

#rms_err_2D, rms_err_3D = compute_one_stereo_pair(frame_left_1_fn, frame_left_2_fn, frame_right_fn, left_segments_1_fn, left_segments_2_fn, right_segments_1_fn)

#print 'RMS error 2D = ', rms_err_2D
#print 'RMS error 3D = ', rms_err_3D



# get quantitative analysis for left frames (t) and (t+1) and right frames (t) and (t+1)

# two adjacent left frames
#frame_left_1_fn = img.open('/home/abramov/Desktop/results_for_paper/sequence_5/input_left_out/img_0041-left.png')
#frame_left_2_fn = img.open('/home/abramov/Desktop/results_for_paper/sequence_5/input_left_out/img_0041-left.png')

# two adjacent right frames
#frame_right_1_fn = img.open('/home/abramov/Desktop/results_for_paper/sequence_5/input_right_out/img_0041-right.png')
#frame_right_2_fn = img.open('/home/abramov/Desktop/results_for_paper/sequence_5/input_right_out/img_0041-right.png')

# left segments
#left_segments_1_fn = np.loadtxt('/home/abramov/Desktop/results_for_paper/sequence_5/clusters_left_values/img_0041_clusters-left.dat')
#left_segments_2_fn = np.loadtxt('/home/abramov/Desktop/results_for_paper/sequence_5/clusters_left_values/img_0041_clusters-left.dat')

# right segments
#right_segments_1_fn = np.loadtxt('/home/abramov/Desktop/results_for_paper/sequence_5/clusters_right_values/img_0041_clusters-right.dat')
#right_segments_2_fn = np.loadtxt('/home/abramov/Desktop/results_for_paper/sequence_5/clusters_right_values/img_0041_clusters-right.dat')

#rms_err_2D_left = compute_RMS_adjacent_frames(frame_left_1_fn, frame_left_2_fn, left_segments_1_fn, left_segments_2_fn)
#print 'RMS error 2D (left) = ', rms_err_2D_left

#rms_err_2D_right = compute_RMS_adjacent_frames(frame_right_1_fn, frame_right_2_fn, right_segments_1_fn, right_segments_2_fn)
#print 'RMS error 2D (right) = ', rms_err_2D_right



# get quantitative analysis of tracking for the whole sequence
#compute_RMS_2D_3D()



# get quantitative analysis of tracking for the whole sequence
compute_RMS_left_right_2D()
