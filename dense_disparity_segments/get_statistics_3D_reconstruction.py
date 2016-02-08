#! /usr/bin/env python
# Alexey Abramov <abramov@physik3.gwdg.de>

import numpy as np
import Image as img
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import math

print('')
print('****************************************************************')
print('*** Get statistics (Alexey Abramov, abramov@physik3.gwdg.de) ***')
print('****************************************************************')
print('')

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


# size of disparity maps that need to be analysed
width = 320
height = 256

# input data for comparison: one of stereo frames and its reconstruction
#frame_original = img.open('./benchmarks/plastic/frame-right.png')
#frame_reconstruction = img.open('./benchmarks/plastic/frame-reconstruction.png')

frame_original = img.open('/home/abramov/Desktop/reconstruction/frame-right.png')
frame_reconstruction = img.open('/home/abramov/Desktop/reconstruction/reconstruction-phase-based_incons.png')

arr_frame_original = np.asarray(frame_original, dtype=float)
arr_frame_reconstruction = np.asarray(frame_reconstruction, dtype=float)

# separate RGB components of the original image
arr_orig_R = arr_frame_original[:,:,0]
arr_orig_G = arr_frame_original[:,:,1]
arr_orig_B = arr_frame_original[:,:,2]

# separate RGB components of the reconstructed image
arr_rec_R = arr_frame_reconstruction[:,:,0]
arr_rec_G = arr_frame_reconstruction[:,:,1]
arr_rec_B = arr_frame_reconstruction[:,:,2]

# define a mask of reconstructed pixels
arr_rec_sum = arr_rec_R + arr_rec_G + arr_rec_B
arr_rec_known_pixels = (arr_rec_sum > 0)
plot_array('Known pixels in the reconstruction', arr_rec_known_pixels, 0, 1, 'known-rec-pixels')

# define a percentage of reconstructed pixels
indices = arr_rec_known_pixels.nonzero()[0]
nmb_rec_pixels = len(indices)
print 'Percentage of reconstructed pixels = ', ( float(nmb_rec_pixels) / float(width * height) ) * 100

# calculate RMS (Root Mean Square) error
cmp_image = (pow((arr_orig_R - arr_rec_R),2) + pow((arr_orig_G - arr_rec_G),2) + pow((arr_orig_B - arr_rec_B),2)) * arr_rec_known_pixels
cmp_image_sum = cmp_image.sum()
rms_error = cmp_image_sum / (width*height)
rms_error = math.sqrt(rms_error)
print 'RMS error = ', rms_error
