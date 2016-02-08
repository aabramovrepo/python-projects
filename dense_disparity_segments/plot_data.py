#! /usr/bin/env python
# Alexey Abramov <abramov@physik3.gwdg.de>

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------------------

def plot_array(title_str, data, scale_min, scale_max, fname):
  """  Plot an array (color-coded). "title_str" is the figure title, "data" is the array being plotted,
       "scale_min" and "scale_max" define a range for the color-code, "fname" is the file name. """

  plt.figure()
  plt.title(title_str)
  plt.imshow(data)
  plt.clim(scale_min, scale_max)
  plt.colorbar()
  plt.savefig(fname + '.png')
  plt.close()

#------------------------------------------------------------------------------------------

def generate_colors():

  segments_nmb = 3000
  segments_colors = []
  
  # generate (r,g,b) color values
  for ind in range(segments_nmb):
    
    r = np.random.randint(255)
    g = np.random.randint(255)
    b = np.random.randint(255)
    
    segments_colors.append((r,g,b))

    """try:
      color_index = segments_colors.index((r,g,b))
      print 'this color is already in the list !'
    except ValueError:
      segments_colors.append((r,g,b))"""

  return segments_colors

#------------------------------------------------------------------------------------------

def save_segments_colored(fname, segments, colors):
  
  arr_R = np.zeros(segments.shape, dtype=float)
  arr_G = np.zeros(segments.shape, dtype=float)
  arr_B = np.zeros(segments.shape, dtype=float)
  
  # segment labels which need to be displayed
  labels = set(np.unique(segments)).difference((0,))
  
  for lb in labels:
    
    segment_mask = (segments == lb)

    arr_R[ segment_mask.nonzero() ] = colors[int(lb)][0]
    arr_G[ segment_mask.nonzero() ] = colors[int(lb)][1]
    arr_B[ segment_mask.nonzero() ] = colors[int(lb)][2]


  arr_R[ (segments == 0) ] = 255
  arr_G[ (segments == 0) ] = 255
  arr_B[ (segments == 0) ] = 255

  sp.misc.imsave(fname, [arr_R, arr_G, arr_B])

#------------------------------------------------------------------------------------------
