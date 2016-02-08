#! /usr/bin/env python
# Alexey Abramov <abramov@physik3.gwdg.de>

import numpy as np
from scipy import ndimage

# local imports
import plot_data as plt

#------------------------------------------------------------------------------------------
  
def remove_small_regions(segments):

  # get segment labels
  labels = set(np.unique(segments)).difference((0,))

  for lb in labels:
    
    area = ((segments == lb)*1).sum()
    
    if (area < 10):
      segments[ (segments == lb).nonzero() ] = 0

#------------------------------------------------------------------------------------------

def detect_small_segments(segments, occlusion_range):

  # get segment labels
  labels = set(np.unique(segments)).difference((0,))
  max_label = max(labels) + 1
  
  # eliminate frame borders from the consideration
  segments[0,:] = max_label
  segments[255,:] = max_label
  segments[:,0] = max_label
  segments[:,319] = max_label
  
  # starting label for gaps
  cur_label = max_label+1
  
  # perform labeling of connected components (finding holes uncovered by segments)
  segments_gaps = np.zeros(segments.shape, dtype=float)
  segments_mask = (segments == 0)*1
  
  seeds, nr_items = ndimage.label(segments_mask)
  labels_list = set(np.unique(seeds)).difference((0,))

  step = 1 # defines size of a small window
  
  # go over found gaps and assign sequential labels
  for label in labels_list:

    segments_gaps[ (seeds == label).nonzero() ] = cur_label

    # for extremely small (1-pixel) gaps take a label from its neighbors
    if (segments_gaps == cur_label).sum() == 1:
      
      # window around the current pixel-gap
      y = (segments_gaps == cur_label).nonzero()[0]
      x = (segments_gaps == cur_label).nonzero()[1]
      
      w_labels = np.zeros([2*step+1, 2*step+1], dtype=segments.dtype)
      w_labels[:,:] = segments[y-step:y+step+1, x-step:x+step+1]

      nmb = 0
      
      # take the most frequent label
      for w_lb in set(np.unique(w_labels)).difference((0,)):
        
        if (w_labels == w_lb).sum() > nmb:
          segments_gaps[ (segments_gaps == cur_label).nonzero() ] = w_lb
          nmb = (w_labels == w_lb).sum()
    
    cur_label += 1
    

  # remove small segments
  remove_small_regions(segments)
  
  segments[ (segments == max_label).nonzero() ] = 0
  
  # fill half-occlusions in all images with zeros
  segments_gaps[:,0:occlusion_range] = 0
  segments[:,0:occlusion_range] = 0
  
  # combine original segments with segments found for gaps
  segments_final = np.zeros(segments.shape, dtype=float)
  segments_final[:] = segments
  
  for lb in set(np.unique(segments_gaps)).difference((0,)):
    segments_final[ (segments_gaps == lb).nonzero() ] = segments_gaps[ (segments_gaps == lb).nonzero() ]
  
  return segments, segments_gaps, segments_final
  
#------------------------------------------------------------------------------------------

def save_segments(fname, segments, colors):

#  segment_colors = plt.generate_colors()
  plt.save_segments_colored(fname + '.png', segments, colors)
  np.savetxt(fname + '.dat', segments)

#------------------------------------------------------------------------------------------
