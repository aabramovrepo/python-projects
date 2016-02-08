#! /usr/bin/env python
# Alexey Abramov <abramov@physik3.gwdg.de>

from PIL import Image
import numpy as np
import scipy as sp
from scipy import ndimage

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
  
def save_segments(fname, segments, colors):
  
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
  
def remove_small_regions(segments_left, segments_right):

  # get segment labels
  labels_left = set(np.unique(segments_left)).difference((0,))
  labels_right = set(np.unique(segments_right)).difference((0,))

  for lb in labels_left:
    
    area = ((segments_left == lb)*1).sum()
    
    if (area < 10):
      segments_left[ (segments_left == lb).nonzero() ] = 0
      segments_right[ (segments_right == lb).nonzero() ] = 0

  for lb in labels_right:
    
    area = ((segments_right == lb)*1).sum()
    
    if (area < 10):
      segments_left[ (segments_left == lb).nonzero() ] = 0
      segments_right[ (segments_right == lb).nonzero() ] = 0

#------------------------------------------------------------------------------------------

def make_segmentation_refinement(frame, segments):

  h = segments.shape[0]
  w = segments.shape[1]
  
  indices_gaps = (segments == 0).nonzero()
  indices_rnd = np.random.permutation(indices_gaps[0])

  for ind in range(indices_rnd.size):
  
    y = indices_gaps[0][ind]
    x = indices_gaps[1][ind]

    if all( [y > 0, y < (h-1), x > 0, x < (w-1)] ):

      window_img = frame[y-1:y+2,x-1:x+2,:]

      window_segments = segments[y-1:y+2,x-1:x+2]
      window_labels = set(np.unique(window_segments)).difference((0,))
      
      if len(window_labels) == 0:
        continue

      labels = []
      
      for lb in window_labels:
        labels.append(lb)

      #if len(window_labels) == 1:
        #segments[y,x] = labels[0]
        #continue

      if len(window_labels) < 2:
        continue

      thres = 500.0
        
      for label in labels:

        r = window_img[0][1,1]
        g = window_img[1][1,1]
        b = window_img[2][1,1]

        color_diff = 0.0
        label_mask = (window_segments == label)*1
        
        diff_r = (window_img[0] - r)*label_mask
        tmp = diff_r*diff_r
        color_diff += float(tmp.sum()) / float(label_mask.sum())

        diff_g = (window_img[1] - g)*label_mask
        tmp = diff_g*diff_g
        color_diff += float(tmp.sum()) / float(label_mask.sum())
        
        diff_b = (window_img[2] - b)*label_mask
        tmp = diff_b*diff_b
        color_diff += float(tmp.sum()) / float(label_mask.sum())
        
        color_diff = math.sqrt(color_diff)
        
        if color_diff < thres:
          thres = color_diff
          segments[y,x] = label

#------------------------------------------------------------------------------------------

def main():

  fname_img_left = './img-left-input.png'
  fname_img_right = './img-right-input.png'

  fname_segments_left = './segments-left.dat'
  fname_segments_right = './segments-right.dat'
  
  # original left and right images
  img_left = Image.open(fname_img_left)
  arr_img_left = np.asarray(img_left, dtype=float)

  img_right = Image.open(fname_img_right)
  arr_img_right = np.asarray(img_right, dtype=float)

  # left and right segments
  arr_segments_left = np.loadtxt(fname_segments_left, dtype=int)
  arr_segments_right = np.loadtxt(fname_segments_right, dtype=int)
  
  segment_colors = generate_colors()

  # fill half-occlusions in both images with zeros
  #arr_segments_left[:,0:32] = 0
  #arr_segments_right[:,295:] = 0

  # get segment labels
  labels_left = set(np.unique(arr_segments_left)).difference((0,))
  labels_right = set(np.unique(arr_segments_right)).difference((0,))
  
  print 'labels_left = ', labels_left
  print 'labels_right = ', labels_right

  max_label = max(labels_left) + 1
  
  arr_segments_left[0,:] = max_label
  arr_segments_left[255,:] = max_label
  arr_segments_left[:,0] = max_label
  arr_segments_left[:,319] = max_label
  
  cur_label = max_label+1
  
  # perform labeling of connected components (finding holes uncovered by segments)
  arr_segments_gaps_left = np.zeros(arr_segments_left.shape, dtype=float)
  segments_mask = (arr_segments_left == 0)*1
  
  seeds, nr_items = ndimage.label(segments_mask)
  labels_list = set(np.unique(seeds)).difference((0,))

  for item_lb in labels_list:

    #arr_segments_left[ (seeds == item_lb).nonzero() ] = cur_label
    arr_segments_gaps_left[ (seeds == item_lb).nonzero() ] = cur_label
    cur_label += 1

  # remove small segments
  remove_small_regions(arr_segments_left, arr_segments_right)
  
  arr_segments_left[ (arr_segments_left == max_label).nonzero() ] = 0
  
  # fill half-occlusions in all images with zeros
  arr_segments_gaps_left[:,0:32] = 0
  arr_segments_left[:,0:32] = 0
  arr_segments_right[:,295:] = 0
  
  save_segments('./tmp/segments-left-original.png', arr_segments_left, segment_colors)
  save_segments('./tmp/segments-left-gaps.png', arr_segments_gaps_left, segment_colors)
  
  save_segments('./tmp/segments-right-original.png', arr_segments_right, segment_colors)
  
  np.savetxt('./tmp/segments-left-original.dat', arr_segments_left)
  np.savetxt('./tmp/segments-left-gaps.dat', arr_segments_gaps_left)

#------------------------------------------------------------------------------------------

# when a Python program is executed directly
if __name__ == "__main__":
  main()

#------------------------------------------------------------------------------------------
