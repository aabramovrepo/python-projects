#! /usr/bin/env python
# Alexey Abramov <abramov@physik3.gwdg.de>

import numpy as np
import os
import time
import sys

# local imports
import view_disparity as vd
import surface_fitting as sf
import get_statistics_ground_truth as sgt


import matplotlib.pyplot as plt
import pylab as p


#------------------------------------------------------------------------------------------

def main():

  database_path = '/media/aa894ccb-f4a4-4799-ae44-21e1f5cb2949/DenseDisparity_dataset/'

  # format of the database entry:
  #
  #  < local path to the stereo pair >
  #  < left image >
  #  < right image >
  #  < local path to the output (results) folder >
  #  < segmentation multiplier for the left image >
  #  < pre-shift for the phase-based stereo >
  #  < occlusion range >
  #  < min disparity >
  #  < max disparity >
  #  < ground truth scale factor >
  #  < texture entropy value >
  
  database = ( ('Middlebury/Baby1/', 'view1.png', 'view5.png', 'output/', 0.9, 70, 32, 0, 38, 3.0, 3.92),             # 0
               ('Middlebury/Baby2/', 'view1.png', 'view5.png', 'output/', 0.88, 70, 32, 0, 36, 3.0, 3.92),            # 1
               ('Middlebury/Baby3/', 'view1.png', 'view5.png', 'output/', 0.88, 70, 32, 10, 39, 3.0, 4.1),            # 2
               ('Middlebury/cones/', 'im2.png', 'im6.png', 'output/', 0.9, 65, 40, 5, 38, 4.0, 4.83),                 # 3
               ('Middlebury/Lampshade1/', 'view1.png', 'view5.png', 'output/', 0.92, 70, 32, 0, 50, 3.0, 2.86),       # 4
               ('Middlebury/Lampshade2/', 'view1.png', 'view5.png', 'output/', 0.92, 70, 32, 0, 50, 3.0, 2.71),       # 5
               ('Middlebury/Midd1/', 'view1.png', 'view5.png', 'output/', 0.7, 70, 32, 10, 40, 3.0, 3.034),           # 6
               ('Middlebury/Midd2/', 'view1.png', 'view5.png', 'output/', 0.7, 70, 32, 5, 45, 3.0, 2.95),             # 7
               ('Middlebury/Monopoly/', 'view1.png', 'view5.png', 'output/', 0.7, 75, 32, 15, 40, 3.0, 3.28),         # 8
               ('Middlebury/Plastic/', 'view1.png', 'view5.png', 'output/', 0.92, 70, 32, 0, 50, 3.0, 2.2),           # 9
               ('Middlebury/Rocks1/', 'view1.png', 'view5.png', 'output/', 0.92, 70, 32, 0, 40, 3.0, 4.78),           # 10
               ('Middlebury/Rocks2/', 'view1.png', 'view5.png', 'output/', 0.80, 70, 32, 15, 45, 3.0, 4.84),          # 11
               ('Middlebury/teddy/', 'im2.png', 'im6.png', 'output/', 0.88, 65, 40, 5, 35, 4.0, 4.5),                 # 12
               ('Middlebury/Wood1/', 'view1.png', 'view5.png', 'output/', 0.92, 70, 32, 0, 45, 3.0, 3.89),            # 13
               ('Middlebury/Wood2/', 'view1.png', 'view5.png', 'output/', 2.0, 85, 32, 5, 50, 3.0, 3.69),             # 14
               ('Middlebury/Aloe/', 'view1.png', 'view5.png', 'output/', 0.92, 60, 32, 5, 40, 3.0, 5.12),             # 15
               ('Middlebury/Bowling1/', 'view1.png', 'view5.png', 'output/', 0.70, 95, 32, 5, 55, 3.0, 3.16),         # 16
               ('Middlebury/Bowling2/', 'view1.png', 'view5.png', 'output/', 0.70, 70, 45, 10, 50, 3.0, 3.99),        # 17
               ('Middlebury/Cloth2/', 'view1.png', 'view5.png', 'output/', 0.92, 90, 32, 5, 60, 3.0, 4.73),           # 18
               ('Middlebury/Cloth3/', 'view1.png', 'view5.png', 'output/', 0.92, 90, 32, 5, 45, 3.0, 4.87),           # 19
               ('Middlebury/Flowerpots/', 'view1.png', 'view5.png', 'output/', 1.3, 80, 32, 0, 45, 3.0, 3.82),        # 20
               
               ('Justus/blue-pan/', 'left.png', 'right.png', 'output/', 0.92, 100, 40, 20, 80, -1.0, 3.4),              # 21
               ('Justus/ice-brick/', 'left.png', 'right.png', 'output/', 0.92, 100, 40, 20, 80, -1.0, 3.29),            # 22
               ('Justus/mouse-pad/', 'left.png', 'right.png', 'output/', 1.2, 100, 40, 20, 80, -1.0, 3.47),             # 23
               ('Justus/red-plate/', 'left.png', 'right.png', 'output/', 1.5, 120, 40, 0, 130, -1.0, 3.3),              # 24
               ('Justus/scene01/', 'scene01_left.png', 'scene01_right.png', 'output/', 1.0, 90, 40, 10, 65, -1.0, 3.6), # 25
               ('Justus/scene03/', 'left.png', 'right.png', 'output/', 0.92, 110, 40, 10, 90, -1.0, 3.92),              # 26
               ('Justus/scene04/', 'left.png', 'right.png', 'output/', 0.92, 90, 40, 10, 90, -1.0, 3.79),               # 27
               ('Justus/scene05/', 'left.png', 'right.png', 'output/', 0.92, 90, 40, 0, 80, -1.0, 3.9),                 # 28
               ('Justus/towel1/', 'left.png', 'right.png', 'output/', 0.92, 90, 40, 0, 60, -1.0, 3.82),                 # 29
               
               ('Garnics/1/', 'left.png', 'right.png', 'output/', 0.92, 95, 40, 40, 70. -1.0, 4.1),                     # 30
               ('Garnics/2/', 'left.png', 'right.png', 'output/', 0.92, 85, 40, 40, 60, -1.0, 4.07),                    # 31
               
               ('Lab/Banana-1/', 'left.png', 'right.png', 'output/', 2.0, 60, 10, 0, 35, -1.0, 2.78),                   # 32
               ('Lab/Box-1/', 'left.png', 'right.png', 'output/', 1.7, 75, 10, 0, 60, -1.0, 3.89),                      # 33
               ('Lab/Box-2/', 'left.png', 'right.png', 'output/', 1.7, 75, 30, 0, 60, -1.0, 3.79),                      # 34
               ('Lab/Breakfast-1/', 'left.png', 'right.png', 'output/', 1.5, 60, 30, 0, 20, -1.0, 2.85),                # 35
               ('Lab/Breakfast-2/', 'left.png', 'right.png', 'output/', 1.5, 52, 30, 0, 20, -1.0, 2.58),                # 36
               ('Lab/Breakfast-3/', 'left.png', 'right.png', 'output/', 1.5, 52, 30, 0, 20, -1.0, 2.6),                 # 37
               ('Lab/Cups-1/', 'left.png', 'right.png', 'output/', 4.0, 52, 30, 0, 20, -1.0, 1.54),                     # 38
               ('Lab/Cups-2/', 'left.png', 'right.png', 'output/', 4.0, 52, 30, 0, 15, -1.0, 1.3),                      # 39
               ('Lab/Sandwich-1/', 'left.png', 'right.png', 'output/', 3.0, 60, 30, 0, 25, -1.0, 2.79),                 # 40
               ('Lab/Sandwich-2/', 'left.png', 'right.png', 'output/', 3.0, 65, 30, 0, 30, -1.0, 2.72),                 # 41
               ('Lab/Table-1/', 'left.png', 'right.png', 'output/', 1.7, 70, 45, 10, 50, -1.0, 3.47)                    # 42
               
#               ('Middlebury/Cloth1/', 'view1.png', 'view5.png', 'output/', 0.70, 80, 32, 0, 45),                # WARNING cannot be over-segmented !!!
#               ('Middlebury/Cloth4/', 'view1.png', 'view5.png', 'output/', 1.5, 90, 32, 0, 45),                 # WARNING cannot be over-segmented !!!
#               ('Justus/scene09/', 'left.png', 'right.png', 'output/', 0.92, 120, 40, 0, 130),                  # WARNING input sparse disparity is very bad !!!
#               ('Justus/scene15/', 'left.png', 'right.png', 'output/', 2.0, 90, 40, 0, 75),                     # WARNING input sparse disparity is very bad !!!
  
             )
  
  db = database[9]
  run(database_path, db)
  sys.exit()

  
  # dictionary: texture value -> [method -> its RMS and percentage values for the current texture value]  
  performance = { 2.5:{}, 3.5:{}, 4.5:{}, 5.5:{} }
  performance = evaluate_performance_ground_truth(performance, database_path, database)
  

  print 'performance = ', performance
  
  
  x_texture = []
  
  y_current_rms = []
  y_bm_rms = []
  y_bp_m_rms = []
  y_bp_s_rms = []
  y_bp_rms = []
  y_csbp_rms = []
  y_expansion_rms = []
  y_icm_rms = []
  y_phase_based_rms = []
  y_swap_rms = []
  y_trw_s_rms = []

  y_current_n = []
  y_bm_n = []
  y_bp_m_n = []
  y_bp_s_n = []
  y_bp_n = []
  y_csbp_n = []
  y_expansion_n = []
  y_icm_n = []
  y_phase_based_n = []
  y_swap_n = []
  y_trw_s_n = []

  # calculate the average for all methods for all points on the texture axis
  for key, value in performance.iteritems():
    
    print 'Texture entropy = ', key
    
    x_texture.append(key)
    
    for key_method, value_method in value.iteritems():
      
      avg_rms = sum(value_method[0]) / float( len(value_method[0]) )
      avg_n = sum(value_method[1]) / float( len(value_method[1]) )
      
      #print '   Method = ', key_method, ' RMS = ', avg_rms, ', Found pixels = ', avg_n

      if key_method == 'Current':
        y_current_rms.append(avg_rms)
        y_current_n.append(avg_n)

      elif key_method == 'BM':
        y_bm_rms.append(avg_rms)
        y_bm_n.append(avg_n)

      elif key_method == 'BP-M':
        y_bp_m_rms.append(avg_rms)
        y_bp_m_n.append(avg_n)

      elif key_method == 'BP-S':
        y_bp_s_rms.append(avg_rms)
        y_bp_s_n.append(avg_n)
      
      elif key_method == 'BP':
        y_bp_rms.append(avg_rms)
        y_bp_n.append(avg_n)

      elif key_method == 'CSBP':
        y_csbp_rms.append(avg_rms)
        y_csbp_n.append(avg_n)

      elif key_method == 'Expansion':
        y_expansion_rms.append(avg_rms)
        y_expansion_n.append(avg_n)

      elif key_method == 'ICM':
        y_icm_rms.append(avg_rms)
        y_icm_n.append(avg_n)

      elif key_method == 'phase-based':
        y_phase_based_rms.append(avg_rms)
        y_phase_based_n.append(avg_n)

      elif key_method == 'Swap':
        y_swap_rms.append(avg_rms)
        y_swap_n.append(avg_n)

      elif key_method == 'TRW-S':
        y_trw_s_rms.append(avg_rms)
        y_trw_s_n.append(avg_n)
        
  print 'x_texture = ', x_texture
  print 'y_current_rms = ', y_current_rms
  print 'y_current_n = ', y_current_n

  fig = plt.figure()
  
  # set scale
  p.plot(2.0, 0.0)
  p.plot(6.0, 15.0) #20.0

  p.plot(x_texture, y_current_rms, color = 'orange',  lw=3)
  p.plot(x_texture, y_bm_rms, color = 'red',  lw=3)
  p.plot(x_texture, y_bp_m_rms, color = 'yellow',  lw=3)
  p.plot(x_texture, y_bp_s_rms, color = 'blue',  lw=3)
  p.plot(x_texture, y_bp_rms, color = 'magenta',  lw=3)
  p.plot(x_texture, y_csbp_rms, color = 'green',  lw=3)
  p.plot(x_texture, y_expansion_rms, color = 'purple',  lw=3)
  p.plot(x_texture, y_icm_rms, color = 'black',  lw=3)
  p.plot(x_texture, y_phase_based_rms, color = 'darkblue',  lw=3)
  p.plot(x_texture, y_swap_rms, color = 'darkred',  lw=3)
  p.plot(x_texture, y_trw_s_rms, color = 'darkgreen',  lw=3)
  
  
  
  fig = plt.figure()
  
  # set scale
  p.plot(2, 100) # 105
  p.plot(6, 100) # 105

  p.plot(x_texture, y_current_n, color = 'orange',  lw=3)
  p.plot(x_texture, y_bm_n, color = 'red',  lw=3)
  p.plot(x_texture, y_bp_m_n, color = 'yellow',  lw=3)
  p.plot(x_texture, y_bp_s_n, color = 'blue',  lw=3)
  p.plot(x_texture, y_bp_n, color = 'magenta',  lw=3)
  p.plot(x_texture, y_csbp_n, color = 'green',  lw=3)
  p.plot(x_texture, y_expansion_n, color = 'purple',  lw=3)
  p.plot(x_texture, y_icm_n, color = 'black',  lw=3)
  p.plot(x_texture, y_phase_based_n, color = 'darkblue',  lw=3)
  p.plot(x_texture, y_swap_n, color = 'darkred',  lw=3)
  p.plot(x_texture, y_trw_s_n, color = 'darkgreen',  lw=3)
  
  p.show()

#------------------------------------------------------------------------------------------
  
def evaluate_performance_ground_truth(performance, database_path, database):
  
  for key, value in performance.iteritems():

    performance[key] = {'Current':([],[]), 'BM':([],[]), 'BP-M':([],[]), 'BP-S':([],[]), 'BP':([],[]),
                        'CSBP':([],[]), 'Expansion':([],[]), 'ICM':([],[]), 'phase-based':([],[]),
                        'Swap':([],[]), 'TRW-S':([],[])
                       }

  print 'performance = ', performance

  # go over all entries in the database
  for db_entry in database[0:21]:

    print 'name = ', db_entry[0]
    
    texture = db_entry[10]
    
    statistics = run(database_path, db_entry)
    
    if texture >= 2.0 and texture < 3.0:
      
      txt_key = 2.5
      add_data(txt_key, performance, statistics)

    elif texture >= 3.0 and texture < 4.0:
      
      txt_key = 3.5
      add_data(txt_key, performance, statistics)
    
    elif texture >= 4.0 and texture < 5.0:

      txt_key = 4.5
      add_data(txt_key, performance, statistics)

    elif texture >= 5.0 and texture < 6.0:

      txt_key = 5.5
      add_data(txt_key, performance, statistics)
      
  return performance

#------------------------------------------------------------------------------------------

def add_data(key_texture, performance, statistics):

  for key, value in statistics.iteritems():

    performance[key_texture][key][0].append(value[0])
    performance[key_texture][key][1].append(value[1])

#------------------------------------------------------------------------------------------

def run(path, data):

  fname_left = path + data[0] + data[1]
  fname_right = path + data[0] + data[2]
  output_path = path + data[0] + data[3]
  multiplier = data[4]
  pre_shift = data[5]
  occl_range = data[6]
  disparity_min = data[7]
  disparity_max = data[8]
  scale_factor = data[9]
   
  # segment input stereo pair  
  os.system("./denseStereo " + fname_left + " " + fname_right + " " + output_path + " " + str(multiplier) + " " + str(pre_shift))
  time.sleep(1)

  # view all intermediate steps
  vd.view_disparity(output_path, disparity_min, disparity_max, occl_range)
  
  # run the surface fitting to find dense disparity map
  disparity = sf.surface_fitting(output_path, occl_range, disparity_min, disparity_max)
  
  return sgt.get_statistics(path+data[0], scale_factor, disparity, disparity_min, disparity_max, occl_range)
  
#------------------------------------------------------------------------------------------

# when a Python program is executed directly
if __name__ == "__main__":
  main()

#------------------------------------------------------------------------------------------
