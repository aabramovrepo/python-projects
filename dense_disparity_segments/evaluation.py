#! /usr/bin/env python
# Alexey Abramov <abramov@physik3.gwdg.de>

import numpy as np
import Image as img
import scipy as sp
from scipy import ndimage
import matplotlib as mpl
import matplotlib.pyplot as plt
import pylab as p
#import math
import sys

#------------------------------------------------------------------------------------------
def texture_disparity_plots():

  # plot RMS for the Middlebury images depending on the texture level
  x = [2.5, 3.5, 4.5, 5.5]

  y_icm =  [16.13, 16.6, 14.2, 12.8]
  y_bpm =  [12.2, 11.5, 8.1, 7.2]
  y_bps =  [10.2, 9.7, 7.1, 7.0]
  y_trws = [10.15, 10.8, 6.8, 4.6]
  y_exp =  [10.4, 11.5, 7.1, 6.85]
  y_swap = [9.6, 10.4, 6.8, 6.8]
  y_bm =   [6.8, 5.7, 3.56, 4.3]
  y_bp =   [16.2, 15.9, 12.6, 9.7]
#  y_csbp = [8.3, 14.8, 8.7, 9.8]
  y_csbp = [8.3, 12, 8.7, 9.8]
  y_phas = [5.9, 6.5, 2.55, 2.55]

  y_lsq_planar = [5.2, 7.4, 4.4, 3.45]
  y_lsq_quadric = [5.3, 8.1, 5.1, 3.9]
  y_nelder_planar = [5.3, 7.1, 3.9, 3.3]
  y_nelder_quadric = [5.6, 7.3, 4.6, 3.6]

  fg = plt.figure()

  # set scale
  p.plot(2.0, 0.0)
  p.plot(6.0, 20.0)

  #p.plot(2.0, 2.0)
  #p.plot(6.0, 9.0)

  #plot_stereo_values(x, y_icm, y_bpm, y_bps, y_trws, y_exp, y_swap, y_bm, y_bp, y_csbp, y_phas)
  plot_stereo_values_ours(x, y_icm, y_bpm, y_bps, y_trws, y_exp, y_swap, y_bm, y_bp, y_csbp, y_phas, y_lsq_planar, y_lsq_quadric, y_nelder_planar, y_nelder_quadric)

  #p.show()

  # plot amount of found pixels for the Middlebury images depending on the texture level
  x = [2.5, 3.5, 4.5, 5.5]

  y_icm =  [85.2, 89.9, 93, 94.4]
  y_bpm =  [88.4, 92.7, 96.6, 97.3]
  y_bps =  [90.6, 91, 95.8, 97]
  y_trws = [88.4, 92.6, 95.4, 97.1]
  y_exp =  [88.5, 93.5, 95.7, 97]
  y_swap = [86.9, 91.8, 95.1, 96.4]
  y_bm =   [46.4, 62.5, 76, 76.3]
  y_bp = [93.9, 96.4, 97.8, 97.9]
  y_csbp = [98.9, 98.8, 98.7, 98.9]
  y_phas = [59.4, 66.8, 78.7, 72]
  
  y_lsq_planar = [88, 86, 85, 85]
  y_lsq_quadric = [89, 85, 86, 86]
  y_nelder_planar = [88, 85, 85, 85]
  y_nelder_quadric = [88, 86, 86, 85.5]

  fg = plt.figure()

  # set scale
  p.plot(2, 105)
  p.plot(6, 105)

  #p.plot(2, 80)
  #p.plot(6, 90)

  #plot_stereo_values(x, y_icm, y_bpm, y_bps, y_trws, y_exp, y_swap, y_bm, y_bp, y_csbp, y_phas)
  plot_stereo_values_ours(x, y_icm, y_bpm, y_bps, y_trws, y_exp, y_swap, y_bm, y_bp, y_csbp, y_phas, y_lsq_planar, y_lsq_quadric, y_nelder_planar, y_nelder_quadric)

  #p.show()

  # plot RMS for images from our database depending on the texture level
  x = [2.5, 3.5, 4.5, 5.5]

  y_icm =  [16.2, 12.4, 13, 12]
  y_bpm =  [36, 16.8, 12.2, 10.5]
  y_bps =  [35.8, 17.4, 12.6, 10.5]
  y_trws = [38, 16.1, 12.6, 10.5]
  y_exp =  [37.3, 16.1, 12.6, 10.5]
  y_swap = [39.7, 17.3, 12.2, 10.5]
  y_bm =   [22.8, 21.4, 18, 16.5]
#  y_bp =   [24.2, 29.6, 34.8, 24.5]
  y_bp =   [24.2, 29.6, 29.6, 24.5]
  y_csbp = [19.5, 18.5, 14.4, 13.5]
  y_phas = [28.2, 26.1, 18.6, 21]

  y_lsq_planar = [29.6, 30, 29.2, 37]
  y_lsq_quadric = [29.1, 29.6, 28.8, 36.5]
  y_nelder_planar = [29.6, 30.8, 29.6, 35.3]
  y_nelder_quadric = [29.5, 30.6, 28.7, 35.4]

  fg = plt.figure()

  # set scale
  p.plot(2, 45)
  p.plot(6, 5)

  #p.plot(2, 40)
  #p.plot(6, 25)

#  plot_stereo_values(x, y_icm, y_bpm, y_bps, y_trws, y_exp, y_swap, y_bm, y_bp, y_csbp, y_phas)
  plot_stereo_values_ours(x, y_icm, y_bpm, y_bps, y_trws, y_exp, y_swap, y_bm, y_bp, y_csbp, y_phas, y_lsq_planar, y_lsq_quadric, y_nelder_planar, y_nelder_quadric)

  #p.show()

  # plot amount of found pixels for images from our database depending on the texture level
  x = [2.5, 3.5, 4.5, 5.5]
  y_icm =  [30.2, 40.6, 52.6, 68]
  y_bpm =  [50.3, 63.8, 81, 89]
  y_bps =  [52.7, 67.5, 82.2, 87]
  y_trws = [52, 67.8, 82.2, 87]
  y_exp =  [55.2, 67.9, 82.8, 87]
  y_swap = [52, 66.4, 82.4, 87]
  y_bm =   [32.8, 50.9, 70.4, 71]
  y_csbp = [59.2, 71.5, 83.2, 86.5]
  y_bp =   [65.2, 75.6, 77.6, 83]
  y_phas = [55.7, 61.3, 77, 70.5]

  y_lsq_planar = [87, 80, 79, 79]
  y_lsq_quadric = [87, 79, 77, 78]
  y_nelder_planar = [87, 79, 76, 78]
  y_nelder_quadric = [87, 79, 78, 80]

  fg = plt.figure()

  # set scale
  p.plot(2, 95)
  p.plot(6, 25)

  #p.plot(2, 70)
  #p.plot(6, 90)

#  plot_stereo_values(x, y_icm, y_bpm, y_bps, y_trws, y_exp, y_swap, y_bm, y_bp, y_csbp, y_phas)
  plot_stereo_values_ours(x, y_icm, y_bpm, y_bps, y_trws, y_exp, y_swap, y_bm, y_bp, y_csbp, y_phas, y_lsq_planar, y_lsq_quadric, y_nelder_planar, y_nelder_quadric)

  p.show()

#------------------------------------------------------------------------------------------
def plot_stereo_values(x, y_icm, y_bpm, y_bps, y_trws, y_exp, y_swap, y_bm, y_bp, y_csbp, y_phas):

  p.plot(x, y_icm, color = 'green',  lw=3)
  p.plot(x, y_bpm, color = 'red',  lw=3)
  p.plot(x, y_bps, color = 'yellow',  lw=3)
  p.plot(x, y_trws, color = 'blue',  lw=3)
  p.plot(x, y_exp, color = 'magenta',  lw=3)
  p.plot(x, y_swap, color = 'orange',  lw=3)
  p.plot(x, y_bm, color = 'purple',  lw=3)
  p.plot(x, y_bp, color = 'black',  lw=3)
  p.plot(x, y_csbp, color = 'darkblue',  lw=3)
  p.plot(x, y_phas, color = 'darkred',  lw=3)

#------------------------------------------------------------------------------------------
def plot_stereo_values_ours(x, y_icm, y_bpm, y_bps, y_trws, y_exp, y_swap, y_bm, y_bp, y_csbp, y_phas, y_lsq_planar, y_lsq_quadric, y_nelder_planar, y_nelder_quadric):

  p.plot(x, y_icm, color = 'green',  lw=3)
  p.plot(x, y_bpm, color = 'red',  lw=3)
  p.plot(x, y_bps, color = 'yellow',  lw=3)
  p.plot(x, y_trws, color = 'blue',  lw=3)
  p.plot(x, y_exp, color = 'magenta',  lw=3)
  p.plot(x, y_swap, color = 'orange',  lw=3)
  p.plot(x, y_bm, color = 'purple',  lw=3)
  p.plot(x, y_bp, color = 'black',  lw=3)
  p.plot(x, y_csbp, color = 'darkblue',  lw=3)
  p.plot(x, y_phas, color = 'darkred',  lw=3)

#  p.plot(x, y_lsq_planar, color = 'blue',  lw=5)
#  p.plot(x, y_lsq_quadric, color = 'green',  lw=5)
  p.plot(x, y_nelder_planar, color = 'red',  lw=5)
#  p.plot(x, y_nelder_quadric, color = 'orange',  lw=5)

#------------------------------------------------------------------------------------------

# images from the Middlebury dataset

#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/aloe-texture.dat')
#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/baby1-texture.dat')
#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/baby2-texture.dat')
#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/baby3-texture.dat')
#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/bowling1-texture.dat')
#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/bowling2-texture.dat')
#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/cloth1-texture.dat')
#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/cloth2-texture.dat')
#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/cloth3-texture.dat')
#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/cloth4-texture.dat')
#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/cones-texture.dat')
#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/flowerpots-texture.dat')
#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/lampshade1-texture.dat')
#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/lampshade2-texture.dat')
#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/mid1-texture.dat')
#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/mid2-texture.dat')
#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/monopoly-texture.dat')
#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/rocks1-texture.dat')
#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/rocks2-texture.dat')
#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/teddy-texture.dat')
#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/tsukuba-texture.dat')
#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/venus-texture.dat')
#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/wood1-texture.dat')

#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/wood2-texture.dat')

# images from our dataset

#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/banana-1-texture.dat')
#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/blue-pan-texture.dat')
#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/box-1-texture.dat')
#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/box-2-texture.dat')
#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/breakfast-1-texture.dat')
#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/breakfast-2-texture.dat')
#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/cups1-texture.dat')
#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/cups2-texture.dat')
#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/garnics1-texture.dat')
#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/garnics2-texture.dat')
#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/garnics3-texture.dat')
#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/garnics4-texture.dat')
#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/ice-brick-texture.dat')
#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/mouse-pad-texture.dat')
#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/plastic-texture.dat')
#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/red-plate-texture.dat')
#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/sandwich-1-texture.dat')
#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/sandwich-2-texture.dat')
#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/scene01-texture.dat')
#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/scene03-texture.dat')
#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/scene04-texture.dat')
#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/scene05-texture.dat')
#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/scene09-texture.dat')
#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/scene15-texture.dat')
#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/table2-texture.dat')
#arr_entropy = np.loadtxt('/home/abramov/Desktop/Thesis-plots/Texture-detection/texture-values/towel1-texture.dat')


#height = arr_entropy.shape[0]
#width = arr_entropy.shape[1]

#entropy_sum = arr_entropy.sum()
#entropy_avg = entropy_sum / (height * width)

#plot_array('Entropy values', arr_entropy, arr_entropy.min(), arr_entropy.max(), 'entropy-values-matlab' + str(entropy_avg))


#texture_disparity_plots()


#------------------------------------------------------------------------------------------

def plot_score(x, y):

  p.plot(x, y, color = 'green',  lw=3)

#------------------------------------------------------------------------------------------
  
def meanstdv(x):
  
  from math import sqrt
  n, mean, std = len(x), 0, 0
  
  for a in x:
    mean = mean + a
    
  mean = mean / float(n)
  
  for a in x:
    std = std + (a - mean)**2
    
  std = sqrt(std / float(n-1))
  
  return mean, std
  
#------------------------------------------------------------------------------------------

def main():

  y_phas = [6.5, 8.0, 3.8, 5.4]
  
  txt = [2.86, 2.95, 2.2, 2.71]
  
  mean, std = meanstdv(y_phas)
  
  mean_txt, std_txt = meanstdv(txt)
  
  print 'mean = ', mean, ' std = ', std
  print 'mean_txt = ', mean_txt, ' std_txt = ', std_txt

#  sys.exit()

  # plot RMS for the Middlebury images depending on the texture level
  x = [2.5, 3.5, 4.5, 5.5]

  y_phas_rms = [5.9, 6.5, 2.55, 2.55]
  y_phas_n = [59.4, 66.8, 78.7, 72.0]
  
  fg = plt.figure()

  p.plot(2.0, 0.0)
  p.plot(6.0, 20.0)

  #p.plot(x, y_phas_rms)
  
  plt.errorbar(x, y_phas_rms, xerr=0.33, yerr=1.77)

  p.show()

  
  sys.exit()
  
  
  y_icm =  [16.13, 16.6, 14.2, 12.8]
  y_bpm =  [12.2, 11.5, 8.1, 7.2]
  y_bps =  [10.2, 9.7, 7.1, 7.0]
  y_trws = [10.15, 10.8, 6.8, 4.6]
  y_exp =  [10.4, 11.5, 7.1, 6.85]
  y_swap = [9.6, 10.4, 6.8, 6.8]
  y_bm =   [6.8, 5.7, 3.56, 4.3]
  y_bp =   [16.2, 15.9, 12.6, 9.7]
  y_csbp = [8.3, 12, 8.7, 9.8]
  y_phas = [5.9, 6.5, 2.55, 2.55]

  y_lsq_planar = [5.2, 7.4, 4.4, 3.45]
  y_lsq_quadric = [5.3, 8.1, 5.1, 3.9]
  y_nelder_planar = [5.3, 7.1, 3.9, 3.3]
  y_nelder_quadric = [5.6, 7.3, 4.6, 3.6]

  fg = plt.figure()

  # set scale
  p.plot(2.0, 0.0)
  p.plot(6.0, 20.0)

  #p.plot(2.0, 2.0)
  #p.plot(6.0, 9.0)

  #plot_stereo_values(x, y_icm, y_bpm, y_bps, y_trws, y_exp, y_swap, y_bm, y_bp, y_csbp, y_phas)
  plot_stereo_values_ours(x, y_icm, y_bpm, y_bps, y_trws, y_exp, y_swap, y_bm, y_bp, y_csbp, y_phas, y_lsq_planar, y_lsq_quadric, y_nelder_planar, y_nelder_quadric)

  #p.show()

  # plot amount of found pixels for the Middlebury images depending on the texture level
  x = [2.5, 3.5, 4.5, 5.5]

  y_icm =  [85.2, 89.9, 93, 94.4]
  y_bpm =  [88.4, 92.7, 96.6, 97.3]
  y_bps =  [90.6, 91, 95.8, 97]
  y_trws = [88.4, 92.6, 95.4, 97.1]
  y_exp =  [88.5, 93.5, 95.7, 97]
  y_swap = [86.9, 91.8, 95.1, 96.4]
  y_bm =   [46.4, 62.5, 76, 76.3]
  y_bp = [93.9, 96.4, 97.8, 97.9]
  y_csbp = [98.9, 98.8, 98.7, 98.9]
  y_phas = [59.4, 66.8, 78.7, 72]
  
  y_lsq_planar = [88, 86, 85, 85]
  y_lsq_quadric = [89, 85, 86, 86]
  y_nelder_planar = [88, 85, 85, 85]
  y_nelder_quadric = [88, 86, 86, 85.5]

  fg = plt.figure()

  # set scale
  p.plot(2, 105)
  p.plot(6, 105)

  #p.plot(2, 80)
  #p.plot(6, 90)

  #plot_stereo_values(x, y_icm, y_bpm, y_bps, y_trws, y_exp, y_swap, y_bm, y_bp, y_csbp, y_phas)
  plot_stereo_values_ours(x, y_icm, y_bpm, y_bps, y_trws, y_exp, y_swap, y_bm, y_bp, y_csbp, y_phas, y_lsq_planar, y_lsq_quadric, y_nelder_planar, y_nelder_quadric)

  p.show()

  sys.exit()
  
  
  
  
  
  
  
  
  
  # plot RMS for images from our database depending on the texture level
  x = [2.5, 3.5, 4.5, 5.5]

  y_icm =  [16.2, 12.4, 13, 12]
  y_bpm =  [36, 16.8, 12.2, 10.5]
  y_bps =  [35.8, 17.4, 12.6, 10.5]
  y_trws = [38, 16.1, 12.6, 10.5]
  y_exp =  [37.3, 16.1, 12.6, 10.5]
  y_swap = [39.7, 17.3, 12.2, 10.5]
  y_bm =   [22.8, 21.4, 18, 16.5]
#  y_bp =   [24.2, 29.6, 34.8, 24.5]
  y_bp =   [24.2, 29.6, 29.6, 24.5]
  y_csbp = [19.5, 18.5, 14.4, 13.5]
  y_phas = [28.2, 26.1, 18.6, 21]

  y_lsq_planar = [29.6, 30, 29.2, 37]
  y_lsq_quadric = [29.1, 29.6, 28.8, 36.5]
  y_nelder_planar = [29.6, 30.8, 29.6, 35.3]
  y_nelder_quadric = [29.5, 30.6, 28.7, 35.4]

  fg = plt.figure()

  # set scale
  p.plot(2, 45)
  p.plot(6, 5)

  #p.plot(2, 40)
  #p.plot(6, 25)

#  plot_stereo_values(x, y_icm, y_bpm, y_bps, y_trws, y_exp, y_swap, y_bm, y_bp, y_csbp, y_phas)
  plot_stereo_values_ours(x, y_icm, y_bpm, y_bps, y_trws, y_exp, y_swap, y_bm, y_bp, y_csbp, y_phas, y_lsq_planar, y_lsq_quadric, y_nelder_planar, y_nelder_quadric)

  #p.show()

  # plot amount of found pixels for images from our database depending on the texture level
  x = [2.5, 3.5, 4.5, 5.5]
  y_icm =  [30.2, 40.6, 52.6, 68]
  y_bpm =  [50.3, 63.8, 81, 89]
  y_bps =  [52.7, 67.5, 82.2, 87]
  y_trws = [52, 67.8, 82.2, 87]
  y_exp =  [55.2, 67.9, 82.8, 87]
  y_swap = [52, 66.4, 82.4, 87]
  y_bm =   [32.8, 50.9, 70.4, 71]
  y_csbp = [59.2, 71.5, 83.2, 86.5]
  y_bp =   [65.2, 75.6, 77.6, 83]
  y_phas = [55.7, 61.3, 77, 70.5]

  y_lsq_planar = [87, 80, 79, 79]
  y_lsq_quadric = [87, 79, 77, 78]
  y_nelder_planar = [87, 79, 76, 78]
  y_nelder_quadric = [87, 79, 78, 80]

  fg = plt.figure()

  # set scale
  p.plot(2, 95)
  p.plot(6, 25)

  #p.plot(2, 70)
  #p.plot(6, 90)

#  plot_stereo_values(x, y_icm, y_bpm, y_bps, y_trws, y_exp, y_swap, y_bm, y_bp, y_csbp, y_phas)
  plot_stereo_values_ours(x, y_icm, y_bpm, y_bps, y_trws, y_exp, y_swap, y_bm, y_bp, y_csbp, y_phas, y_lsq_planar, y_lsq_quadric, y_nelder_planar, y_nelder_quadric)

  p.show()

#------------------------------------------------------------------------------------------

# when a Python program is executed directly
if __name__ == "__main__":
  main()

#------------------------------------------------------------------------------------------
