#! /usr/bin/env python
# Alexey Abramov <abramov@physik3.gwdg.de>

import numpy as np
import plot_data as plt

#------------------------------------------------------------------------------------------

def main():

  fnames = ['aloe-texture', 'cones-texture', 'rocks1-texture', 'teddy-texture', 'garnics1-texture', 'box-1-texture', 'scene05-texture', 'scene01-texture',
            'table1-texture', 'blue-pan-texture', 'red-plate-texture', 'lampshade1-texture', 'breakfast-1-texture', 'plastic-texture', 'cups1-texture']
  path = './texture-evaluation/texture-values/'

  for fname in fnames:
    arr = np.loadtxt(path+fname+'.dat', dtype=float)
    plt.plot_array('Texture', arr, 2, 6, path + 'plots/' + fname)

#------------------------------------------------------------------------------------------

# when a Python program is executed directly
if __name__ == "__main__":
  main()

#------------------------------------------------------------------------------------------
