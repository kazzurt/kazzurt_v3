from __future__ import print_function
from __future__ import division
import time
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import config
import microphone
import dsp
import led
import sys
import numpy as np
from numpy import random as rn
import config
from color_pal import pallette
import viz_mf
import cmdfun
import pygame

gain     = dsp.ExpFilter(np.tile(0.01, config.N_FFT_BINS), alpha_decay=0.001, alpha_rise=0.99)
pix      = config.N_PIXELS // 2 - 1
co2      = pallette.pal(0)
p        = np.tile(1.0, (3, config.N_PIXELS))

arxo      = np.linspace(1,19,10).astype(int)
arxe      = np.linspace(0,20,10).astype(int)
aryo      = np.linspace(0,49,50).astype(int)
arye      = np.linspace(0,49,20).astype(int)
arr1 = np.zeros((40,25))
arr2 = np.zeros((40,25))
numy = 39
numx = 24
class special:
    
    def teeth(y):
        global p, ttim, pix, arr1, arr2, numy, numx
        
        for i in arxo:
            for j in aryo:
                arr[i,j] = 255

        
        p[:,:] = viz_mf.flatMatHardMode(arr)
        numy-=1
        numx-=1
        
        return p
