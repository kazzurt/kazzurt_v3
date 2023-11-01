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
import kzbutfun
import quadratize
p      = np.tile(1.0, (3, config.N_PIXELS ))#// 2))
gain   = dsp.ExpFilter(np.tile(0.01, config.N_FFT_BINS), alpha_decay=0.001, alpha_rise=0.99)

ends = np.linspace(0,950,20).astype(int)
#mids = np.linspace(25,975,20).astype(int)
mids = ends+49
mids      = np.tile(1.0, (3, len(ends )))
mids[0,:] = np.linspace(25,975,20).astype(int)
mids[1,:] = ends+35

n1 = ends
n = 0
c = 0
c1 = 0
rtim = 0
class newt2:

    def newt2(y):
        """Effect that originates in the center and scrolls outwards"""
        global p, ends, mids, n, c, c1, n1, rtim
        rtim +=1/y
        n1 = np.linspace(0,1300-1,20).astype(int)
        for i in range(0,len(n1)-1):
            if n1[i]<=1300-1:
                n1[i]+=1
        
        p[0, n1]    = (.5*np.sin(rtim/10)+.5)*255
        p[1, n1]    = (.5*np.sin(rtim/10+np.pi/3*(np.sin(rtim/20)))+.5)*255
        p[2, n1]    = (.5*np.sin(rtim/10+4*np.pi/3*(np.sin(rtim/20)+np.pi/3))+.5)*255

        
        n = 1
        p[:, n:] = p[:, :-n]#**1.01
        if rtim%10 == 0:
          p[:, rn.randint(0,1300-1,size=rn.randint(0,int(20*np.sin(rtim/3)+50)))]   = 0

        #p[:,:] = 255
        #p = p / np.max(p) * 255
        #p = gaussian_filter1d(p, sigma=0.2)
        return p


    


