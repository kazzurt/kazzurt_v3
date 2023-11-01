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

p      = np.tile(1.0, (3, config.N_PIXELS ))#// 2))
gain   = dsp.ExpFilter(np.tile(0.01, config.N_FFT_BINS), alpha_decay=0.001, alpha_rise=0.99)

ends = np.linspace(0,950,20).astype(int)
#mids = np.linspace(25,975,20).astype(int)
mids = ends+49
mids      = np.tile(1.0, (3, len(ends )))
mids[0,:] = np.linspace(25,975,20).astype(int)
mids[1,:] = ends+35


n1 = np.linspace(0,999,50).astype(int)
n = 0
c = 0
c1 = 0
rtim = 0
class newt4:

    def newt4(y):
        """Effect that originates in the center and scrolls outwards"""
        global p, ends, mids, n, c, c1, n1,rtim

        y = y**2
        gain.update(y)
        y /= gain.value
        y *= 255.0
        rtim+=1
        g = 2*int(np.max(y[:len(y) // 3]))
        b = 2*int(np.max(y[len(y) // 2: 2 * len(y) // 2]))
        r = 2*int(np.max(y[2 * len(y) // 3:]))
        c+=1
        trig = np.mean((r+g+b)/3)
      
        if trig > 50 and c>10: #or n[0] == mids.any:
            for i in range(0,len(n1) ):
                if n1[i]>994:
                    n1[i] = 0    
                else:
                   n1[i] +=5
                
           
            c1+=1
        
        if n1[30] == 0:
            n1 = np.linspace(0,999,50).astype(int)
        elif c>30:# c1 > 15:
            #n1 = mids[rn.randint(0,2),:].astype(int)
           
            c = 0
            #c = 0
            #c1 = 0
        #n1 = np.linspace(0,999,50).astype(int)
        #n2 = np.linspace(0,999,10).astype(int)
        p[0, n1]    = b
        p[1, n1]    = g*(.5*np.sin(rtim/20)+.5)
        p[2, n1]    = r
        #p[:,n2] = 0
        
        n = 1
        p[:, n:] = p[:, :-n]#**1.01
        p *= .1*np.sin(rtim/35)+.8 + (r+b+g)/3/255 * .1
        #p =p/(np.max(p))*255
        #p[:,:] = 150
        #p = p / np.max(p) * 255
        #p = gaussian_filter1d(p, sigma=0.2)
        return p


    



