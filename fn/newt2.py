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

n1 = ends
n = 0
c = 0
c1 = 0
rtim = 0
class newt2:

    def newt2(y):
        """Effect that originates in the center and scrolls outwards"""
        global p, ends, mids, n, c, c1, n1,rtim

        y = y**2
        gain.update(y)
        y /= gain.value
        y *= 255.0
        rtim+=1
#         g = 2*int(np.max(y[:len(y) // 3]))
#         b = 2*int(np.max(y[len(y) // 2: 2 * len(y) // 2]))
#         r = 2*int(np.max(y[2 * len(y) // 3:]))
        g = 255*(.5*np.sin(rtim/20)+.5)
        r = 255*(.5*np.sin(rtim/20+2*np.pi/3)+.5)
        b = 255*(.5*np.sin(rtim/20+4*np.pi/3)+.5)
        c+=1
        trig = np.mean((r+g+b)/3)
        c1+=1
#         if trig > 50 and c>15: #or n[0] == mids.any:
#             n1 = ends
#            
#         elif c>30:# c1 > 15:
#             n1 = mids[rn.randint(0,2),:].astype(int)
#             c = 0
        n1 = np.linspace(0,999,50).astype(int)
        if c1>22 or c1==1:
            p[0, n1]    = b#b+25
            p[1, n1]    = g#g*(.5*np.sin(rtim/20)+.5)
            p[2, n1]    = r#r+25
            c1 = 2        
        n = 1
        p[:, n:] = p[:, :-n]#**1.01
        p *= 0.995
        #p =p/(np.max(p))*255
        
        #p = p / np.max(p) * 255
        #p = gaussian_filter1d(p, sigma=0.2)
        return p


    


