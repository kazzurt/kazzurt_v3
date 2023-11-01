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

gain     = dsp.ExpFilter(np.tile(0.01, config.N_FFT_BINS), alpha_decay=0.001, alpha_rise=0.99)
pix      = config.N_PIXELS // 2 - 1
p        = np.tile(1.0, (3, config.N_PIXELS))
coo      = np.array([1,1,1]).astype(float)


rtim        = 0
rtim3       = 0
cnt3        = 0
cy          = 0
ard         = 1
cyc         = 0
bthe        = 0
thresh_bthe = 0.5
timeCount   = 1
countUp     = True

arx         = np.linspace(0,config.ARX-1,config.ARX).astype(int)
ary         = np.linspace(0,config.ARY-1,config.ARY).astype(int)
red_ar      = np.zeros((config.ARX,config.ARY))
gre_ar      = np.zeros((config.ARX,config.ARY))
blu_ar      = np.zeros((config.ARX,config.ARY))


bdir        = 1
nuu         = 50 #defines speed for kuwave2 (higher is slower)
mat_map     = 1
sparkle     = 0
rtim4       = 0
rtim5       = 0
y_off       = 14
x_off       = 2
sparkle2    = 0
sparkle3    = 0

right       = 1
left        = 0
rig         = 0
lig         = 0
numx        = 1
numy        = 0
phas        = np.pi
sec         = 4
ydi         = 10
xdi         = 5
thresh      = .4
swit        = 1
#Colorwave1
rtim11      = 0
rtim31      = 0
coo11       = np.array([1,1,1]).astype(float)
cy11        = 0
y_off11     = 14
x_off11     = 2

#Colorwave6
rtim36 = 0

class colorwave2:
    def colorwave2(y):
        global p, rtim, pix, arx, ary, rtim3, coo, xdiv, ydiv, cy, ard, cyc, bthe, thresh_bthe, bdir, red_ar, gre_ar, blu_ar, rtim4, C, rtim5, y_off, x_off, \
               coms, right, left, rig, lig, numx, numy
        
        y = y**2
        gain.update(y)
        y /= gain.value
        y *= 255.
        ty = np.mean(y[len(y)//2::])
       
        rtim+=.2

        xf = x_off
        yf = y_off
        
            
            
        if cyc==0:
            for i in arx:
                red_ar[i,ary] =  (.5*np.sin(rtim/4 + ary/yf + i/xf            )+.5)*255
                gre_ar[i,ary] =  (.5*np.sin(rtim/4 + ary/yf + i/xf + 2*np.pi/3)+.5)*255
                blu_ar[i,ary] =  (.5*np.sin(rtim/4 + ary/yf + i/xf + 4*np.pi/3)+.5)*255
        if cy>1:
            red_ar = np.fliplr(red_ar)
            if cy>2:
                gre_ar = np.fliplr(gre_ar)
                if cy>3:
                    blu_ar = np.fliplr(blu_ar)
                    if cy>4:
                        bdir=-1 #this will make us start stepping cy backwards
                        #sparkle = 1
                        
        p[0,:] = coo[0]*viz_mf.flatMatHardMode(red_ar)
        p[1,:] = coo[1]*viz_mf.flatMatHardMode(gre_ar)
        p[2,:] = coo[2]*viz_mf.flatMatHardMode(blu_ar)
 
        return p