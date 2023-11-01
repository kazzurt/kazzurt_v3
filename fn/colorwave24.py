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

arx         = np.linspace(0,39,40).astype(int)
ary         = np.linspace(0,24,25).astype(int)
red_ar      = np.zeros((40,25))
gre_ar      = np.zeros((40,25))
blu_ar      = np.zeros((40,25))
red_ar2     = np.zeros((40,25))
gre_ar2     = np.zeros((40,25))
blu_ar2     = np.zeros((40,25))
inten1      = np.zeros((40,25))
inten2      = np.zeros((40,25))

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

class colorwave24:
    def colorwave24(y): #function 51
        global p, rtim, pix, arx, ary, rtim3, coo, xdiv, ydiv, cy, oods, ard, cyc, bthe, thresh_bthe, bdir, red_ar, gre_ar, blu_ar, rtim4, C, rtim5, y_off, x_off, \
               coms, right, left, rig, lig, numx, numy, phas, red_ar2, blu_ar2, gre_ar2, sec
        y = y**2
        gain.update(y)
        y /= gain.value
        y *= 255.
        ty = np.mean(y[len(y)//2::])
       
        rtim3+=1
        rtim5+=1
        bthe+=1
        rtim +=1
        xf = x_off
        yf = y_off
        
        for i in arx:
            red_ar[i,ary] =  (.5*np.sin(rtim/4 + ary/yf + i/xf            )+.5)
            gre_ar[i,ary] =  (.5*np.sin(rtim/4 + ary/yf + i/xf + 2*np.pi/3+phas)+.5)
            blu_ar[i,ary] =  (.5*np.sin(rtim/4 + ary/yf + i/xf + 4*np.pi/3+2*phas)+.5)
        
        for i in arx:
            red_ar2[i,ary] =  (.5*np.sin(rtim/sec + ary/yf + i/xf + 2*np.pi/3+phas)+.5)
            gre_ar2[i,ary] =  (.5*np.sin(rtim/sec + ary/yf + i/xf + 4*np.pi/3+2*phas)+.5)
            blu_ar2[i,ary] =  (.5*np.sin(rtim/sec + ary/yf + i/xf )+.5)  
                        
        p[0,:] = (viz_mf.flatMatHardMode(red_ar)+viz_mf.flatMatHardMode(red_ar2)/2) * 255
        p[1,:] = (viz_mf.flatMatHardMode(gre_ar)+viz_mf.flatMatHardMode(gre_ar2)/2) * 255
        p[2,:] = (viz_mf.flatMatHardMode(blu_ar)+viz_mf.flatMatHardMode(blu_ar2)/2) * 255
        
        return p
