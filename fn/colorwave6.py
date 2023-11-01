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
import quadratize
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
sec = 10
class colorwave6:
    def colorwave6(y):
        global p, rtim, pix, arx, ary, rtim36, coo, xdiv, ydiv, cy, oods, ard, cyc, bthe, thresh_bthe, bdir, red_ar, gre_ar, blu_ar, sparkle, rtim4, C, rtim5, y_off, x_off, \
               sparkle2, sparkle3, coms, sec
        y = y**2
        gain.update(y)
        y /= gain.value
        y *= 255.
        ty = np.mean(y[len(y)//2::])
        
        rtim+=.5
        rtim36+=.5
        rtim5+=.5
        bthe+=.5
        
        num = 8*np.sin(rtim/20)+10
        
        #If our x and y division sine wave is at its peak, let's increment cy, which determines color remapping below
        #rtim3 prevents going into this function a bunch of times while num sits close to its maximum
        #bdir controls whether we're stepping cy upwards or downwards
        if num > 10 and rtim36 > 15:
            rtim36 = 0
            if bdir == 1: 
                cy+=1
            elif bdir == -1:
                cy-=1
                if cy<=0:
                    bdir = 1
                    #sparkle = 0
                    rtim4 = 0
        
        #copied from breathe2
        #This is the best wave function, dependent on loop number (rtim), x direction (i in arx) and y direction (ary)
        #Would be nice if both directions could be a matrix operation, but idk how to do that. Choosing smaller direction for the for loop
        xf = x_off+num+(3*np.sin(rtim/10+np.pi)+9)
        yf = y_off-num+(3*np.sin(rtim/10)+10)
        if cyc==0:
            for i in arx:
                red_ar[i,ary] =  (.5*np.sin(rtim/sec + ary/yf + i/xf            )+.5)*255
                gre_ar[i,ary] =  (.5*np.sin(rtim/sec + ary/yf + i/xf + np.pi/3)+.5)*255
                blu_ar[i,ary] =  (.5*np.sin(rtim/sec + ary/yf + i/xf + 2*np.pi/3)+.5)*255
#         if cy>1:
#             red_ar = np.fliplr(red_ar)
#             if cy>2:
#                 gre_ar = np.fliplr(gre_ar)
#                 if cy>3:
#                     blu_ar = np.fliplr(blu_ar)
#                     if cy>4:
#                         bdir=-1 #this will make us start stepping cy backwards
#                         #sparkle = 1
                        
        p[0,:] = (.5*(np.sin(rtim/10)**3)+.5)                  *coo[0]*quadratize.flatMatQuads(red_ar)
        p[1,:] = (.5*(np.sin(rtim/10+np.pi/3+np.pi/6)**3)  +.5)*coo[1]*quadratize.flatMatQuads(gre_ar)
        p[2,:] = (.5*(np.sin(rtim/10+np.pi*2/3+np.pi/3)**3)+.5)*coo[2]*quadratize.flatMatQuads(blu_ar)
        
        return p
