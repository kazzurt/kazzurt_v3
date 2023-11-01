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

arx         = np.linspace(0,38,20).astype(int)
arx2        = np.linspace(1,39,20).astype(int)

ary         = np.linspace(0,6,7).astype(int)
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
phas        = np.pi/2
sec         = 4
ydi         = 20
xdi         = 15
thresh      = .4
swit        = 1
#Colorwave1
rtim11      = 0
rtim31      = 0
coo11       = np.array([1,1,1]).astype(float)
cy11        = 0
y_off11     = 14
x_off11     = 2
inc = 1
#Colorwave6
rtim36 = 0
class pointwave2:
    def pointwave2(y):  
        global p, rtim, pix, arx, ary, rtim3, coo, xdiv, ydiv, cy, oods, ard, cyc, bthe, thresh_bthe, bdir, red_ar, gre_ar, blu_ar, rtim4, C, rtim5, y_off, x_off, \
               right, left, rig, lig, numx, numy, phas, red_ar2, blu_ar2, gre_ar2, sec, inten1, inten2, ydi, xdi, rtim36, inc, arx2
        y = y**2
        gain.update(y)
        y /= gain.value
        y *= 255.
        ty = np.mean(y[len(y)//2::])
       
        rtim3+=1
        rtim5+=1
        rtim36 +=1
        if rtim36>1:
            rtim36 = 0
            if np.max(ary)>=23:
                inc = -1
            elif np.min(ary) <=0:
                inc = 1
            ary +=inc
        bthe+=1
        rtim-=1
        xf = x_off
        yf = y_off
        phas = np.pi*np.sin(rtim/20) + 1.5*np.pi
       
        xdi = 5*np.sin(np.pi*rtim/20)+6
        ydi = 10*np.sin(np.pi*rtim/10)+15
        
        for i in arx:
            inten1[i,ary]  = ((.5*np.sin(np.pi*ary/ydi - 25*np.pi)+.5)/2 + (.5*np.sin(np.pi*i/xdi - 10*np.pi)+.5)/2)*(.5*np.sin(rtim5*2*np.pi/25)+.5) #wtf kurt
            inten2[i,ary]  = ((.5*np.sin(np.pi*ary/ydi)+.5)/2 + (.5*np.sin(np.pi*i/xdi)+.5)/2)*(.5*np.sin(rtim5*2*np.pi/25-np.pi)+.5)
            
            red_ar[i,ary] =  (.5*np.sin(rtim/4 + ary/ydi + i/xdi            )+.4)
            gre_ar[i,ary] =  (.5*np.sin(rtim/4 + ary/ydi + i/xdi + 2*np.pi/3+phas)+.4)
            blu_ar[i,ary] =  (.5*np.sin(rtim/4 + ary/ydi + i/xdi + 4*np.pi/3+2*phas)+.4)        
 

        p[0,:] = viz_mf.flatMatHardMode(red_ar)
        p[1,:] = viz_mf.flatMatHardMode(gre_ar)
        p[2,:] = viz_mf.flatMatHardMode(blu_ar)
        
        p = p/np.max(p)*255
        return p

