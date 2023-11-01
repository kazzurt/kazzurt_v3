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
R = np.linspace(0,config.ARY-1,config.ARY).astype(int)
C = np.linspace(0,config.ARX-1,config.ARX).astype(int)
class colorscroll:
    
    def colorscroll(y):
            global p, rtim11, pix, arx, ary, rtim31, coo11, cy11, bdir, red_ar, gre_ar, blu_ar, y_off11, x_off11, R, C
            y = y**2
            gain.update(y)
            y /= gain.value
            y *= 255.
            ty = np.mean(y[len(y)//2::])
            
            rtim11-=.5
            rtim31+=.5
            nn = 25
            dx = int(nn/2*np.sin(rtim11/25)+nn*2)
            dx2 = int(nn/2*np.cos(rtim11/25)+nn*2)
            
            dy = int(nn/2*np.sin(rtim11/25)+nn*2)
            dy2 = int(nn/2*np.cos(rtim11/25)+nn*2)
            
            C = np.linspace(0,config.ARX-1,dx).astype(int)
            C2 = np.linspace(0,config.ARX-1,dx2).astype(int)
            
            R = np.linspace(0,config.ARY-1,dy).astype(int)
            R2 = np.linspace(0,config.ARY-1,dy2).astype(int)
            
            num = 4*np.sin(rtim11/25)+3

            xf = x_off11 + num
            yf = y_off11 - num
           
            #for i in range(0,config.ARX):
            for i in range(0,len(R)):
                blu_ar[C,R[i]] = 255
                red_ar[C,R[i]] = 255
                gre_ar[C,R[i]] = 255
                
            for i in range(0,len(C2)):
                blu_ar[C2[i],R2] = 0
                red_ar[C2[i],R2] = 0
                gre_ar[C2[i],R2] = 0
                

                            
            p[0,:] = (.5*np.sin(rtim11/40)+.5)          *coo11[0]*quadratize.flatMatQuads(red_ar)
            p[1,:] = (.5*np.sin(rtim11/40+2*np.pi/3)+.5)*coo11[1]*quadratize.flatMatQuads(gre_ar)
            p[2,:] = (.5*np.sin(rtim11/40+4*np.pi/3)+.5)*coo11[2]*quadratize.flatMatQuads(blu_ar)
            p[0, :] = gaussian_filter1d(p[0, :], sigma=2)
            p[1, :] = gaussian_filter1d(p[1, :], sigma=2)
            p[2, :] = gaussian_filter1d(p[2, :], sigma=2)
            return p



