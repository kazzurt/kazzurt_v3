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
import floor_quad
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
y_off11     = 3
x_off11     = 3
fl = 1

class colorwave0:
    
    def colorwave0(y):
            global p, rtim11, pix, arx, ary, rtim31, coo11, cy11, bdir, red_ar, gre_ar, blu_ar, y_off11, x_off11, fl
            y = y**2
            gain.update(y)
            y /= gain.value
            y *= 255.
            ty = np.mean(y[len(y)//2::])
            
            rtim11+=1
            rtim31+=1
            
            num = 1 #5*np.sin(rtim11/4)+6
            
            
            
            #copied from breathe2
            #This is the best wave function, dependent on loop number (rtim), x direction (i in arx) and y direction (ary)
            #Would be nice if both directions could be a matrix operation, but idk how to do that. Choosing smaller direction for the for loop
            xf = x_off11 + num + (.5*np.sin(rtim11/100+np.pi/2)+.5)*1.9
            yf = y_off11 - num - (.5*np.sin(rtim11/100)+.5)*1.9

            ph1 = .5*np.sin(rtim11/10)+1
            ph2 = .5*np.sin(rtim11/10+np.pi/2)+3
            for i in arx:
#                 red_ar[i,ary] =  (.5*np.sin(rtim11/12 + ary/yf + i/xf            )+.5)*255
#                 gre_ar[i,ary] =  (.5*np.sin(rtim11/12 + ary/yf + i/xf + ph1*np.pi/3)+.5)*255
#                 blu_ar[i,ary] =  (.5*np.sin(rtim11/12 + ary/yf + i/xf + ph2*np.pi/3)+.5)*255
                red_ar[i,ary] =  (.5*np.sin(rtim11/12  + ary/yf          )+.5)*255
                gre_ar[i,ary] =  (.5*np.sin(rtim11/12 + ary/yf + ph1*np.pi/3)+.5)*255
                blu_ar[i,ary] =  (.5*np.sin(rtim11/12 + ary/yf + ph2*np.pi/3)+.5)*255            
#             p[0,:] = coo11[0]*viz_mf.flatMatHardMode(red_ar) #/ np.max(p[0,:]) * 255
#             p[1,:] = coo11[1]*viz_mf.flatMatHardMode(gre_ar) #/ np.max(p[1,:]) * 255
#             p[2,:] = coo11[2]*viz_mf.flatMatHardMode(blu_ar) #/ np.max(p[2,:]) * 255
#             p[0,:] = red_ar.flatten()#*(.2*np.sin(rtim11/1000)+.8)
#             p[1,:] = gre_ar.flatten()#*(.2*np.sin(rtim11/1000+ ph1*np.pi/3+.1)+.8)
#             p[2,:] = blu_ar.flatten()#*(.2*np.sin(rtim11/1000+ ph1*np.pi/3+.2)+.8)
            x = np.linspace(0,51,52).astype(int)
            y = 14

            if y>=25:
                rtim11=0
            red_ar[:] = 0
            gre_ar[:] = 0
            blu_ar[:] = 0
            red_ar[x,y] = 255
            gre_ar[x,y] = 255
            blu_ar[x,y] = 255
            p[0,:] = quadratize.flatMatQuads(red_ar)
            p[1,:] = quadratize.flatMatQuads(gre_ar)
            p[2,:] = quadratize.flatMatQuads(blu_ar)
            #p = (p/np.max(p))*255
#             time.sleep(3)
            
            return p

