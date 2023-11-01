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

class colorwave25:
    
    def colorwave25(y):
            global p, rtim11, pix, arx, ary, rtim31, coo11, cy11, bdir, red_ar, gre_ar, blu_ar, y_off11, x_off11
           
            
            rtim11-=.5/y
            rtim31+=.5/y
            
            num = 4*np.sin(rtim11/25)+3
           
            
            #If our x and y division sine wave is at its peak, let's increment cy, which determines color remapping below
            #rtim3 prevents going into this function a bunch of times while num sits close to its maximum
            #bdir controls whether we're stepping cy upwards or downwards
#             if num > 10 and rtim31 > 15:
#                 rtim31 = 0
#                 if bdir == 1: 
#                     cy11+=1
#                 elif bdir == -1:
#                     cy11-=1
#                     if cy11<=0:
#                         bdir = 1
#                         #sparkle = 0
#                         rtim4 = 0
            
            #copied from breathe2
            #This is the best wave function, dependent on loop number (rtim), x direction (i in arx) and y direction (ary)
            #Would be nice if both directions could be a matrix operation, but idk how to do that. Choosing smaller direction for the for loop
            xf = x_off11 + num
            yf = y_off11 - num
            
            for i in arx:
                red_ar[i,ary] =  (.5*np.sin(rtim11/5 + ary/(.1*yf) + i/(.2*xf)            )+.5)*255
                gre_ar[i,ary] =  (.5*np.sin(rtim11/10 + ary/(.2*yf) + i/(.3*xf) + 2*np.pi/3)+.5)*255
                blu_ar[i,ary] =  (.5*np.sin(rtim11/15 + ary/(.3*yf) + i/(.4*xf) + 4*np.pi/3)+.5)*255
#             if cy11>1:
#                 red_ar = np.fliplr(red_ar)
#                 if cy11>2:
#                     gre_ar = np.fliplr(gre_ar)
#                     if cy11>3:
#                         blu_ar = np.fliplr(blu_ar)
#                         if cy11>4:
#                             bdir=-1 #this will make us start stepping cy backwards
                            
            p[0,:] = (.5*np.sin(rtim11/20/4)+.5)          *coo11[0]*quadratize.flatMatQuads(red_ar)
            p[1,:] = (.5*np.sin(rtim11/30/4+2*np.pi/3)+.5)*coo11[1]*quadratize.flatMatQuads(gre_ar)
            p[2,:] = (.5*np.sin(rtim11/40/4+4*np.pi/3)+.5)*coo11[2]*quadratize.flatMatQuads(blu_ar)
            
            return p




