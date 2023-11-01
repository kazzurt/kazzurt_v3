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
import PIL
from PIL import Image, ImageOps
from fn.rotatee import rotatee
#import vis_fresh
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

class colorwave7:
    
    def colorwave7(y):
            global p, rtim11, pix, arx, ary, rtim31, coo11, cy11, bdir, red_ar, gre_ar, blu_ar, y_off11, x_off11
       
            
            rtim11-=.5/y
            rtim31+=.5/y
            
            num = 4*np.sin(rtim11/25)+3

            xf = x_off11 + num
            yf = y_off11 - num
            
            for i in arx:
                blu_ar[i,ary] =  (.5*np.sin(rtim11/4 + ary/(.1*yf) + i/(.2*xf)            )+.5)*255
                red_ar[i,ary] =  (.5*np.sin(rtim11/4 + ary/(.2*yf) + i/(.3*xf) + 2*np.pi/3)+.5)*255
                gre_ar[i,ary] =  (.5*np.sin(rtim11/4 + ary/(.3*yf) + i/(.4*xf) + 5*np.pi/3)+.5)*255
            n = 2*np.sin(rtim11/10)+4
#             blu_ar = rotatee(blu_ar, rtim31, 2)
#             gre_ar = rotatee(gre_ar, rtim31, 2)
#             red_ar = rotatee(red_ar, rtim31, 2)
            
            p[0,:] = (.5*np.sin(rtim11/40/4)+.5)          *coo11[0]*quadratize.flatMatQuads(red_ar)
            p[1,:] = (.5*np.sin(rtim11/40/4+3*np.pi/3)+.5)*coo11[1]*quadratize.flatMatQuads(gre_ar)
            p[2,:] = (.5*np.sin(rtim11/40/4+n*np.pi/3)+.5)*coo11[2]*quadratize.flatMatQuads(blu_ar)
            
            return p


