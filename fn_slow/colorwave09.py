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

class colorwave09:
    
    def colorwave09(y):
            global p, rtim11, pix, arx, ary, rtim31, coo11, cy11, bdir, red_ar, gre_ar, blu_ar, y_off11, x_off11
          
            
            rtim11+=1/y
            rtim31+=1/y
            
            num = 3*np.sin(rtim11/6)+8 + 1*np.cos(rtim11/50)
            
            n = 2*np.sin(rtim11/25)+1
            n2 =  2*np.cos(rtim11/25)+1
            xf = x_off11 + num
            yf = y_off11 - num
            f1 = .25*np.cos(rtim11/25)+1
            f2 = .25*np.cos(rtim11/25+np.pi/2)+1
            gr = .5*np.sin(rtim11/3)+.5
            
            for i in arx:
                if i%2==0:
                    red_ar[i,ary] =  gr*(.5*np.sin(rtim11/(4+n) + ary/yf*n + i/xf            )+.5)*255
                    gre_ar[i,ary] =  gr*(.5*np.sin(rtim11/(4+n) + ary/yf*n + i/xf + 2*np.pi/3*f1)+.5)*255
                    blu_ar[i,ary] =  gr*(.5*np.sin(rtim11/(4+n) + ary/yf*n + i/xf + 4*np.pi/3*f2)+.5)*255 #+ 4*np.pi/3
                else: 
                    red_ar[i,ary] =  (1-gr)*(.5*np.sin(rtim11/(4+n) + ary/yf*n + i/xf            )+.5)*255
                    gre_ar[i,ary] =  (1-gr)*(.5*np.sin(rtim11/(4+n) + ary/yf*n + i/xf + 2*np.pi/3*f1)+.5)*255
                    blu_ar[i,ary] =  (1-gr)*(.5*np.sin(rtim11/(4+n) + ary/yf*n + i/xf + 4*np.pi/3*f2)+.5)*255 #+ 4*np.pi/3
#             red_ar = 0
#             blu_ar = 0
#             gre_ar = 0
            #L = np.linspace(0,1300,1300/2).astype(int)
            p[0,:] = quadratize.flatMatQuads(red_ar)
            p[1,:] = quadratize.flatMatQuads(gre_ar)
            p[2,:] = quadratize.flatMatQuads(blu_ar)
           # p[:,L] = 0
            if rtim11>3000/y:
               rtim11=0
            return p



